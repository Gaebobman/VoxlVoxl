package ai.omnivoice.poc

import android.os.Debug
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runners.MethodSorters
import java.io.File

/**
 * The on-device ladder from the brief, run head-less so results are scriptable:
 *
 *   Level 2  every ONNX session loads on the phone
 *   Level 3  reference + transcript + target text -> a real WAV
 *   Level 4  the same with radios off (enforced by the manifest having no
 *            INTERNET permission — see AndroidManifest.xml)
 *   Level 5  CPU / XNNPACK / NNAPI comparison
 *
 * Run:
 *   ./gradlew :app:connectedDebugAndroidTest
 *   adb shell am instrument -w -e class ai.omnivoice.poc.DeviceBenchmark \
 *       ai.omnivoice.poc.test/androidx.test.runner.AndroidJUnitRunner
 *
 * Parameters come from instrumentation args so a sweep needs no rebuild:
 *   -e steps 16  -e threads 6  -e backend CPU  -e text "..."
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class DeviceBenchmark {

    companion object {
        const val TAG = "OmniVoice.Bench"
    }

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val args = InstrumentationRegistry.getArguments()

    /**
     * Internal storage, as docs/architecture.md specifies. The app-specific
     * *external* dir looks writable to `adb push` but the app process cannot
     * read files `shell` created there — DAC permissions allow it and SELinux
     * does not (run-as works because it runs under a different context). Models
     * are therefore staged in /data/local/tmp and copied in via run-as; see
     * scripts/push_models.sh.
     */
    private val modelDir: File get() = File(ctx.filesDir, "models")
    private val outDir: File get() = File(ctx.getExternalFilesDir(null), "out").apply { mkdirs() }

    private fun arg(name: String, def: String) = args.getString(name) ?: def

    /** `-e qnn_opts "htp_arch=79,soc_model=0"` */
    private fun providerOptions(): Map<String, String> =
        arg("qnn_opts", "").split(",").mapNotNull {
            val kv = it.split("=", limit = 2)
            if (kv.size == 2 && kv[0].isNotBlank()) kv[0].trim() to kv[1].trim() else null
        }.toMap()

    private fun bench(tag: String, msg: String) = Log.i(TAG, "RESULT $tag $msg")

    /**
     * `-e text "..."` cannot carry spaces through `adb shell am instrument`, so
     * long inputs are read from a file instead: `-e textfile long.txt`, relative
     * to the model directory.
     */
    private fun inputText(default: String): String {
        val name = arg("textfile", "")
        if (name.isNotBlank()) {
            val f = File(modelDir, name)
            if (!f.isFile) throw IllegalStateException("textfile $name not found in $modelDir")
            return f.readText(Charsets.UTF_8).trim()
        }
        return arg("text", default)
    }

    @Test
    fun t01_modelsArePresent() {
        assertTrue("$modelDir missing — adb push models/android/", modelDir.isDirectory)
        var total = 0L
        for (n in listOf("omnivoice_lm.onnx", "omnivoice_lm.onnx.data",
                         "higgs_decoder.onnx", "tokenizer.json")) {
            val f = File(modelDir, n)
            assertTrue("$n missing", f.isFile)
            total += f.length()
            bench("file", "$n ${f.length()}")
        }
        bench("bundle_bytes", "$total")
    }

    /**
     * Level 4 — offline. Stronger than an airplane-mode run: the manifest
     * declares no INTERNET permission, so the kernel refuses this process a
     * socket at all. Airplane mode shows "the radios were off during this run";
     * this shows "this process cannot reach the network, ever".
     */
    @Test
    fun t06_cannotReachTheNetwork() {
        var threw: Exception? = null
        try {
            java.net.Socket().use { it.connect(java.net.InetSocketAddress("1.1.1.1", 53), 1500) }
        } catch (e: Exception) {
            threw = e
        }
        bench("offline", "socket_attempt=${threw?.javaClass?.simpleName ?: "SUCCEEDED"} " +
            "msg=${threw?.message?.take(80)}")
        assertTrue(
            "the app opened a socket — the INTERNET permission must have crept back in",
            threw != null,
        )

        val pm = ctx.packageManager
            .getPackageInfo(ctx.packageName, android.content.pm.PackageManager.GET_PERMISSIONS)
        val declared = pm.requestedPermissions?.toList() ?: emptyList()
        bench("permissions", declared.joinToString(",").ifEmpty { "(none)" })
        assertTrue("INTERNET is declared", declared.none { it == "android.permission.INTERNET" })
    }

    /** Thermal behaviour and drift across repeated generations. */
    @Test
    fun t07_thermalRuns() {
        val runs = arg("runs", "5").toInt()
        val steps = arg("steps", "16").toInt()
        val threads = arg("threads", "6").toInt()
        val vp7 = VoicePrompt.load(File(modelDir, "voice_prompt.bin"))
        val prompt = ai.omnivoice.poc.core.VoiceProfile(
            id = "bench", displayName = "bench", codes = vp7.codes,
            refText = vp7.refText, refRms = vp7.refRms, sampleRate = vp7.sampleRate)
        val power = ctx.getSystemService(android.os.PowerManager::class.java)

        OmniVoiceEngine(modelDir, Backend.CPU, threads,
                        cfg = GenConfig(numStep = steps)).use { engine ->
            engine.load()
            for (i in 1..runs) {
                val r = engine.synthesize(
                    text = inputText("오늘 회의를 시작하겠습니다."),
                    voiceProfile = prompt,
                    style = ai.omnivoice.poc.core.VoiceStyle(language = "ko"),
                )
                val thermal = runCatching { power.currentThermalStatus }.getOrDefault(-1)
                bench("thermal", "run=$i/$runs total_ms=${r.metrics.totalMillis} " +
                    "gen_ms=${r.metrics.generateMillis} " +
                    "rtf=${"%.3f".format(r.metrics.rtf)} thermal_status=$thermal")
            }
        }
    }

    /**
     * Level 5 evidence — which nodes an execution provider actually claimed.
     * Run one forward with profiling on and report the per-EP node counts;
     * "NNAPI is enabled" and "NNAPI is running the model" are different claims.
     */
    @Test
    fun t05_partitioning() {
        val backend = Backend.valueOf(arg("backend", "NNAPI"))
        val profDir = File(outDir, "profile").apply { mkdirs() }
        val lmFile = arg("model", "omnivoice_lm.onnx")
        OnnxModelRunner(modelDir, backend, arg("threads", "6").toInt(),
                        profileDir = profDir, lmFileName = lmFile,
                        providerOptions = providerOptions()).use { r ->
            val prompt = VoicePrompt.load(File(modelDir, "voice_prompt.bin"))
            val s = 188
            val ids = Array(OV.NUM_CODEBOOKS) { LongArray(s) }
            for (c in 0 until OV.NUM_CODEBOOKS) {
                for (t in 0 until prompt.numFrames) ids[c][37 + t] = prompt.codes[c][t].toLong()
                for (i in 140 until s) ids[c][i] = OV.AUDIO_MASK_ID.toLong()
            }
            r.forward(ids, BooleanArray(s) { it >= 37 })
            val path = r.endProfiling()
            bench("profile", "backend=$backend model=$lmFile file=$path")
        }
    }

    /** Level 2 — every session loads on the device. */
    @Test
    fun t02_sessionsLoad() {
        val backend = Backend.valueOf(arg("backend", "CPU"))
        val threads = arg("threads", "0").toInt()
        OnnxModelRunner(modelDir, backend, threads,
                        verbose = arg("verbose", "false").toBoolean(),
                        lmFileName = arg("model", "omnivoice_lm.onnx"),
                        providerOptions = providerOptions()).use { r ->
            r.lm
            r.vocoder
            bench("load", "backend=$backend threads=$threads " +
                "lm_ms=${r.lmLoadMillis} vocoder_ms=${r.vocoderLoadMillis}")
            assertTrue("backbone reported zero load time", r.lmLoadMillis >= 0)
        }
    }

    /** The Kotlin tokenizer must agree with the ids PyTorch used. */
    @Test
    fun t03_tokenizerMatchesReference() {
        val tok = QwenBpeTokenizer.fromFile(File(modelDir, "tokenizer.json"))
        assertEquals(listOf(151669), tok.encode(OV.TOK_DENOISE).toList())
        assertEquals(listOf(151674), tok.encode(OV.TOK_TEXT_START).toList())
        assertEquals(listOf(151675), tok.encode(OV.TOK_TEXT_END).toList())
        val ko = tok.encode("오늘 회의를 시작하겠습니다.")
        bench("tokenizer", "korean_ids=${ko.size}")
        assertTrue(ko.isNotEmpty())
    }

    /** Level 3 — full voice cloning on the device. */
    @Test
    fun t04_generate() {
        val promptFile = File(modelDir, "voice_prompt.bin")
        assertTrue("voice_prompt.bin missing — adb push it next to the models",
            promptFile.isFile)
        val vp = VoicePrompt.load(promptFile)
        val prompt = ai.omnivoice.poc.core.VoiceProfile(
            id = "bench", displayName = "bench", codes = vp.codes,
            refText = vp.refText, refRms = vp.refRms, sampleRate = vp.sampleRate)
        bench("prompt", "frames=${prompt.frames} " +
            "seconds=${"%.2f".format(prompt.frames.toFloat() / OV.FRAME_RATE)} rms=${prompt.refRms}")

        val backend = Backend.valueOf(arg("backend", "CPU"))
        val threads = arg("threads", "0").toInt()
        val steps = arg("steps", "16").toInt()
        val text = inputText("오늘 회의를 시작하겠습니다.")
        val cfg = GenConfig(numStep = steps)

        Runtime.getRuntime().gc()
        val before = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }

        val lmFile = arg("model", "omnivoice_lm.onnx")
        OmniVoiceEngine(modelDir, backend, threads,
                        verbose = arg("verbose", "false").toBoolean(),
                        lmFileName = lmFile, cfg = cfg,
                        thermalStatus = {
                            runCatching {
                                ctx.getSystemService(android.os.PowerManager::class.java)
                                    .currentThermalStatus
                            }.getOrDefault(-1)
                        }).use { engine ->
            engine.load()
            val r = engine.synthesize(
                text = text, voiceProfile = prompt,
                style = ai.omnivoice.poc.core.VoiceStyle(language = arg("language", "ko")),
                onProgress = { p ->
                    if (p.step % 4 == 0 || p.step == 1) {
                        Log.i(TAG, "  chunk ${p.chunk}/${p.totalChunks} step ${p.step}/" +
                            "${p.totalSteps}, ${p.cellsRemaining} cells masked")
                    }
                },
                deterministic = arg("deterministic", "false").toBoolean(),
            )

            val after = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
            val tag = lmFile.substringBefore('/').ifEmpty { "dyn" }
                .let { if (it.endsWith(".onnx")) "dyn" else it }
            val name = "gen_${tag}_${backend}_s${steps}_t${threads}.wav"
            WavIo.write(File(outDir, name), r.samples, r.sampleRate)
            // raw codes so the PC side can score this run with the same
            // re-masking judge used on the desktop outputs

            val m = r.metrics
            bench("generate",
                "backend=$backend threads=$threads steps=$steps " +
                    "S=${m.sequenceLength} frames=${m.targetFrames} chunks=${m.chunks} " +
                    "gen_ms=${m.generateMillis} voc_ms=${m.decodeMillis} " +
                    "post_ms=${m.postMillis} total_ms=${m.totalMillis} " +
                    "audio_s=${"%.3f".format(m.audioSeconds)} rtf=${"%.3f".format(m.rtf)} " +
                    "load_ms=${m.modelLoadMillis} thermal=${m.thermalStatus} " +
                    "pss_before_kb=${before.totalPss} pss_after_kb=${after.totalPss} " +
                    "native_kb=${after.nativePss} out=$name")

            assertTrue("no audio produced", r.samples.isNotEmpty())
            assertTrue("audio shorter than 0.3s", m.audioSeconds > 0.3)
        }
    }

    /** F-A01 / F-A02 — enrollment on the device itself. */
    @Test
    fun t08_enroll() {
        org.junit.Assume.assumeTrue(
            "encoder graphs absent — this build cannot enroll",
            OmniVoiceEnroller.available(modelDir))

        val wav = File(modelDir, arg("refwav", "reference.wav"))
        assertTrue("${wav.name} missing — push sample/reference.wav", wav.isFile)
        val audio = wav.inputStream().use { WavIo.read(it) }
        val refText = inputText("안녕하세요. 이것은 제 목소리를 등록하기 위한 테스트 음성입니다.")

        val threads = arg("threads", "6").toInt()
        val profile = OmniVoiceEnroller(modelDir, threads).use { enroller ->
            enroller.enroll(audio.samples, audio.sampleRate, refText, "On-device")
                .also { bench("enroll", "ms=${enroller.lastEncodeMillis} frames=${it.frames}") }
        }

        val mgr = FileVoiceProfileManager(File(ctx.filesDir, "voices"))
        mgr.save(profile)
        val back = mgr.get(profile.id)
        assertTrue("profile did not round-trip through storage", back != null)
        assertEquals(profile.frames, back!!.frames)
        bench("profiles", "count=${mgr.list().size} dir=${mgr.directory()}")

        // agreement against the profile the PC produced from the same clip
        val ref = VoicePrompt.load(File(modelDir, "voice_prompt.bin"))
        val n = minOf(ref.numFrames, profile.frames)
        var same = 0
        var cb0 = 0
        for (c in 0 until OV.NUM_CODEBOOKS) {
            for (t in 0 until n) {
                if (ref.codes[c][t] == profile.codes[c][t]) {
                    same++
                    if (c == 0) cb0++
                }
            }
        }
        bench("enroll_agreement",
            "frames=${profile.frames} vs_pc=${ref.numFrames} " +
                "all=${"%.2f".format(100.0 * same / (OV.NUM_CODEBOOKS * n))} " +
                "cb0=${"%.2f".format(100.0 * cb0 / n)}")
    }

    /** Cancellation must abort promptly and must not yield a partial clip. */
    @Test
    fun t09_cancel() {
        val vp = VoicePrompt.load(File(modelDir, "voice_prompt.bin"))
        val prompt = ai.omnivoice.poc.core.VoiceProfile(
            id = "bench", displayName = "bench", codes = vp.codes,
            refText = vp.refText, refRms = vp.refRms, sampleRate = vp.sampleRate)
        val cancelAfter = arg("cancel_after", "3").toInt()

        OmniVoiceEngine(modelDir, Backend.CPU, arg("threads", "6").toInt(),
                        cfg = GenConfig(numStep = 16)).use { engine ->
            engine.load()
            var steps = 0
            val t0 = System.nanoTime()
            var threw = false
            try {
                engine.synthesize(
                    text = inputText("오늘 회의를 시작하겠습니다."),
                    voiceProfile = prompt,
                    onProgress = { steps = it.step },
                    isActive = { steps < cancelAfter },
                )
            } catch (e: ai.omnivoice.poc.core.SynthesisCancelledException) {
                threw = true
            }
            val ms = (System.nanoTime() - t0) / 1_000_000
            bench("cancel", "requested_after_step=$cancelAfter observed_steps=$steps " +
                "elapsed_ms=$ms threw=$threw")
            assertTrue("cancellation did not throw", threw)
            assertTrue("cancelled too late: $steps steps", steps <= cancelAfter + 1)
        }
    }

    /** Long text must split, and the seams must not lose audio. */
    @Test
    fun t10_chunking() {
        val estimate = { t: String -> DurationEstimator.estimateFrames(t, "안녕하세요.", 103) }
        val short = TextChunker.chunk("오늘 회의를 시작하겠습니다.", estimate)
        assertEquals(1, short.size)

        val long = buildString { repeat(14) { append("오늘 회의를 시작하겠습니다. ") } }
        val parts = TextChunker.chunk(long, estimate)
        bench("chunking", "estimated_frames=${estimate(long)} chunks=${parts.size} " +
            "sizes=${parts.map { estimate(it) }}")
        assertTrue("long text was not split", parts.size > 1)

        val a = FloatArray(24_000) { 0.5f }
        val b = FloatArray(24_000) { 0.5f }
        val joined = TextChunker.crossFade(listOf(a, b), OV.SR_24K)
        assertEquals(a.size + b.size - (0.05f * OV.SR_24K).toInt(), joined.size)
        var dip = 1.0f
        for (v in joined) dip = minOf(dip, Math.abs(v))
        bench("crossfade", "len=${joined.size} min_abs=${"%.4f".format(dip)}")
        assertTrue("equal-power cross-fade dipped to $dip", dip > 0.45f)
    }

    /**
     * How fast can enrollment be VERIFIED?
     *
     * The spec (§6) wants a test sentence synthesised right after enrollment, but
     * that costs a full generation — ~20 s on top of the recording. There is a
     * much cheaper check available: decode the enrolled codes straight back
     * through the vocoder. That is literally what the model will hear as the
     * reference, so it catches a clipped recording, an over-trimmed one, or a
     * mis-levelled one — and it needs no LM pass at all.
     */
    @Test
    fun t11_enrollEcho() {
        org.junit.Assume.assumeTrue(OmniVoiceEnroller.available(modelDir))
        val wav = File(modelDir, "reference.wav")
        assertTrue("reference.wav missing", wav.isFile)
        val audio = wav.inputStream().use { WavIo.read(it) }

        val t0 = System.nanoTime()
        val profile = OmniVoiceEnroller(modelDir, 6).use {
            it.enroll(audio.samples, audio.sampleRate, "테스트", "echo")
        }
        val encodeMs = (System.nanoTime() - t0) / 1_000_000

        // vocoder only — no backbone
        val t1 = System.nanoTime()
        val codes = Array(OV.NUM_CODEBOOKS) { c ->
            LongArray(profile.frames) { t -> profile.codes[c][t].toLong() }
        }
        val echo = OnnxModelRunner(modelDir, Backend.CPU, 6).use { it.decodeCodes(codes) }
        val decodeMs = (System.nanoTime() - t1) / 1_000_000

        WavIo.write(File(outDir, "enroll_echo.wav"), echo, OV.SR_24K)
        bench("enroll_echo", "encode_ms=$encodeMs decode_ms=$decodeMs " +
            "total_ms=${encodeMs + decodeMs} in_s=${"%.2f".format(audio.samples.size.toFloat() / audio.sampleRate)} " +
            "out_s=${"%.2f".format(echo.size.toFloat() / OV.SR_24K)}")
        assertTrue("no echo audio", echo.isNotEmpty())
    }

    // =======================================================================
    // t12 — runtime-level levers (no change to the model's mathematics).
    //
    //   -e lever fusion     one batch-1 forward with a block-diagonal mask,
    //                       replacing the two separate CFG forwards
    //   -e lever sessopts   session-option variants against a fixed baseline
    //   -e lever optmodel   offline graph optimization: save once, reload
    //
    // Every variant is measured against a baseline re-measured at the END of
    // the run, because docs/benchmark.md §7.6 shows thermal drift is a larger
    // effect than anything measured here.
    // =======================================================================

    private class Prompt(
        val ids: Array<LongArray>, val mask: BooleanArray,
        val genStart: Int, val s: Int, val tGen: Int,
    )

    private fun buildPrompt(text: String): Prompt {
        val vp = VoicePrompt.load(File(modelDir, "voice_prompt.bin"))
        val tok = QwenBpeTokenizer.fromFile(File(modelDir, "tokenizer.json"))
        val tGen = DurationEstimator.estimateFrames(text, vp.refText, vp.numFrames, 1.0f)
        val styleText = OV.TOK_DENOISE + OV.TOK_LANG_START + "ko" + OV.TOK_LANG_END +
            OV.TOK_INSTRUCT_START + "None" + OV.TOK_INSTRUCT_END
        val textIds: IntArray = tok.encode(styleText) +
            tok.encodeWithNonverbal(OV.TOK_TEXT_START + TextUtils.combine(text, vp.refText) + OV.TOK_TEXT_END)
        val nText = textIds.size
        val tRef = vp.numFrames
        val s = nText + tRef + tGen
        val genStart = nText + tRef
        val ids = Array(OV.NUM_CODEBOOKS) { c ->
            LongArray(s) { i ->
                when {
                    i < nText -> textIds[i].toLong()
                    i < genStart -> vp.codes[c][i - nText].toLong()
                    else -> OV.AUDIO_MASK_ID.toLong()
                }
            }
        }
        return Prompt(ids, BooleanArray(s) { it >= nText }, genStart, s, tGen)
    }

    private fun longTensor(env: ai.onnxruntime.OrtEnvironment, v: LongArray, shape: LongArray) =
        ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.ByteBuffer.allocateDirect(v.size * 8)
                .order(java.nio.ByteOrder.nativeOrder()).asLongBuffer()
                .put(v).also { it.rewind() }, shape)

    private fun boolTensor(env: ai.onnxruntime.OrtEnvironment, v: ByteArray, shape: LongArray) =
        ai.onnxruntime.OnnxTensor.createTensor(env,
            java.nio.ByteBuffer.allocateDirect(v.size)
                .order(java.nio.ByteOrder.nativeOrder()).put(v).also { it.rewind() },
            shape, ai.onnxruntime.OnnxJavaType.BOOL)

    /** feeds for one forward over [rows] positions. */
    private fun feed(
        env: ai.onnxruntime.OrtEnvironment,
        ids: Array<LongArray>, mask: BooleanArray, attn: ByteArray, pos: LongArray,
    ): Map<String, ai.onnxruntime.OnnxTensor> {
        val s = mask.size
        val flat = LongArray(OV.NUM_CODEBOOKS * s)
        for (c in 0 until OV.NUM_CODEBOOKS) System.arraycopy(ids[c], 0, flat, c * s, s)
        return mapOf(
            "input_ids" to longTensor(env, flat, longArrayOf(1, OV.NUM_CODEBOOKS.toLong(), s.toLong())),
            "audio_mask" to boolTensor(env, ByteArray(s) { if (mask[it]) 1 else 0 }, longArrayOf(1, s.toLong())),
            "attention_mask" to boolTensor(env, attn, longArrayOf(1, 1, s.toLong(), s.toLong())),
            "position_ids" to longTensor(env, pos, longArrayOf(1, s.toLong())),
        )
    }

    private fun splitFeeds(env: ai.onnxruntime.OrtEnvironment, p: Prompt) = Pair(
        feed(env, p.ids, p.mask, ByteArray(p.s * p.s) { 1 }, LongArray(p.s) { it.toLong() }),
        feed(env,
            Array(OV.NUM_CODEBOOKS) { c -> p.ids[c].copyOfRange(p.genStart, p.s) },
            BooleanArray(p.tGen) { true },
            ByteArray(p.tGen * p.tGen) { 1 },
            LongArray(p.tGen) { it.toLong() }),
    )

    private fun concatFeeds(env: ai.onnxruntime.OrtEnvironment, p: Prompt):
        Map<String, ai.onnxruntime.OnnxTensor> {
        val n = p.s + p.tGen
        val ids = Array(OV.NUM_CODEBOOKS) { c -> p.ids[c] + p.ids[c].copyOfRange(p.genStart, p.s) }
        val mask = BooleanArray(n) { if (it < p.s) p.mask[it] else true }
        // block diagonal: the two branches must not see each other
        val attn = ByteArray(n * n)
        for (i in 0 until p.s) java.util.Arrays.fill(attn, i * n, i * n + p.s, 1.toByte())
        for (i in p.s until n) java.util.Arrays.fill(attn, i * n + p.s, i * n + n, 1.toByte())
        val pos = LongArray(n) { if (it < p.s) it.toLong() else (it - p.s).toLong() }
        return feed(env, ids, mask, attn, pos)
    }

    private fun runOnce(sess: ai.onnxruntime.OrtSession,
                        f: Map<String, ai.onnxruntime.OnnxTensor>): Pair<Long, FloatArray> {
        val t0 = System.nanoTime()
        sess.run(f).use { out ->
            val fb = (out[0] as ai.onnxruntime.OnnxTensor).floatBuffer
            val r = FloatArray(fb.remaining()); fb.get(r)
            return Pair((System.nanoTime() - t0) / 1_000_000, r)
        }
    }

    private fun sessionOptions(
        threads: Int, entries: Map<String, String> = emptyMap(),
        optimizedPath: String? = null,
    ): ai.onnxruntime.OrtSession.SessionOptions {
        val so = ai.onnxruntime.OrtSession.SessionOptions()
        so.setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.ALL_OPT)
        so.setExecutionMode(ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        if (threads > 0) so.setIntraOpNumThreads(threads)
        for ((k, v) in entries) so.addConfigEntry(k, v)
        if (optimizedPath != null) so.setOptimizedModelFilePath(optimizedPath)
        return so
    }

    @Test
    fun t12_levers() {
        val lever = arg("lever", "fusion")
        val threads = arg("threads", "6").toInt()
        val reps = arg("reps", "3").toInt()
        val text = inputText("오늘 회의를 시작하겠습니다.")
        val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
        val lmFile = File(modelDir, arg("model", "omnivoice_lm.onnx")).absolutePath
        val p = buildPrompt(text)
        bench("lever_setup", "lever=$lever S=${p.s} gen_start=${p.genStart} " +
            "t_gen=${p.tGen} threads=$threads reps=$reps thermal=${thermal()}")

        when (lever) {
            "fusion" -> leverFusion(env, lmFile, threads, reps, p)
            "kvcache" -> leverKvCache(env, threads, reps, p)
            "qnn" -> leverQnn(env, threads, reps, p)
            "sessopts" -> leverSessOpts(env, lmFile, threads, reps, p)
            "models" -> leverModels(env, threads, reps, p)
            "optmodel" -> leverOptModel(env, lmFile, threads, reps, p)
            "loadtime" -> {
                val opt = File(modelDir, "omnivoice_lm.opt.onnx")
                assertTrue("run -e lever optmodel first", opt.isFile)
                for (r in 0 until reps) {
                    for ((tag, f) in listOf("original" to lmFile, "optimized" to opt.absolutePath)) {
                        val t0 = System.nanoTime()
                        env.createSession(f, sessionOptions(threads)).close()
                        bench("lever_loadtime", "r=$r variant=$tag " +
                            "load_ms=${(System.nanoTime() - t0) / 1_000_000}")
                        System.gc()
                    }
                }
            }
            else -> throw IllegalArgumentException("unknown lever $lever")
        }
    }

    private fun thermal(): Int = try {
        val pm = ctx.getSystemService(android.os.PowerManager::class.java)
        pm.currentThermalStatus
    } catch (e: Throwable) { -1 }

    /**
     * Approximate prefix KV cache against the shipping graph, interleaved.
     *
     * The prefix -- style tokens, both transcripts, the reference codes -- is
     * identical on every step, so the KV graph computes its K/V once (prefill)
     * and each later conditional forward runs only the T generated positions
     * against that cache. The desktop could not answer whether this pays on the
     * shipping configuration: it has no VNNI, so its int8 compute path behaves
     * nothing like i8mm. Both sessions share one weight blob and are measured
     * round-robin with the order alternated, because DVFS moves a single forward
     * by 30 % between adjacent measurements.
     */
    private fun leverKvCache(
        env: ai.onnxruntime.OrtEnvironment, threads: Int, reps: Int, p: Prompt,
    ) {
        val layers = 28; val heads = 8; val dim = 128
        val plain = env.createSession(
            File(modelDir, arg("model", "omnivoice_lm.onnx")).absolutePath, sessionOptions(threads))
        val kv = env.createSession(
            File(modelDir, arg("kvmodel", "omnivoice_lm_kv.onnx")).absolutePath, sessionOptions(threads))
        val owned = ArrayList<ai.onnxruntime.OnnxTensor>()
        fun floatTensor(v: FloatArray, shape: LongArray) =
            ai.onnxruntime.OnnxTensor.createTensor(env,
                java.nio.ByteBuffer.allocateDirect(v.size * 4)
                    .order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer()
                    .put(v).also { it.rewind() }, shape).also { owned.add(it) }
        try {
            val pLen = p.genStart; val t = p.tGen; val s = p.s
            val emptyShape = longArrayOf(layers.toLong(), 1, heads.toLong(), 0, dim.toLong())
            val emptyK = floatTensor(FloatArray(0), emptyShape)
            val emptyV = floatTensor(FloatArray(0), emptyShape)
            val (fc, fu) = splitFeeds(env, p)
            val prefill = fc + mapOf("past_key" to emptyK, "past_value" to emptyV)
            val kvUncond = fu + mapOf("past_key" to emptyK, "past_value" to emptyV)

            // prefill once to build the cache, keeping only the prefix positions
            val block = s * dim; val keep = pLen * dim
            val (prefillLogits, pastK, pastV) = kv.run(prefill).use { out ->
                fun sliced(name: String): FloatArray {
                    val fb = (out.get(name).get() as ai.onnxruntime.OnnxTensor).floatBuffer
                    val full = FloatArray(fb.remaining()); fb.get(full)
                    val r = FloatArray(layers * heads * keep)
                    for (lh in 0 until layers * heads) System.arraycopy(full, lh * block, r, lh * keep, keep)
                    return r
                }
                val lfb = (out[0] as ai.onnxruntime.OnnxTensor).floatBuffer
                val lg = FloatArray(lfb.remaining()); lfb.get(lg)
                Triple(lg, sliced("present_key"), sliced("present_value"))
            }
            val pastShape = longArrayOf(layers.toLong(), 1, heads.toLong(), pLen.toLong(), dim.toLong())
            val genIds = LongArray(OV.NUM_CODEBOOKS * t)
            for (c in 0 until OV.NUM_CODEBOOKS) System.arraycopy(p.ids[c], pLen, genIds, c * t, t)
            val cached = mapOf(
                "input_ids" to longTensor(env, genIds, longArrayOf(1, OV.NUM_CODEBOOKS.toLong(), t.toLong())),
                "audio_mask" to boolTensor(env, ByteArray(t) { 1 }, longArrayOf(1, t.toLong())),
                "attention_mask" to boolTensor(env, ByteArray(t * (pLen + t)) { 1 },
                    longArrayOf(1, 1, t.toLong(), (pLen + t).toLong())),
                "position_ids" to longTensor(env, LongArray(t) { (pLen + it).toLong() }, longArrayOf(1, t.toLong())),
                "past_key" to floatTensor(pastK, pastShape),
                "past_value" to floatTensor(pastV, pastShape),
            )

            // correctness before timing: with the cache built from this very state
            // the cached forward should reproduce the plain graph's gen-region logits
            val v = OV.AUDIO_VOCAB_SIZE
            val plainC = runOnce(plain, fc).second
            val cachedC = runOnce(kv, cached).second
            fun compare(got: FloatArray, gotRows: Int, gotStart: Int): Pair<Float, Int> {
                var mx = 0f; var agree = 0
                for (c in 0 until OV.NUM_CODEBOOKS) for (i in 0 until t) {
                    val a0 = ((c * s) + pLen + i) * v
                    val b0 = ((c * gotRows) + gotStart + i) * v
                    var ba = 0; var bb = 0
                    var va = Float.NEGATIVE_INFINITY; var vb = Float.NEGATIVE_INFINITY
                    for (k in 0 until v) {
                        val x = plainC[a0 + k]; val y = got[b0 + k]
                        val d = Math.abs(x - y); if (d > mx) mx = d
                        if (x > va) { va = x; ba = k }
                        if (y > vb) { vb = y; bb = k }
                    }
                    if (ba == bb) agree++
                }
                return Pair(mx, agree)
            }
            val (dPre, aPre) = compare(prefillLogits, s, pLen)
            val (dCache, aCache) = compare(cachedC, t, 0)
            bench("lever_kvcache_correctness", "cells=${OV.NUM_CODEBOOKS * t} " +
                "prefill_vs_plain max_abs=$dPre argmax=$aPre " +
                "cached_vs_plain max_abs=$dCache argmax=$aCache")

            runOnce(plain, fu); runOnce(kv, kvUncond); runOnce(kv, prefill)   // warm
            val pc = ArrayList<Long>(); val pu = ArrayList<Long>()
            val kc = ArrayList<Long>(); val ku = ArrayList<Long>(); val kp = ArrayList<Long>()
            for (r in 0 until reps) {
                fun plainPair() {
                    pc.add(runOnce(plain, fc).first); pu.add(runOnce(plain, fu).first)
                }
                fun kvTrio() {
                    kc.add(runOnce(kv, cached).first); ku.add(runOnce(kv, kvUncond).first)
                    kp.add(runOnce(kv, prefill).first)
                }
                if (r % 2 == 0) { plainPair(); kvTrio() } else { kvTrio(); plainPair() }
                bench("lever_kvcache_round", "r=$r plain_c=${pc.last()} plain_u=${pu.last()} " +
                    "kv_c=${kc.last()} kv_u=${ku.last()} kv_prefill=${kp.last()} thermal=${thermal()}")
            }
            fun med(xs: List<Long>) = xs.sorted()[xs.size / 2].toDouble()
            val steps = arg("steps", "16").toInt()
            val mpc = med(pc); val mpu = med(pu); val mkc = med(kc); val mku = med(ku); val mkp = med(kp)
            val plainGen = steps * (mpc + mpu)
            // step 1's conditional forward IS the prefill; every later one rides the cache
            val cacheKvU = mkp + (steps - 1) * mkc + steps * mku
            val cachePlainU = mkp + (steps - 1) * mkc + steps * mpu
            bench("lever_kvcache", "S=$s P=$pLen T=$t steps=$steps " +
                "med_ms plain_c=$mpc plain_u=$mpu kv_c=$mkc kv_u=$mku kv_prefill=$mkp " +
                "gen_ms plain=${"%.0f".format(plainGen)} " +
                "cache_kv_uncond=${"%.0f".format(cacheKvU)} " +
                "cache_plain_uncond=${"%.0f".format(cachePlainU)} " +
                "speedup_kv_uncond=${"%.3f".format(plainGen / cacheKvU)} " +
                "speedup_plain_uncond=${"%.3f".format(plainGen / cachePlainU)}")
            (fc.values + fu.values + cached.values).toSet()
                .filter { it !in owned }.forEach { it.close() }
        } finally {
            owned.forEach { it.close() }
            plain.close(); kv.close()
        }
    }

    /**
     * The Hexagon NPU through QNN, against the CPU path that ships.
     *
     * Needs the static-shape QDQ graph from scripts/export_qnn.py (S = 188). The
     * first session compiles that graph for HTP -- measured 33.9 s -- and, with
     * `ep.context_enable`, writes a QNN context binary; later sessions load the
     * binary instead. The CPU side is the shipping KV graph: full forward
     * (prefill), cached forward, and the unconditional branch. The unconditional
     * branch (T = 48) has no static NPU graph here, so the NPU estimate keeps it
     * on CPU. Round-robin with the order alternated, as every lever here.
     */
    private fun leverQnn(
        env: ai.onnxruntime.OrtEnvironment, threads: Int, reps: Int, p: Prompt,
    ) {
        check(p.s == 188) { "the static QNN graph is S=188; this prompt is S=${p.s}" }
        val qdq = File(modelDir, arg("qnnmodel", "qnn/omnivoice_lm_s188.onnx"))
        val ctx = File(qdq.parentFile, qdq.nameWithoutExtension + "_ctx.onnx")
        fun qnn(so: ai.onnxruntime.OrtSession.SessionOptions) =
            so.also { it.addQnn(mapOf("backend_path" to "libQnnHtp.so")) }

        if (!ctx.isFile || arg("recompile", "false").toBoolean()) {
            ctx.parentFile?.listFiles()?.filter { it.name.startsWith(ctx.nameWithoutExtension) }
                ?.forEach { it.delete() }
            val so = qnn(sessionOptions(threads))
            so.addConfigEntry("ep.context_enable", "1")
            so.addConfigEntry("ep.context_file_path", ctx.absolutePath)
            so.addConfigEntry("ep.context_embed_mode", "0")
            val t0 = System.nanoTime()
            env.createSession(qdq.absolutePath, so).close()
            val written = ctx.parentFile?.listFiles()
                ?.filter { it.name.startsWith(ctx.nameWithoutExtension) } ?: emptyList()
            bench("lever_qnn_compile", "compile_ms=${(System.nanoTime() - t0) / 1_000_000} " +
                "files=${written.map { it.name + ":" + it.length() }}")
        }
        val fromCtx = ctx.isFile
        val tl = System.nanoTime()
        val npu = env.createSession((if (fromCtx) ctx else qdq).absolutePath, qnn(sessionOptions(threads)))
        bench("lever_qnn_load", "from=${if (fromCtx) "context_binary" else "qdq_graph"} " +
            "load_ms=${(System.nanoTime() - tl) / 1_000_000}")
        val kv = env.createSession(
            File(modelDir, arg("kvmodel", "omnivoice_lm.onnx")).absolutePath, sessionOptions(threads))

        val layers = 28; val heads = 8; val dim = 128
        val owned = ArrayList<ai.onnxruntime.OnnxTensor>()
        fun floatTensor(v: FloatArray, shape: LongArray) =
            ai.onnxruntime.OnnxTensor.createTensor(env,
                java.nio.ByteBuffer.allocateDirect(v.size * 4)
                    .order(java.nio.ByteOrder.nativeOrder()).asFloatBuffer()
                    .put(v).also { it.rewind() }, shape).also { owned.add(it) }
        try {
            val pLen = p.genStart; val t = p.tGen; val s = p.s
            val emptyShape = longArrayOf(layers.toLong(), 1, heads.toLong(), 0, dim.toLong())
            val emptyK = floatTensor(FloatArray(0), emptyShape)
            val emptyV = floatTensor(FloatArray(0), emptyShape)
            val (fc, fu) = splitFeeds(env, p)
            val prefill = fc + mapOf("past_key" to emptyK, "past_value" to emptyV)
            val kvUncond = fu + mapOf("past_key" to emptyK, "past_value" to emptyV)
            val block = s * dim; val keep = pLen * dim
            val (cpuLogits, pastK, pastV) = kv.run(prefill).use { out ->
                fun sliced(name: String): FloatArray {
                    val fb = (out.get(name).get() as ai.onnxruntime.OnnxTensor).floatBuffer
                    val full = FloatArray(fb.remaining()); fb.get(full)
                    val r = FloatArray(layers * heads * keep)
                    for (lh in 0 until layers * heads) System.arraycopy(full, lh * block, r, lh * keep, keep)
                    return r
                }
                val lfb = (out[0] as ai.onnxruntime.OnnxTensor).floatBuffer
                val lg = FloatArray(lfb.remaining()); lfb.get(lg)
                Triple(lg, sliced("present_key"), sliced("present_value"))
            }
            val pastShape = longArrayOf(layers.toLong(), 1, heads.toLong(), pLen.toLong(), dim.toLong())
            val genIds = LongArray(OV.NUM_CODEBOOKS * t)
            for (c in 0 until OV.NUM_CODEBOOKS) System.arraycopy(p.ids[c], pLen, genIds, c * t, t)
            val cached = mapOf(
                "input_ids" to longTensor(env, genIds, longArrayOf(1, OV.NUM_CODEBOOKS.toLong(), t.toLong())),
                "audio_mask" to boolTensor(env, ByteArray(t) { 1 }, longArrayOf(1, t.toLong())),
                "attention_mask" to boolTensor(env, ByteArray(t * (pLen + t)) { 1 },
                    longArrayOf(1, 1, t.toLong(), (pLen + t).toLong())),
                "position_ids" to longTensor(env, LongArray(t) { (pLen + it).toLong() }, longArrayOf(1, t.toLong())),
                "past_key" to floatTensor(pastK, pastShape),
                "past_value" to floatTensor(pastV, pastShape),
            )

            // how far the a16w8 NPU graph is from the int4 CPU graph, on one forward
            val t0 = System.nanoTime()
            val npuLogits = runOnce(npu, fc).second
            val firstRunMs = (System.nanoTime() - t0) / 1_000_000
            val v = OV.AUDIO_VOCAB_SIZE
            var mx = 0f; var agree = 0
            for (c in 0 until OV.NUM_CODEBOOKS) for (i in 0 until t) {
                val o0 = ((c * s) + pLen + i) * v
                var ba = 0; var bb = 0
                var va = Float.NEGATIVE_INFINITY; var vb = Float.NEGATIVE_INFINITY
                for (k in 0 until v) {
                    val a = cpuLogits[o0 + k]; val b = npuLogits[o0 + k]
                    val d = Math.abs(a - b); if (d > mx) mx = d
                    if (a > va) { va = a; ba = k }
                    if (b > vb) { vb = b; bb = k }
                }
                if (ba == bb) agree++
            }
            bench("lever_qnn_agreement", "first_npu_run_ms=$firstRunMs cells=${OV.NUM_CODEBOOKS * t} " +
                "argmax_vs_cpu=$agree max_abs=$mx")

            runOnce(npu, fc); runOnce(kv, cached); runOnce(kv, kvUncond)   // warm
            val nf = ArrayList<Long>(); val cf = ArrayList<Long>()
            val cc = ArrayList<Long>(); val cu = ArrayList<Long>()
            for (r in 0 until reps) {
                fun npuRun() { nf.add(runOnce(npu, fc).first) }
                fun cpuRun() {
                    cf.add(runOnce(kv, prefill).first); cc.add(runOnce(kv, cached).first)
                    cu.add(runOnce(kv, kvUncond).first)
                }
                if (r % 2 == 0) { npuRun(); cpuRun() } else { cpuRun(); npuRun() }
                bench("lever_qnn_round", "r=$r npu_full=${nf.last()} cpu_full=${cf.last()} " +
                    "cpu_cached=${cc.last()} cpu_uncond=${cu.last()} thermal=${thermal()}")
            }
            fun med(xs: List<Long>) = xs.sorted()[xs.size / 2].toDouble()
            val steps = arg("steps", "16").toInt()
            val mnf = med(nf); val mcf = med(cf); val mcc = med(cc); val mcu = med(cu)
            val cpuShip = mcf + (steps - 1) * mcc + steps * mcu
            val npuCond = steps * mnf + steps * mcu
            bench("lever_qnn", "S=$s steps=$steps med_ms npu_full=$mnf cpu_full=$mcf " +
                "cpu_cached=$mcc cpu_uncond=$mcu " +
                "gen_ms cpu_kv_shipping=${"%.0f".format(cpuShip)} " +
                "npu_cond_cpu_uncond=${"%.0f".format(npuCond)} " +
                "npu_vs_cpu_full=${"%.3f".format(mcf / mnf)} " +
                "npu_vs_shipping=${"%.3f".format(cpuShip / npuCond)}")
            (fc.values + fu.values + cached.values).toSet()
                .filter { it !in owned }.forEach { it.close() }
        } finally {
            owned.forEach { it.close() }
            npu.close(); kv.close()
        }
    }

    /** Two forwards vs one block-diagonal forward, interleaved to cancel drift. */
    private fun leverFusion(
        env: ai.onnxruntime.OrtEnvironment, lmFile: String,
        threads: Int, reps: Int, p: Prompt,
    ) {
        val sess = env.createSession(lmFile, sessionOptions(threads))
        try {
            val (fc, fu) = splitFeeds(env, p)
            val fx = concatFeeds(env, p)
            val v = OV.AUDIO_VOCAB_SIZE

            // warm + correctness
            val lc = runOnce(sess, fc).second
            val lu = runOnce(sess, fu).second
            val lx = runOnce(sess, fx).second
            val n = p.s + p.tGen
            var maxCond = 0.0f; var maxUnc = 0.0f; var agree = 0; var cells = 0
            for (c in 0 until OV.NUM_CODEBOOKS) {
                for (t in 0 until p.tGen) {
                    var bestA = 0; var bestB = 0
                    var va = Float.NEGATIVE_INFINITY; var vb = Float.NEGATIVE_INFINITY
                    for (i in 0 until v) {
                        val a = lc[((c * p.s) + p.genStart + t) * v + i]
                        val b = lx[((c * n) + p.genStart + t) * v + i]
                        val d = Math.abs(a - b); if (d > maxCond) maxCond = d
                        if (a > va) { va = a; bestA = i }
                        if (b > vb) { vb = b; bestB = i }
                        val ua = lu[((c * p.tGen) + t) * v + i]
                        val ub = lx[((c * n) + p.s + t) * v + i]
                        val du = Math.abs(ua - ub); if (du > maxUnc) maxUnc = du
                    }
                    cells++; if (bestA == bestB) agree++
                }
            }
            bench("lever_fusion_correctness",
                "max_abs_cond=$maxCond max_abs_uncond=$maxUnc argmax_agree=$agree/$cells")

            var splitMs = 0L; var fuseMs = 0L
            val splitAll = ArrayList<Long>(); val fuseAll = ArrayList<Long>()
            for (r in 0 until reps) {
                val a = runOnce(sess, fc).first + runOnce(sess, fu).first
                val b = runOnce(sess, fx).first
                splitMs += a; fuseMs += b; splitAll.add(a); fuseAll.add(b)
                bench("lever_fusion_round", "r=$r split_ms=$a fused_ms=$b thermal=${thermal()}")
            }
            bench("lever_fusion", "split_ms=${splitMs / reps} fused_ms=${fuseMs / reps} " +
                "speedup=${"%.3f".format(splitMs.toDouble() / fuseMs)} " +
                "split_all=$splitAll fused_all=$fuseAll")
            (fc.values + fu.values + fx.values).forEach { it.close() }
        } finally { sess.close() }
    }

    /**
     * Session-option variants. DVFS on this device moves a single forward by
     * 30 % between adjacent measurements, so a sequential A-then-B comparison
     * is worthless: both sessions are held open at once and the variants are
     * measured round-robin, which is the only way the delta survives the drift.
     */
    private fun leverSessOpts(
        env: ai.onnxruntime.OrtEnvironment, lmFile: String,
        threads: Int, reps: Int, p: Prompt,
    ) {
        // cpu0-5 are the 3.63 GHz Oryon cores, cpu6-7 the 4.74 GHz prime pair;
        // ORT affinity ids are 1-based and cover intra_op_num_threads-1 workers.
        val all = linkedMapOf(
            "baseline" to Pair(threads, emptyMap<String, String>()),
            "affinity_gold" to Pair(6, mapOf("session.intra_op_thread_affinities" to "2;3;4;5;6")),
            "affinity_all8" to Pair(8, mapOf("session.intra_op_thread_affinities" to "2;3;4;5;6;7;8")),
            "affinity_prime" to Pair(6, mapOf("session.intra_op_thread_affinities" to "4;5;6;7;8")),
            "dbb4" to Pair(threads, mapOf("session.dynamic_block_base" to "4")),
            "dbb4_affinity_gold" to Pair(6, mapOf(
                "session.dynamic_block_base" to "4",
                "session.intra_op_thread_affinities" to "2;3;4;5;6")),
            "dbb2" to Pair(threads, mapOf("session.dynamic_block_base" to "2")),
            "no_spinning" to Pair(threads, mapOf("session.intra_op.allow_spinning" to "0")),
            "threads5" to Pair(5, emptyMap()),
            "threads7" to Pair(7, emptyMap()),
        )
        val want = arg("variants", "baseline,dbb4").split(",").map { it.trim() }.filter { it.isNotEmpty() }
        val names = ArrayList<String>()
        val sessions = ArrayList<ai.onnxruntime.OrtSession>()
        for (n in want) {
            val (th, entries) = all[n] ?: throw IllegalArgumentException("unknown variant $n")
            val t0 = System.nanoTime()
            try {
                sessions.add(env.createSession(lmFile, sessionOptions(th, entries)))
            } catch (e: Throwable) {
                bench("lever_sessopt", "variant=$n FAILED ${e.javaClass.simpleName}: ${e.message}")
                continue
            }
            names.add(n)
            bench("lever_sessopt_load", "variant=$n threads=$th load_ms=${(System.nanoTime() - t0) / 1_000_000}")
        }
        try {
            val (fc, fu) = splitFeeds(env, p)
            val totals = LongArray(names.size)
            val perRound = Array(names.size) { ArrayList<Long>() }
            for (i in names.indices) { runOnce(sessions[i], fc); runOnce(sessions[i], fu) }
            for (r in 0 until reps) {
                for (i in names.indices) {
                    val ms = runOnce(sessions[i], fc).first + runOnce(sessions[i], fu).first
                    totals[i] += ms; perRound[i].add(ms)
                }
                bench("lever_sessopt_round", "r=$r " +
                    names.indices.joinToString(" ") { "${names[it]}=${perRound[it][r]}" } +
                    " thermal=${thermal()}")
            }
            val base = totals[0].toDouble()
            for (i in names.indices) {
                bench("lever_sessopt", "variant=${names[i]} step_ms=${totals[i] / reps} " +
                    "median=${perRound[i].sorted()[reps / 2]} " +
                    "speedup_vs_${names[0]}=${"%.3f".format(base / totals[i])} all=${perRound[i]}")
            }
            (fc.values + fu.values).forEach { it.close() }
        } finally { sessions.forEach { it.close() } }
    }

    /**
     * Two model FILES interleaved in one process. Used for the MatMulNBits
     * accuracy_level comparison: the two graphs differ only in that attribute
     * and share one byte-identical `omnivoice_lm.onnx.data`, so the delta is
     * the int4 GEMM compute path and nothing else.
     *
     *   -e lever models -e models omnivoice_lm.onnx,omnivoice_lm_acc4.onnx
     */
    private fun leverModels(
        env: ai.onnxruntime.OrtEnvironment, threads: Int, reps: Int, p: Prompt,
    ) {
        val files = arg("models", "omnivoice_lm.onnx,omnivoice_lm_acc4.onnx")
            .split(",").map { it.trim() }.filter { it.isNotEmpty() }
        val names = ArrayList<String>()
        val sessions = ArrayList<ai.onnxruntime.OrtSession>()
        bench("lever_models_env", "ort=${ai.onnxruntime.OrtEnvironment.getAvailableProviders()}")
        for (f in files) {
            val file = File(modelDir, f)
            assertTrue("$f not in $modelDir", file.isFile)
            val t0 = System.nanoTime()
            val sess = try {
                env.createSession(file.absolutePath, sessionOptions(threads))
            } catch (e: Throwable) {
                bench("lever_models", "model=$f LOAD_FAILED ${e.javaClass.simpleName}: ${e.message}")
                continue
            }
            names.add(f); sessions.add(sess)
            bench("lever_models_load", "model=$f load_ms=${(System.nanoTime() - t0) / 1_000_000} " +
                "inputs=${sess.inputNames} outputs=${sess.outputNames}")
        }
        try {
            val (fc, fu) = splitFeeds(env, p)
            val totals = LongArray(names.size)
            val perRound = Array(names.size) { ArrayList<Long>() }
            val first = arrayOfNulls<FloatArray>(names.size)
            for (i in names.indices) {
                runOnce(sessions[i], fc); first[i] = runOnce(sessions[i], fu).second
            }
            // numerical agreement between the two compute paths, on real logits
            if (names.size > 1 && first[0] != null && first[1] != null) {
                val a = first[0]!!; val b = first[1]!!
                var maxAbs = 0.0f; var agree = 0; var cells = 0
                val v = OV.AUDIO_VOCAB_SIZE
                for (c in 0 until OV.NUM_CODEBOOKS) for (t in 0 until p.tGen) {
                    var ba = 0; var bb = 0
                    var va = Float.NEGATIVE_INFINITY; var vb = Float.NEGATIVE_INFINITY
                    for (i in 0 until v) {
                        val x = a[((c * p.tGen) + t) * v + i]; val y = b[((c * p.tGen) + t) * v + i]
                        val d = Math.abs(x - y); if (d > maxAbs) maxAbs = d
                        if (x > va) { va = x; ba = i }; if (y > vb) { vb = y; bb = i }
                    }
                    cells++; if (ba == bb) agree++
                }
                bench("lever_models_agreement", "a=${names[0]} b=${names[1]} " +
                    "max_abs=$maxAbs argmax_agree=$agree/$cells")
            }
            for (r in 0 until reps) {
                for (i in names.indices) {
                    val ms = runOnce(sessions[i], fc).first + runOnce(sessions[i], fu).first
                    totals[i] += ms; perRound[i].add(ms)
                }
                bench("lever_models_round", "r=$r " +
                    names.indices.joinToString(" ") { "${names[it]}=${perRound[it][r]}" } +
                    " thermal=${thermal()}")
            }
            val base = totals[0].toDouble()
            for (i in names.indices) {
                val sorted = perRound[i].sorted()
                bench("lever_models", "model=${names[i]} mean_step_ms=${totals[i] / reps} " +
                    "median_step_ms=${sorted[reps / 2]} min_step_ms=${sorted.first()} " +
                    "speedup_vs_${names[0]}=${"%.3f".format(base / totals[i])} " +
                    "est_lm_s_16steps=${"%.1f".format(sorted[reps / 2] * 16 / 1000.0)} all=${perRound[i]}")
            }
            (fc.values + fu.values).forEach { it.close() }
        } finally { sessions.forEach { it.close() } }
    }

    /** Offline graph optimization: does saving the optimized graph cut load time? */
    private fun leverOptModel(
        env: ai.onnxruntime.OrtEnvironment, lmFile: String,
        threads: Int, reps: Int, p: Prompt,
    ) {
        // The optimized graph keeps the external-data REFERENCE (`*.onnx.data`,
        // resolved relative to the model file) and does not copy the 421 MB of
        // weights, so it has to be written next to the original or it cannot be
        // reopened. That also makes it cheap to ship: 0.7 MB beside the weights.
        val optDir = modelDir
        val optFile = File(optDir, "omnivoice_lm.opt.onnx")

        fun measure(path: String, tag: String, save: String? = null) {
            val t0 = System.nanoTime()
            val sess = env.createSession(path, sessionOptions(threads, optimizedPath = save))
            val loadMs = (System.nanoTime() - t0) / 1_000_000
            try {
                val (fc, fu) = splitFeeds(env, p)
                runOnce(sess, fc); runOnce(sess, fu)
                var c = 0L; var u = 0L
                for (r in 0 until reps) { c += runOnce(sess, fc).first; u += runOnce(sess, fu).first }
                bench("lever_optmodel", "variant=$tag load_ms=$loadMs cond_ms=${c / reps} " +
                    "unc_ms=${u / reps} step_ms=${(c + u) / reps} thermal=${thermal()}")
                (fc.values + fu.values).forEach { it.close() }
            } finally { sess.close(); System.gc() }
        }

        measure(lmFile, "original_first")
        if (!optFile.isFile) measure(lmFile, "original_saving", save = optFile.absolutePath)
        bench("lever_optmodel", "saved=${optFile.isFile} opt_bytes=${optFile.length()}")
        if (optFile.isFile) measure(optFile.absolutePath, "optimized")
        measure(lmFile, "original_again")
    }

    /**
     * End-to-end A/B of the fused CFG forward, alternating within one process so
     * DVFS drift falls out of the comparison. Deterministic + same seed, so the
     * two paths must produce byte-identical audio.
     */
    @Test
    fun t13_fusionEndToEnd() {
        val vp = VoicePrompt.load(File(modelDir, "voice_prompt.bin"))
        val profile = ai.omnivoice.poc.core.VoiceProfile(
            id = "bench", displayName = "bench", codes = vp.codes,
            refText = vp.refText, refRms = vp.refRms, sampleRate = vp.sampleRate)
        val steps = arg("steps", "16").toInt()
        val threads = arg("threads", "6").toInt()
        val rounds = arg("rounds", "2").toInt()
        val text = inputText("오늘 회의를 시작하겠습니다.")

        // ONE session, the flag flipped between calls: two engines mean two
        // 422 MB sessions and the device's memory manager kills the process.
        val engine = OmniVoiceEngine(modelDir, Backend.CPU, threads,
            cfg = GenConfig(numStep = steps), thermalStatus = { thermal() })
        try {
            engine.load()
            var sTot = 0L; var fTot = 0L; var sLm = 0L; var fLm = 0L
            var identical = true
            var samples = 0
            for (r in 0 until rounds) {
                // split first on even rounds, fused first on odd: whichever runs
                // first pays the colder-core advantage, so alternate the order
                val order = if (r % 2 == 0) listOf(false, true) else listOf(true, false)
                var a: ai.omnivoice.poc.core.AudioResult? = null
                var b: ai.omnivoice.poc.core.AudioResult? = null
                for (fuse in order) {
                    engine.fuseCfg = fuse
                    val res = engine.synthesize(text, profile, seed = 1234L, deterministic = true)
                    if (fuse) b = res else a = res
                }
                sTot += a!!.metrics.totalMillis; fTot += b!!.metrics.totalMillis
                sLm += a.metrics.generateMillis; fLm += b.metrics.generateMillis
                if (a.samples.size != b.samples.size ||
                    !a.samples.indices.all { a.samples[it] == b.samples[it] }) identical = false
                samples = a.samples.size
                bench("fusion_e2e_round", "r=$r order=${order.first()} " +
                    "split_total=${a.metrics.totalMillis} fused_total=${b.metrics.totalMillis} " +
                    "split_lm=${a.metrics.generateMillis} fused_lm=${b.metrics.generateMillis} " +
                    "split_rtf=${"%.2f".format(a.metrics.rtf)} " +
                    "fused_rtf=${"%.2f".format(b.metrics.rtf)} thermal=${thermal()}")
            }
            bench("fusion_e2e", "steps=$steps threads=$threads rounds=$rounds " +
                "split_lm_ms=${sLm / rounds} fused_lm_ms=${fLm / rounds} " +
                "lm_speedup=${"%.3f".format(sLm.toDouble() / fLm)} " +
                "split_total_ms=${sTot / rounds} fused_total_ms=${fTot / rounds} " +
                "total_speedup=${"%.3f".format(sTot.toDouble() / fTot)} " +
                "audio_identical=$identical samples=$samples")
            assertTrue("fused CFG changed the audio", identical)
        } finally { engine.close() }
    }
}
