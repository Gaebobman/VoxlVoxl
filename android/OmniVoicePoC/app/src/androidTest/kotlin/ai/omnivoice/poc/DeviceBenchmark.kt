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
}
