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

    private fun bench(tag: String, msg: String) = Log.i(TAG, "RESULT $tag $msg")

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
        val prompt = VoicePrompt.load(File(modelDir, "voice_prompt.bin"))
        val power = ctx.getSystemService(android.os.PowerManager::class.java)

        OmniVoiceEngine(modelDir, Backend.CPU, threads).use { engine ->
            engine.preload()
            for (i in 1..runs) {
                val r = engine.generate(
                    text = arg("text", "오늘 회의를 시작하겠습니다."),
                    prompt = prompt, language = "ko", cfg = GenConfig(numStep = steps),
                )
                val thermal = runCatching { power.currentThermalStatus }.getOrDefault(-1)
                bench("thermal", "run=$i/$runs total_ms=${r.totalMillis} " +
                    "lm_ms=${r.lmMillis} rtf=${"%.3f".format(r.rtf)} thermal_status=$thermal")
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
        OnnxModelRunner(modelDir, backend, arg("threads", "6").toInt(),
                        profileDir = profDir).use { r ->
            val prompt = VoicePrompt.load(File(modelDir, "voice_prompt.bin"))
            val s = 188
            val ids = Array(OV.NUM_CODEBOOKS) { LongArray(s) }
            for (c in 0 until OV.NUM_CODEBOOKS) {
                for (t in 0 until prompt.numFrames) ids[c][37 + t] = prompt.codes[c][t].toLong()
                for (i in 140 until s) ids[c][i] = OV.AUDIO_MASK_ID.toLong()
            }
            r.forward(ids, BooleanArray(s) { it >= 37 })
            val path = r.endProfiling()
            bench("profile", "backend=$backend file=$path")
        }
    }

    /** Level 2 — every session loads on the device. */
    @Test
    fun t02_sessionsLoad() {
        val backend = Backend.valueOf(arg("backend", "CPU"))
        val threads = arg("threads", "0").toInt()
        OnnxModelRunner(modelDir, backend, threads,
                        verbose = arg("verbose", "false").toBoolean()).use { r ->
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
        val prompt = VoicePrompt.load(promptFile)
        bench("prompt", "frames=${prompt.numFrames} seconds=${"%.2f".format(prompt.durationSeconds)} " +
            "rms=${prompt.refRms}")

        val backend = Backend.valueOf(arg("backend", "CPU"))
        val threads = arg("threads", "0").toInt()
        val steps = arg("steps", "16").toInt()
        val text = arg("text", "오늘 회의를 시작하겠습니다.")
        val cfg = GenConfig(numStep = steps)

        Runtime.getRuntime().gc()
        val before = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }

        OmniVoiceEngine(modelDir, backend, threads,
                        verbose = arg("verbose", "false").toBoolean()).use { engine ->
            engine.preload()
            val r = engine.generate(
                text = text, prompt = prompt, language = "ko", cfg = cfg,
                deterministic = arg("deterministic", "false").toBoolean(),
            ) { step, total, left ->
                if (step % 4 == 0 || step == 1) Log.i(TAG, "  step $step/$total, $left cells masked")
            }

            val after = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
            val name = "gen_${backend}_s${steps}_t${threads}.wav"
            WavIo.write(File(outDir, name), r.waveform, r.sampleRate)
            // raw codes so the PC side can score this run with the same
            // re-masking judge used on the desktop outputs
            File(outDir, name.removeSuffix(".wav") + ".codes.bin").outputStream().use { os ->
                val bb = java.nio.ByteBuffer
                    .allocate(OV.NUM_CODEBOOKS * r.targetFrames * 2)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                for (c in 0 until OV.NUM_CODEBOOKS) {
                    for (t in 0 until r.targetFrames) bb.putShort(r.codes[c][t].toShort())
                }
                os.write(bb.array())
            }

            bench("generate",
                "backend=$backend threads=$threads steps=$steps " +
                    "S=${r.sequenceLength} frames=${r.targetFrames} " +
                    "lm_calls=${r.lmCalls} lm_ms=${r.lmMillis} voc_ms=${r.vocoderMillis} " +
                    "post_ms=${r.postMillis} total_ms=${r.totalMillis} " +
                    "audio_s=${"%.3f".format(r.audioSeconds)} rtf=${"%.3f".format(r.rtf)} " +
                    "load_lm_ms=${r.lmLoadMillis} load_voc_ms=${r.vocoderLoadMillis} " +
                    "pss_before_kb=${before.totalPss} pss_after_kb=${after.totalPss} " +
                    "native_kb=${after.nativePss} out=$name")

            assertTrue("no audio produced", r.waveform.isNotEmpty())
            assertTrue("audio shorter than 0.3s", r.audioSeconds > 0.3)
        }
    }
}
