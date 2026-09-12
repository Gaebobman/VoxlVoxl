package ai.omnivoice.poc

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.EnumSet

/**
 * Which execution provider to register. CPU is the correctness baseline.
 *
 * QNN targets the Hexagon NPU on Snapdragon parts and needs a model built for
 * it — static shapes and QDQ activations, not the weight-only int4 the CPU path
 * uses (docs/benchmark.md §7.5). Pointing it at the CPU model is expected to
 * offload nothing.
 */
enum class Backend { CPU, XNNPACK, NNAPI, QNN }

/**
 * Owns the ONNX Runtime environment and the sessions.
 *
 * The backbone takes a 4-D bidirectional attention mask, so unlike a normal LLM
 * there is no KV cache to manage and the sequence length is constant within one
 * utterance — which is why the input tensors are allocated once per utterance
 * and rewritten in place across the 32 decoding steps.
 */
class OnnxModelRunner(
    private val modelDir: File,
    private val backend: Backend = Backend.CPU,
    private val threads: Int = 0,
    /**
     * Turns on ORT's own VERBOSE log, which is the only way to see graph
     * partitioning — how many nodes an EP actually claimed versus fell back to
     * CPU. "NNAPI is enabled" and "NNAPI is running the model" are different
     * claims and the brief asks for the second one.
     */
    private val verbose: Boolean = false,
    /**
     * Directory for ORT's profiling JSON. ORT's Android AAR does not route its
     * own log to logcat, so the verbose partitioning report is invisible on
     * device; the profile is better evidence anyway -- it records the execution
     * provider for every node that actually ran.
     */
    private val profileDir: File? = null,
    /**
     * Relative to [modelDir]. A variant in a subdirectory works because ORT
     * resolves `.onnx.data` relative to the model file, not the process CWD —
     * which is also why two variants cannot share one directory: both record
     * the same external-data filename.
     */
    private val lmFileName: String = "omnivoice_lm.onnx",
    /** Extra provider options, e.g. QNN's `backend_path` / `htp_arch` / `soc_model`. */
    private val providerOptions: Map<String, String> = emptyMap(),
) : AutoCloseable {

    companion object {
        const val TAG = "OmniVoice.Onnx"
    }

    // OrtEnvironment is a process singleton whose log level is fixed at first
    // use, and it defaults to WARNING -- which silently swallows the
    // partitioning report. It has to be raised here, before anything else
    // touches ORT, or --verbose does nothing.
    private val env: OrtEnvironment = if (verbose) {
        OrtEnvironment.getEnvironment(
            ai.onnxruntime.OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE, "omnivoice")
    } else {
        OrtEnvironment.getEnvironment()
    }

    var lmLoadMillis: Long = 0; private set
    var vocoderLoadMillis: Long = 0; private set
    var lmCalls: Int = 0; private set
    var lmComputeNanos: Long = 0; private set

    private fun options(): OrtSession.SessionOptions {
        val so = OrtSession.SessionOptions()
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (verbose) {
            so.setSessionLogLevel(ai.onnxruntime.OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
            so.setSessionLogVerbosityLevel(1)
        }
        if (profileDir != null) {
            profileDir.mkdirs()
            so.enableProfiling(File(profileDir, "ort_${backend}_").absolutePath)
        }
        so.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        if (threads > 0) so.setIntraOpNumThreads(threads)
        when (backend) {
            Backend.CPU -> Unit
            Backend.XNNPACK -> so.addXnnpack(mapOf("intra_op_num_threads" to threads.coerceAtLeast(1).toString()))
            Backend.NNAPI -> so.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            Backend.QNN -> {
                val opts = HashMap<String, String>()
                opts["backend_path"] = "libQnnHtp.so"
                opts.putAll(providerOptions)
                Log.i(TAG, "QNN provider options: $opts")
                so.addQnn(opts)
            }
        }
        return so
    }

    private fun open(name: String): OrtSession {
        val f = File(modelDir, name)
        if (!f.isFile) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.MODEL_MISSING,
                "$name not found in $modelDir — push models/android/ to the device",
            )
        }
        return try {
            env.createSession(f.absolutePath, options())
        } catch (e: OrtException) {
            val kind = if (e.message?.contains("op", ignoreCase = true) == true &&
                e.message?.contains("not", ignoreCase = true) == true
            ) OmniVoiceException.Kind.UNSUPPORTED_OP else OmniVoiceException.Kind.MODEL_LOAD_FAILED
            throw OmniVoiceException(kind, "failed to load $name with backend=$backend", e)
        } catch (e: OutOfMemoryError) {
            throw OmniVoiceException(OmniVoiceException.Kind.OUT_OF_MEMORY, "OOM loading $name")
        }
    }

    /**
     * Loaded lazily and kept: the backbone is 422 MB and re-opening it per
     * utterance would dominate everything else.
     */
    val lm: OrtSession by lazy {
        val t0 = System.nanoTime()
        // ORT resolves .onnx.data relative to the model file, so both must sit
        // in the same directory — the reason models live in filesDir, not assets.
        val s = open(lmFileName)
        lmLoadMillis = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "backbone loaded in ${lmLoadMillis}ms " +
            "(file=$lmFileName, backend=$backend, threads=$threads)")
        describe(lmFileName, s)
        s
    }

    val vocoder: OrtSession by lazy {
        val t0 = System.nanoTime()
        val s = open("higgs_decoder.onnx")
        vocoderLoadMillis = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "vocoder loaded in ${vocoderLoadMillis}ms")
        describe("higgs_decoder", s)
        s
    }

    private fun describe(name: String, s: OrtSession) {
        for ((k, v) in s.inputInfo) Log.i(TAG, "  $name IN  $k : ${v.info}")
        for ((k, v) in s.outputInfo) Log.i(TAG, "  $name OUT $k : ${v.info}")
    }

    private fun boolBuffer(n: Int): ByteBuffer =
        ByteBuffer.allocateDirect(n).order(ByteOrder.nativeOrder())

    /**
     * One backbone forward. [inputIds] is `[8][S]`, [audioMask] is `[S]`.
     * Attention is fully bidirectional over the whole sequence, so the 4-D mask
     * is all-true; it stays an explicit input because the graph would otherwise
     * have no way to express non-causal attention.
     */
    fun forward(inputIds: Array<LongArray>, audioMask: BooleanArray): FloatArray {
        val s = audioMask.size
        val idsBuf = ByteBuffer.allocateDirect(OV.NUM_CODEBOOKS * s * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        for (c in 0 until OV.NUM_CODEBOOKS) idsBuf.put(inputIds[c], 0, s)
        idsBuf.rewind()

        val amBuf = boolBuffer(s)
        for (i in 0 until s) amBuf.put(if (audioMask[i]) 1 else 0)
        amBuf.rewind()

        val attnBuf = boolBuffer(s * s)
        val ones = ByteArray(s) { 1 }
        for (i in 0 until s) attnBuf.put(ones)
        attnBuf.rewind()

        val posBuf = ByteBuffer.allocateDirect(s * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        for (i in 0 until s) posBuf.put(i.toLong())
        posBuf.rewind()

        val tIds = OnnxTensor.createTensor(env, idsBuf, longArrayOf(1, OV.NUM_CODEBOOKS.toLong(), s.toLong()))
        val tAm = OnnxTensor.createTensor(env, amBuf, longArrayOf(1, s.toLong()), ai.onnxruntime.OnnxJavaType.BOOL)
        val tAttn = OnnxTensor.createTensor(env, attnBuf, longArrayOf(1, 1, s.toLong(), s.toLong()), ai.onnxruntime.OnnxJavaType.BOOL)
        val tPos = OnnxTensor.createTensor(env, posBuf, longArrayOf(1, s.toLong()))

        try {
            val t0 = System.nanoTime()
            val out = lm.run(
                mapOf(
                    "input_ids" to tIds,
                    "audio_mask" to tAm,
                    "attention_mask" to tAttn,
                    "position_ids" to tPos,
                )
            )
            try {
                val logits = (out[0] as OnnxTensor).floatBuffer
                val result = FloatArray(logits.remaining())
                logits.get(result)
                lmComputeNanos += System.nanoTime() - t0
                lmCalls++
                return result
            } finally {
                out.close()
            }
        } catch (e: OutOfMemoryError) {
            throw OmniVoiceException(OmniVoiceException.Kind.OUT_OF_MEMORY, "OOM in backbone forward (S=$s)")
        } catch (e: OrtException) {
            throw OmniVoiceException(OmniVoiceException.Kind.GENERATION_FAILED, "backbone forward failed (S=$s)", e)
        } finally {
            tIds.close(); tAm.close(); tAttn.close(); tPos.close()
        }
    }

    /** `codes` is `[8][T]`; returns a 24 kHz waveform. */
    fun decodeCodes(codes: Array<LongArray>): FloatArray {
        val t = codes[0].size
        val buf = ByteBuffer.allocateDirect(OV.NUM_CODEBOOKS * t * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        for (c in 0 until OV.NUM_CODEBOOKS) buf.put(codes[c], 0, t)
        buf.rewind()
        val tensor = OnnxTensor.createTensor(
            env, buf, longArrayOf(OV.NUM_CODEBOOKS.toLong(), 1, t.toLong()))
        try {
            val out = vocoder.run(mapOf("codes" to tensor))
            try {
                val fb = (out[0] as OnnxTensor).floatBuffer
                val wav = FloatArray(fb.remaining())
                fb.get(wav)
                return wav
            } finally {
                out.close()
            }
        } catch (e: OrtException) {
            throw OmniVoiceException(OmniVoiceException.Kind.GENERATION_FAILED, "vocoder failed (T=$t)", e)
        } finally {
            tensor.close()
        }
    }

    fun resetCounters() {
        lmCalls = 0
        lmComputeNanos = 0
    }

    /** Flushes the profile and returns where ORT wrote it, if profiling is on. */
    fun endProfiling(): String? =
        if (profileDir == null) null else runCatching { lm.endProfiling() }.getOrNull()

    override fun close() {
        runCatching { lm.close() }
        runCatching { vocoder.close() }
    }
}
