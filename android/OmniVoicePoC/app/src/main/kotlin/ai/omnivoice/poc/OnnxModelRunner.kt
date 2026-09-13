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
 * nothing in the generated region can be cached. The prefix is the exception —
 * see [prefill] — when the graph was exported with `past_key`/`past_value` I/O.
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
        val feeds = inputs(inputIds, 0, s, audioMask, past = 0, pos0 = 0)
        // A KV-capable backbone still serves plain calls -- the unconditional
        // branch has no prefix to cache -- with an empty past.
        val empty = if (hasKvCache) pastTensor(emptyFloats(), 0) else null
        try {
            return runLogits(
                if (empty == null) feeds
                else feeds + mapOf("past_key" to empty, "past_value" to empty), s)
        } finally {
            feeds.values.forEach { it.close() }
            empty?.close()
        }
    }

    // ── approximate prefix KV cache ───────────────────────────────────────
    //
    // The prefix -- style tokens, both transcripts, the reference codes -- is
    // identical on every un-masking step, yet the plain graph re-encodes it on
    // every forward. A backbone exported with `past_key`/`past_value` I/O
    // encodes it once and runs later forwards over the generated positions
    // only. Measured on device: 1.93x on a short sentence, 1.30x on a long
    // one; the gain tracks the prefix share. research-notes.md §7.1.

    /** True when the backbone was exported with `export_onnx.py --kv-cache`. */
    val hasKvCache: Boolean by lazy { lm.inputInfo.containsKey("past_key") }

    /** `[layers, batch, kv_heads, past, head_dim]`, with -1 for the dynamic axes. */
    private val kvShape: LongArray by lazy {
        (lm.inputInfo.getValue("past_key").info as ai.onnxruntime.TensorInfo).shape
    }

    private var pastK: OnnxTensor? = null
    private var pastV: OnnxTensor? = null

    private fun emptyFloats() =
        ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asFloatBuffer()

    private fun pastTensor(data: java.nio.FloatBuffer, len: Int): OnnxTensor =
        OnnxTensor.createTensor(env, data,
            longArrayOf(kvShape[0], 1, kvShape[2], len.toLong(), kvShape[4]))

    /**
     * A full conditional forward that also keeps the prefix's K/V. Returns the
     * same `[8][S]` logits as [forward]; call it on the first step.
     */
    fun prefill(inputIds: Array<LongArray>, audioMask: BooleanArray, prefixLen: Int): FloatArray {
        check(hasKvCache) { "prefill needs a backbone exported with --kv-cache" }
        clearCache()
        val s = audioMask.size
        val feeds = inputs(inputIds, 0, s, audioMask, past = 0, pos0 = 0)
        val empty = pastTensor(emptyFloats(), 0)
        try {
            val t0 = System.nanoTime()
            lm.run(feeds + mapOf("past_key" to empty, "past_value" to empty)).use { out ->
                val fb = (out[0] as OnnxTensor).floatBuffer
                val logits = FloatArray(fb.remaining()); fb.get(logits)
                pastK = slicePrefix(out.get("present_key").get() as OnnxTensor, s, prefixLen)
                pastV = slicePrefix(out.get("present_value").get() as OnnxTensor, s, prefixLen)
                lmComputeNanos += System.nanoTime() - t0
                lmCalls++
                return logits
            }
        } catch (e: OutOfMemoryError) {
            throw OmniVoiceException(OmniVoiceException.Kind.OUT_OF_MEMORY, "OOM in backbone prefill (S=$s)")
        } catch (e: OrtException) {
            throw OmniVoiceException(OmniVoiceException.Kind.GENERATION_FAILED, "backbone prefill failed (S=$s)", e)
        } finally {
            feeds.values.forEach { it.close() }
            empty.close()
        }
    }

    /**
     * A conditional forward over the generated positions `[genStart, S)` only,
     * attending over the cached prefix. Returns `[8][T]` logits.
     */
    fun forwardCached(inputIds: Array<LongArray>, audioMask: BooleanArray, genStart: Int): FloatArray {
        val k = checkNotNull(pastK) { "forwardCached before prefill" }
        val v = checkNotNull(pastV)
        val q = audioMask.size - genStart
        val feeds = inputs(inputIds, genStart, q, audioMask, past = genStart, pos0 = genStart)
        try {
            return runLogits(feeds + mapOf("past_key" to k, "past_value" to v), q)
        } finally {
            feeds.values.forEach { it.close() }
        }
    }

    /** The cache is ~27 MB per tensor on a typical prompt; free it between chunks. */
    fun clearCache() {
        pastK?.close(); pastV?.close()
        pastK = null; pastV = null
    }

    private fun slicePrefix(present: OnnxTensor, s: Int, keep: Int): OnnxTensor {
        val lh = (kvShape[0] * kvShape[2]).toInt()
        val block = s * kvShape[4].toInt()
        val k = keep * kvShape[4].toInt()
        val src = present.floatBuffer
        val dst = ByteBuffer.allocateDirect(lh * k * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val row = FloatArray(k)
        for (i in 0 until lh) {
            src.position(i * block); src.get(row, 0, k); dst.put(row)
        }
        dst.rewind()
        return pastTensor(dst, keep)
    }

    /** The four backbone inputs for positions `[from, from+q)` attending over `past + q` keys. */
    private fun inputs(
        inputIds: Array<LongArray>, from: Int, q: Int, audioMask: BooleanArray,
        past: Int, pos0: Int,
    ): Map<String, OnnxTensor> {
        val idsBuf = ByteBuffer.allocateDirect(OV.NUM_CODEBOOKS * q * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        for (c in 0 until OV.NUM_CODEBOOKS) idsBuf.put(inputIds[c], from, q)
        idsBuf.rewind()

        val amBuf = boolBuffer(q)
        for (i in 0 until q) amBuf.put(if (audioMask[from + i]) 1 else 0)
        amBuf.rewind()

        // bidirectional: every query attends to every key, cached or fresh
        val kv = past + q
        val attnBuf = boolBuffer(q * kv)
        val ones = ByteArray(kv) { 1 }
        for (i in 0 until q) attnBuf.put(ones)
        attnBuf.rewind()

        val posBuf = ByteBuffer.allocateDirect(q * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        for (i in 0 until q) posBuf.put((pos0 + i).toLong())
        posBuf.rewind()

        return mapOf(
            "input_ids" to OnnxTensor.createTensor(env, idsBuf,
                longArrayOf(1, OV.NUM_CODEBOOKS.toLong(), q.toLong())),
            "audio_mask" to OnnxTensor.createTensor(env, amBuf,
                longArrayOf(1, q.toLong()), ai.onnxruntime.OnnxJavaType.BOOL),
            "attention_mask" to OnnxTensor.createTensor(env, attnBuf,
                longArrayOf(1, 1, q.toLong(), kv.toLong()), ai.onnxruntime.OnnxJavaType.BOOL),
            "position_ids" to OnnxTensor.createTensor(env, posBuf, longArrayOf(1, q.toLong())),
        )
    }

    private fun runLogits(feeds: Map<String, OnnxTensor>, rows: Int): FloatArray {
        try {
            val t0 = System.nanoTime()
            // Only the logits: a KV-capable graph also offers present_key/value,
            // and materialising those on every step would copy the whole cache out.
            lm.run(feeds, setOf("logits")).use { out ->
                val fb = (out[0] as OnnxTensor).floatBuffer
                val result = FloatArray(fb.remaining()); fb.get(result)
                lmComputeNanos += System.nanoTime() - t0
                lmCalls++
                return result
            }
        } catch (e: OutOfMemoryError) {
            throw OmniVoiceException(OmniVoiceException.Kind.OUT_OF_MEMORY, "OOM in backbone forward (rows=$rows)")
        } catch (e: OrtException) {
            throw OmniVoiceException(OmniVoiceException.Kind.GENERATION_FAILED, "backbone forward failed (rows=$rows)", e)
        }
    }

    /**
     * Both classifier-free-guidance branches in ONE forward.
     *
     * The conditional branch is the whole sequence; the unconditional branch is
     * the trailing MASK block alone. Running them as two sessions.run() calls
     * pays ORT's per-call fixed cost (thread-pool wake-up plus ~2 400 shape and
     * gather nodes whose cost does not scale with S) twice. Concatenating them
     * into one sequence of S + T with a BLOCK-DIAGONAL attention mask — so
     * neither branch can see the other — and per-branch position_ids is
     * arithmetically identical: verified bit-identical logits on PC and device
     * (docs/benchmark.md §7.10). Note this is not the same as batching the two
     * branches, which would have to pad the short one up to S and would cost
     * 2*S rows instead of S + T.
     *
     * Returns the logits over the concatenated sequence, `[8][S + T][1025]`
     * flattened; conditional cell (c, t) is at `((c * (S+T)) + genStart + t)`,
     * unconditional cell (c, t) at `((c * (S+T)) + S + t)`.
     */
    fun forwardFused(inputIds: Array<LongArray>, audioMask: BooleanArray,
                     genStart: Int): FloatArray {
        val s = audioMask.size
        val t = s - genStart
        val n = s + t

        val idsBuf = ByteBuffer.allocateDirect(OV.NUM_CODEBOOKS * n * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        for (c in 0 until OV.NUM_CODEBOOKS) {
            idsBuf.put(inputIds[c], 0, s)
            idsBuf.put(inputIds[c], genStart, t)
        }
        idsBuf.rewind()

        val amBuf = boolBuffer(n)
        for (i in 0 until s) amBuf.put(if (audioMask[i]) 1 else 0)
        for (i in 0 until t) amBuf.put(1)          // the MASK block is all audio
        amBuf.rewind()

        val attnBuf = boolBuffer(n * n)
        val condRow = ByteArray(n).also { java.util.Arrays.fill(it, 0, s, 1.toByte()) }
        val uncRow = ByteArray(n).also { java.util.Arrays.fill(it, s, n, 1.toByte()) }
        for (i in 0 until s) attnBuf.put(condRow)
        for (i in 0 until t) attnBuf.put(uncRow)
        attnBuf.rewind()

        val posBuf = ByteBuffer.allocateDirect(n * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        for (i in 0 until s) posBuf.put(i.toLong())
        for (i in 0 until t) posBuf.put(i.toLong())
        posBuf.rewind()

        val tIds = OnnxTensor.createTensor(env, idsBuf, longArrayOf(1, OV.NUM_CODEBOOKS.toLong(), n.toLong()))
        val tAm = OnnxTensor.createTensor(env, amBuf, longArrayOf(1, n.toLong()), ai.onnxruntime.OnnxJavaType.BOOL)
        val tAttn = OnnxTensor.createTensor(env, attnBuf, longArrayOf(1, 1, n.toLong(), n.toLong()), ai.onnxruntime.OnnxJavaType.BOOL)
        val tPos = OnnxTensor.createTensor(env, posBuf, longArrayOf(1, n.toLong()))

        try {
            val t0 = System.nanoTime()
            val out = lm.run(mapOf("input_ids" to tIds, "audio_mask" to tAm,
                                   "attention_mask" to tAttn, "position_ids" to tPos))
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
            throw OmniVoiceException(OmniVoiceException.Kind.OUT_OF_MEMORY, "OOM in fused forward (N=$n)")
        } catch (e: OrtException) {
            throw OmniVoiceException(OmniVoiceException.Kind.GENERATION_FAILED, "fused forward failed (N=$n)", e)
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
        clearCache()
        runCatching { lm.close() }
        runCatching { vocoder.close() }
    }
}
