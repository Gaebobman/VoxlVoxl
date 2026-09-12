package ai.omnivoice.poc

import android.util.Log
import java.io.File
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * The whole on-device pipeline. 1:1 with `scripts/infer_onnx.py`, which is the
 * specification and is itself validated against PyTorch.
 */
class OmniVoiceEngine(
    private val modelDir: File,
    backend: Backend = Backend.CPU,
    threads: Int = 0,
    verbose: Boolean = false,
    lmFileName: String = "omnivoice_lm.onnx",
) : AutoCloseable {

    companion object {
        const val TAG = "OmniVoice.Engine"
    }

    private val runner = OnnxModelRunner(modelDir, backend, threads, verbose,
        lmFileName = lmFileName)
    private val tokenizer: QwenBpeTokenizer by lazy {
        val f = File(modelDir, "tokenizer.json")
        if (!f.isFile) throw OmniVoiceException(
            OmniVoiceException.Kind.MODEL_MISSING, "tokenizer.json not found in $modelDir")
        QwenBpeTokenizer.fromFile(f)
    }

    data class Result(
        val waveform: FloatArray,
        val sampleRate: Int,
        val codes: Array<LongArray>,
        val sequenceLength: Int,
        val targetFrames: Int,
        val lmCalls: Int,
        val lmMillis: Long,
        val vocoderMillis: Long,
        val postMillis: Long,
        val totalMillis: Long,
        val lmLoadMillis: Long,
        val vocoderLoadMillis: Long,
    ) {
        val audioSeconds: Double get() = waveform.size.toDouble() / sampleRate
        val rtf: Double get() = if (audioSeconds > 0) totalMillis / 1000.0 / audioSeconds else Double.NaN
    }

    /** Warms both sessions so load time is not charged to the first generate. */
    fun preload() {
        runner.lm
        runner.vocoder
    }

    fun generate(
        text: String,
        prompt: VoicePrompt?,
        language: String? = "ko",
        instruct: String? = null,
        cfg: GenConfig = GenConfig(),
        frames: Int? = null,
        seed: Long = 1234L,
        deterministic: Boolean = false,
        onStep: ((step: Int, total: Int, cellsLeft: Int) -> Unit)? = null,
    ): Result {
        val wallStart = System.nanoTime()
        runner.resetCounters()
        val rng = Random(seed)
        val positionTemperature = if (deterministic) 0.0f else cfg.positionTemperature

        // --- target length -------------------------------------------------
        val tGen = frames ?: DurationEstimator.estimateFrames(
            text, prompt?.refText, prompt?.numFrames ?: 0)

        // --- prompt ---------------------------------------------------------
        val style = buildString {
            if (cfg.denoise && prompt != null) append(OV.TOK_DENOISE)
            append(OV.TOK_LANG_START).append(language ?: "None").append(OV.TOK_LANG_END)
            append(OV.TOK_INSTRUCT_START).append(instruct ?: "None").append(OV.TOK_INSTRUCT_END)
        }
        val textIds = tokenizer.encode(style) + tokenizer.encodeWithNonverbal(
            OV.TOK_TEXT_START + TextUtils.combine(text, prompt?.refText) + OV.TOK_TEXT_END)

        val nText = textIds.size
        val tRef = prompt?.numFrames ?: 0
        val s = nText + tRef + tGen
        val genStart = nText + tRef

        val inputIds = Array(OV.NUM_CODEBOOKS) { LongArray(s) }
        for (c in 0 until OV.NUM_CODEBOOKS) {
            for (i in 0 until nText) inputIds[c][i] = textIds[i].toLong()
            if (prompt != null) {
                for (t in 0 until tRef) inputIds[c][nText + t] = prompt.codes[c][t].toLong()
            }
            for (i in genStart until s) inputIds[c][i] = OV.AUDIO_MASK_ID.toLong()
        }
        // audio_mask covers the reference codes too — they are audio tokens, and
        // treating them as text is what breaks the community implementation.
        val audioMask = BooleanArray(s) { it >= nText }

        Log.i(TAG, "S=$s (text $nText + ref $tRef + gen $tGen), steps=${cfg.numStep}, " +
            "guidance=${cfg.guidanceScale}")

        // --- unconditional branch: the MASK block alone ----------------------
        val uIds = Array(OV.NUM_CODEBOOKS) { c -> inputIds[c].copyOfRange(genStart, s) }
        val uMask = BooleanArray(tGen) { true }

        // --- iterative un-masking -------------------------------------------
        val tokens = Array(OV.NUM_CODEBOOKS) { LongArray(tGen) { OV.AUDIO_MASK_ID.toLong() } }
        val schedule = UnmaskSchedule.schedule(cfg.numStep, cfg.tShift, tGen * OV.NUM_CODEBOOKS)
        val v = OV.AUDIO_VOCAB_SIZE
        val real = OV.AUDIO_MASK_ID   // 1024 real codes, index 1024 is MASK

        val logProbs = FloatArray(OV.NUM_CODEBOOKS * tGen * v)
        val pred = IntArray(OV.NUM_CODEBOOKS * tGen)
        val scores = FloatArray(OV.NUM_CODEBOOKS * tGen)

        for (step in schedule.indices) {
            val k = schedule[step]
            if (k <= 0) continue

            val cLogits = runner.forward(inputIds, audioMask)     // [1,8,S,1025]
            val uLogits = if (cfg.guidanceScale != 0.0f) runner.forward(uIds, uMask) else null

            for (c in 0 until OV.NUM_CODEBOOKS) {
                for (t in 0 until tGen) {
                    val cOff = ((c * s) + genStart + t) * v
                    val dst = ((c * tGen) + t) * v
                    if (uLogits == null) {
                        logSoftmaxInto(cLogits, cOff, logProbs, dst, v)
                    } else {
                        val uOff = ((c * tGen) + t) * v
                        logSoftmaxInto(cLogits, cOff, logProbs, dst, v)
                        val uTmp = FloatArray(v)
                        logSoftmaxInto(uLogits, uOff, uTmp, 0, v)
                        // cfg: log_softmax(c + guidance * (c - u))
                        for (i in 0 until v) {
                            val cl = logProbs[dst + i]
                            logProbs[dst + i] = cl + cfg.guidanceScale * (cl - uTmp[i])
                        }
                        logSoftmaxInPlace(logProbs, dst, v)
                    }
                    logProbs[dst + real] = Float.NEGATIVE_INFINITY   // never emit MASK

                    var best = 0
                    var bestV = Float.NEGATIVE_INFINITY
                    for (i in 0 until real) {
                        val x = logProbs[dst + i]
                        if (x > bestV) { bestV = x; best = i }
                    }
                    val cell = (c * tGen) + t
                    pred[cell] = best
                    var sc = bestV - c * cfg.layerPenaltyFactor
                    if (positionTemperature > 0f) {
                        val u = rng.nextFloat()
                        sc = sc / positionTemperature + (-ln(-ln(u + 1e-10f) + 1e-10f))
                    }
                    scores[cell] = if (tokens[c][t] != OV.AUDIO_MASK_ID.toLong()) {
                        Float.NEGATIVE_INFINITY
                    } else sc
                }
            }

            for (cell in topK(scores, k)) {
                val c = cell / tGen
                val t = cell % tGen
                tokens[c][t] = pred[cell].toLong()
            }

            var left = 0
            for (c in 0 until OV.NUM_CODEBOOKS) {
                for (t in 0 until tGen) {
                    inputIds[c][genStart + t] = tokens[c][t]
                    uIds[c][t] = tokens[c][t]
                    if (tokens[c][t] == OV.AUDIO_MASK_ID.toLong()) left++
                }
            }
            onStep?.invoke(step + 1, cfg.numStep, left)
        }

        var leftover = 0
        for (c in 0 until OV.NUM_CODEBOOKS) {
            for (t in 0 until tGen) {
                if (tokens[c][t] == OV.AUDIO_MASK_ID.toLong()) { tokens[c][t] = 0; leftover++ }
            }
        }
        if (leftover > 0) Log.w(TAG, "$leftover cells still masked after the loop; clamped to 0")

        // --- vocoder ---------------------------------------------------------
        val tVoc = System.nanoTime()
        val raw = runner.decodeCodes(tokens)
        val vocMillis = (System.nanoTime() - tVoc) / 1_000_000

        // --- post ------------------------------------------------------------
        val tPost = System.nanoTime()
        val wav = SilenceUtils.postProcess(raw, prompt?.refRms, cfg)
        val postMillis = (System.nanoTime() - tPost) / 1_000_000

        if (wav.isEmpty()) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.GENERATION_FAILED,
                "${"%.2f".format(raw.size.toDouble() / OV.SR_24K)}s of vocoder output was " +
                    "entirely below the -50 dBFS silence floor — the model produced silence. " +
                    "Most common cause: classifier-free guidance disabled.",
            )
        }

        val total = (System.nanoTime() - wallStart) / 1_000_000
        return Result(
            waveform = wav, sampleRate = OV.SR_24K, codes = tokens,
            sequenceLength = s, targetFrames = tGen,
            lmCalls = runner.lmCalls, lmMillis = runner.lmComputeNanos / 1_000_000,
            vocoderMillis = vocMillis, postMillis = postMillis, totalMillis = total,
            lmLoadMillis = runner.lmLoadMillis, vocoderLoadMillis = runner.vocoderLoadMillis,
        )
    }

    // --- helpers -----------------------------------------------------------

    private fun logSoftmaxInto(src: FloatArray, srcOff: Int, dst: FloatArray, dstOff: Int, n: Int) {
        var m = Float.NEGATIVE_INFINITY
        for (i in 0 until n) { val x = src[srcOff + i]; if (x > m) m = x }
        var sum = 0.0
        for (i in 0 until n) sum += exp((src[srcOff + i] - m).toDouble())
        val logSum = ln(sum).toFloat() + m
        for (i in 0 until n) dst[dstOff + i] = src[srcOff + i] - logSum
    }

    private fun logSoftmaxInPlace(a: FloatArray, off: Int, n: Int) {
        var m = Float.NEGATIVE_INFINITY
        for (i in 0 until n) { val x = a[off + i]; if (x > m) m = x }
        var sum = 0.0
        for (i in 0 until n) sum += exp((a[off + i] - m).toDouble())
        val logSum = ln(sum).toFloat() + m
        for (i in 0 until n) a[off + i] -= logSum
    }

    /** Indices of the [k] largest entries. Partial selection, not a full sort. */
    private fun topK(a: FloatArray, k: Int): IntArray {
        val n = a.size
        val kk = minOf(k, n)
        if (kk <= 0) return IntArray(0)
        val idx = (0 until n).sortedByDescending { a[it] }
        return IntArray(kk) { idx[it] }
    }

    override fun close() = runner.close()
}
