package ai.omnivoice.poc

import ai.omnivoice.poc.core.AudioResult
import ai.omnivoice.poc.core.SpeechSynthesizer
import ai.omnivoice.poc.core.SynthesisCancelledException
import ai.omnivoice.poc.core.SynthesisMetrics
import ai.omnivoice.poc.core.SynthesisProgress
import ai.omnivoice.poc.core.VoiceProfile
import ai.omnivoice.poc.core.VoiceStyle
import android.util.Log
import java.io.File
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * The OmniVoice implementation of [SpeechSynthesizer]. 1:1 with
 * `scripts/infer_onnx.py`, which is validated against PyTorch.
 */
class OmniVoiceEngine(
    private val modelDir: File,
    private val backend: Backend = Backend.CPU,
    private val threads: Int = 0,
    verbose: Boolean = false,
    lmFileName: String = "omnivoice_lm.onnx",
    private val cfg: GenConfig = GenConfig(),
    private val thermalStatus: () -> Int = { -1 },
    private val peakPssKb: () -> Int = { -1 },
) : SpeechSynthesizer {

    companion object {
        const val TAG = "OmniVoice.Engine"
    }

    override val backendName: String get() = "OmniVoice/$backend"

    private val runner = OnnxModelRunner(modelDir, backend, threads, verbose,
        lmFileName = lmFileName)

    private val tokenizer: QwenBpeTokenizer by lazy {
        val f = File(modelDir, "tokenizer.json")
        if (!f.isFile) throw OmniVoiceException(
            OmniVoiceException.Kind.MODEL_MISSING, "tokenizer.json not found in $modelDir")
        QwenBpeTokenizer.fromFile(f)
    }

    override fun load() {
        runner.lm
        runner.vocoder
    }

    override fun synthesize(
        text: String,
        voiceProfile: VoiceProfile?,
        style: VoiceStyle?,
        onProgress: ((SynthesisProgress) -> Unit)?,
        isActive: () -> Boolean,
    ): AudioResult = synthesize(text, voiceProfile, style, onProgress, isActive, 1234L, false)

    fun synthesize(
        text: String,
        voiceProfile: VoiceProfile?,
        style: VoiceStyle? = null,
        onProgress: ((SynthesisProgress) -> Unit)? = null,
        isActive: () -> Boolean = { true },
        seed: Long = 1234L,
        deterministic: Boolean = false,
    ): AudioResult {
        val wallStart = System.nanoTime()
        runner.resetCounters()

        val speed = style?.speed ?: 1.0f
        val refText = voiceProfile?.refText
        val refFrames = voiceProfile?.frames ?: 0

        // Chunk only when the sequence would get long enough for memory to bind.
        // Splitting earlier would COST time — see TextChunker's note.
        val chunks = TextChunker.chunk(
            text,
            estimate = { DurationEstimator.estimateFrames(it, refText, refFrames, speed) },
        )
        if (chunks.size > 1) {
            Log.i(TAG, "text split into ${chunks.size} chunks to bound peak memory")
        }

        val rng = Random(seed)
        val parts = ArrayList<FloatArray>(chunks.size)
        var totalFrames = 0
        var lastS = 0
        var decodeMillis = 0L

        for ((ci, chunkText) in chunks.withIndex()) {
            if (!isActive()) throw SynthesisCancelledException()
            val codes = generateCodes(
                chunkText, voiceProfile, style, cfg, rng, deterministic,
                chunk = ci + 1, totalChunks = chunks.size,
                onProgress = onProgress, isActive = isActive,
            ) { s -> lastS = s }
            totalFrames += codes[0].size

            val tv = System.nanoTime()
            parts.add(runner.decodeCodes(codes))
            decodeMillis += (System.nanoTime() - tv) / 1_000_000
        }

        val raw = TextChunker.crossFade(parts, OV.SR_24K)
        val tPost = System.nanoTime()
        val wav = SilenceUtils.postProcess(raw, voiceProfile?.refRms, cfg)
        val postMillis = (System.nanoTime() - tPost) / 1_000_000

        if (wav.isEmpty()) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.GENERATION_FAILED,
                "${"%.2f".format(raw.size.toDouble() / OV.SR_24K)}s of vocoder output was " +
                    "entirely below the -50 dBFS silence floor — the model produced silence, " +
                    "not a short clip. The usual cause is classifier-free guidance disabled.",
            )
        }

        val total = (System.nanoTime() - wallStart) / 1_000_000
        val metrics = SynthesisMetrics(
            sequenceLength = lastS,
            targetFrames = totalFrames,
            chunks = chunks.size,
            modelLoadMillis = runner.lmLoadMillis + runner.vocoderLoadMillis,
            generateMillis = runner.lmComputeNanos / 1_000_000,
            decodeMillis = decodeMillis,
            postMillis = postMillis,
            totalMillis = total,
            audioSeconds = wav.size.toDouble() / OV.SR_24K,
            peakPssKb = peakPssKb(),
            thermalStatus = thermalStatus(),
        )
        Log.i(TAG, "synthesized ${"%.2f".format(metrics.audioSeconds)}s in ${total}ms " +
            "(RTF ${"%.2f".format(metrics.rtf)}, ${runner.lmCalls} LM calls, " +
            "${chunks.size} chunk(s), thermal=${metrics.thermalStatus})")

        return AudioResult(wav, OV.SR_24K, metrics)
    }

    /** One chunk: prompt assembly + the CFG un-masking loop. */
    private fun generateCodes(
        text: String,
        profile: VoiceProfile?,
        style: VoiceStyle?,
        cfg: GenConfig,
        rng: Random,
        deterministic: Boolean,
        chunk: Int,
        totalChunks: Int,
        onProgress: ((SynthesisProgress) -> Unit)?,
        isActive: () -> Boolean,
        reportS: (Int) -> Unit,
    ): Array<LongArray> {
        val positionTemperature = if (deterministic) 0.0f else cfg.positionTemperature
        val speed = style?.speed ?: 1.0f
        val language = style?.language
        // Measured: an instruct is ignored whenever a clone prompt is present
        // (docs/feature-validation.md §4), so it is only emitted without one.
        val instruct = if (profile == null) style?.instruct else null
        if (profile != null && style?.instruct != null) {
            Log.w(TAG, "instruct='${style.instruct}' dropped: the voice profile overrides " +
                "voice-design attributes. Use a differently-recorded profile instead.")
        }

        val tGen = DurationEstimator.estimateFrames(text, profile?.refText, profile?.frames ?: 0, speed)

        val styleText = buildString {
            if (cfg.denoise && profile != null) append(OV.TOK_DENOISE)
            append(OV.TOK_LANG_START).append(language ?: "None").append(OV.TOK_LANG_END)
            append(OV.TOK_INSTRUCT_START).append(instruct ?: "None").append(OV.TOK_INSTRUCT_END)
        }
        val textIds = tokenizer.encode(styleText) + tokenizer.encodeWithNonverbal(
            OV.TOK_TEXT_START + TextUtils.combine(text, profile?.refText) + OV.TOK_TEXT_END)

        val nText = textIds.size
        val tRef = profile?.frames ?: 0
        val s = nText + tRef + tGen
        val genStart = nText + tRef
        reportS(s)

        val inputIds = Array(OV.NUM_CODEBOOKS) { LongArray(s) }
        for (c in 0 until OV.NUM_CODEBOOKS) {
            for (i in 0 until nText) inputIds[c][i] = textIds[i].toLong()
            if (profile != null) {
                for (t in 0 until tRef) inputIds[c][nText + t] = profile.codes[c][t].toLong()
            }
            for (i in genStart until s) inputIds[c][i] = OV.AUDIO_MASK_ID.toLong()
        }
        // The reference codes are audio, so audio_mask covers them too. Treating
        // them as text is what breaks the community implementation.
        val audioMask = BooleanArray(s) { it >= nText }

        Log.i(TAG, "chunk $chunk/$totalChunks: S=$s (text $nText + ref $tRef + gen $tGen), " +
            "steps=${cfg.numStep}, guidance=${cfg.guidanceScale}, lang=${language ?: "None"}")

        val uIds = Array(OV.NUM_CODEBOOKS) { c -> inputIds[c].copyOfRange(genStart, s) }
        val uMask = BooleanArray(tGen) { true }

        val tokens = Array(OV.NUM_CODEBOOKS) { LongArray(tGen) { OV.AUDIO_MASK_ID.toLong() } }
        val schedule = UnmaskSchedule.schedule(cfg.numStep, cfg.tShift, tGen * OV.NUM_CODEBOOKS)
        val v = OV.AUDIO_VOCAB_SIZE
        val real = OV.AUDIO_MASK_ID

        val logProbs = FloatArray(OV.NUM_CODEBOOKS * tGen * v)
        val uTmp = FloatArray(v)
        val pred = IntArray(OV.NUM_CODEBOOKS * tGen)
        val scores = FloatArray(OV.NUM_CODEBOOKS * tGen)

        for (step in schedule.indices) {
            // Cancellation is checked between steps rather than inside them: one
            // step is 0.5-2 s, which is a responsive enough granularity, and the
            // forward itself is a single opaque ORT call.
            if (!isActive()) throw SynthesisCancelledException()
            val k = schedule[step]
            if (k <= 0) continue

            val cLogits = runner.forward(inputIds, audioMask)
            val uLogits = if (cfg.guidanceScale != 0.0f) runner.forward(uIds, uMask) else null

            for (c in 0 until OV.NUM_CODEBOOKS) {
                for (t in 0 until tGen) {
                    val cOff = ((c * s) + genStart + t) * v
                    val dst = ((c * tGen) + t) * v
                    logSoftmaxInto(cLogits, cOff, logProbs, dst, v)
                    if (uLogits != null) {
                        logSoftmaxInto(uLogits, ((c * tGen) + t) * v, uTmp, 0, v)
                        for (i in 0 until v) {
                            val cl = logProbs[dst + i]
                            logProbs[dst + i] = cl + cfg.guidanceScale * (cl - uTmp[i])
                        }
                        logSoftmaxInPlace(logProbs, dst, v)
                    }
                    logProbs[dst + real] = Float.NEGATIVE_INFINITY

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
                    scores[cell] =
                        if (tokens[c][t] != OV.AUDIO_MASK_ID.toLong()) Float.NEGATIVE_INFINITY
                        else sc
                }
            }

            for (cell in topK(scores, k)) {
                tokens[cell / tGen][cell % tGen] = pred[cell].toLong()
            }

            var left = 0
            for (c in 0 until OV.NUM_CODEBOOKS) {
                for (t in 0 until tGen) {
                    inputIds[c][genStart + t] = tokens[c][t]
                    uIds[c][t] = tokens[c][t]
                    if (tokens[c][t] == OV.AUDIO_MASK_ID.toLong()) left++
                }
            }
            onProgress?.invoke(
                SynthesisProgress(chunk, totalChunks, step + 1, cfg.numStep, left))
        }

        var leftover = 0
        for (c in 0 until OV.NUM_CODEBOOKS) {
            for (t in 0 until tGen) {
                if (tokens[c][t] == OV.AUDIO_MASK_ID.toLong()) { tokens[c][t] = 0; leftover++ }
            }
        }
        if (leftover > 0) Log.w(TAG, "$leftover cells still masked after the loop; clamped to 0")
        return tokens
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

    /** Indices of the [k] largest entries; partial selection, not a full sort. */
    private fun topK(a: FloatArray, k: Int): IntArray {
        val n = a.size
        val kk = minOf(k, n)
        if (kk <= 0) return IntArray(0)
        val idx = (0 until n).sortedByDescending { a[it] }
        return IntArray(kk) { idx[it] }
    }

    override fun close() = runner.close()
}
