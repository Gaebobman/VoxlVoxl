package ai.omnivoice.poc

import kotlin.math.log10
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Port of omnivoice/utils/audio.py, which delegates to pydub (MIT).
 *
 * Reimplemented because the device has no pydub, and approximating it is not an
 * option: a hand-rolled RMS gate trimmed the reference clip to 104 frames
 * instead of 103 and codec agreement against the reference prompt collapsed from
 * 92 % to 2.55 %. The Python side of this port is bit-exact against pydub
 * (`validate_onnx.py --stage dsp`), and this is a transcription of it.
 *
 * pydub works on int16 samples at 1 ms granularity, so everything here is in
 * milliseconds on the int16 scale, with `maxPossibleAmplitude = 32768`.
 */
object SilenceUtils {

    private const val MAX_AMP = 32768.0

    private fun msToSample(ms: Int, sr: Int): Int = (ms.toLong() * sr / 1000L).toInt()

    /** `audioop.rms` truncates to int; match that so thresholds tie-break identically. */
    private fun msRms(x: ShortArray, sr: Int, startMs: Int, lenMs: Int): Double {
        val a = msToSample(startMs, sr)
        val b = msToSample(startMs + lenMs, sr)
        val lo = a.coerceIn(0, x.size)
        val hi = b.coerceIn(0, x.size)
        if (hi <= lo) return 0.0
        var acc = 0.0
        for (i in lo until hi) {
            val v = x[i].toDouble()
            acc += v * v
        }
        return sqrt(acc / (hi - lo)).toInt().toDouble()
    }

    /** Port of `pydub.silence.detect_silence`. Ranges are `[startMs, endMs)`. */
    fun detectSilence(
        x: ShortArray, sr: Int, minSilenceLen: Int,
        silenceThreshDb: Double, seekStep: Int = 1,
    ): List<IntArray> {
        val segLen = (x.size.toLong() * 1000L / sr).toInt()
        if (segLen < minSilenceLen) return emptyList()
        val thresh = Math.pow(10.0, silenceThreshDb / 20.0) * MAX_AMP

        val lastStart = segLen - minSilenceLen
        val starts = ArrayList<Int>()
        var i = 0
        while (i <= lastStart) { starts.add(i); i += seekStep }
        if (lastStart % seekStep != 0) starts.add(lastStart)

        val silent = ArrayList<Int>()
        for (s in starts) if (msRms(x, sr, s, minSilenceLen) <= thresh) silent.add(s)
        if (silent.isEmpty()) return emptyList()

        val ranges = ArrayList<IntArray>()
        var prev = silent[0]
        var cur = prev
        for (k in 1 until silent.size) {
            val v = silent[k]
            val contiguous = v == prev + seekStep
            val hasGap = v > prev + minSilenceLen
            if (!contiguous && hasGap) {
                ranges.add(intArrayOf(cur, prev + minSilenceLen))
                cur = v
            }
            prev = v
        }
        ranges.add(intArrayOf(cur, prev + minSilenceLen))
        return ranges
    }

    /** Port of `pydub.silence.detect_nonsilent`. */
    fun detectNonsilent(
        x: ShortArray, sr: Int, minSilenceLen: Int,
        silenceThreshDb: Double, seekStep: Int = 1,
    ): List<IntArray> {
        var silent = detectSilence(x, sr, minSilenceLen, silenceThreshDb, seekStep)
        val length = (x.size.toLong() * 1000L / sr).toInt()
        if (silent.isEmpty()) return listOf(intArrayOf(0, length))
        if (silent[0][0] == 0 && silent[0][1] == length) return emptyList()

        val out = ArrayList<IntArray>()
        var prevEnd = 0
        if (silent[0][0] == 0) {
            prevEnd = silent[0][1]
            silent = silent.drop(1)
        }
        for (r in silent) {
            out.add(intArrayOf(prevEnd, r[0]))
            prevEnd = r[1]
        }
        if (prevEnd < length) out.add(intArrayOf(prevEnd, length))
        return out
    }

    /** Port of `pydub.silence.detect_leading_silence` — dBFS, not raw rms. */
    private fun detectLeadingSilence(
        x: ShortArray, sr: Int, threshDb: Double = -50.0, chunkMs: Int = 10,
    ): Int {
        val length = (x.size.toLong() * 1000L / sr).toInt()
        var trim = 0
        while (trim < length) {
            val rms = msRms(x, sr, trim, chunkMs)
            val dbfs = if (rms <= 0.0) Double.NEGATIVE_INFINITY else 20.0 * log10(rms / MAX_AMP)
            if (dbfs >= threshDb) break
            trim += chunkMs
        }
        return trim
    }

    private fun toI16(x: FloatArray): ShortArray =
        ShortArray(x.size) { (x[it] * 32768.0f).coerceIn(-32768.0f, 32767.0f).toInt().toShort() }

    private fun toF32(x: ShortArray): FloatArray =
        FloatArray(x.size) { x[it] / 32768.0f }

    /** Port of `omnivoice.utils.audio.remove_silence`. */
    fun removeSilence(
        x: FloatArray, sr: Int, midSil: Int = 300, leadSil: Int = 100,
        trailSil: Int = 300, silenceThreshDb: Double = -50.0,
    ): FloatArray {
        if (x.isEmpty()) return x
        var xi = toI16(x)

        if (midSil > 0) {
            val nonsilent = detectNonsilent(xi, sr, midSil, silenceThreshDb, seekStep = 10)
            val length = (xi.size.toLong() * 1000L / sr).toInt()
            var total = 0
            val pieces = ArrayList<IntArray>(nonsilent.size)
            for (r in nonsilent) {
                val a = msToSample(maxOf(0, r[0] - midSil), sr).coerceIn(0, xi.size)
                val b = msToSample(minOf(length, r[1] + midSil), sr).coerceIn(0, xi.size)
                if (b > a) { pieces.add(intArrayOf(a, b)); total += b - a }
            }
            val merged = ShortArray(total)
            var p = 0
            for (r in pieces) {
                System.arraycopy(xi, r[0], merged, p, r[1] - r[0])
                p += r[1] - r[0]
            }
            xi = merged
        }

        if (xi.isNotEmpty()) {
            val head = maxOf(0, detectLeadingSilence(xi, sr, silenceThreshDb) - leadSil)
            val off = msToSample(head, sr).coerceIn(0, xi.size)
            xi = xi.copyOfRange(off, xi.size)
        }
        if (xi.isNotEmpty()) {
            val rev = ShortArray(xi.size) { xi[xi.size - 1 - it] }
            val tail = maxOf(0, detectLeadingSilence(rev, sr, silenceThreshDb) - trailSil)
            val off = msToSample(tail, sr).coerceIn(0, rev.size)
            val cut = rev.copyOfRange(off, rev.size)
            xi = ShortArray(cut.size) { cut[cut.size - 1 - it] }
        }
        return toF32(xi)
    }

    /** Port of `omnivoice.utils.audio.fade_and_pad_audio`. */
    fun fadeAndPad(
        x: FloatArray, sr: Int = OV.SR_24K, padSeconds: Float = 0.1f,
        fadeSeconds: Float = 0.1f,
    ): FloatArray {
        if (x.isEmpty()) return x
        val out = x.copyOf()
        val k = minOf((fadeSeconds * sr).toInt(), out.size / 2)
        if (k > 0) {
            for (i in 0 until k) {
                val w = i.toFloat() / (k - 1).coerceAtLeast(1)
                out[i] *= w
                out[out.size - 1 - i] *= w
            }
        }
        val p = (padSeconds * sr).toInt()
        if (p <= 0) return out
        val padded = FloatArray(out.size + 2 * p)
        System.arraycopy(out, 0, padded, p, out.size)
        return padded
    }

    /** Port of `_post_process_audio`. [refRms] null means "no reference". */
    fun postProcess(x: FloatArray, refRms: Float?, cfg: GenConfig): FloatArray {
        var y = x
        if (cfg.postprocessOutput) y = removeSilence(y, OV.SR_24K, 500, 100, 100)
        if (refRms != null && refRms < 0.1f) {
            y = FloatArray(y.size) { y[it] * refRms / 0.1f }
        } else if (refRms == null) {
            var peak = 0.0f
            for (v in y) peak = maxOf(peak, Math.abs(v))
            if (peak > 1e-6f) y = FloatArray(y.size) { y[it] / peak * 0.5f }
        }
        return fadeAndPad(y, OV.SR_24K, cfg.padSeconds, cfg.fadeSeconds)
    }

    /** Round the sample count down to a whole codec frame (hop 960). */
    fun clipToHop(x: FloatArray): FloatArray {
        val n = (x.size / OV.HOP_LENGTH) * OV.HOP_LENGTH
        return if (n == x.size) x else x.copyOf(n)
    }

    fun rms(x: FloatArray): Float {
        if (x.isEmpty()) return 0.0f
        var acc = 0.0
        for (v in x) acc += v.toDouble() * v
        return sqrt(acc / x.size).toFloat()
    }
}
