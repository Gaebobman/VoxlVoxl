package ai.omnivoice.poc.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Bar waveform, in three jobs: the enrolled reference on a profile card, the
 * generated clip with a playback position, and a live level meter while
 * recording.
 *
 * Peaks are computed once per waveform and cached — a 10 s clip is 240 000
 * samples and re-scanning it on every frame of playback would be absurd.
 */
class WaveformView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    enum class Tone { AMBER, VIOLET, REC }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val rect = RectF()

    private var peaks: FloatArray = FloatArray(0)
    private var buckets = 30

    var tone: Tone = Tone.AMBER
        set(v) { field = v; invalidate() }

    /** 0..1; bars before it are lit, after it dimmed. 1 lights everything. */
    var progress: Float = 1f
        set(v) { field = v.coerceIn(0f, 1f); invalidate() }

    /** Live meter: pushes one level and scrolls. */
    private var live: FloatArray? = null
    private var liveAt = 0

    fun setWaveform(samples: FloatArray, buckets: Int = 30) {
        this.buckets = buckets.coerceAtLeast(4)
        live = null
        peaks = if (samples.isEmpty()) FloatArray(0) else computePeaks(samples, this.buckets)
        progress = 1f
        invalidate()
    }

    fun startLive(buckets: Int = 24) {
        this.buckets = buckets
        live = FloatArray(buckets)
        liveAt = 0
        peaks = FloatArray(0)
        invalidate()
    }

    fun pushLevel(rms: Float) {
        val l = live ?: return
        l[liveAt % l.size] = rms.coerceIn(0f, 1f)
        liveAt++
        invalidate()
    }

    fun clear() {
        peaks = FloatArray(0)
        live = null
        invalidate()
    }

    /** RMS per bucket, normalised so a quiet clip still reads as a waveform. */
    private fun computePeaks(x: FloatArray, n: Int): FloatArray {
        val out = FloatArray(n)
        val per = (x.size / n).coerceAtLeast(1)
        var max = 0f
        for (i in 0 until n) {
            val from = i * per
            val to = minOf(x.size, from + per)
            var acc = 0.0
            for (j in from until to) acc += x[j].toDouble() * x[j]
            val v = if (to > from) sqrt(acc / (to - from)).toFloat() else 0f
            out[i] = v
            if (v > max) max = v
        }
        if (max > 1e-6f) for (i in out.indices) out[i] = (out[i] / max).coerceIn(0f, 1f)
        return out
    }

    override fun onDraw(canvas: Canvas) {
        val l = live
        val values: FloatArray = when {
            l != null -> FloatArray(l.size) { i ->
                // newest on the right
                l[((liveAt - l.size + i) % l.size + l.size) % l.size]
            }
            peaks.isNotEmpty() -> peaks
            else -> return
        }
        val n = values.size
        if (n == 0 || width == 0) return

        val dp = resources.displayMetrics.density
        val gap = 2f * dp
        val bw = ((width - gap * (n - 1)) / n).coerceAtLeast(1.5f * dp)
        val h = height.toFloat()
        val cy = h / 2f
        val lit = if (l != null) (liveAt.coerceAtMost(n)).toFloat() / n else progress

        val (a, b) = when (tone) {
            Tone.AMBER -> 0xFFFFD29A.toInt() to 0xFFEDA13F.toInt()
            Tone.VIOLET -> 0xFFC0AEEE.toInt() to 0xFF9B84DF.toInt()
            Tone.REC -> 0xFFFFB3A6.toInt() to 0xFFFF7A6B.toInt()
        }

        for (i in 0 until n) {
            val x = i * (bw + gap)
            val bh = (h * (0.10f + 0.88f * values[i])).coerceAtLeast(2f * dp)
            rect.set(x, cy - bh / 2f, x + bw, cy + bh / 2f)
            if (i.toFloat() / n < lit) {
                paint.shader = LinearGradient(0f, cy - bh / 2f, 0f, cy + bh / 2f, a, b, Shader.TileMode.CLAMP)
            } else {
                paint.shader = null
                paint.color = 0x29FFFFFF
            }
            canvas.drawRoundRect(rect, bw / 2f, bw / 2f, paint)
        }
        paint.shader = null
    }
}
