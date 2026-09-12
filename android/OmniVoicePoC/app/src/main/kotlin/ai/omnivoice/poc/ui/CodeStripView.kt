package ai.omnivoice.poc.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import ai.omnivoice.poc.OV

/**
 * The stored voice, drawn as what it actually is: an 8 × T grid of codec codes.
 *
 * The first attempt here was a waveform derived from the code values, and it was
 * a lie — an RVQ index is a cluster id, not an amplitude, so averaging a few
 * codebooks yields a near-constant and every profile drew the same. Since the
 * recording is deliberately not kept, there is no waveform to draw; this shows
 * the 1.8 kB that IS kept, and two different recordings genuinely differ in it.
 */
class CodeStripView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    enum class Tone { AMBER, VIOLET }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val rect = RectF()
    private var codes: Array<ShortArray>? = null

    var tone: Tone = Tone.AMBER
        set(v) { field = v; invalidate() }

    fun setCodes(c: Array<ShortArray>?) {
        codes = c
        invalidate()
    }

    /** Code value to colour: constant hue per tone, lightness carries the value. */
    private fun colorFor(code: Int, row: Int): Int {
        val t = (code.coerceIn(0, 1023) / 1023f)
        val hsv = FloatArray(3)
        when (tone) {
            Tone.AMBER -> { hsv[0] = 34f; hsv[1] = 0.62f - 0.30f * (row / 7f) }
            Tone.VIOLET -> { hsv[0] = 258f; hsv[1] = 0.46f - 0.22f * (row / 7f) }
        }
        // deeper codebooks are residuals and carry less of the identity, so they
        // fade — the same ordering the decoder's layer penalty implies
        hsv[2] = (0.35f + 0.55f * t) * (1f - 0.45f * (row / 7f))
        return Color.HSVToColor(hsv)
    }

    override fun onDraw(canvas: Canvas) {
        val c = codes ?: return
        if (c.isEmpty() || c[0].isEmpty() || width == 0) return

        val rows = OV.NUM_CODEBOOKS
        val dp = resources.displayMetrics.density
        val gapY = 1f * dp
        val cellH = (height - gapY * (rows - 1)) / rows
        if (cellH <= 0f) return

        // one column per ~2dp, sampled from the frames
        val cols = (width / (3f * dp)).toInt().coerceIn(8, c[0].size.coerceAtLeast(8))
        val gapX = 1f * dp
        val cellW = (width - gapX * (cols - 1)) / cols
        val frames = c[0].size

        for (r in 0 until rows) {
            val row = c.getOrNull(r) ?: continue
            for (col in 0 until cols) {
                val f = (col.toFloat() / cols * frames).toInt().coerceIn(0, frames - 1)
                paint.color = colorFor(row[f].toInt(), r)
                val x = col * (cellW + gapX)
                val y = r * (cellH + gapY)
                rect.set(x, y, x + cellW, y + cellH)
                canvas.drawRoundRect(rect, cellW * 0.35f, cellW * 0.35f, paint)
            }
        }
    }
}
