package ai.omnivoice.poc.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BlurMaskFilter
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import android.util.AttributeSet
import android.view.View
import kotlin.math.min
import kotlin.random.Random

/**
 * The layer everything else sits on: three coloured lights and a soft spectrum,
 * under a veil that keeps type readable.
 *
 * The spectrum is what makes the translucent panels mean anything — there is a
 * voice behind the glass rather than a texture. It is drawn once into a bitmap
 * and only redrawn on resize, so the blur it carries costs nothing per frame.
 * That is deliberate: a warm device already runs generation about twice as slow
 * (RTF 11.30 -> 20.89 measured), and continuous GPU blur during a 20-80 s
 * CPU-bound job would buy the look back in seconds of waiting.
 */
class GroundView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private var cache: Bitmap? = null
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        // in pixels, so it has to scale with density — an 18px blur that softens
        // the band on a 1x screen leaves hard edges on a 3.5x phone
        maskFilter = BlurMaskFilter(
            22f * resources.displayMetrics.density, BlurMaskFilter.Blur.NORMAL)
    }

    /** Colour stops for the spectrum, warm on the left through to cool. */
    private val spectrum = intArrayOf(
        0xFFFFB65E.toInt(), 0xFFFFAE6D.toInt(), 0xFFFF9F86.toInt(), 0xFFF48FA2.toInt(),
        0xFFDE85BE.toInt(), 0xFFC07FD2.toInt(), 0xFF9B84DF.toInt(), 0xFF7E8BE2.toInt(),
        0xFF6C92E0.toInt(),
    )

    init {
        // BlurMaskFilter is a software effect; the layer is cached, not per-frame
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) {
        super.onSizeChanged(w, h, ow, oh)
        cache?.recycle()
        cache = if (w > 0 && h > 0) render(w, h) else null
    }

    override fun onDraw(canvas: Canvas) {
        val b = cache ?: return
        canvas.drawBitmap(b, 0f, 0f, null)
    }

    private fun render(w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(0xFF0A0910.toInt())

        fun light(cx: Float, cy: Float, r: Float, color: Int) {
            paint.shader = RadialGradient(cx, cy, r, color, color and 0x00FFFFFF, Shader.TileMode.CLAMP)
            c.drawRect(0f, 0f, w.toFloat(), h.toFloat(), paint)
        }
        val d = min(w, h).toFloat()
        light(w * 0.15f, h * 0.12f, d * 1.15f, 0x6BEDA13F)
        light(w * 0.88f, h * 0.30f, d * 0.95f, 0x57C454A8)
        light(w * 0.50f, h * 0.92f, d * 1.05f, 0x4D4678D2)

        // the voice: a soft, static spectrum band a third of the way down
        val rng = Random(7)
        val top = h * 0.40f
        val band = h * 0.20f
        val n = 22
        val gap = w * 0.012f
        val bw = (w - gap * (n + 1)) / n
        for (i in 0 until n) {
            val t = i / (n - 1f)
            val hgt = band * (0.20f + 0.78f * rng.nextFloat())
            barPaint.color = spectrum[(t * (spectrum.size - 1)).toInt()]
            barPaint.alpha = 96
            val x = gap + i * (bw + gap)
            val y = top + (band - hgt) / 2f
            c.drawRoundRect(RectF(x, y, x + bw, y + hgt), bw / 2f, bw / 2f, barPaint)
        }

        paint.shader = LinearGradient(
            0f, 0f, 0f, h.toFloat(),
            intArrayOf(0x610A0910, 0x940A0910.toInt(), 0xD60A0910.toInt()),
            floatArrayOf(0f, 0.46f, 1f), Shader.TileMode.CLAMP,
        )
        c.drawRect(0f, 0f, w.toFloat(), h.toFloat(), paint)
        paint.shader = null
        paint.color = Color.TRANSPARENT
        return bmp
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        cache?.recycle()
        cache = null
    }
}
