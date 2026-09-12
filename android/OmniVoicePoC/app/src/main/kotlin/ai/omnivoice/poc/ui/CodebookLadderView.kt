package ai.omnivoice.poc.ui

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.util.AttributeSet
import android.view.View
import ai.omnivoice.poc.OV

/**
 * Eight bars, one per codebook, showing how much of each has been un-masked.
 *
 * This is the model's real state, not a decorative progress bar: the un-masking
 * loop subtracts `layer_penalty_factor` per codebook index, so layer 0 resolves
 * first and layer 7 last, and the staircase that produces is worth showing.
 */
class CodebookLadderView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0,
) : View(context, attrs, defStyle) {

    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x1AFFFFFF }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x7AF6F1EA
        textSize = 9.5f * resources.displayMetrics.scaledDensity
    }

    /**
     * The rows are codebooks, and an index is meaningless to anyone who has not
     * read the model card. But the residual quantiser IS coarse-to-fine — layer 0
     * is the roughest approximation of the voice and each one after it adds
     * detail — so the axis says that instead of numbering it.
     */
    var topLabel: String = ""
    var bottomLabel: String = ""

    /** Warm at layer 0 shading to violet at layer 7, matching the design. */
    private val ends = arrayOf(
        0xFFFFD29A.toInt() to 0xFFEDA13F.toInt(),
        0xFFFFD29A.toInt() to 0xFFEDA13F.toInt(),
        0xFFF6B884.toInt() to 0xFFE0913C.toInt(),
        0xFFE8A98C.toInt() to 0xFFCE8147.toInt(),
        0xFFD89A96.toInt() to 0xFFB87257.toInt(),
        0xFFC68CA4.toInt() to 0xFF9F6470.toInt(),
        0xFFB27FB4.toInt() to 0xFF875A86.toInt(),
        0xFF9B84DF.toInt() to 0xFF6E5FA6.toInt(),
    )

    private val progress = FloatArray(OV.NUM_CODEBOOKS)
    private val from = FloatArray(OV.NUM_CODEBOOKS)
    private val target = FloatArray(OV.NUM_CODEBOOKS)
    private val rect = RectF()
    private var anim: ValueAnimator? = null

    /**
     * @param filled per-codebook fill in 0..1
     *
     * The loop reports once per un-masking step, roughly once a second, so the
     * bars used to stand still and then jump. They are interpolated between two
     * reported states rather than extrapolated: nothing here shows progress the
     * model has not already made.
     */
    fun setProgress(filled: FloatArray) {
        anim?.cancel()
        System.arraycopy(progress, 0, from, 0, progress.size)
        for (i in 0 until minOf(filled.size, target.size)) {
            target[i] = filled[i].coerceIn(0f, 1f)
        }
        anim = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 360L
            interpolator = Motion.standard
            addUpdateListener {
                val t = it.animatedValue as Float
                for (i in progress.indices) progress[i] = from[i] + (target[i] - from[i]) * t
                invalidate()
            }
            start()
        }
    }

    fun reset() {
        anim?.cancel()
        progress.fill(0f)
        from.fill(0f)
        target.fill(0f)
        invalidate()
    }

    override fun onMeasure(widthSpec: Int, heightSpec: Int) {
        val w = MeasureSpec.getSize(widthSpec)
        val dp = resources.displayMetrics.density
        // 8 bars of 7dp with 5dp gaps
        val h = (OV.NUM_CODEBOOKS * 7 + (OV.NUM_CODEBOOKS - 1) * 5) * dp
        setMeasuredDimension(w, h.toInt())
    }

    override fun onDraw(canvas: Canvas) {
        val dp = resources.displayMetrics.density
        val barH = 7 * dp
        val gap = 5 * dp
        val labelW = 30 * dp
        val left = labelW
        val right = width.toFloat()
        val radius = barH / 2f

        for (i in 0 until OV.NUM_CODEBOOKS) {
            val top = i * (barH + gap)
            if (i == 0 && topLabel.isNotEmpty()) {
                canvas.drawText(topLabel, 0f, top + barH - 0.5f * dp, label)
            }
            if (i == OV.NUM_CODEBOOKS - 1 && bottomLabel.isNotEmpty()) {
                canvas.drawText(bottomLabel, 0f, top + barH - 0.5f * dp, label)
            }
            rect.set(left, top, right, top + barH)
            canvas.drawRoundRect(rect, radius, radius, track)

            val p = progress[i]
            if (p <= 0f) continue
            val end = left + (right - left) * p
            val (a, b) = ends[i]
            fill.shader = LinearGradient(left, 0f, right, 0f, a, b, Shader.TileMode.CLAMP)
            rect.set(left, top, end, top + barH)
            canvas.drawRoundRect(rect, radius, radius, fill)
        }
        fill.shader = null
    }
}
