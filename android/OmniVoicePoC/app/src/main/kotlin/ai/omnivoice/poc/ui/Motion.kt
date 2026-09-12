package ai.omnivoice.poc.ui

import android.animation.ValueAnimator
import android.transition.ChangeBounds
import android.transition.Fade
import android.transition.TransitionManager
import android.transition.TransitionSet
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.animation.PathInterpolator
import android.widget.EditText
import android.widget.TextView

import ai.omnivoice.poc.R

/**
 * The one place that decides how things move.
 *
 * Every control in the app changed state instantly: a button re-coloured under
 * the thumb with no motion, a screen replaced another in a single frame, and
 * the codebook ladder jumped a step at a time. Instant is not fast — it reads
 * as a redraw rather than a response, and it gives the eye nothing to follow
 * from the old state to the new one.
 *
 * The rules here are deliberately narrow. Motion is used only where it carries
 * meaning — acknowledging a touch, showing where a screen came from, joining
 * two values of the same number — and never as decoration. Durations stay under
 * a quarter second so nothing ever feels like a wait, and everything decelerates
 * into place rather than easing symmetrically, which is what makes a movement
 * read as physical instead of animated.
 */
object Motion {

    /** Material's emphasized-decelerate curve: leaves fast, settles gently. */
    val standard = PathInterpolator(0.2f, 0f, 0f, 1f)

    /** Symmetric, for things that leave rather than arrive. */
    val exit = PathInterpolator(0.4f, 0f, 1f, 1f)

    const val PRESS = 80L
    const val RELEASE = 220L
    const val SCREEN = 260L
    const val LAYOUT = 220L

    /**
     * A touch dip. The colour change alone was doing all the work of saying
     * "received", which is why the buttons felt dead: a state-list drawable
     * swaps between two frames with nothing in between.
     *
     * Returning false from the touch listener leaves the real click handling to
     * `View.onTouchEvent`, so this never intercepts a tap or a scroll.
     */
    fun View.pressable(depth: Float = 0.965f) {
        setOnTouchListener { v, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN ->
                    if (v.isEnabled) v.animate().scaleX(depth).scaleY(depth)
                        .setDuration(PRESS).setInterpolator(standard).start()
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL ->
                    v.animate().scaleX(1f).scaleY(1f)
                        .setDuration(RELEASE).setInterpolator(overshoot).start()
            }
            false
        }
    }

    /** A touch of overshoot on release so the button springs back rather than creeping. */
    private val overshoot = android.view.animation.OvershootInterpolator(2.2f)

    /**
     * Apply the dip to every tappable control under [this], skipping text fields
     * and containers, whose "clickable" flag exists only to swallow taps.
     */
    fun ViewGroup.pressableTree() {
        for (i in 0 until childCount) {
            val c = getChildAt(i)
            if (c is EditText) continue
            // A whole card dips less than a button: the same scale on a large
            // surface reads as the layout moving rather than the control.
            if (c.isClickable) c.pressable(if (c is ViewGroup) 0.985f else 0.965f)
            if (c is ViewGroup) c.pressableTree()
        }
    }

    /**
     * A screen arriving. Forward moves rise a few dp into place and back moves
     * settle down onto it, so the direction of travel is visible without a
     * transition animation between two live screens — which would mean keeping
     * both measured and drawn, and these screens are not cheap.
     */
    fun View.enterScreen(forward: Boolean) {
        alpha = 0f
        translationY = if (forward) 14f * resources.displayMetrics.density
        else -10f * resources.displayMetrics.density
        animate().alpha(1f).translationY(0f)
            .setDuration(SCREEN).setInterpolator(standard).start()
    }

    /** Re-lay-out this container smoothly instead of snapping. */
    fun ViewGroup.animateNextLayout(duration: Long = LAYOUT) {
        TransitionManager.beginDelayedTransition(this, TransitionSet().apply {
            ordering = TransitionSet.ORDERING_TOGETHER
            addTransition(Fade(Fade.OUT).setDuration(duration / 2))
            addTransition(ChangeBounds().setDuration(duration))
            addTransition(Fade(Fade.IN).setDuration(duration))
            interpolator = standard
        })
    }

    /**
     * Move a readout from the number it shows to the number it should show.
     *
     * The percentage only changes once per un-masking step — about once a second
     * — so without this it sits still and then jumps six points. The animation
     * invents no progress: it only draws the interval between two values the
     * model actually reported.
     */
    fun TextView.animateInt(to: Int, duration: Long = 420L) {
        val from = (getTag(R.id.motion_value) as? Int) ?: to.also { text = "$to" }
        (getTag(R.id.motion_anim) as? ValueAnimator)?.cancel()
        setTag(R.id.motion_value, to)
        if (from == to || duration <= 0L) { text = "$to"; return }
        val a = ValueAnimator.ofInt(from, to).apply {
            this.duration = duration
            interpolator = standard
            addUpdateListener { text = "${it.animatedValue}" }
        }
        setTag(R.id.motion_anim, a)
        a.start()
    }
}
