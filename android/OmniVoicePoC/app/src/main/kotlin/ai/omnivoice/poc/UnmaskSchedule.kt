package ai.omnivoice.poc

/**
 * Port of `_get_time_steps` and the schedule loop in `_generate_iterative`
 * (Apache-2.0). Pinned by android/fixtures/schedule.json.
 *
 * With the default `tShift = 0.1` the schedule is strongly back-loaded in work:
 * at 16 steps over 384 cells it is
 * `[3,3,4,4,5,6,6,8,9,12,15,20,28,43,73,145]`, so the last step commits 38 % of
 * all cells. That is why halving the step count costs so little quality and
 * quartering it costs a lot.
 */
object UnmaskSchedule {

    fun timeSteps(numStep: Int, tShift: Double): DoubleArray {
        val out = DoubleArray(numStep + 1)
        val s = tShift
        for (i in 0..numStep) {
            val t = i.toDouble() / numStep
            out[i] = s * t / (1.0 + (s - 1.0) * t)
        }
        return out
    }

    /** Cells to un-mask at each step; always sums to [total]. */
    fun schedule(numStep: Int, tShift: Double, total: Int): IntArray {
        val t = timeSteps(numStep, tShift)
        val out = IntArray(numStep)
        var remaining = total
        for (i in 0 until numStep) {
            val n = if (i == numStep - 1) {
                remaining
            } else {
                minOf(Math.ceil(total * (t[i + 1] - t[i])).toInt(), remaining)
            }
            out[i] = n
            remaining -= n
        }
        return out
    }
}
