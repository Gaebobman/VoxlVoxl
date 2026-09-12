package ai.omnivoice.poc

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class UnmaskScheduleTest {

    @Test
    fun `matches python timesteps and schedule`() {
        val cases = Fixtures.load("schedule.json").getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val numStep = c.getInt("num_step")
            val tShift = c.getDouble("t_shift")
            val total = c.getInt("total")

            val wantT = c.getJSONArray("timesteps")
            val gotT = UnmaskSchedule.timeSteps(numStep, tShift)
            assertEquals(wantT.length(), gotT.size)
            for (k in 0 until wantT.length()) {
                assertEquals(wantT.getDouble(k), gotT[k], 1e-9, "timesteps[$k]")
            }

            val wantS = c.getJSONArray("schedule")
            val gotS = UnmaskSchedule.schedule(numStep, tShift, total)
            assertArrayEquals(
                IntArray(wantS.length()) { wantS.getInt(it) }, gotS,
                "schedule(numStep=$numStep, tShift=$tShift, total=$total)",
            )
            assertEquals(total, gotS.sum(), "schedule must account for every cell")
        }
    }
}
