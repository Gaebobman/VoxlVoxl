package ai.omnivoice.poc

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class DurationEstimatorTest {

    @Test
    fun `matches python character weights and frame estimates`() {
        val cases = Fixtures.load("duration.json").getJSONArray("cases")
        var weightChecks = 0
        var frameChecks = 0
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val text = c.getString("text")
            if (c.has("weight")) {
                assertEquals(
                    c.getDouble("weight"),
                    DurationEstimator.textWeight(text),
                    1e-6,
                    "textWeight(${text.take(30)})",
                )
                weightChecks++
            } else {
                val refText = c.getString("ref_text")
                val frames = DurationEstimator.estimateFrames(
                    text,
                    refText,
                    c.getInt("ref_frames"),
                    c.getDouble("speed").toFloat(),
                )
                assertEquals(
                    c.getInt("frames"), frames,
                    "estimateFrames(${text.take(24)}, ref=${refText.take(16)}, " +
                        "speed=${c.getDouble("speed")})",
                )
                frameChecks++
            }
        }
        check(weightChecks > 0 && frameChecks > 0) { "fixture file looks empty" }
    }
}
