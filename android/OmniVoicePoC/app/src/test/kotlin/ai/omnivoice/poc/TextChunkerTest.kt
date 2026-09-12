package ai.omnivoice.poc

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.abs

class TextChunkerTest {

    private val estimate = { t: String ->
        DurationEstimator.estimateFrames(t, "안녕하세요. 테스트입니다.", 60)
    }

    @Test
    fun `short text is never split`() {
        val parts = TextChunker.chunk("오늘 회의를 시작하겠습니다.", estimate)
        assertEquals(1, parts.size)
    }

    @Test
    fun `long text splits and keeps every sentence`() {
        val sentence = "오늘 회의를 시작하겠습니다. "
        val long = sentence.repeat(30)
        val parts = TextChunker.chunk(long, estimate)
        assertTrue(parts.size > 1, "long text was not split")
        val rejoined = parts.joinToString(" ").replace(Regex("\\s+"), " ").trim()
        val original = long.replace(Regex("\\s+"), " ").trim()
        assertEquals(original, rejoined)
    }

    @Test
    fun `text with no sentence boundary stays whole rather than being cut mid-word`() {
        val runOn = "가".repeat(4000)
        assertEquals(1, TextChunker.chunk(runOn, estimate).size)
    }

    @Test
    fun `cross-fade holds level through the seam`() {
        // equal-power: two constant 0.5 signals must not dip at the join
        val a = FloatArray(24_000) { 0.5f }
        val b = FloatArray(24_000) { 0.5f }
        val joined = TextChunker.crossFade(listOf(a, b), OV.SR_24K)
        assertEquals(a.size + b.size - (0.05f * OV.SR_24K).toInt(), joined.size)
        var minAbs = 1.0f
        for (v in joined) minAbs = minOf(minAbs, abs(v))
        assertTrue(minAbs > 0.45f, "cross-fade dipped to $minAbs")
    }

    @Test
    fun `a single part passes through untouched`() {
        val a = FloatArray(100) { it / 100f }
        assertTrue(TextChunker.crossFade(listOf(a), OV.SR_24K).contentEquals(a))
        assertEquals(0, TextChunker.crossFade(emptyList(), OV.SR_24K).size)
    }
}
