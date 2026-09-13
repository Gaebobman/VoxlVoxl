package ai.omnivoice.poc

import ai.omnivoice.poc.core.TextTimeline
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.PI
import kotlin.math.sin

/**
 * Script following during playback rests on two maps: output samples back to
 * vocoder samples through removed silences, and vocoder samples to characters.
 * The first must not change the audio at all -- the silence removal is a
 * bit-exact port of upstream's pydub code -- and neither may lose track of time.
 */
class TextTimelineTest {

    private val sr = 24_000

    /** 0.3 s tone, [gap] s of silence, 0.3 s tone. */
    private fun toneGapTone(gap: Double): FloatArray {
        val tone = (0.3 * sr).toInt()
        val silence = (gap * sr).toInt()
        return FloatArray(tone * 2 + silence) { i ->
            if (i < tone || i >= tone + silence) (0.3 * sin(2 * PI * 220 * i / sr)).toFloat() else 0f
        }
    }

    @Test
    fun `mapped silence removal returns exactly the same audio`() {
        for (gap in doubleArrayOf(0.2, 1.0, 2.5)) {
            val x = toneGapTone(gap)
            val plain = SilenceUtils.removeSilence(x, sr, 500, 100, 100)
            val (mapped, segs) = SilenceUtils.removeSilenceMapped(x, sr, 500, 100, 100)
            assertArrayEquals(plain, mapped, "gap=$gap: the mapping must not change a sample")

            var covered = 0
            for (k in 0 until segs.size / 2) {
                val a = segs[2 * k]; val b = segs[2 * k + 1]
                assertTrue(a in 0..b && b <= x.size, "gap=$gap: segment [$a,$b) outside the input")
                covered += b - a
            }
            assertEquals(mapped.size, covered, "gap=$gap: segments must account for every output sample")
        }
    }

    @Test
    fun `a long silence is actually removed, so the map is needed`() {
        val x = toneGapTone(2.5)
        val (y, _) = SilenceUtils.removeSilenceMapped(x, sr, 500, 100, 100)
        assertTrue(y.size < x.size - sr, "expected over a second removed, got ${x.size - y.size} samples")
    }

    private fun uniform(n: Int) = DoubleArray(n) { (it + 1).toDouble() / n }

    @Test
    fun `nothing is spoken during the leading pad, everything by the end`() {
        val text = "가나다라마바사"
        val pad = 2_400
        val t = TextTimeline(text, intArrayOf(pad, 0, 48_000),
            listOf(TextTimeline.Chunk(0, text.length, 0, 48_000, uniform(text.length))))
        assertEquals(0, t.spokenThrough(0))
        assertEquals(0, t.spokenThrough(pad - 1))
        assertEquals(text.length, t.spokenThrough(pad + 48_000 + pad - 1))

        var last = 0
        for (o in 0 until pad + 48_000 + pad step 97) {
            val c = t.spokenThrough(o)
            assertTrue(c >= last, "went backwards at sample $o: $last -> $c")
            last = c
        }
    }

    @Test
    fun `a removed silence holds position instead of drifting`() {
        // output [0,1000) came from source [0,1000); output [1000,2000) from [5000,6000)
        val text = "abcdef"
        val t = TextTimeline(text, intArrayOf(0, 0, 1000, 1000, 5000, 1000),
            listOf(TextTimeline.Chunk(0, 6, 0, 6000, uniform(6))))
        assertEquals(1, t.spokenThrough(999), "source 999 of 6000 is inside the first character")
        assertEquals(6, t.spokenThrough(1500), "source 5500 of 6000 is inside the last character")
    }

    @Test
    fun `chunk boundaries are exact`() {
        val text = "하나. 둘."
        val t = TextTimeline(text, intArrayOf(0, 0, 20_000), listOf(
            TextTimeline.Chunk(0, 4, 0, 10_000, uniform(4)),
            TextTimeline.Chunk(4, 7, 10_000, 20_000, uniform(3)),
        ))
        assertEquals(4, t.spokenThrough(9_999), "the first chunk ends with its own audio")
        assertEquals(5, t.spokenThrough(10_000), "the second starts exactly where its audio does")
    }

    @Test
    fun `crossFadeStarts agrees with where crossFade puts each part`() {
        val parts = listOf(FloatArray(30_000) { 1f }, FloatArray(20_000) { 2f }, FloatArray(25_000) { 3f })
        val out = TextChunker.crossFade(parts, sr)
        val starts = TextChunker.crossFadeStarts(IntArray(parts.size) { parts[it].size }, sr)
        val fade = (0.05f * sr).toInt()
        for (i in 1 until parts.size) {
            // just past the fade, the joined signal is the new part alone
            assertEquals(parts[i][0], out[starts[i] + fade], "part $i misplaced")
        }
        assertEquals(out.size, starts.last() + parts.last().size)
    }
}
