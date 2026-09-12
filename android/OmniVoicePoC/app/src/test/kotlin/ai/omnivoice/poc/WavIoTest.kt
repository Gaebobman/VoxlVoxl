package ai.omnivoice.poc

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File

class WavIoTest {

    @Test
    fun `reads the repo reference clip`() {
        val wav = Fixtures.repoFile("sample/reference.wav")
        assumeTrue(wav.isFile, "sample/reference.wav missing")
        val a = wav.inputStream().use { WavIo.read(it) }
        assertEquals(OV.SR_24K, a.sampleRate)
        assertEquals(102000, a.samples.size)
    }

    @Test
    fun `round-trips within one int16 LSB`() {
        val n = 24_000
        val x = FloatArray(n) { (Math.sin(it * 0.05) * 0.4).toFloat() }
        val tmp = File.createTempFile("wav", ".wav")
        try {
            WavIo.write(tmp, x, OV.SR_24K)
            val back = tmp.inputStream().use { WavIo.read(it) }
            assertEquals(OV.SR_24K, back.sampleRate)
            assertEquals(n, back.samples.size)
            var maxErr = 0.0f
            for (i in 0 until n) maxErr = maxOf(maxErr, Math.abs(x[i] - back.samples[i]))
            assertTrue(maxErr < 1e-4f, "max error $maxErr")
        } finally {
            tmp.delete()
        }
    }

    @Test
    fun `resampling 24k to 16k lands on the expected length`() {
        val wav = Fixtures.repoFile("sample/reference.wav")
        assumeTrue(wav.isFile)
        val a = wav.inputStream().use { WavIo.read(it) }
        val y = WavIo.resample(a.samples, OV.SR_24K, OV.SR_16K)
        assertEquals(a.samples.size * 2 / 3, y.size)
    }
}
