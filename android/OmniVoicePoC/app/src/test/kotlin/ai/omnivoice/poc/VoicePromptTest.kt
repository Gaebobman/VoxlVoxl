package ai.omnivoice.poc

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File

class VoicePromptTest {

    @Test
    fun `reads the file written by the python side`() {
        val f = Fixtures.load("voice_prompt.json")
        val bin = Fixtures.repoFile(f.getString("file"))
        assumeTrue(bin.isFile, "${f.getString("file")} missing — run infer_onnx.py enroll")

        val vp = VoicePrompt.load(bin)
        assertEquals(f.getInt("num_frames"), vp.numFrames)
        assertEquals(f.getString("ref_text"), vp.refText)
        assertEquals(f.getInt("sample_rate"), vp.sampleRate)
        assertEquals(f.getDouble("ref_rms"), vp.refRms.toDouble(), 1e-6)

        val head = f.getJSONArray("codes_head")
        for (c in 0 until OV.NUM_CODEBOOKS) {
            val row = head.getJSONArray(c)
            for (t in 0 until row.length()) {
                assertEquals(row.getInt(t), vp.codes[c][t].toInt(), "codes[$c][$t]")
            }
        }
        var sum = 0L
        for (c in 0 until OV.NUM_CODEBOOKS) for (t in 0 until vp.numFrames) sum += vp.codes[c][t]
        assertEquals(f.getLong("codes_sum"), sum)
    }

    @Test
    fun `round-trips through save and load`(): Unit {
        val codes = Array(OV.NUM_CODEBOOKS) { c -> ShortArray(37) { t -> ((c * 37 + t) % 1024).toShort() } }
        val vp = VoicePrompt(codes, "테스트 transcript.", 0.0731f)
        val tmp = File.createTempFile("vprompt", ".bin")
        try {
            vp.save(tmp)
            val back = VoicePrompt.load(tmp)
            assertEquals(vp.numFrames, back.numFrames)
            assertEquals(vp.refText, back.refText)
            assertEquals(vp.refRms, back.refRms)
            for (c in 0 until OV.NUM_CODEBOOKS) {
                for (t in 0 until vp.numFrames) assertEquals(codes[c][t], back.codes[c][t])
            }
        } finally {
            tmp.delete()
        }
    }
}
