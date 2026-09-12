package ai.omnivoice.poc

import ai.omnivoice.poc.core.VoiceProfile
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File

/**
 * The display name is the one part of a profile that is not in `voice_prompt.bin`
 * — it lives in a sidecar file beside it. That split is what makes a rename cheap,
 * and it is also what makes it easy to get wrong, so it is tested rather than
 * eyeballed on a screen.
 */
class FileVoiceProfileManagerTest {

    private fun profile(id: String, name: String) = VoiceProfile(
        id = id,
        displayName = name,
        // 8 codebooks x 6 frames of arbitrary but distinguishable codes
        codes = Array(8) { c -> ShortArray(6) { t -> (c * 100 + t).toShort() } },
        refText = "이것은 제 목소리를 등록하기 위한 짧은 문장입니다.",
        refRms = 0.0904f,
        sampleRate = OV.SR_24K,
    )

    @Test
    fun `a rename survives a reload and leaves the codes untouched`(@TempDir dir: File) {
        val profiles = FileVoiceProfileManager(dir)
        val original = profile("v1", "9월 12일 21:05")
        profiles.save(original)

        val before = profiles.get("v1")!!
        assertEquals("9월 12일 21:05", before.displayName)

        profiles.save(before.copy(displayName = "Studio mic"))

        // reload through a fresh manager: nothing may be held in memory
        val after = FileVoiceProfileManager(dir).get("v1")!!
        assertEquals("Studio mic", after.displayName, "the new name did not persist")
        assertEquals(original.refText, after.refText)
        assertEquals(original.sampleRate, after.sampleRate)
        assertEquals(original.frames, after.frames)
        for (c in original.codes.indices) {
            assertTrue(
                original.codes[c].contentEquals(after.codes[c]),
                "codebook $c changed during a rename",
            )
        }
    }

    @Test
    fun `a profile with no sidecar falls back to its id rather than failing`(@TempDir dir: File) {
        val profiles = FileVoiceProfileManager(dir)
        profiles.save(profile("v1", "이름"))
        assertTrue(File(dir, "v1.name").delete())

        val p = profiles.get("v1")!!
        assertEquals("v1", p.displayName)
        assertEquals(6, p.frames, "the codes must still load without the sidecar")
    }

    @Test
    fun `delete removes the sidecar too, so a reused id cannot inherit a stale name`(
        @TempDir dir: File,
    ) {
        val profiles = FileVoiceProfileManager(dir)
        profiles.save(profile("v1", "Studio mic"))
        assertTrue(profiles.delete("v1"))
        assertNull(profiles.get("v1"))
        assertTrue(dir.listFiles()!!.isEmpty(), "left behind: ${dir.list()!!.joinToString()}")

        profiles.save(profile("v1", "9월 13일 09:00"))
        assertEquals("9월 13일 09:00", profiles.get("v1")!!.displayName)
    }

    @Test
    fun `list returns every saved profile with its own name`(@TempDir dir: File) {
        val profiles = FileVoiceProfileManager(dir)
        profiles.save(profile("v1", "Studio mic"))
        profiles.save(profile("v2", "Phone call"))

        val byId = profiles.list().associateBy { it.id }
        assertEquals(2, byId.size)
        assertEquals("Studio mic", byId["v1"]?.displayName)
        assertEquals("Phone call", byId["v2"]?.displayName)
    }
}
