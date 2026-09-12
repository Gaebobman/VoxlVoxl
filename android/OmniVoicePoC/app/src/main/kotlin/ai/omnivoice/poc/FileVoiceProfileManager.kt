package ai.omnivoice.poc

import ai.omnivoice.poc.core.VoiceProfile
import ai.omnivoice.poc.core.VoiceProfileManager
import android.util.Log
import java.io.File

/**
 * Profiles on app-private internal storage, one `voice_prompt.bin` each.
 * Spec F-A03 and F-E02: no raw audio is kept, only the codec codes, so a leaked
 * profile is not a leaked recording.
 *
 * At ~1.8 kB per profile there is no reason to cap how many a user keeps.
 */
class FileVoiceProfileManager(private val root: File) : VoiceProfileManager {

    companion object {
        const val TAG = "OmniVoice.Profiles"
        private const val EXT = ".bin"
        private val SAFE = Regex("[^A-Za-z0-9_-]")
    }

    init {
        root.mkdirs()
    }

    override fun directory(): File = root

    private fun fileFor(id: String) = File(root, sanitize(id) + EXT)

    private fun sanitize(id: String) = SAFE.replace(id, "_").take(64).ifEmpty { "voice" }

    override fun list(): List<VoiceProfile> =
        (root.listFiles { f -> f.isFile && f.name.endsWith(EXT) } ?: emptyArray())
            .sortedBy { it.name }
            .mapNotNull { f ->
                runCatching { toProfile(f) }
                    .onFailure { Log.w(TAG, "skipping ${f.name}: ${it.message}") }
                    .getOrNull()
            }

    override fun get(id: String): VoiceProfile? {
        val f = fileFor(id)
        return if (f.isFile) runCatching { toProfile(f) }.getOrNull() else null
    }

    override fun save(profile: VoiceProfile) {
        val vp = VoicePrompt(profile.codes, profile.refText, profile.refRms, profile.sampleRate)
        val f = fileFor(profile.id)
        vp.save(f)
        File(root, sanitize(profile.id) + ".name").writeText(profile.displayName)
        Log.i(TAG, "saved ${profile.id} (${profile.frames} frames, ${f.length()} bytes)")
    }

    override fun delete(id: String): Boolean {
        File(root, sanitize(id) + ".name").delete()
        val ok = fileFor(id).delete()
        Log.i(TAG, "delete $id -> $ok")
        return ok
    }

    private fun toProfile(f: File): VoiceProfile {
        val vp = VoicePrompt.load(f)
        val id = f.name.removeSuffix(EXT)
        val nameFile = File(root, "$id.name")
        return VoiceProfile(
            id = id,
            displayName = if (nameFile.isFile) nameFile.readText().trim() else id,
            codes = vp.codes,
            refText = vp.refText,
            refRms = vp.refRms,
            sampleRate = vp.sampleRate,
            createdAtMillis = f.lastModified(),
        )
    }
}
