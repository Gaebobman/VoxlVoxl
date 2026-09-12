package ai.omnivoice.poc.core

import java.io.File

/**
 * The abstractions from the feature spec §13. Nothing above this layer names
 * OmniVoice.
 *
 * This is not speculative future-proofing. The measured RTF on the target device
 * is ~7.5 at best (docs/feature-validation.md §1), which is fine for asynchronous
 * synthesis and disqualifying for Phase 4 Conversation and Phase 5 Telephony.
 * A second [SpeechSynthesizer] is therefore a scheduled requirement, not a
 * hypothetical one.
 */

/** An enrolled speaker. The encoded representation, never the raw audio. */
data class VoiceProfile(
    val id: String,
    val displayName: String,
    /** `[codebooks][frames]` codec codes. */
    val codes: Array<ShortArray>,
    val refText: String,
    val refRms: Float,
    val sampleRate: Int,
    val createdAtMillis: Long = System.currentTimeMillis(),
) {
    val frames: Int get() = codes[0].size

    override fun equals(other: Any?) = other is VoiceProfile && other.id == id
    override fun hashCode() = id.hashCode()
}

/**
 * How the voice should sound.
 *
 * Measured (docs/feature-validation.md §4): [instruct] only takes effect when
 * there is **no** voice profile. With a profile present the clone conditioning
 * completely overrides it — every style landed within ±9 Hz of baseline F0. So
 * style for a cloned voice means *choosing a differently-recorded profile*, which
 * is what [StyleController] exposes.
 */
data class VoiceStyle(
    /** Voice-design attributes, e.g. "female, low pitch". Closed vocabulary. */
    val instruct: String? = null,
    val speed: Float = 1.0f,
    val language: String? = null,
)

/** Everything a caller can learn about one synthesis. Spec F-F02. */
data class SynthesisMetrics(
    val sequenceLength: Int,
    val targetFrames: Int,
    val chunks: Int,
    val modelLoadMillis: Long,
    val generateMillis: Long,
    val decodeMillis: Long,
    val postMillis: Long,
    val totalMillis: Long,
    val audioSeconds: Double,
    val peakPssKb: Int,
    val thermalStatus: Int,
) {
    val rtf: Double get() = if (audioSeconds > 0) totalMillis / 1000.0 / audioSeconds else Double.NaN
}

data class AudioResult(
    val samples: FloatArray,
    val sampleRate: Int,
    val metrics: SynthesisMetrics,
) {
    override fun equals(other: Any?) = this === other
    override fun hashCode() = System.identityHashCode(this)
}

/** Reported while a long synthesis runs, so the UI can show real progress. */
data class SynthesisProgress(
    val chunk: Int,
    val totalChunks: Int,
    val step: Int,
    val totalSteps: Int,
    val cellsRemaining: Int,
    val cellsTotal: Int,
    /**
     * Fraction of each codebook that is resolved, 0..1, index 0 first.
     *
     * Measured rather than modelled: the un-masking loop subtracts
     * `layerPenaltyFactor` per codebook index, so lower codebooks really do
     * resolve first, and a UI that derives this from the total instead of
     * counting it gets the shape wrong.
     */
    val perCodebook: FloatArray,
) {
    /** 0..1 over the whole job, chunks included. */
    val fraction: Float
        get() = ((chunk - 1) + step.toFloat() / totalSteps.coerceAtLeast(1)) /
            totalChunks.coerceAtLeast(1)

    override fun equals(other: Any?) = this === other
    override fun hashCode() = System.identityHashCode(this)
}

interface SpeechSynthesizer : AutoCloseable {
    val backendName: String

    /** Opens sessions. Safe to call more than once. */
    fun load()

    /**
     * @param onProgress called from the synthesis thread; keep it cheap.
     * @param isActive polled between steps; return false to cancel. A cancelled
     *   call throws [SynthesisCancelledException] rather than returning a partial
     *   result, so a caller cannot mistake a stub for a finished utterance.
     */
    fun synthesize(
        text: String,
        voiceProfile: VoiceProfile?,
        style: VoiceStyle? = null,
        onProgress: ((SynthesisProgress) -> Unit)? = null,
        isActive: () -> Boolean = { true },
    ): AudioResult
}

class SynthesisCancelledException : Exception("synthesis cancelled")

/** Turns a recording into a [VoiceProfile]. Spec F-A01 / F-A02. */
interface VoiceEnroller : AutoCloseable {
    /**
     * @param samples mono float32 at [sampleRate]
     * @param refText the transcript; typed by the user in v1, per the spec
     */
    fun enroll(
        samples: FloatArray,
        sampleRate: Int,
        refText: String,
        displayName: String,
    ): VoiceProfile
}

/** Persistence for profiles. Spec F-A03, F-E02. */
interface VoiceProfileManager {
    fun list(): List<VoiceProfile>
    fun get(id: String): VoiceProfile?
    fun save(profile: VoiceProfile)
    fun delete(id: String): Boolean
    fun directory(): File
}

/**
 * Spec F-C02. Split deliberately in two, because the measurements say they are
 * different mechanisms:
 *
 *  - with a voice profile, style = which profile (F-C01 transfers speaking
 *    manner through the reference recording: a whispered enrollment stays
 *    whispered, voiced fraction 20.3 % → 24.1 %)
 *  - without one, style = a voice-design instruct (F-C03, which moves F0 by up
 *    to ±85 Hz)
 */
interface StyleController {
    /** Design attributes offered when no profile is selected. */
    fun designPresets(): List<StylePreset>

    /** Validates against the model's closed vocabulary; null if unsupported. */
    fun resolveInstruct(items: List<String>): String?
}

data class StylePreset(val label: String, val instruct: String)
