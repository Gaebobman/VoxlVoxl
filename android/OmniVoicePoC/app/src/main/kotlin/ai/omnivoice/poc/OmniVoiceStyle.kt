package ai.omnivoice.poc

import ai.omnivoice.poc.core.StyleController
import ai.omnivoice.poc.core.StylePreset

/**
 * Port of `_resolve_instruct` from omnivoice/models/omnivoice.py (Apache-2.0).
 *
 * The vocabulary is closed and the model raises on anything outside it, so the
 * UI must offer a picker rather than a text field. Only the English set is
 * exposed here; the Chinese set exists upstream and is a separate feature.
 *
 * Only presets whose effect was actually measured appear in [designPresets] —
 * the spec says "실제 모델이 안정적으로 제어 가능한 Style만 사용자 UI에 제공한다",
 * and docs/feature-validation.md §4 is that measurement.
 */
object OmniVoiceStyle : StyleController {

    val GENDER = listOf("male", "female")
    val AGE = listOf("child", "teenager", "young adult", "middle-aged", "elderly")
    val PITCH = listOf(
        "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch")
    val STYLE = listOf("whisper")
    val ACCENT = listOf(
        "american accent", "australian accent", "british accent", "canadian accent",
        "chinese accent", "indian accent", "japanese accent", "korean accent",
        "portuguese accent", "russian accent",
    )

    /** Every item the model accepts, lower-cased. */
    val VOCABULARY: Set<String> = (GENDER + AGE + PITCH + STYLE + ACCENT).toSet()

    /**
     * Measured F0 shift versus a no-instruct baseline of 193.5 Hz, in voice-design
     * mode (docs/feature-validation.md §4). `whisper` is listed by its voiced
     * fraction instead, 72.4 % → 20.3 %, because F0 is meaningless once the
     * excitation is unvoiced.
     */
    override fun designPresets(): List<StylePreset> = listOf(
        StylePreset("Neutral", "moderate pitch"),
        StylePreset("Whisper", "whisper"),                 // voiced 72.4 % -> 20.3 %
        StylePreset("Very low", "very low pitch"),         // -82.9 Hz
        StylePreset("Low", "low pitch"),                   // +32.9 Hz (weak, non-monotonic)
        StylePreset("High", "high pitch"),                 // +59.1 Hz
        StylePreset("Very high", "very high pitch"),       // +85.5 Hz
        StylePreset("Elderly", "elderly"),                 // -79.8 Hz
        StylePreset("Child", "child"),                     // +41.7 Hz
        StylePreset("Female", "female"),                   // +22.7 Hz
        StylePreset("Male", "male"),
    )

    /**
     * Joins and validates. Upstream uses ", " for English items and raises on
     * unknown ones; we return null instead so the caller can report it.
     */
    override fun resolveInstruct(items: List<String>): String? {
        val cleaned = items.map { it.trim().lowercase() }.filter { it.isNotEmpty() }
        if (cleaned.isEmpty()) return null
        if (cleaned.any { it !in VOCABULARY }) return null
        return cleaned.joinToString(", ")
    }

    fun unsupported(items: List<String>): List<String> =
        items.map { it.trim().lowercase() }.filter { it.isNotEmpty() && it !in VOCABULARY }

    /** Non-verbal tags, verified acoustically (docs/feature-validation.md §5). */
    val NON_VERBAL = listOf(
        "laughter", "sigh", "surprise-ah", "surprise-oh", "surprise-wa",
        "surprise-yo", "question-en", "question-ah", "question-oh", "question-ei",
        "question-yi", "confirmation-en", "dissatisfaction-hnn",
    )

    /**
     * The subset measured to have a clear effect. `dissatisfaction-hnn` moved the
     * voiced fraction only +1.3σ and is held back until checked by ear; the
     * remaining tags were not individually probed.
     */
    val NON_VERBAL_VERIFIED = listOf(
        "laughter", "sigh", "surprise-ah", "surprise-oh", "question-en",
        "confirmation-en",
    )
}
