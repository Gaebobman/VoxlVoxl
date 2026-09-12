package ai.omnivoice.poc

/**
 * Port of `_combine_text` and `add_punctuation` from
 * omnivoice/models/omnivoice.py (Apache-2.0).
 *
 * Pinned by android/fixtures/text.json.
 */
object TextUtils {

    private val NEWLINES = Regex("[\\r\\n]+")
    private val SPACES = Regex("[ \\t]+")
    // remove whitespace adjacent to a CJK ideograph, on either side
    private val AROUND_CJK = Regex("(?<=[\\u4e00-\\u9fff])\\s+|\\s+(?=[\\u4e00-\\u9fff])")

    /**
     * Builds the single text field OmniVoice expects: the reference transcript
     * immediately followed by the target text. This concatenation is what lets
     * the model align the reference codec prefix with the reference speech, so
     * it is not optional when cloning.
     */
    fun combine(text: String, refText: String?): String {
        var full = if (!refText.isNullOrEmpty()) {
            refText.trim() + " " + text.trim()
        } else {
            text.trim()
        }
        full = NEWLINES.replace(full, "")
        full = full.replace('（', '(').replace('）', ')')
        full = SPACES.replace(full, " ")
        return AROUND_CJK.replace(full, "")
    }

    private const val TERMINALS = ".。!！?？,，;；:："

    fun addPunctuation(s: String): String {
        val t = s.trim()
        return if (t.isNotEmpty() && TERMINALS.indexOf(t[t.length - 1]) >= 0) t else "$t."
    }
}
