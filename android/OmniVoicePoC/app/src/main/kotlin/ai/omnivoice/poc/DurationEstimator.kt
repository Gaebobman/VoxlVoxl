package ai.omnivoice.poc

/**
 * Port of `RuleDurationEstimator` from omnivoice/utils/duration.py (Apache-2.0).
 *
 * Pure table lookup, no model. It decides how many audio frames to generate, so
 * a drift here makes every utterance the wrong length. Pinned by
 * android/fixtures/duration.json.
 */
object DurationEstimator {

    private const val W_CJK = 3.0
    private const val W_HANGUL = 2.5
    private const val W_KANA = 2.2
    private const val W_ETHIOPIC = 3.0
    private const val W_YI = 3.0
    private const val W_INDIC = 1.8
    private const val W_THAI_LAO = 1.5
    private const val W_KHMER_MYANMAR = 1.8
    private const val W_ARABIC = 1.5
    private const val W_HEBREW = 1.5
    private const val W_LATIN = 1.0
    private const val W_CYRILLIC = 1.0
    private const val W_GREEK = 1.0
    private const val W_ARMENIAN = 1.0
    private const val W_GEORGIAN = 1.0
    private const val W_PUNCT = 0.5
    private const val W_SPACE = 0.2
    private const val W_DIGIT = 3.5
    private const val W_MARK = 0.0
    private const val W_DEFAULT = 1.0

    // (inclusive upper code point, weight) — binary-searched, in ascending order.
    private val BREAKS = intArrayOf(
        0x02AF, 0x03FF, 0x052F, 0x058F, 0x05FF, 0x077F, 0x089F, 0x08FF, 0x097F,
        0x09FF, 0x0A7F, 0x0AFF, 0x0B7F, 0x0BFF, 0x0C7F, 0x0CFF, 0x0D7F, 0x0DFF,
        0x0EFF, 0x0FFF, 0x109F, 0x10FF, 0x11FF, 0x137F, 0x139F, 0x13FF, 0x167F,
        0x169F, 0x16FF, 0x171F, 0x173F, 0x175F, 0x177F, 0x17FF, 0x18AF, 0x18FF,
        0x194F, 0x19DF, 0x19FF, 0x1A1F, 0x1AAF, 0x1B7F, 0x1BBF, 0x1BFF, 0x1C4F,
        0x1C7F, 0x1C8F, 0x1CBF, 0x1CCF, 0x1CFF, 0x1D7F, 0x1DBF, 0x1DFF, 0x1EFF,
        0x309F, 0x30FF, 0x312F, 0x318F, 0x9FFF, 0xA4CF, 0xA4FF, 0xA63F, 0xA69F,
        0xA6FF, 0xA7FF, 0xA82F, 0xA87F, 0xA8DF, 0xA8FF, 0xA92F, 0xA95F, 0xA97F,
        0xA9DF, 0xA9FF, 0xAA5F, 0xAA7F, 0xAADF, 0xAAFF, 0xAB2F, 0xAB6F, 0xABBF,
        0xABFF, 0xD7AF, 0xFAFF, 0xFDFF, 0xFE6F, 0xFEFF, 0xFFEF,
    )
    private val WEIGHTS = doubleArrayOf(
        W_LATIN, W_GREEK, W_CYRILLIC, W_ARMENIAN, W_HEBREW, W_ARABIC, W_ARABIC,
        W_ARABIC, W_INDIC, W_INDIC, W_INDIC, W_INDIC, W_INDIC, W_INDIC, W_INDIC,
        W_INDIC, W_INDIC, W_INDIC, W_THAI_LAO, W_INDIC, W_KHMER_MYANMAR,
        W_GEORGIAN, W_HANGUL, W_ETHIOPIC, W_ETHIOPIC, W_DEFAULT, W_DEFAULT,
        W_DEFAULT, W_DEFAULT, W_DEFAULT, W_DEFAULT, W_DEFAULT, W_DEFAULT,
        W_KHMER_MYANMAR, W_DEFAULT, W_DEFAULT, W_INDIC, W_INDIC,
        W_KHMER_MYANMAR, W_INDIC, W_INDIC, W_INDIC, W_INDIC, W_INDIC, W_INDIC,
        W_INDIC, W_CYRILLIC, W_GEORGIAN, W_INDIC, W_INDIC, W_LATIN, W_LATIN,
        W_DEFAULT, W_LATIN, W_KANA, W_KANA, W_CJK, W_HANGUL, W_CJK, W_YI,
        W_DEFAULT, W_DEFAULT, W_CYRILLIC, W_DEFAULT, W_LATIN, W_INDIC,
        W_DEFAULT, W_INDIC, W_INDIC, W_INDIC, W_INDIC, W_HANGUL, W_INDIC,
        W_KHMER_MYANMAR, W_INDIC, W_KHMER_MYANMAR, W_INDIC, W_INDIC,
        W_ETHIOPIC, W_LATIN, W_DEFAULT, W_INDIC, W_HANGUL, W_CJK, W_ARABIC,
        W_DEFAULT, W_ARABIC, W_LATIN,
    )

    init {
        require(BREAKS.size == WEIGHTS.size) { "duration tables out of sync" }
    }

    /** Weight of a single code point. Mirrors `_get_char_weight`. */
    fun charWeight(code: Int): Double {
        if ((code in 65..90) || (code in 97..122)) return W_LATIN
        if (code == 32) return W_SPACE
        if (code == 0x0640) return W_MARK          // Arabic tatweel

        when (Character.getType(code).toByte()) {
            Character.NON_SPACING_MARK, Character.ENCLOSING_MARK,
            Character.COMBINING_SPACING_MARK -> return W_MARK

            Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION,
            Character.START_PUNCTUATION, Character.END_PUNCTUATION,
            Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
            Character.OTHER_PUNCTUATION, Character.MATH_SYMBOL,
            Character.CURRENCY_SYMBOL, Character.MODIFIER_SYMBOL,
            Character.OTHER_SYMBOL -> return W_PUNCT

            Character.SPACE_SEPARATOR, Character.LINE_SEPARATOR,
            Character.PARAGRAPH_SEPARATOR -> return W_SPACE

            Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER,
            Character.OTHER_NUMBER -> return W_DIGIT
        }

        val i = lowerBound(code)
        if (i < BREAKS.size) return WEIGHTS[i]
        if (code > 0x20000) return W_CJK
        return W_DEFAULT
    }

    /** Python's `bisect_left` over [BREAKS]. */
    private fun lowerBound(code: Int): Int {
        var lo = 0
        var hi = BREAKS.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (BREAKS[mid] < code) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** Sum of code-point weights. Iterates by code point, not by UTF-16 unit. */
    fun textWeight(s: String): Double {
        var total = 0.0
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            total += charWeight(cp)
            i += Character.charCount(cp)
        }
        return total
    }

    /**
     * Frames of audio to generate. Mirrors `_estimate_target_tokens` plus
     * `estimate_duration`; [refFrames] is a frame count, not seconds.
     */
    fun estimateFrames(
        text: String,
        refText: String?,
        refFrames: Int,
        speed: Float = 1.0f,
        lowThreshold: Double = 50.0,
        boostStrength: Double = 3.0,
    ): Int {
        var ref = refText ?: ""
        var frames = refFrames
        if (ref.isEmpty() || frames <= 0) {
            ref = "Nice to meet you."
            frames = 25
        }
        val refWeight = textWeight(ref)
        if (refWeight == 0.0) return 1

        var est = textWeight(text) / (refWeight / frames)
        if (speed > 0f && speed != 1.0f) est /= speed
        if (est < lowThreshold) {
            est = lowThreshold * Math.pow(est / lowThreshold, 1.0 / boostStrength)
        }
        return maxOf(1, est.toInt())
    }
}
