package ai.omnivoice.poc

/**
 * Picks the `<|lang_start|>` value from the script of the text.
 *
 * Language was a Developer-settings dropdown pinned to "ko", which is wrong the
 * moment a Korean user types an English sentence. The Unicode block tables that
 * decide this already exist in [DurationEstimator] — the same ranges that weight
 * a character's speaking time identify its script — so detection costs nothing
 * new and needs no model.
 *
 * Deliberately coarse: it distinguishes SCRIPTS, not languages, so it returns
 * null rather than guessing between languages that share one (Cyrillic could be
 * Russian or Ukrainian; Latin could be anything). The caller then sends "None",
 * which is what OmniVoice expects for language-agnostic generation.
 */
object LanguageDetector {

    /** Scripts where one script maps to one dominant language for our purposes. */
    private const val MIN_SHARE = 0.30f

    data class Detection(val code: String?, val label: String, val confident: Boolean)

    fun detect(text: String): Detection {
        var hangul = 0
        var kana = 0
        var han = 0
        var latin = 0
        var cyrillic = 0
        var arabic = 0
        var thai = 0
        var devanagari = 0
        var total = 0

        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            i += Character.charCount(cp)
            if (!Character.isLetter(cp)) continue
            total++
            when {
                cp in 0xAC00..0xD7A3 || cp in 0x1100..0x11FF || cp in 0x3130..0x318F -> hangul++
                cp in 0x3040..0x30FF -> kana++
                cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF -> han++
                cp in 0x0400..0x052F -> cyrillic++
                cp in 0x0600..0x06FF || cp in 0x0750..0x077F -> arabic++
                cp in 0x0E00..0x0E7F -> thai++
                cp in 0x0900..0x097F -> devanagari++
                cp < 0x0250 || cp in 0x1E00..0x1EFF -> latin++
            }
        }
        if (total == 0) return Detection(null, "auto", false)

        fun share(n: Int) = n.toFloat() / total

        // Japanese first: kana is decisive even in small amounts, because
        // Japanese text is mostly han and would otherwise read as Chinese.
        if (kana > 0 && share(kana + han) > MIN_SHARE) return Detection("ja", "일본어", true)
        if (share(hangul) > MIN_SHARE) return Detection("ko", "한국어", true)
        if (share(han) > MIN_SHARE) return Detection("zh", "중국어", true)
        if (share(thai) > MIN_SHARE) return Detection("th", "태국어", true)
        if (share(devanagari) > MIN_SHARE) return Detection("hi", "힌디어", true)
        if (share(arabic) > MIN_SHARE) return Detection("ar", "아랍어", true)
        // Latin and Cyrillic are shared by too many languages to call from script
        if (share(latin) > MIN_SHARE) return Detection("en", "영어", false)
        if (share(cyrillic) > MIN_SHARE) return Detection("ru", "러시아어", false)
        return Detection(null, "auto", false)
    }
}
