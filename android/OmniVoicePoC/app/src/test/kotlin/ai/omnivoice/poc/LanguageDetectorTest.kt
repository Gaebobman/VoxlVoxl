package ai.omnivoice.poc

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class LanguageDetectorTest {

    private fun code(s: String) = LanguageDetector.detect(s).code

    @Test
    fun `identifies the scripts the app actually sees`() {
        assertEquals("ko", code("오늘 회의를 시작하겠습니다."))
        assertEquals("en", code("Let us begin today's meeting."))
        assertEquals("ja", code("こんにちは、元気ですか？"))
        assertEquals("zh", code("你好，世界！今天天气很好。"))
        assertEquals("ru", code("Привет, как дела?"))
        assertEquals("ar", code("مرحبا بالعالم"))
        assertEquals("hi", code("नमस्ते दुनिया"))
        assertEquals("th", code("สวัสดีครับ"))
    }

    @Test
    fun `kana decides Japanese even when most characters are han`() {
        // Japanese prose is mostly kanji; without the kana rule this reads as zh
        assertEquals("ja", code("今日は会議を始めます。"))
    }

    @Test
    fun `mixed Korean and English follows the majority`() {
        assertEquals("ko", code("오늘 meeting을 시작하겠습니다."))
        assertEquals("en", code("Please read 안녕 out loud and continue in English."))
    }

    @Test
    fun `admits when the script cannot settle the language`() {
        // Latin and Cyrillic are shared by too many languages to call from script
        assertTrue(!LanguageDetector.detect("Hello there").confident)
        assertTrue(LanguageDetector.detect("안녕하세요").confident)
    }

    @Test
    fun `no letters means no guess`() {
        assertEquals(null, code(""))
        assertEquals(null, code("123 456 !!!"))
    }
}
