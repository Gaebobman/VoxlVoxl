package ai.omnivoice.poc

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TextUtilsTest {

    @Test
    fun `matches python combine_text and add_punctuation`() {
        val cases = Fixtures.load("text.json").getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val text = c.getString("text")
            val refText = if (c.isNull("ref_text")) null else c.getString("ref_text")
            assertEquals(
                c.getString("combined"), TextUtils.combine(text, refText),
                "combine(${text.take(30)}, ref=${refText?.take(16)})",
            )
            assertEquals(
                c.getString("punctuated"), TextUtils.addPunctuation(text),
                "addPunctuation(${text.take(30)})",
            )
        }
    }
}
