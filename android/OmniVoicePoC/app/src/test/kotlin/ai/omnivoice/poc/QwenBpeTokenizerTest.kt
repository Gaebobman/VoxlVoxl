package ai.omnivoice.poc

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class QwenBpeTokenizerTest {

    private val tok by lazy { QwenBpeTokenizer.fromJson(Fixtures.tokenizerJson) }

    @Test
    fun `matches huggingface tokenizer on every fixture case`() {
        val cases = Fixtures.load("tokenizer.json").getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val text = c.getString("text")
            val want = c.getJSONArray("ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
            assertArrayEquals(want, tok.encode(text), "encode(${text.take(40)})")

            val wantNv = c.getJSONArray("ids_nonverbal")
                .let { a -> IntArray(a.length()) { a.getInt(it) } }
            assertArrayEquals(
                wantNv, tok.encodeWithNonverbal(text),
                "encodeWithNonverbal(${text.take(40)})",
            )
        }
    }

    @Test
    fun `omnivoice special tokens have the ids the model was trained with`() {
        val expected = mapOf(
            OV.TOK_DENOISE to 151669,
            OV.TOK_LANG_START to 151670, OV.TOK_LANG_END to 151671,
            OV.TOK_INSTRUCT_START to 151672, OV.TOK_INSTRUCT_END to 151673,
            OV.TOK_TEXT_START to 151674, OV.TOK_TEXT_END to 151675,
        )
        for ((token, id) in expected) {
            assertArrayEquals(intArrayOf(id), tok.encode(token), token)
        }
    }

    @Test
    fun `assembles the exact prompt PyTorch built`() {
        val f = Fixtures.load("prompt.json")
        val refText = f.getString("ref_text")
        val text = f.getString("text")
        val style = buildString {
            if (f.getBoolean("denoise")) append(OV.TOK_DENOISE)
            append(OV.TOK_LANG_START).append(f.getString("language")).append(OV.TOK_LANG_END)
            append(OV.TOK_INSTRUCT_START).append("None").append(OV.TOK_INSTRUCT_END)
        }
        val ids = tok.encode(style) +
            tok.encodeWithNonverbal(
                OV.TOK_TEXT_START + TextUtils.combine(text, refText) + OV.TOK_TEXT_END)

        val want = f.getJSONArray("text_row_ids")
            .let { a -> IntArray(a.length()) { a.getInt(it) } }
        assertArrayEquals(want, ids, "assembled prompt token ids")
        assertEquals(f.getInt("text_token_count"), ids.size)
    }
}
