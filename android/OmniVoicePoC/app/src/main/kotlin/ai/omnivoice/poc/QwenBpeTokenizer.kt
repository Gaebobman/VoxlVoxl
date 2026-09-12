package ai.omnivoice.poc

import org.json.JSONObject
import java.io.File

/**
 * Byte-level BPE for the Qwen2 tokenizer shipped with k2-fsa/OmniVoice.
 *
 * Reimplemented rather than pulled in as a dependency: the HuggingFace
 * `tokenizers` crate has no Android artifact, and onnxruntime-extensions would
 * add a second native library for one text pass. The pieces that matter:
 *
 *  - NFC normalisation
 *  - the GPT-2 split regex, applied in "Isolated" mode
 *  - byte-level mapping of UTF-8 bytes into the printable BMP range
 *  - greedy lowest-rank merges over the ranked merge table
 *  - added/special tokens matched before any of the above
 *
 * Pinned by android/fixtures/tokenizer.json.
 */
class QwenBpeTokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val merges: Map<Long, Int>,
    private val addedTokens: List<Pair<String, Int>>,   // longest-first
) {

    companion object {
        /** GPT-2 / Qwen2 pre-tokenizer pattern, from tokenizer.json. */
        private val SPLIT = Regex(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)" +
                "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" +
                "|\\p{N}" +
                "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +
                "|\\s*[\\r\\n]+" +
                "|\\s+(?!\\S)" +
                "|\\s+",
            RegexOption.IGNORE_CASE,
        )

        /** GPT-2 byte <-> unicode table. */
        private val BYTE_TO_UNICODE: IntArray = run {
            val out = IntArray(256)
            val taken = BooleanArray(256)
            var n = 0
            fun span(from: Int, to: Int) {
                for (b in from..to) { out[b] = b; taken[b] = true }
            }
            span('!'.code, '~'.code)
            span('¡'.code, '¬'.code)
            span('®'.code, 'ÿ'.code)
            for (b in 0..255) {
                if (!taken[b]) { out[b] = 256 + n; n++ }
            }
            out
        }

        fun fromFile(path: File): QwenBpeTokenizer = fromJson(path.readText())

        fun fromJson(text: String): QwenBpeTokenizer {
            val root = JSONObject(text)
            val model = root.getJSONObject("model")

            val vocabJson = model.getJSONObject("vocab")
            val vocab = HashMap<String, Int>(vocabJson.length() * 2)
            for (k in vocabJson.keys()) vocab[k] = vocabJson.getInt(k)

            val mergesJson = model.getJSONArray("merges")
            val merges = HashMap<Long, Int>(mergesJson.length() * 2)
            for (i in 0 until mergesJson.length()) {
                // tokenizers >= 0.20 writes merges as ["a", "b"]; older ones as "a b"
                val left: String
                val right: String
                val item = mergesJson.get(i)
                if (item is org.json.JSONArray) {
                    left = item.getString(0); right = item.getString(1)
                } else {
                    val s = item as String
                    val sp = s.indexOf(' ')
                    left = s.substring(0, sp); right = s.substring(sp + 1)
                }
                val a = vocab[left] ?: continue
                val b = vocab[right] ?: continue
                merges[pack(a, b)] = i
            }

            val added = ArrayList<Pair<String, Int>>()
            val addedJson = root.optJSONArray("added_tokens")
            if (addedJson != null) {
                for (i in 0 until addedJson.length()) {
                    val t = addedJson.getJSONObject(i)
                    added.add(t.getString("content") to t.getInt("id"))
                }
            }
            added.sortByDescending { it.first.length }
            return QwenBpeTokenizer(vocab, merges, added)
        }

        private fun pack(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
    }

    /** Encode without adding any special tokens (OmniVoice adds its own). */
    fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val out = ArrayList<Int>(text.length)
        splitOnAddedTokens(text) { piece, addedId ->
            if (addedId != null) {
                out.add(addedId)
            } else {
                val normalized = java.text.Normalizer.normalize(piece, java.text.Normalizer.Form.NFC)
                for (m in SPLIT.findAll(normalized)) {
                    encodeChunk(m.value, out)
                }
            }
        }
        return out.toIntArray()
    }

    /**
     * Mirrors `_tokenize_with_nonverbal_tags`: each `[laughter]`-style tag is
     * tokenized standalone so its ids never depend on the surrounding language.
     */
    fun encodeWithNonverbal(text: String): IntArray {
        val out = ArrayList<Int>(text.length)
        var last = 0
        for (m in NONVERBAL.findAll(text)) {
            if (m.range.first > last) out.addAll(encode(text.substring(last, m.range.first)).toList())
            out.addAll(encode(m.value).toList())
            last = m.range.last + 1
        }
        if (last < text.length) out.addAll(encode(text.substring(last)).toList())
        if (out.isEmpty()) return encode(text)
        return out.toIntArray()
    }

    private val NONVERBAL = Regex(
        "\\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|" +
            "question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|" +
            "surprise-yo|dissatisfaction-hnn)\\]"
    )

    private inline fun splitOnAddedTokens(text: String, emit: (String, Int?) -> Unit) {
        var i = 0
        var plainStart = 0
        outer@ while (i < text.length) {
            for ((content, id) in addedTokens) {
                if (text.startsWith(content, i)) {
                    if (i > plainStart) emit(text.substring(plainStart, i), null)
                    emit(content, id)
                    i += content.length
                    plainStart = i
                    continue@outer
                }
            }
            i++
        }
        if (plainStart < text.length) emit(text.substring(plainStart), null)
    }

    private fun encodeChunk(chunk: String, out: MutableList<Int>) {
        if (chunk.isEmpty()) return
        val bytes = chunk.toByteArray(Charsets.UTF_8)

        // byte-level mapping, one symbol per byte
        var ids = ArrayList<Int>(bytes.size)
        for (b in bytes) {
            val sym = String(Character.toChars(BYTE_TO_UNICODE[b.toInt() and 0xFF]))
            val id = vocab[sym] ?: throw OmniVoiceException(
                OmniVoiceException.Kind.TOKENIZER_FAILED,
                "byte symbol $sym missing from vocab",
            )
            ids.add(id)
        }

        // greedy lowest-rank merge
        while (ids.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestAt = -1
            for (i in 0 until ids.size - 1) {
                val r = merges[pack(ids[i], ids[i + 1])] ?: continue
                if (r < bestRank) { bestRank = r; bestAt = i }
            }
            if (bestAt < 0) break
            val merged = ArrayList<Int>(ids.size - 1)
            var i = 0
            while (i < ids.size) {
                if (i == bestAt) {
                    val a = keyOf(ids[i]); val b = keyOf(ids[i + 1])
                    merged.add(
                        vocab[a + b] ?: throw OmniVoiceException(
                            OmniVoiceException.Kind.TOKENIZER_FAILED,
                            "merge result '$a$b' missing from vocab",
                        )
                    )
                    i += 2
                } else {
                    merged.add(ids[i]); i++
                }
            }
            ids = merged
        }
        out.addAll(ids)
    }

    private val idToToken: Array<String?> by lazy {
        val max = vocab.values.max()
        val arr = arrayOfNulls<String>(max + 1)
        for ((k, v) in vocab) arr[v] = k
        arr
    }

    private fun keyOf(id: Int): String = idToToken[id]
        ?: throw OmniVoiceException(OmniVoiceException.Kind.TOKENIZER_FAILED, "unknown id $id")
}
