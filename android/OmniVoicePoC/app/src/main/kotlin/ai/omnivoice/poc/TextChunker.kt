package ai.omnivoice.poc

/**
 * Splits long text so peak memory stays bounded.
 *
 * This is a **memory** guard, not a speed optimisation, and the distinction
 * matters because the arithmetic runs the other way. Measured on device: peak PSS
 * went 594 MB at S = 188 to 882 MB at S = 444, while RTF *improved* from 11.21 to
 * 7.48 — the reference prefix is re-paid at every forward, so it amortises over
 * longer outputs. Splitting a paragraph into sentences roughly doubles the work.
 *
 * So: chunk as late as possible, and never below [minFramesPerChunk].
 */
object TextChunker {

    /**
     * Beyond this many generated frames the sequence gets long enough that peak
     * memory becomes the binding constraint. 750 frames = 30 s, matching
     * upstream's `audio_chunk_threshold`.
     */
    const val DEFAULT_THRESHOLD_FRAMES = 750

    /** Target size once splitting is unavoidable. 375 frames = 15 s. */
    const val DEFAULT_CHUNK_FRAMES = 375

    /** Below this a chunk pays more prefix overhead than it saves in memory. */
    const val MIN_FRAMES_PER_CHUNK = 150

    /** Sentence-final punctuation across the scripts OmniVoice targets. */
    private val BOUNDARY = Regex("(?<=[.!?。！？…])\\s+|(?<=[\\uAC00-\\uD7A3][.!?])\\s+")

    /**
     * @param estimate maps a text fragment to its predicted frame count
     * @return the original text as a single element when no split is needed
     */
    fun chunk(
        text: String,
        estimate: (String) -> Int,
        thresholdFrames: Int = DEFAULT_THRESHOLD_FRAMES,
        chunkFrames: Int = DEFAULT_CHUNK_FRAMES,
    ): List<String> {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return listOf(trimmed)
        if (estimate(trimmed) <= thresholdFrames) return listOf(trimmed)

        val sentences = splitSentences(trimmed)
        if (sentences.size <= 1) return listOf(trimmed)   // nothing to split on

        val out = ArrayList<String>()
        val sb = StringBuilder()
        var acc = 0
        for (s in sentences) {
            val n = estimate(s)
            if (acc > 0 && acc + n > chunkFrames && acc >= MIN_FRAMES_PER_CHUNK) {
                out.add(sb.toString().trim())
                sb.setLength(0)
                acc = 0
            }
            if (sb.isNotEmpty()) sb.append(' ')
            sb.append(s)
            acc += n
        }
        if (sb.isNotEmpty()) out.add(sb.toString().trim())
        return out.ifEmpty { listOf(trimmed) }
    }

    fun splitSentences(text: String): List<String> =
        BOUNDARY.split(text).map { it.trim() }.filter { it.isNotEmpty() }

    /**
     * Equal-power cross-fade, used to join chunk boundaries. Upstream applies the
     * same idea in `cross_fade_chunks`.
     */
    /**
     * Where each part begins in [crossFade]'s output. Kept beside it, with the
     * same overlap rule, so the two cannot drift apart.
     */
    fun crossFadeStarts(sizes: IntArray, sr: Int, fadeSeconds: Float = 0.05f): IntArray {
        val starts = IntArray(sizes.size)
        if (sizes.isEmpty()) return starts
        val fade = (fadeSeconds * sr).toInt().coerceAtLeast(1)
        var pos = sizes[0]
        for (i in 1 until sizes.size) {
            val n = minOf(fade, sizes[i], pos)
            starts[i] = pos - n
            pos += sizes[i] - n
        }
        return starts
    }

    fun crossFade(parts: List<FloatArray>, sr: Int, fadeSeconds: Float = 0.05f): FloatArray {
        if (parts.isEmpty()) return FloatArray(0)
        if (parts.size == 1) return parts[0]
        val fade = (fadeSeconds * sr).toInt().coerceAtLeast(1)
        var total = parts.sumOf { it.size } - fade * (parts.size - 1)
        if (total <= 0) total = parts.sumOf { it.size }
        val out = FloatArray(total)
        var pos = 0
        for ((i, p) in parts.withIndex()) {
            if (i == 0) {
                System.arraycopy(p, 0, out, 0, p.size)
                pos = p.size
                continue
            }
            val n = minOf(fade, p.size, pos)
            val start = pos - n
            for (k in 0 until n) {
                // equal power: sin/cos keeps perceived loudness flat through the seam
                val t = (k + 0.5) / n
                val a = Math.cos(t * Math.PI / 2).toFloat()
                val b = Math.sin(t * Math.PI / 2).toFloat()
                out[start + k] = out[start + k] * a + p[k] * b
            }
            val rest = p.size - n
            if (rest > 0) System.arraycopy(p, n, out, pos, rest)
            pos += rest
        }
        return if (pos == out.size) out else out.copyOf(pos)
    }
}
