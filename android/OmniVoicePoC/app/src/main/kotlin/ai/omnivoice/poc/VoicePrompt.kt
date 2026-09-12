package ai.omnivoice.poc

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * `voice_prompt.bin` — a speaker enrolled once, reusable forever.
 *
 * Holding the codec codes rather than the audio is what lets the generation
 * build ship without the 654 MB of Higgs encoder graphs, and removes ~1.7 s of
 * encoder work per utterance. Layout in docs/plan.md §3; mirrors
 * scripts/_common.py and android/fixtures/voice_prompt.json.
 */
class VoicePrompt(
    /** `[8][frames]`, values 0..1023. */
    val codes: Array<ShortArray>,
    val refText: String,
    val refRms: Float,
    val sampleRate: Int = OV.SR_24K,
) {
    val numFrames: Int get() = codes[0].size
    val durationSeconds: Float get() = numFrames.toFloat() / OV.FRAME_RATE

    companion object {
        private const val MAGIC = 0x4F565650   // "OVVP" big-endian read as int
        const val VERSION = 1

        fun load(file: File): VoicePrompt {
            val raw = file.readBytes()
            if (raw.size < 28) {
                throw OmniVoiceException(OmniVoiceException.Kind.VOICE_PROMPT_INVALID, "file too short")
            }
            if (raw[0] != 'O'.code.toByte() || raw[1] != 'V'.code.toByte() ||
                raw[2] != 'V'.code.toByte() || raw[3] != 'P'.code.toByte()
            ) {
                throw OmniVoiceException(OmniVoiceException.Kind.VOICE_PROMPT_INVALID, "bad magic")
            }
            val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
            val version = bb.getInt(4)
            if (version != VERSION) {
                throw OmniVoiceException(
                    OmniVoiceException.Kind.VOICE_PROMPT_INVALID, "unsupported version $version")
            }
            val numCodebooks = bb.getInt(8)
            val numFrames = bb.getInt(12)
            val rms = bb.getFloat(16)
            val sr = bb.getInt(20)
            val textLen = bb.getInt(24)
            if (numCodebooks != OV.NUM_CODEBOOKS) {
                throw OmniVoiceException(
                    OmniVoiceException.Kind.VOICE_PROMPT_INVALID,
                    "expected ${OV.NUM_CODEBOOKS} codebooks, got $numCodebooks")
            }
            val textOff = 28
            val text = String(raw, textOff, textLen, Charsets.UTF_8)
            val codesOff = textOff + textLen
            val need = codesOff + numCodebooks * numFrames * 2
            if (raw.size < need) {
                throw OmniVoiceException(
                    OmniVoiceException.Kind.VOICE_PROMPT_INVALID,
                    "truncated: need $need bytes, have ${raw.size}")
            }
            val codes = Array(numCodebooks) { ShortArray(numFrames) }
            var p = codesOff
            for (c in 0 until numCodebooks) {
                for (t in 0 until numFrames) {
                    codes[c][t] = bb.getShort(p)
                    p += 2
                }
            }
            return VoicePrompt(codes, text, rms, sr)
        }
    }

    fun save(file: File) {
        val text = refText.toByteArray(Charsets.UTF_8)
        val n = numFrames
        val bb = ByteBuffer.allocate(28 + text.size + OV.NUM_CODEBOOKS * n * 2)
            .order(ByteOrder.LITTLE_ENDIAN)
        bb.put("OVVP".toByteArray(Charsets.US_ASCII))
        bb.putInt(VERSION).putInt(OV.NUM_CODEBOOKS).putInt(n).putFloat(refRms)
            .putInt(sampleRate).putInt(text.size)
        bb.put(text)
        for (c in 0 until OV.NUM_CODEBOOKS) for (t in 0 until n) bb.putShort(codes[c][t])
        file.parentFile?.mkdirs()
        file.writeBytes(bb.array())
    }
}
