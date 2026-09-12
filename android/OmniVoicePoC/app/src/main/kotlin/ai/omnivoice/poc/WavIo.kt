package ai.omnivoice.poc

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Minimal 16-bit PCM RIFF reader/writer. Mirrors scripts/_common.py. */
object WavIo {

    class Audio(val samples: FloatArray, val sampleRate: Int)

    fun read(stream: InputStream): Audio {
        val raw = stream.readBytes()
        if (raw.size < 44 || String(raw, 0, 4, Charsets.US_ASCII) != "RIFF" ||
            String(raw, 8, 4, Charsets.US_ASCII) != "WAVE"
        ) {
            throw OmniVoiceException(OmniVoiceException.Kind.WAV_DECODE_FAILED, "not a RIFF/WAVE file")
        }
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12
        var channels = 0
        var sampleRate = 0
        var bits = 0
        var format = 0
        var dataOff = -1
        var dataLen = 0
        while (pos + 8 <= raw.size) {
            val id = String(raw, pos, 4, Charsets.US_ASCII)
            val size = bb.getInt(pos + 4)
            if (size < 0 || pos + 8 + size > raw.size + 1) break
            when (id) {
                "fmt " -> {
                    format = bb.getShort(pos + 8).toInt()
                    channels = bb.getShort(pos + 10).toInt()
                    sampleRate = bb.getInt(pos + 12)
                    bits = bb.getShort(pos + 22).toInt()
                }
                "data" -> { dataOff = pos + 8; dataLen = minOf(size, raw.size - pos - 8) }
            }
            pos += 8 + size + (size and 1)
        }
        if (dataOff < 0 || channels == 0) {
            throw OmniVoiceException(OmniVoiceException.Kind.WAV_DECODE_FAILED, "missing fmt or data chunk")
        }
        if (format != 1 || bits != 16) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.WAV_DECODE_FAILED,
                "only 16-bit PCM is supported (format=$format, bits=$bits)",
            )
        }
        val n = dataLen / 2
        val interleaved = FloatArray(n)
        for (i in 0 until n) interleaved[i] = bb.getShort(dataOff + i * 2) / 32768.0f
        if (channels == 1) return Audio(interleaved, sampleRate)

        val frames = n / channels
        val mono = FloatArray(frames)
        for (i in 0 until frames) {
            var acc = 0.0f
            for (c in 0 until channels) acc += interleaved[i * channels + c]
            mono[i] = acc / channels
        }
        return Audio(mono, sampleRate)
    }

    fun write(file: File, samples: FloatArray, sampleRate: Int) {
        val dataLen = samples.size * 2
        val bb = ByteBuffer.allocate(44 + dataLen).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray(Charsets.US_ASCII)).putInt(36 + dataLen)
        bb.put("WAVE".toByteArray(Charsets.US_ASCII))
        bb.put("fmt ".toByteArray(Charsets.US_ASCII)).putInt(16)
        bb.putShort(1).putShort(1).putInt(sampleRate)
        bb.putInt(sampleRate * 2).putShort(2).putShort(16)
        bb.put("data".toByteArray(Charsets.US_ASCII)).putInt(dataLen)
        for (s in samples) {
            val v = (s.coerceIn(-1.0f, 1.0f) * 32767.0f).toInt()
            bb.putShort(v.toShort())
        }
        file.parentFile?.mkdirs()
        file.writeBytes(bb.array())
    }

    /**
     * Polyphase-free windowed-sinc resampler. Kept modest deliberately: the
     * measured effect of resampler quality on codec agreement is 91.3 % (linear)
     * vs 93.3 % (torchaudio), and codebook 0 stays at 99-100 % either way
     * (docs/benchmark.md §5).
     */
    fun resample(x: FloatArray, srIn: Int, srOut: Int, halfWidth: Int = 16): FloatArray {
        if (srIn == srOut) return x
        val ratio = srOut.toDouble() / srIn
        val n = Math.round(x.size * ratio).toInt()
        val out = FloatArray(n)
        val cutoff = minOf(1.0, ratio)
        for (i in 0 until n) {
            val center = i / ratio
            val i0 = Math.floor(center).toInt()
            var acc = 0.0
            var norm = 0.0
            for (k in (i0 - halfWidth + 1)..(i0 + halfWidth)) {
                if (k < 0 || k >= x.size) continue
                val d = center - k
                val w = sinc(cutoff * d) * blackman(d, halfWidth)
                acc += w * x[k]
                norm += w
            }
            out[i] = if (norm != 0.0) (acc / norm).toFloat() else 0.0f
        }
        return out
    }

    private fun sinc(v: Double): Double =
        if (Math.abs(v) < 1e-12) 1.0 else Math.sin(Math.PI * v) / (Math.PI * v)

    private fun blackman(d: Double, halfWidth: Int): Double {
        val t = (d + halfWidth) / (2.0 * halfWidth)
        if (t < 0.0 || t > 1.0) return 0.0
        return 0.42 - 0.5 * Math.cos(2 * Math.PI * t) + 0.08 * Math.cos(4 * Math.PI * t)
    }
}
