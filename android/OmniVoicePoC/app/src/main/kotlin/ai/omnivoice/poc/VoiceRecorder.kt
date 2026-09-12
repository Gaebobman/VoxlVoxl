package ai.omnivoice.poc

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlin.concurrent.thread

/**
 * Spec F-A01 — microphone capture for enrollment.
 *
 * Records mono 16-bit at 24 kHz, the codec's own rate, so the enrollment path
 * does no resampling at all. `VOICE_RECOGNITION` rather than `MIC` because it
 * asks the platform for the least processed signal: AGC and noise suppression
 * would alter exactly the timbre we are trying to capture.
 */
class VoiceRecorder {

    companion object {
        const val TAG = "OmniVoice.Rec"
        const val SAMPLE_RATE = OV.SR_24K
        /** Below this the reference is too short to characterise a speaker. */
        const val MIN_SECONDS = 3.0f
        /** Upstream warns past 20 s: slower generation, no quality gain. */
        const val RECOMMENDED_MAX_SECONDS = 12.0f
    }

    private var record: AudioRecord? = null
    private var worker: Thread? = null
    @Volatile private var running = false
    private val collected = ArrayList<ShortArray>()
    @Volatile private var collectedCount = 0

    val isRecording: Boolean get() = running
    val seconds: Float get() = collectedCount.toFloat() / SAMPLE_RATE

    /** @param onLevel 0..1 RMS for a level meter; called ~20x/second. */
    fun start(onLevel: ((Float) -> Unit)? = null) {
        if (running) return
        val min = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (min <= 0) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.BAD_SAMPLE_RATE,
                "device does not support ${SAMPLE_RATE}Hz mono 16-bit capture")
        }
        val buf = maxOf(min, SAMPLE_RATE / 10 * 2)
        val r = try {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, buf)
        } catch (e: SecurityException) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.WAV_DECODE_FAILED,
                "RECORD_AUDIO permission not granted", e)
        }
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            r.release()
            throw OmniVoiceException(
                OmniVoiceException.Kind.WAV_DECODE_FAILED, "AudioRecord failed to initialise")
        }

        collected.clear()
        collectedCount = 0
        record = r
        running = true
        r.startRecording()
        worker = thread(name = "voice-recorder") {
            val chunk = ShortArray(SAMPLE_RATE / 20)     // 50 ms
            while (running) {
                val n = r.read(chunk, 0, chunk.size)
                if (n <= 0) continue
                collected.add(chunk.copyOf(n))
                collectedCount += n
                if (onLevel != null) {
                    var acc = 0.0
                    for (i in 0 until n) {
                        val v = chunk[i] / 32768.0
                        acc += v * v
                    }
                    onLevel(Math.sqrt(acc / n).toFloat())
                }
            }
        }
        Log.i(TAG, "recording at ${SAMPLE_RATE}Hz")
    }

    /** @return mono float32 at [SAMPLE_RATE]. */
    fun stop(): FloatArray {
        running = false
        worker?.join(1000)
        worker = null
        runCatching { record?.stop() }
        runCatching { record?.release() }
        record = null

        val out = FloatArray(collectedCount)
        var p = 0
        for (c in collected) {
            for (v in c) out[p++] = v / 32768.0f
        }
        collected.clear()
        Log.i(TAG, "recorded ${"%.2f".format(out.size.toFloat() / SAMPLE_RATE)}s")
        return out
    }

    fun cancel() {
        running = false
        worker?.join(1000)
        worker = null
        runCatching { record?.stop() }
        runCatching { record?.release() }
        record = null
        collected.clear()
        collectedCount = 0
    }
}
