package ai.omnivoice.poc

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import java.io.File

/**
 * Spec F-B03 / F-B04 — Play / Pause / Stop / Replay, and save to WAV.
 *
 * Uses [AudioTrack] in static mode: the whole clip is already in memory by the
 * time synthesis finishes (a minute of 24 kHz mono float is under 6 MB), so
 * streaming buys nothing and static mode makes pause/replay exact.
 */
class AudioOutput {

    companion object {
        const val TAG = "OmniVoice.Audio"
    }

    enum class State { IDLE, PLAYING, PAUSED }

    private var track: AudioTrack? = null
    private var samples: FloatArray = FloatArray(0)
    private var sampleRate: Int = OV.SR_24K

    var state: State = State.IDLE
        private set

    /** Called on the main thread when the clip reaches its end. */
    var onFinished: (() -> Unit)? = null

    /** 0..1, or 0 when nothing is loaded. */
    val progress: Float
        get() {
            val t = track ?: return 0f
            val n = samples.size
            return if (n == 0) 0f else (t.playbackHeadPosition.toFloat() / n).coerceIn(0f, 1f)
        }

    val durationSeconds: Float
        get() = if (samples.isEmpty()) 0f else samples.size.toFloat() / sampleRate

    fun load(pcm: FloatArray, sr: Int) {
        release()
        samples = pcm
        sampleRate = sr
        if (pcm.isEmpty()) return

        val pcm16 = ShortArray(pcm.size) {
            (pcm[it].coerceIn(-1f, 1f) * 32767f).toInt().toShort()
        }
        val t = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sr)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(pcm16.size * 2)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        t.write(pcm16, 0, pcm16.size)
        // A static track that runs off the end just sits there; without a marker
        // nothing knows playback finished, so the UI kept showing a play button
        // that did nothing.
        t.setNotificationMarkerPosition(pcm16.size)
        t.setPlaybackPositionUpdateListener(object : AudioTrack.OnPlaybackPositionUpdateListener {
            override fun onMarkerReached(unused: AudioTrack?) {
                stop()
                onFinished?.invoke()
            }
            override fun onPeriodicNotification(unused: AudioTrack?) = Unit
        })
        track = t
        state = State.IDLE
        Log.i(TAG, "loaded ${"%.2f".format(durationSeconds)}s @ ${sr}Hz")
    }

    fun play() {
        val t = track ?: return
        // MODE_STATIC rewinds with reloadStaticData(), not setPlaybackHeadPosition:
        // the latter is for streaming tracks and silently fails here, which is why
        // the second press produced nothing.
        if (state == State.IDLE) {
            t.pause()
            val r = t.reloadStaticData()
            if (r != AudioTrack.SUCCESS) Log.w(TAG, "reloadStaticData -> $r")
        }
        t.play()
        state = State.PLAYING
    }

    fun pause() {
        val t = track ?: return
        if (state != State.PLAYING) return
        t.pause()
        state = State.PAUSED
    }

    fun stop() {
        val t = track ?: return
        t.pause()
        // Never flush() a static track — it is a streaming-mode call and it
        // throws away the buffer this class deliberately keeps.
        runCatching { t.reloadStaticData() }
        state = State.IDLE
    }

    fun replay() {
        stop()
        play()
    }

    /** Spec F-B04. */
    fun save(file: File) {
        if (samples.isEmpty()) {
            throw OmniVoiceException(
                OmniVoiceException.Kind.GENERATION_FAILED, "nothing to save")
        }
        WavIo.write(file, samples, sampleRate)
        Log.i(TAG, "saved ${file.absolutePath} (${file.length()} bytes)")
    }

    fun release() {
        runCatching { track?.pause() }
        runCatching { track?.flush() }
        runCatching { track?.release() }
        track = null
        state = State.IDLE
    }
}
