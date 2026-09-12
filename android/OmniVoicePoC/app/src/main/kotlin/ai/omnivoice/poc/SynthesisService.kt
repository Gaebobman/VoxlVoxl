package ai.omnivoice.poc

import ai.omnivoice.poc.core.AudioResult
import ai.omnivoice.poc.core.SynthesisCancelledException
import ai.omnivoice.poc.core.SynthesisProgress
import ai.omnivoice.poc.core.VoiceProfile
import ai.omnivoice.poc.core.VoiceStyle
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File

/**
 * Synthesis runs here, not in an Activity.
 *
 * This is forced by measurement, not by taste: 10.57 s of audio takes 79 s on
 * this device, and a foreground-only job dies the moment the user switches app
 * or the screen locks. A foreground service with a notification is the only way
 * Android lets a minute of CPU work survive.
 */
class SynthesisService : Service() {

    companion object {
        const val TAG = "OmniVoice.Service"
        private const val CHANNEL = "synthesis"
        private const val NOTIFICATION_ID = 1

        /** Measured best on the S26 Ultra; 8 threads is slower than 6. */
        const val DEFAULT_THREADS = 6
    }

    sealed interface Status {
        data object Idle : Status
        data object Loading : Status
        data class Running(val progress: SynthesisProgress, val etaSeconds: Float?) : Status
        data class Done(val result: AudioResult) : Status
        data class Failed(val error: Throwable) : Status
        data object Cancelled : Status
    }

    inner class LocalBinder : Binder() {
        val service: SynthesisService get() = this@SynthesisService
    }

    private val binder = LocalBinder()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var job: Job? = null
    private var engine: OmniVoiceEngine? = null

    private val _status = MutableStateFlow<Status>(Status.Idle)
    val status: StateFlow<Status> = _status

    val modelDir: File get() = File(filesDir, "models")
    val voiceDir: File get() = File(filesDir, "voices")

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL, "Speech synthesis",
                    NotificationManager.IMPORTANCE_LOW))
        }
    }

    /**
     * @param steps 16 is the measured knee — half the latency of 32 for a quality
     *   drop smaller than the model's own sampling noise (docs/benchmark.md §2).
     */
    fun synthesize(
        text: String,
        profile: VoiceProfile?,
        style: VoiceStyle? = null,
        steps: Int = 16,
        threads: Int = DEFAULT_THREADS,
    ) {
        if (job?.isActive == true) {
            Log.w(TAG, "a synthesis is already running; ignoring")
            return
        }
        startForegroundSafely("Preparing…", 0)
        _status.value = Status.Loading

        job = scope.launch {
            val started = System.nanoTime()
            try {
                val e = engine ?: OmniVoiceEngine(
                    modelDir, Backend.CPU, threads,
                    cfg = GenConfig(numStep = steps),
                    thermalStatus = {
                        runCatching {
                            getSystemService(PowerManager::class.java).currentThermalStatus
                        }.getOrDefault(-1)
                    },
                ).also { engine = it }
                e.load()

                val result = e.synthesize(
                    text = text,
                    voiceProfile = profile,
                    style = style,
                    onProgress = { p ->
                        val elapsed = (System.nanoTime() - started) / 1e9f
                        val eta = if (p.fraction > 0.02f)
                            elapsed * (1f - p.fraction) / p.fraction else null
                        _status.value = Status.Running(p, eta)
                        notify("Generating… ${(p.fraction * 100).toInt()}%" +
                            (eta?.let { "  ~${it.toInt()}s left" } ?: ""),
                            (p.fraction * 100).toInt())
                    },
                    isActive = { this.isActive },
                )
                _status.value = Status.Done(result)
                notify("Done — ${"%.1f".format(result.metrics.audioSeconds)}s", 100)
            } catch (e: SynthesisCancelledException) {
                _status.value = Status.Cancelled
            } catch (e: Throwable) {
                Log.e(TAG, "synthesis failed", e)
                _status.value = Status.Failed(e)
            } finally {
                stopForegroundCompat()
            }
        }
    }

    fun cancel() {
        job?.cancel()
        job = null
    }

    sealed interface EnrollStatus {
        data object Idle : EnrollStatus
        data object Encoding : EnrollStatus
        data class Ready(val profile: VoiceProfile, val echo: FloatArray, val millis: Long) : EnrollStatus
        data class Failed(val error: Throwable) : EnrollStatus
    }

    private val _enroll = MutableStateFlow<EnrollStatus>(EnrollStatus.Idle)
    val enrollStatus: StateFlow<EnrollStatus> = _enroll

    /**
     * Enroll, then immediately decode the result back to audio so the user can
     * hear what the model will hear. The generation engine is released first:
     * the encoders need ~654 MB and the backbone ~590 MB, and there is no reason
     * for both to be resident.
     */
    fun enroll(samples: FloatArray, sampleRate: Int, refText: String, displayName: String) {
        if (job?.isActive == true) {
            Log.w(TAG, "a synthesis is running; not enrolling")
            return
        }
        _enroll.value = EnrollStatus.Encoding
        job = scope.launch {
            try {
                releaseEngine()
                val t0 = System.nanoTime()
                val (profile, echo) = OmniVoiceEnroller(modelDir, DEFAULT_THREADS).use { e ->
                    val p = e.enroll(samples, sampleRate, refText, displayName)
                    p to e.echo(p)
                }
                val ms = (System.nanoTime() - t0) / 1_000_000
                Log.i(TAG, "enrolled '$displayName' + echo in ${ms}ms")
                _enroll.value = EnrollStatus.Ready(profile, echo, ms)
            } catch (e: Throwable) {
                Log.e(TAG, "enrollment failed", e)
                _enroll.value = EnrollStatus.Failed(e)
            }
        }
    }

    fun clearEnrollment() {
        _enroll.value = EnrollStatus.Idle
    }

    /** Frees ~590 MB. Worth doing when the app goes to background for a while. */
    fun releaseEngine() {
        cancel()
        engine?.close()
        engine = null
    }

    private fun buildNotification(text: String, progress: Int): Notification {
        val b = Notification.Builder(this, CHANNEL)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
        if (progress in 0..100) b.setProgress(100, progress, progress == 0)
        return b.build()
    }

    private fun startForegroundSafely(text: String, progress: Int) {
        val n = buildNotification(text, progress)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, n,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, n)
        }
    }

    private fun notify(text: String, progress: Int) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, buildNotification(text, progress))
    }

    private fun stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION") stopForeground(true)
        }
    }

    override fun onDestroy() {
        scope.cancel()
        engine?.close()
        engine = null
        super.onDestroy()
    }
}
