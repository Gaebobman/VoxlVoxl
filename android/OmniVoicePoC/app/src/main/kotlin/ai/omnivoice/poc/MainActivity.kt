package ai.omnivoice.poc

import ai.omnivoice.poc.core.AudioResult
import ai.omnivoice.poc.core.VoiceProfile
import ai.omnivoice.poc.core.VoiceStyle
import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.lifecycle.Lifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * The whole UI. Three groups: enroll a voice, synthesise text, inspect metrics.
 *
 * Shape driven by measurement rather than taste — a generation takes 20-80 s on
 * this device, so progress, an ETA and Cancel are the load-bearing controls, and
 * the work itself lives in [SynthesisService] so it survives app switches.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        const val TAG = "OmniVoice.UI"
        private val STEP_CHOICES = listOf(
            "8 fast" to 8,        // RTF 5.4, quality below the fp32 noise floor
            "16 std" to 16,       // RTF 11.2, the measured knee
            "32 best" to 32,      // RTF 24.6, ties fp32
        )
        private val LANGUAGES = listOf(
            "auto" to "auto", "ko" to "ko", "en" to "en", "ja" to "ja", "zh" to "zh",
        )
    }

    private lateinit var voiceSpinner: Spinner
    private lateinit var recordButton: Button
    private lateinit var deleteVoiceButton: Button
    private lateinit var recordStatus: TextView
    private lateinit var refTextInput: EditText
    private lateinit var enrollButton: Button
    private lateinit var targetTextInput: EditText
    private lateinit var tagRow: LinearLayout
    private lateinit var generateButton: Button
    private lateinit var cancelButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var statusText: TextView
    private lateinit var playButton: Button
    private lateinit var replayButton: Button
    private lateinit var saveButton: Button
    private lateinit var stepsSpinner: Spinner
    private lateinit var languageSpinner: Spinner
    private lateinit var styleSpinner: Spinner
    private lateinit var metricsText: TextView

    private var script: EnrollmentScripts.Script = EnrollmentScripts.ALL[0]
    private val recorder = VoiceRecorder()
    private val player = AudioOutput()
    private lateinit var profiles: FileVoiceProfileManager
    private var profileList: List<VoiceProfile> = emptyList()
    private var recorded: FloatArray? = null
    private var lastResult: AudioResult? = null

    private var service: SynthesisService? = null
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as SynthesisService.LocalBinder).service
            observeStatus()
        }
        override fun onServiceDisconnected(name: ComponentName?) { service = null }
    }

    private val micPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startRecording()
        else toast("Microphone permission is required to enroll a voice")
    }

    private val notifPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* a denied notification only costs the progress display */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        applyWindowInsets()
        bindViews()

        profiles = FileVoiceProfileManager(File(filesDir, "voices"))
        // The transcript is the pipeline's silent failure mode, so the app hands
        // the user a sentence to read instead of asking them to transcribe. The
        // field stays editable for anyone who wants to record their own words.
        refTextInput.setText(script.text)
        refreshProfiles()
        setupSpinners()
        setupTagRow()
        wireButtons()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            notifPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        bindService(Intent(this, SynthesisService::class.java), connection,
            Context.BIND_AUTO_CREATE)

        if (!File(File(filesDir, "models"), "omnivoice_lm.onnx").isFile) {
            statusText.text = "Models are missing. Push them with scripts/push_models.sh."
            generateButton.isEnabled = false
        }
        if (!OmniVoiceEnroller.available(File(filesDir, "models"))) {
            recordStatus.text =
                "Encoder models absent — this build can play a voice but not enroll one."
            recordButton.isEnabled = false
        }
    }

    /**
     * targetSdk 36 enforces edge-to-edge, so the content would otherwise start at
     * y=0 — behind the status bar, with the ActionBar drawn over the first
     * control. Pad the scroll container by the system bars instead.
     */
    private fun applyWindowInsets() {
        val root = findViewById<View>(R.id.rootScroll)
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(
                androidx.core.view.WindowInsetsCompat.Type.systemBars() or
                    androidx.core.view.WindowInsetsCompat.Type.ime())
            v.setPadding(bars.left, 0, bars.right, bars.bottom)
            insets
        }
    }

    private fun bindViews() {
        voiceSpinner = findViewById(R.id.voiceSpinner)
        recordButton = findViewById(R.id.recordButton)
        deleteVoiceButton = findViewById(R.id.deleteVoiceButton)
        recordStatus = findViewById(R.id.recordStatus)
        refTextInput = findViewById(R.id.refTextInput)
        enrollButton = findViewById(R.id.enrollButton)
        targetTextInput = findViewById(R.id.targetTextInput)
        tagRow = findViewById(R.id.tagRow)
        generateButton = findViewById(R.id.generateButton)
        cancelButton = findViewById(R.id.cancelButton)
        progressBar = findViewById(R.id.progressBar)
        statusText = findViewById(R.id.statusText)
        playButton = findViewById(R.id.playButton)
        replayButton = findViewById(R.id.replayButton)
        saveButton = findViewById(R.id.saveButton)
        stepsSpinner = findViewById(R.id.stepsSpinner)
        languageSpinner = findViewById(R.id.languageSpinner)
        styleSpinner = findViewById(R.id.styleSpinner)
        metricsText = findViewById(R.id.metricsText)
    }

    private fun <T> spinner(view: Spinner, labels: List<String>, select: Int = 0) {
        view.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        if (labels.isNotEmpty()) view.setSelection(select.coerceIn(0, labels.size - 1))
    }

    private fun setupSpinners() {
        spinner<Int>(stepsSpinner, STEP_CHOICES.map { it.first }, 1)
        spinner<String>(languageSpinner, LANGUAGES.map { it.second }, 0)
        // Design styles only apply when no profile is selected — measured:
        // with a clone prompt the instruct is ignored entirely.
        spinner<String>(styleSpinner,
            listOf("no style") + OmniVoiceStyle.designPresets().map { it.label }, 0)
    }

    private fun setupTagRow() {
        // Tags append an event rather than colouring the sentence, so they
        // insert at the cursor.
        for (tag in OmniVoiceStyle.NON_VERBAL_VERIFIED) {
            tagRow.addView(Button(this).apply {
                text = tag.substringBefore('-')
                textSize = 11f
                setOnClickListener { insertAtCursor("[$tag]") }
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            })
        }
    }

    private fun insertAtCursor(s: String) {
        val at = targetTextInput.selectionStart.coerceAtLeast(0)
        targetTextInput.text.insert(at, s)
    }

    private fun wireButtons() {
        recordButton.setOnClickListener {
            if (recorder.isRecording) stopRecording()
            else if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
            ) startRecording()
            else micPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        refTextInput.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) { updateEnrollEnabled() }
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        })

        enrollButton.setOnClickListener { enroll() }
        deleteVoiceButton.setOnClickListener { deleteSelectedVoice() }
        deleteVoiceButton.setOnLongClickListener { testSelectedVoice(); true }
        generateButton.setOnClickListener { generate() }
        cancelButton.setOnClickListener { service?.cancel() }

        playButton.setOnClickListener {
            if (player.state == AudioOutput.State.PLAYING) {
                player.pause(); playButton.setText(R.string.play)
            } else {
                player.play(); playButton.setText(R.string.pause)
            }
        }
        replayButton.setOnClickListener {
            player.replay(); playButton.setText(R.string.pause)
        }
        saveButton.setOnClickListener { save() }
    }

    // --- enrollment ------------------------------------------------------

    private fun startRecording() {
        runCatching {
            recorder.start { level ->
                runOnUiThread {
                    recordStatus.text = "Recording %.1fs  level %.2f".format(
                        recorder.seconds, level)
                }
            }
        }.onFailure { toast(it.message ?: "recording failed"); return }
        recorded = null
        recordButton.setText(R.string.stop_recording)
        updateEnrollEnabled()
    }

    private fun stopRecording() {
        val pcm = recorder.stop()
        recorded = pcm
        recordButton.setText(R.string.record)
        val secs = pcm.size.toFloat() / VoiceRecorder.SAMPLE_RATE
        recordStatus.text = when {
            secs < VoiceRecorder.MIN_SECONDS ->
                "Only %.1fs — record at least %.0fs".format(secs, VoiceRecorder.MIN_SECONDS)
            secs > VoiceRecorder.RECOMMENDED_MAX_SECONDS ->
                ("%.1fs recorded. Longer references slow every generation " +
                    "without improving the voice.").format(secs)
            else -> "%.1fs recorded. Now type exactly what you said.".format(secs)
        }
        updateEnrollEnabled()
    }

    private fun updateEnrollEnabled() {
        val pcm = recorded
        enrollButton.isEnabled = pcm != null &&
            pcm.size >= VoiceRecorder.MIN_SECONDS * VoiceRecorder.SAMPLE_RATE &&
            refTextInput.text.isNotBlank()
    }

    private fun enroll() {
        val pcm = recorded ?: return
        val svc = service ?: return toast("Synthesis service is not bound yet")
        enrollButton.isEnabled = false
        recordStatus.text = "목소리를 변환하는 중…"
        svc.enroll(pcm, VoiceRecorder.SAMPLE_RATE, refTextInput.text.toString(),
            SimpleDateFormat("MMM d HH:mm", Locale.getDefault())
                .format(System.currentTimeMillis()))
    }

    /**
     * Enrollment finishes with the echo, not with a test synthesis: the echo is
     * what the model will hear as its reference, costs ~0.5 s warm against ~20 s
     * for a generation, and reveals clipping, over-trimming and level problems
     * that a test sentence would only hint at. "Test voice" stays as the next,
     * optional step.
     */
    private fun renderEnroll(s: SynthesisService.EnrollStatus) {
        when (s) {
            is SynthesisService.EnrollStatus.Idle -> Unit
            is SynthesisService.EnrollStatus.Encoding -> {
                recordStatus.text = "목소리를 변환하는 중…"
                enrollButton.isEnabled = false
            }
            is SynthesisService.EnrollStatus.Ready -> {
                profiles.save(s.profile)
                refreshProfiles()
                voiceSpinner.setSelection(profileList.indexOfFirst { it.id == s.profile.id } + 1)
                player.load(s.echo, OV.SR_24K)
                playButton.isEnabled = true
                replayButton.isEnabled = true
                recordStatus.text = "등록 완료 · %.1f초. 재생을 눌러 시스템이 들은 소리를 확인하세요 (%d ms)"
                    .format(s.profile.frames.toFloat() / OV.FRAME_RATE, s.millis)
                recorded = null
                refTextInput.setText(script.text)
                service?.clearEnrollment()
                updateEnrollEnabled()
            }
            is SynthesisService.EnrollStatus.Failed -> {
                recordStatus.text = s.error.message ?: "등록 실패"
                Log.e(TAG, "enrollment failed", s.error)
                service?.clearEnrollment()
                updateEnrollEnabled()
            }
        }
    }

    private fun refreshProfiles() {
        profileList = profiles.list()
        val labels = listOf(getString(R.string.no_voice)) +
            profileList.map { "${it.displayName}  (%.1fs)".format(it.frames.toFloat() / OV.FRAME_RATE) }
        val keep = voiceSpinner.selectedItemPosition.coerceIn(0, labels.size - 1)
        spinner<String>(voiceSpinner, labels, keep)
        deleteVoiceButton.isEnabled = profileList.isNotEmpty()
    }

    private fun selectedProfile(): VoiceProfile? =
        profileList.getOrNull(voiceSpinner.selectedItemPosition - 1)

    /** Spec §6 "Test Voice" — the optional full synthesis, after the echo. */
    private fun testSelectedVoice() {
        val p = selectedProfile() ?: return toast("No voice selected")
        service?.synthesize(
            text = EnrollmentScripts.TEST_SENTENCE, profile = p,
            style = VoiceStyle(language = "ko"), steps = 16,
        ) ?: toast("Synthesis service is not bound yet")
    }

    private fun deleteSelectedVoice() {
        val p = selectedProfile() ?: return toast("No voice selected")
        AlertDialog.Builder(this)
            .setTitle("Delete ${p.displayName}?")
            .setMessage("The voice profile is removed from this device. This cannot be undone.")
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                profiles.delete(p.id)
                refreshProfiles()
            }
            .show()
    }

    // --- synthesis -------------------------------------------------------

    private fun generate() {
        val text = targetTextInput.text.toString().trim()
        if (text.isEmpty()) return toast("Enter some text")

        val profile = selectedProfile()
        val styleIndex = styleSpinner.selectedItemPosition - 1
        val instruct = if (profile == null && styleIndex >= 0)
            OmniVoiceStyle.designPresets()[styleIndex].instruct else null
        if (profile != null && styleIndex >= 0) {
            toast("Style presets apply only without a voice — record a different " +
                "reference to change how the voice speaks")
        }

        val chosen = LANGUAGES[languageSpinner.selectedItemPosition].first
        val language = if (chosen == "auto") {
            // The Unicode tables that weight speaking time also identify the
            // script, so detection is free and beats a dropdown pinned to "ko".
            LanguageDetector.detect(text).code ?: "None"
        } else chosen
        val style = VoiceStyle(instruct = instruct, language = language)
        service?.synthesize(
            text = text, profile = profile, style = style,
            steps = STEP_CHOICES[stepsSpinner.selectedItemPosition].second,
        ) ?: toast("Synthesis service is not bound yet")
    }

    private fun observeStatus() {
        val svc = service ?: return
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { svc.status.collect { render(it) } }
                launch { svc.enrollStatus.collect { renderEnroll(it) } }
            }
        }
    }

    private fun render(s: SynthesisService.Status) {
        val busy = s is SynthesisService.Status.Running || s is SynthesisService.Status.Loading
        generateButton.isEnabled = !busy
        cancelButton.isEnabled = busy
        when (s) {
            is SynthesisService.Status.Idle -> statusText.text = "Idle"
            is SynthesisService.Status.Loading -> {
                statusText.text = "Loading models…"
                progressBar.isIndeterminate = true
            }
            is SynthesisService.Status.Running -> {
                progressBar.isIndeterminate = false
                progressBar.progress = (s.progress.fraction * 100).toInt()
                statusText.text = buildString {
                    append("Generating  step ${s.progress.step}/${s.progress.totalSteps}")
                    if (s.progress.totalChunks > 1) {
                        append("  chunk ${s.progress.chunk}/${s.progress.totalChunks}")
                    }
                    s.etaSeconds?.let { append("  ~${it.toInt()}s left") }
                }
            }
            is SynthesisService.Status.Done -> {
                progressBar.isIndeterminate = false
                progressBar.progress = 100
                lastResult = s.result
                player.load(s.result.samples, s.result.sampleRate)
                playButton.isEnabled = true
                replayButton.isEnabled = true
                saveButton.isEnabled = true
                playButton.setText(R.string.play)
                val m = s.result.metrics
                statusText.text = "Done — %.2fs of audio".format(m.audioSeconds)
                metricsText.text = buildString {
                    appendLine("RTF          %.2f".format(m.rtf))
                    appendLine("total        %d ms".format(m.totalMillis))
                    appendLine("  generate   %d ms".format(m.generateMillis))
                    appendLine("  vocoder    %d ms".format(m.decodeMillis))
                    appendLine("  post       %d ms".format(m.postMillis))
                    appendLine("model load   %d ms".format(m.modelLoadMillis))
                    appendLine("sequence     %d  frames %d  chunks %d"
                        .format(m.sequenceLength, m.targetFrames, m.chunks))
                    append("thermal      %d %s".format(m.thermalStatus,
                        if (m.thermalStatus > 0) "(warm — expect ~2x slower)" else ""))
                }
            }
            is SynthesisService.Status.Cancelled -> {
                progressBar.isIndeterminate = false
                progressBar.progress = 0
                statusText.text = "Cancelled"
            }
            is SynthesisService.Status.Failed -> {
                progressBar.isIndeterminate = false
                progressBar.progress = 0
                statusText.text = s.error.message ?: "Generation failed"
                Log.e(TAG, "generation failed", s.error)
            }
        }
    }

    private fun save() {
        val r = lastResult ?: return
        val dir = File(getExternalFilesDir(null), "out").apply { mkdirs() }
        val name = "voxlvoxl_%s.wav".format(
            SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis()))
        runCatching {
            WavIo.write(File(dir, name), r.samples, r.sampleRate)
        }.onSuccess {
            toast("Saved ${File(dir, name).absolutePath}")
        }.onFailure {
            toast(it.message ?: "save failed")
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    override fun onStop() {
        super.onStop()
        if (recorder.isRecording) recorder.cancel()
        player.pause()
    }

    override fun onDestroy() {
        player.release()
        runCatching { unbindService(connection) }
        super.onDestroy()
    }
}
