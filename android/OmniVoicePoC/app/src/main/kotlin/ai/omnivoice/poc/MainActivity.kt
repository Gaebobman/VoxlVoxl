package ai.omnivoice.poc

import ai.omnivoice.poc.core.AudioResult
import ai.omnivoice.poc.core.VoiceProfile
import ai.omnivoice.poc.core.VoiceStyle
import ai.omnivoice.poc.ui.CodeStripView
import ai.omnivoice.poc.ui.CodebookLadderView
import ai.omnivoice.poc.ui.Motion
import ai.omnivoice.poc.ui.Motion.animateInt
import ai.omnivoice.poc.ui.Motion.animateNextLayout
import ai.omnivoice.poc.ui.Motion.enterScreen
import ai.omnivoice.poc.ui.Motion.pressableTree
import ai.omnivoice.poc.core.TextTimeline
import ai.omnivoice.poc.ui.WaveformView
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
import android.text.method.ScrollingMovementMethod
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.*
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * One activity, eight screens, a state machine.
 *
 * The flow is linear — install, enroll, verify, voices, compose, generating,
 * result — so a navigation graph would add indirection without adding clarity.
 * The shape of each screen comes from design/; the reasons are in
 * docs/feature-validation.md.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        const val TAG = "OmniVoice.UI"
        // Measured on the S26 Ultra with int8 compute and the prefix KV cache
        // (2026-09-13); "—" where the current path has not been measured.
        private val STEPS = listOf(8 to "—", 16 to "1.8×", 32 to "—")
        private val THREADS = listOf(1 to "—", 2 to "—", 4 to "—", 6 to "1.8×", 8 to "—")
        private val BACKENDS = listOf(
            Triple(Backend.CPU, "1.8×", ""),
            Triple(Backend.XNNPACK, "slower", "0/2785 nodes"),
            Triple(Backend.NNAPI, "slower", "0/2785 nodes"),
        )
    }

    private enum class Screen { FIRST_RUN, LIBRARY, ENROLL, VERIFY, COMPOSE, GENERATING, RESULT, DEV }

    private lateinit var screens: FrameLayout
    private val views = HashMap<Screen, View>()
    private var current = Screen.LIBRARY
    private var previous = Screen.LIBRARY

    private val recorder = VoiceRecorder()
    private val player = AudioOutput()
    private lateinit var profiles: FileVoiceProfileManager
    private var profileList: List<VoiceProfile> = emptyList()
    private var selected: VoiceProfile? = null
    private var reRecordingFor: VoiceProfile? = null
    private var recorded: FloatArray? = null
    private var pendingProfile: VoiceProfile? = null
    private var lastResult: AudioResult? = null
    private val queue = ArrayList<String>()

    private var steps = 16
    private var threads = SynthesisService.DEFAULT_THREADS
    private var backend = Backend.CPU

    private var service: SynthesisService? = null
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as SynthesisService.LocalBinder).service
            observe()
        }
        override fun onServiceDisconnected(name: ComponentName?) { service = null }
    }

    private val micPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) beginRecording() else toast("마이크 권한이 필요합니다") }

    private val notifPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()) { }

    // ── lifecycle ────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        applyInsets()

        screens = findViewById(R.id.screens)
        views[Screen.FIRST_RUN] = findViewById(R.id.screenFirstRun)
        views[Screen.LIBRARY] = findViewById(R.id.screenLibrary)
        views[Screen.ENROLL] = findViewById(R.id.screenEnroll)
        views[Screen.VERIFY] = findViewById(R.id.screenVerify)
        views[Screen.COMPOSE] = findViewById(R.id.screenCompose)
        views[Screen.GENERATING] = findViewById(R.id.screenGenerating)
        views[Screen.RESULT] = findViewById(R.id.screenResult)
        views[Screen.DEV] = findViewById(R.id.screenDev)

        profiles = FileVoiceProfileManager(File(filesDir, "voices"))
        wire()
        screens.pressableTree()
        // Set after pressableTree: a movement method makes a TextView clickable,
        // and a scrolling script should not dip like a button.
        id<TextView>(R.id.resultText).movementMethod = ScrollingMovementMethod()
        id<TextView>(R.id.genText).movementMethod = ScrollingMovementMethod()
        buildDevControls()
        refreshVoices()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) notifPermission.launch(Manifest.permission.POST_NOTIFICATIONS)

        bindService(Intent(this, SynthesisService::class.java), connection, Context.BIND_AUTO_CREATE)

        // Compose is the screen you use every time; the voice library is the one
        // you use once. Opening on the library meant scrolling past it to reach
        // the thing you actually came for.
        show(when {
            !modelsReady() -> Screen.FIRST_RUN
            profileList.isEmpty() -> Screen.LIBRARY
            else -> Screen.COMPOSE
        })

        // Debug builds accept the target text on the launch intent:
        //   adb shell am start -n … --es target_text "오늘 회의를 시작하겠습니다."
        // `adb shell input text` cannot type Hangul, so scripted runs and
        // screenshot capture have no other way to put a real Korean sentence in
        // the field. Release builds ignore it.
        if (BuildConfig.DEBUG) intent?.getStringExtra("target_text")?.let {
            id<EditText>(R.id.targetText).setText(it)
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = goBack()
        })
    }

    /** targetSdk 36 enforces edge-to-edge; pad the screens, not the ground. */
    private fun applyInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.screens)) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            val up = insets.isVisible(WindowInsetsCompat.Type.ime())
            if (up != imeUp) { imeUp = up; onKeyboard(up) }
            insets
        }
    }

    private var imeUp = false

    /**
     * The voice chip and the batch note fold away to leave the field its size,
     * and used to do it on focus. But back dismisses the keyboard without taking
     * focus off the field, so they stayed folded with nothing covering the space
     * they had given up. The keyboard is what takes the room, so the keyboard is
     * what they should follow.
     */
    private fun onKeyboard(up: Boolean) {
        val folded = up && current == Screen.COMPOSE
        id<ViewGroup>(R.id.composeScrollBody).animateNextLayout()
        id<View>(R.id.composeVoice).visibility = if (folded) View.GONE else View.VISIBLE
        id<View>(R.id.composeBatchHint).visibility = if (folded) View.GONE else View.VISIBLE
        if (!up) {
            id<EditText>(R.id.targetText).clearFocus()
            id<EditText>(R.id.queueInput).clearFocus()
        }
    }

    private fun modelsReady() = File(File(filesDir, "models"), "omnivoice_lm.onnx").isFile

    /**
     * Swap screens.
     *
     * Both screens are never animated at once: each is a full hierarchy with a
     * blurred ground behind it, and cross-fading two of them costs more than the
     * transition is worth. The arriving screen fades up from a few dp below on
     * the way forward and from above on the way back, which is enough to say
     * which direction you moved.
     */
    private fun show(s: Screen, forward: Boolean = true) {
        val changed = s != current
        if (s != Screen.DEV) previous = current
        current = s
        for ((k, v) in views) v.visibility = if (k == s) View.VISIBLE else View.GONE
        if (changed) views[s]?.enterScreen(forward)
    }

    private fun goBack() {
        when (current) {
            Screen.DEV -> show(previous, forward = false)
            Screen.ENROLL -> { cancelRecording(); show(Screen.LIBRARY, forward = false) }
            Screen.VERIFY -> show(Screen.ENROLL, forward = false)
            Screen.COMPOSE -> if (profileList.isEmpty()) show(Screen.LIBRARY, forward = false) else finish()
            Screen.RESULT -> show(Screen.COMPOSE, forward = false)
            Screen.LIBRARY -> if (profileList.isEmpty()) finish() else show(Screen.COMPOSE, forward = false)
            Screen.GENERATING -> toast("생성 중입니다 — 취소를 누르세요")
            else -> finish()
        }
    }

    // ── wiring ───────────────────────────────────────────────────────────

    private fun <T : View> id(i: Int): T = findViewById(i)

    private fun wire() {
        id<ImageButton>(R.id.openDev).setOnClickListener { show(Screen.DEV) }
        id<ImageButton>(R.id.composeDev).setOnClickListener { show(Screen.DEV) }
        id<ImageButton>(R.id.devBack).setOnClickListener { show(previous) }
        id<ImageButton>(R.id.enrollBack).setOnClickListener { goBack() }
        id<ImageButton>(R.id.composeBack).setOnClickListener { goBack() }
        id<ImageButton>(R.id.resultBack).setOnClickListener { goBack() }
        id<Button>(R.id.installContinue).setOnClickListener { show(Screen.LIBRARY) }

        id<Button>(R.id.addVoice).setOnClickListener { startEnroll(null) }
        id<Button>(R.id.goCompose).setOnClickListener {
            if (profileList.isEmpty()) toast("먼저 목소리를 등록하세요") else show(Screen.COMPOSE)
        }
        id<ImageButton>(R.id.composeBack).setOnClickListener { show(Screen.LIBRARY) }

        id<Button>(R.id.enrollRecord).setOnClickListener {
            if (recorder.isRecording) finishRecording()
            else if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED) beginRecording()
            else micPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        id<ImageButton>(R.id.verifyPlay).setOnClickListener { togglePlay(R.id.verifyPlay) }
        id<Button>(R.id.verifyAgain).setOnClickListener {
            pendingProfile = null; player.release(); show(Screen.ENROLL)
        }
        id<Button>(R.id.verifyKeep).setOnClickListener { keepProfile() }

        id<Button>(R.id.generate).setOnClickListener { startGeneration() }
        id<Button>(R.id.genCancel).setOnClickListener { service?.cancel() }
        id<ImageButton>(R.id.queueAdd).setOnClickListener { enqueue() }
        id<EditText>(R.id.queueInput).setOnEditorActionListener { _, _, _ -> enqueue(); true }

        id<ImageButton>(R.id.resultPlay).setOnClickListener { togglePlay(R.id.resultPlay) }
        id<ImageButton>(R.id.resultReplay).setOnClickListener {
            player.replay(); setPlayIcon(R.id.resultPlay, true); startTicking(R.id.resultPlay)
        }
        id<Button>(R.id.resultSave).setOnClickListener { saveWav() }
        id<Button>(R.id.resultShare).setOnClickListener { shareWav() }

        // Folding the chrome is driven by the keyboard (see onKeyboard); focus
        // only has to keep the field itself in view once the fold has happened.
        id<EditText>(R.id.targetText).setOnFocusChangeListener { _, focused ->
            if (focused) {
                id<androidx.core.widget.NestedScrollView>(R.id.composeScroll)
                    .post { id<View>(R.id.targetText).let { v -> v.parent.requestChildFocus(v, v) } }
            }
        }
        // Tapping anywhere outside gives the field back its chrome.
        id<View>(R.id.screenCompose).setOnClickListener {
            id<EditText>(R.id.targetText).clearFocus()
            hideKeyboard()
        }

        id<EditText>(R.id.targetText).addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) = updateEstimate()
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        })

        val tagRow = id<LinearLayout>(R.id.tagRow)
        // surprise-ah and surprise-oh both read "놀람", so the row showed it twice
        // and pushed the last chip off the edge. One chip per distinct label.
        for (tag in OmniVoiceStyle.NON_VERBAL_VERIFIED.distinctBy { tagLabel(it) }) {
            tagRow.addView(chip(tagLabel(tag)) { insertTag("[$tag]") }, chipParams())
        }
    }

    private fun tagLabel(tag: String) = when (tag) {
        "laughter" -> "웃음"; "sigh" -> "한숨"
        "surprise-ah", "surprise-oh" -> "놀람"
        "question-en" -> "되물음"; "confirmation-en" -> "맞장구"
        else -> tag
    }

    private fun chip(text: String, onClick: () -> Unit) = Button(this).apply {
        setText(text)
        setTextAppearance(R.style.Voxl_Chip)
        background = getDrawable(R.drawable.glass_chip)
        setTextColor(getColor(R.color.text))
        isAllCaps = false
        stateListAnimator = null
        textSize = 12f
        // five chips have to fit 390dp without the last one being sliced in half
        val pad = (11 * resources.displayMetrics.density).toInt()
        minWidth = 0
        minimumWidth = 0
        setPadding(pad, 0, pad, 0)
        setOnClickListener { onClick() }
    }

    private fun chipParams() = LinearLayout.LayoutParams(
        ViewGroup.LayoutParams.WRAP_CONTENT, (44 * resources.displayMetrics.density).toInt()
    ).apply { marginEnd = (6 * resources.displayMetrics.density).toInt() }

    private fun insertTag(s: String) {
        val f = id<EditText>(R.id.targetText)
        f.text.insert(f.selectionStart.coerceAtLeast(0), s)
    }

    // ── voices ───────────────────────────────────────────────────────────

    private fun refreshVoices() {
        id<ViewGroup>(R.id.voiceList).animateNextLayout()
        profileList = profiles.list()
        if (selected == null || profileList.none { it.id == selected!!.id }) {
            selected = profileList.firstOrNull()
        }
        id<TextView>(R.id.voicesBody).visibility =
            if (profileList.isEmpty()) View.VISIBLE else View.GONE
        id<View>(R.id.privacyRow).visibility =
            if (profileList.isEmpty()) View.VISIBLE else View.GONE
        val list = id<LinearLayout>(R.id.voiceList)
        list.removeAllViews()
        if (profileList.isEmpty()) {
            list.addView(TextView(this).apply {
                setText(R.string.voices_none)
                setTextAppearance(R.style.Voxl_Caption)
                setPadding(0, (24 * resources.displayMetrics.density).toInt(), 0, 0)
                gravity = android.view.Gravity.CENTER
            })
        }
        for (p in profileList) list.addView(voiceCard(p, list))
        list.pressableTree()
        updateComposeVoice()
    }

    private fun voiceCard(p: VoiceProfile, parent: ViewGroup): View {
        val v = LayoutInflater.from(this).inflate(R.layout.item_voice, parent, false)
        val expanded = p.id == selected?.id
        v.isSelected = expanded
        v.findViewById<View>(R.id.voiceExpanded).visibility =
            if (expanded) View.VISIBLE else View.GONE
        v.findViewById<TextView>(R.id.voiceName).text = p.displayName
        v.findViewById<TextView>(R.id.voiceDuration).text =
            "%.1f초".format(p.frames.toFloat() / OV.FRAME_RATE)
        v.findViewById<TextView>(R.id.voiceQuote).text = "“${p.refText}”"
        v.findViewById<TextView>(R.id.voiceBadge).apply {
            text = "사용 중"
            visibility = if (expanded) View.VISIBLE else View.GONE
        }
        // The recording is deliberately not kept, so there is no waveform to
        // draw. The card shows the 1.8 kB that IS kept — the codec codes.
        v.findViewById<CodeStripView>(R.id.voiceWave).apply {
            tone = if (expanded) CodeStripView.Tone.AMBER else CodeStripView.Tone.VIOLET
            setCodes(p.codes)
        }
        // Tapping an already-selected voice is the confirmation, so it goes
        // straight back to composing rather than needing the button below.
        v.setOnClickListener {
            if (expanded) show(Screen.COMPOSE) else { selected = p; refreshVoices() }
        }
        v.findViewById<Button>(R.id.voiceTest).setOnClickListener { testVoice(p) }
        v.findViewById<Button>(R.id.voiceRerecord).setOnClickListener { startEnroll(p) }
        v.findViewById<Button>(R.id.voiceRename).setOnClickListener { renameVoice(p) }
        // The name is the obvious thing to tap to change the name.
        v.findViewById<TextView>(R.id.voiceName).setOnClickListener { renameVoice(p) }
        v.findViewById<Button>(R.id.voiceDelete).setOnClickListener { confirmDelete(p) }
        return v
    }

    /**
     * The display name lives in a sidecar file next to the codes, so renaming
     * is a re-save of the same profile — the 1.8 kB of codec codes are untouched.
     */
    private fun renameVoice(p: VoiceProfile) {
        val field = EditText(this).apply {
            setTextAppearance(R.style.Voxl_Field)
            setText(p.displayName)
            // Renaming almost always means replacing, not appending — and the
            // caret otherwise lands at the start, so typing wrote in front of
            // the old name. Select it all and let the first keystroke take over.
            setSelectAllOnFocus(true)
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_VARIATION_PERSON_NAME
            maxLines = 1
            val pad = (18 * resources.displayMetrics.density).toInt()
            setPadding(pad, pad, pad, pad)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.voice_rename_title)
            .setView(field)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val name = field.text.toString().trim()
                if (name.isEmpty()) return@setPositiveButton
                profiles.save(p.copy(displayName = name))
                if (selected?.id == p.id) selected = p.copy(displayName = name)
                refreshVoices()
            }
            .show().also {
                // A dialog whose only control is a text field should open ready
                // to type in.
                it.window?.setSoftInputMode(
                    WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE)
                field.requestFocus()
            }
    }

    private fun confirmDelete(p: VoiceProfile) {
        AlertDialog.Builder(this)
            .setTitle("${p.displayName} 삭제")
            .setMessage("이 기기에서 지워집니다. 되돌릴 수 없습니다.")
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                profiles.delete(p.id); refreshVoices()
            }.show()
    }

    private fun testVoice(p: VoiceProfile) {
        selected = p
        startGeneration(EnrollmentScripts.TEST_SENTENCE)
    }

    // ── enrollment ───────────────────────────────────────────────────────

    private fun startEnroll(replacing: VoiceProfile?) {
        if (!OmniVoiceEnroller.available(File(filesDir, "models"))) {
            toast("등록 모델이 없습니다 — 이 빌드로는 재생만 가능합니다"); return
        }
        reRecordingFor = replacing
        recorded = null
        id<EditText>(R.id.scriptText).setText(replacing?.refText ?: EnrollmentScripts.ALL[0].text)
        id<TextView>(R.id.enrollStatus).text = ""
        id<WaveformView>(R.id.enrollMeter).clear()
        id<TextView>(R.id.enrollSeconds).text = ""
        setElapsed(0f)
        setEnrollBusy(false)
        show(Screen.ENROLL)
    }

    private fun beginRecording() {
        val meter = id<WaveformView>(R.id.enrollMeter).apply { tone = WaveformView.Tone.REC }
        meter.startLive()
        runCatching {
            recorder.start { level ->
                runOnUiThread {
                    meter.pushLevel(level * 3f)
                    val s = recorder.seconds
                    id<TextView>(R.id.enrollSeconds).text = "%.1f초 · %s".format(s, getString(R.string.enroll_band))
                    setElapsed(s / VoiceRecorder.RECOMMENDED_MAX_SECONDS)
                }
            }
        }.onFailure { toast(it.message ?: "녹음 실패"); return }
        id<Button>(R.id.enrollRecord).setText(R.string.enroll_stop)
    }

    private fun setElapsed(fraction: Float) {
        val bar = id<View>(R.id.enrollElapsed)
        val parent = bar.parent as View
        bar.layoutParams = (bar.layoutParams as FrameLayout.LayoutParams).apply {
            width = (parent.width * fraction.coerceIn(0f, 1f)).toInt()
        }
        bar.requestLayout()
    }

    private fun cancelRecording() {
        if (recorder.isRecording) recorder.cancel()
        id<Button>(R.id.enrollRecord).setText(R.string.enroll_start)
    }

    private fun finishRecording() {
        val pcm = recorder.stop()
        recorded = pcm
        id<Button>(R.id.enrollRecord).setText(R.string.enroll_start)
        val secs = pcm.size.toFloat() / VoiceRecorder.SAMPLE_RATE
        if (secs < VoiceRecorder.MIN_SECONDS) {
            id<TextView>(R.id.enrollStatus).text =
                "%.1f초뿐입니다 — %.0f초 이상 녹음하세요".format(secs, VoiceRecorder.MIN_SECONDS)
            return
        }
        val refText = id<EditText>(R.id.scriptText).text.toString()
        if (refText.isBlank()) { toast("읽은 문장이 필요합니다"); return }
        setEnrollBusy(true)
        service?.enroll(pcm, VoiceRecorder.SAMPLE_RATE, refText,
            reRecordingFor?.displayName
                ?: SimpleDateFormat("M월 d일 HH:mm", Locale.KOREA).format(System.currentTimeMillis()))
            ?: toast("서비스가 아직 준비되지 않았습니다")
    }

    /**
     * Encoding takes ~3 s and nothing on screen said so: the record button kept
     * its full colour and stayed tappable, so a second press started a new
     * recording on top of the enrollment.
     */
    private fun setEnrollBusy(busy: Boolean) {
        id<Button>(R.id.enrollRecord).apply {
            isEnabled = !busy
            setText(if (busy) R.string.enroll_working else R.string.enroll_start)
        }
        id<EditText>(R.id.scriptText).isEnabled = !busy
        id<ImageButton>(R.id.enrollBack).isEnabled = !busy
        id<TextView>(R.id.enrollStatus).text =
            if (busy) getString(R.string.enroll_working_note) else ""
    }

    private fun onEnrolled(p: VoiceProfile, echo: FloatArray, millis: Long) {
        setEnrollBusy(false)
        pendingProfile = p
        player.onFinished = { setPlayIcon(R.id.verifyPlay, false); stopTicking() }
        player.load(echo, OV.SR_24K)
        id<WaveformView>(R.id.verifyWave).apply {
            tone = WaveformView.Tone.AMBER
            setWaveform(echo, buckets = 44)
        }
        id<TextView>(R.id.verifyMeta).text =
            "%.1fs · T_ref=%d frames · encode %d ms".format(p.frames.toFloat() / OV.FRAME_RATE, p.frames, millis)
        // Pre-filled with the timestamp so keeping a voice is still one tap, but
        // this is the one moment the user knows what they just recorded.
        id<EditText>(R.id.verifyName).setText(p.displayName)
        buildChecks(p)
        setPlayIcon(R.id.verifyPlay, false)
        show(Screen.VERIFY)
    }

    private fun buildChecks(p: VoiceProfile) {
        val box = id<LinearLayout>(R.id.verifyChecks)
        box.removeAllViews()
        val dp = resources.displayMetrics.density
        val secs = p.frames.toFloat() / OV.FRAME_RATE
        val items = listOf(
            "duration %.1fs ≥ 3s".format(secs) to (secs >= VoiceRecorder.MIN_SECONDS),
            "RMS %.3f ∈ [0.01, 0.5]".format(p.refRms) to (p.refRms in 0.01f..0.5f),
            "reference transcript 있음" to p.refText.isNotBlank(),
        )
        for ((text, ok) in items) {
            box.addView(LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, (4 * dp).toInt(), 0, (4 * dp).toInt())
                addView(ImageView(this@MainActivity).apply {
                    setImageResource(if (ok) R.drawable.ic_check else R.drawable.ic_close)
                    layoutParams = LinearLayout.LayoutParams((15 * dp).toInt(), (15 * dp).toInt())
                })
                addView(TextView(this@MainActivity).apply {
                    setText(text)
                    setTextAppearance(R.style.Voxl_Body)
                    setTextColor(getColor(if (ok) R.color.text else R.color.rec))
                    textSize = 13f
                    setPadding((9 * dp).toInt(), 0, 0, 0)
                })
            })
        }
    }

    private fun keepProfile() {
        val typed = id<EditText>(R.id.verifyName).text.toString().trim()
        val p = (pendingProfile ?: return).let {
            if (typed.isEmpty() || typed == it.displayName) it else it.copy(displayName = typed)
        }
        reRecordingFor?.let { if (it.id != p.id) profiles.delete(it.id) }
        profiles.save(p)
        selected = p
        pendingProfile = null
        reRecordingFor = null
        player.release()
        refreshVoices()
        show(Screen.LIBRARY)
    }

    // ── generation ───────────────────────────────────────────────────────

    private fun updateComposeVoice() {
        val p = selected
        id<TextView>(R.id.composeVoiceName).text = p?.displayName ?: getString(R.string.no_voice)
        id<TextView>(R.id.composeVoiceMeta).text =
            p?.let { "ref %.1fs · T_ref=%d".format(it.frames.toFloat() / OV.FRAME_RATE, it.frames) } ?: ""
        id<LinearLayout>(R.id.composeVoice).setOnClickListener { show(Screen.LIBRARY) }
        id<TextView>(R.id.composeVoiceHint).text = getString(R.string.change_voice)
        updateEstimate()
    }

    /**
     * The estimate uses the same duration model the engine uses, and the RTF the
     * device actually measured, so the number on screen is the number the user
     * will live through.
     */
    private val prefs by lazy { getSharedPreferences("voxlvoxl", MODE_PRIVATE) }

    /**
     * RTF for the compose estimate. The table this replaced was measured before
     * int8 compute and the prefix KV cache, and told users ~60 s for a sentence
     * that takes 9. So: what this device last measured at this step count, and
     * until it has, the 16-step RTF measured on the S26 Ultra (1.8) scaled by the
     * step count -- the un-masking loop is nearly all of the time, and RTF was
     * measured to scale linearly with steps (5.4 / 11.2 / 24.6 at 8 / 16 / 32).
     */
    private fun estimateRtf(stepCount: Int): Float {
        val measured = prefs.getFloat("rtf_s$stepCount", -1f)
        return if (measured > 0f) measured else 1.8f * stepCount / 16f
    }

    private fun updateEstimate() {
        val text = id<EditText>(R.id.targetText).text.toString()
        val det = LanguageDetector.detect(text)
        id<TextView>(R.id.composeLanguage).text = det.label
        if (text.isBlank()) { id<TextView>(R.id.composeEstimate).text = ""; return }
        val p = selected
        val frames = DurationEstimator.estimateFrames(text, p?.refText, p?.frames ?: 0)
        val audio = frames.toFloat() / OV.FRAME_RATE
        val rtf = estimateRtf(steps)
        id<TextView>(R.id.composeEstimate).text =
            "≈%.1fs audio · RTF %.1f → %ds".format(audio, rtf, (audio * rtf).toInt())
    }

    private fun startGeneration(overrideText: String? = null) {
        val text = overrideText ?: id<EditText>(R.id.targetText).text.toString().trim()
        if (text.isEmpty()) return toast("문장을 입력하세요")
        val svc = service ?: return toast("서비스가 아직 준비되지 않았습니다")
        val det = LanguageDetector.detect(text)
        id<TextView>(R.id.genText).text = text
        id<CodebookLadderView>(R.id.ladder).apply {
            topLabel = getString(R.string.ladder_top)
            bottomLabel = getString(R.string.ladder_bottom)
            reset()
        }
        svc.synthesize(
            text = text, profile = selected,
            style = VoiceStyle(language = det.code ?: "None"),
            steps = steps, threads = threads,
        )
        id<EditText>(R.id.targetText).clearFocus()
        hideKeyboard()
        show(Screen.GENERATING)
    }

    private fun enqueue() {
        val f = id<EditText>(R.id.queueInput)
        val t = f.text.toString().trim()
        if (t.isEmpty()) return
        queue.add(t)
        f.setText("")
        renderQueue()
    }

    private fun renderQueue() {
        val box = id<LinearLayout>(R.id.queueList)
        box.animateNextLayout()
        box.removeAllViews()
        val dp = resources.displayMetrics.density
        queue.forEachIndexed { i, t ->
            box.addView(LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                background = getDrawable(R.drawable.glass_chip)
                setPadding((14 * dp).toInt(), (12 * dp).toInt(), (14 * dp).toInt(), (12 * dp).toInt())
                addView(TextView(this@MainActivity).apply {
                    text = "${i + 1}"
                    setTextAppearance(R.style.Voxl_Readout)
                    width = (18 * dp).toInt()
                })
                addView(TextView(this@MainActivity).apply {
                    setText(t)
                    setTextAppearance(R.style.Voxl_Body)
                    setTextColor(getColor(R.color.text))
                    textSize = 12.5f
                    maxLines = 1
                    ellipsize = android.text.TextUtils.TruncateAt.END
                    layoutParams = LinearLayout.LayoutParams(0,
                        ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                })
                addView(TextView(this@MainActivity).apply {
                    setText("대기")
                    setTextAppearance(R.style.Voxl_Readout)
                    textSize = 10.5f
                })
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = (8 * dp).toInt()
            })
        }
    }

    private fun observe() {
        val svc = service ?: return
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { svc.status.collect { render(it) } }
                launch {
                    svc.enrollStatus.collect { s ->
                        when (s) {
                            is SynthesisService.EnrollStatus.Encoding -> setEnrollBusy(true)
                            is SynthesisService.EnrollStatus.Ready -> {
                                onEnrolled(s.profile, s.echo, s.millis)
                                svc.clearEnrollment()
                            }
                            is SynthesisService.EnrollStatus.Failed -> {
                                setEnrollBusy(false)
                                id<TextView>(R.id.enrollStatus).text = s.error.message ?: "등록 실패"
                                svc.clearEnrollment()
                            }
                            else -> Unit
                        }
                    }
                }
            }
        }
    }

    private fun render(s: SynthesisService.Status) {
        when (s) {
            is SynthesisService.Status.Loading -> {
                id<TextView>(R.id.genEta).animateInt(0, duration = 0L)
                id<TextView>(R.id.genEtaUnit).text = "%  session load"
            }
            is SynthesisService.Status.Running -> {
                val p = s.progress
                // Percent rather than a countdown: the seconds estimate is honest
                // but it moves around as the device warms, and a number that
                // jumps back up reads as the app being wrong.
                id<TextView>(R.id.genEta).animateInt((p.fraction * 100).toInt())
                id<TextView>(R.id.genEtaUnit).text = "%"
                id<TextView>(R.id.genStep).text = "step ${p.step} / ${p.totalSteps}"
                id<TextView>(R.id.genCells).text =
                    "${p.cellsTotal - p.cellsRemaining} / ${p.cellsTotal} tokens unmasked"
                id<CodebookLadderView>(R.id.ladder).setProgress(p.perCodebook)
            }
            is SynthesisService.Status.Done -> {
                lastResult = s.result
                prefs.edit().putFloat("rtf_s$steps", s.result.metrics.rtf.toFloat()).apply()
                updateEstimate()
                player.onFinished = {
                    setPlayIcon(R.id.resultPlay, false)
                    stopTicking()
                    id<WaveformView>(R.id.resultWave).progress = 0f
                    highlightSpoken(0)
                    id<TextView>(R.id.resultTime).text =
                        "0:00 / %s".format(clock(s.result.metrics.audioSeconds))
                }
                player.load(s.result.samples, s.result.sampleRate)
                showResult(s.result)
                if (queue.isNotEmpty()) {
                    val next = queue.removeAt(0)
                    renderQueue()
                    id<EditText>(R.id.targetText).setText(next)
                }
            }
            is SynthesisService.Status.Cancelled -> show(Screen.COMPOSE, forward = false)
            is SynthesisService.Status.Failed -> {
                toast(s.error.message ?: "생성 실패")
                Log.e(TAG, "generation failed", s.error)
                show(Screen.COMPOSE, forward = false)
            }
            else -> Unit
        }
    }

    private var resultTimeline: TextTimeline? = null
    private var spokenUpTo = -1
    private val spokenSpan by lazy { ForegroundColorSpan(getColor(R.color.accent_light)) }

    /**
     * Follow along in the script while the result plays. The model emits no
     * alignment, so this is exact at chunk boundaries and estimated inside a
     * chunk from DurationEstimator's per-character weights: close at phrase
     * level, not at syllable level. Keeps the spoken line in the upper third.
     */
    private fun highlightSpoken(upTo: Int) {
        if (upTo == spokenUpTo) return
        spokenUpTo = upTo
        val tv = id<TextView>(R.id.resultText)
        val sp = tv.text as? android.text.Spannable ?: return
        sp.removeSpan(spokenSpan)
        if (upTo <= 0) { tv.scrollTo(0, 0); return }
        val end = upTo.coerceAtMost(sp.length)
        sp.setSpan(spokenSpan, 0, end, android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        val layout = tv.layout ?: return
        val visible = tv.height - tv.totalPaddingTop - tv.totalPaddingBottom
        val y = layout.getLineTop(layout.getLineForOffset(end))
        val target = (y - visible / 3).coerceIn(0, maxOf(0, layout.height - visible))
        if (target != tv.scrollY) tv.scrollTo(0, target)
    }

    private fun showResult(r: AudioResult) {
        resultTimeline = r.timeline
        spokenUpTo = -1
        id<TextView>(R.id.resultText).apply {
            setText(r.timeline?.text ?: id<TextView>(R.id.genText).text, TextView.BufferType.SPANNABLE)
            scrollTo(0, 0)
        }
        id<WaveformView>(R.id.resultWave).apply {
            tone = WaveformView.Tone.AMBER
            setWaveform(r.samples, buckets = 52)
            progress = 0f
        }
        id<TextView>(R.id.resultTime).text = "0:00 / %s".format(clock(r.metrics.audioSeconds))
        setPlayIcon(R.id.resultPlay, false)
        val m = r.metrics
        id<TextView>(R.id.metricsRtf).text = "%.1f".format(m.rtf)
        id<TextView>(R.id.metricsWall).text =
            "%.2fs audio / %.1fs wall".format(m.audioSeconds, m.totalMillis / 1000.0)
        id<TextView>(R.id.metricsConfig).text = "$backend ×$threads · int4"
        val table = id<TableLayout>(R.id.metricsTable)
        table.removeAllViews()
        val dp = resources.displayMetrics.density
        fun row(a: String, av: String, b: String, bv: String) {
            table.addView(TableRow(this).apply {
                addView(cell(a, false))
                addView(cell(av, true).apply { setPadding((10 * dp).toInt(), (3 * dp).toInt(), 0, (3 * dp).toInt()) })
                addView(cell(b, false).apply { setPadding((20 * dp).toInt(), (3 * dp).toInt(), 0, (3 * dp).toInt()) })
                addView(cell(bv, true).apply { setPadding((10 * dp).toInt(), (3 * dp).toInt(), 0, (3 * dp).toInt()) })
            })
        }
        row("LM decode", "${m.generateMillis} ms", "Vocoder", "${m.decodeMillis} ms")
        row("Session load", "${m.modelLoadMillis} ms", "Post-proc", "${m.postMillis} ms")
        row("S (seq len)", "${m.sequenceLength}", "Chunks", "${m.chunks}")
        id<LinearLayout>(R.id.genThermal).visibility =
            if (m.thermalStatus > 0) View.VISIBLE else View.GONE
        show(Screen.RESULT)
    }

    private fun cell(text: String, value: Boolean) = TextView(this).apply {
        setText(text)
        setTextAppearance(if (value) R.style.Voxl_Readout_Value else R.style.Voxl_Readout)
        textSize = 11f
        setPadding(0, (3 * resources.displayMetrics.density).toInt(), 0, (3 * resources.displayMetrics.density).toInt())
    }

    private fun clock(seconds: Double): String {
        val s = seconds.toInt()
        return "%d:%02d".format(s / 60, s % 60)
    }

    // ── playback ─────────────────────────────────────────────────────────

    private val ticker = android.os.Handler(android.os.Looper.getMainLooper())
    private var tick: Runnable? = null

    private fun togglePlay(buttonId: Int) {
        if (player.state == AudioOutput.State.PLAYING) {
            player.pause()
            setPlayIcon(buttonId, false)
            stopTicking()
        } else {
            player.play()
            setPlayIcon(buttonId, true)
            startTicking(buttonId)
        }
    }

    /**
     * AudioTrack reports the head position but tells nobody, so the waveform and
     * the clock have to be driven. 60 ms is well under a frame and costs a single
     * invalidate on a view that is already on screen.
     */
    private fun startTicking(buttonId: Int) {
        stopTicking()
        val wave = if (buttonId == R.id.resultPlay)
            id<WaveformView>(R.id.resultWave) else id<WaveformView>(R.id.verifyWave)
        val clock = if (buttonId == R.id.resultPlay) id<TextView>(R.id.resultTime) else null
        val total = player.durationSeconds
        val r = object : Runnable {
            override fun run() {
                val p = player.progress
                wave.progress = p
                if (buttonId == R.id.resultPlay) resultTimeline?.let {
                    highlightSpoken(it.spokenThrough((p * total * OV.SR_24K).toInt()))
                }
                clock?.text = "%s / %s".format(clock(p * total.toDouble()), clock(total.toDouble()))
                if (player.state != AudioOutput.State.PLAYING) {
                    setPlayIcon(buttonId, false)
                    return
                }
                ticker.postDelayed(this, 60)
            }
        }
        tick = r
        ticker.post(r)
    }

    private fun stopTicking() {
        tick?.let { ticker.removeCallbacks(it) }
        tick = null
    }

    private fun setPlayIcon(buttonId: Int, playing: Boolean) {
        id<ImageButton>(buttonId).setImageResource(
            if (playing) R.drawable.ic_pause else R.drawable.ic_play)
    }

    private fun outFile(): File {
        val dir = File(getExternalFilesDir(null), "out").apply { mkdirs() }
        return File(dir, "voxlvoxl_%s.wav".format(
            SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())))
    }

    private fun saveWav() {
        val r = lastResult ?: return
        runCatching { outFile().also { WavIo.write(it, r.samples, r.sampleRate) } }
            .onSuccess { toast("저장됨: ${it.absolutePath}") }
            .onFailure { toast(it.message ?: "저장 실패") }
    }

    private fun shareWav() {
        val r = lastResult ?: return
        runCatching {
            val f = outFile()
            WavIo.write(f, r.samples, r.sampleRate)
            val uri = androidx.core.content.FileProvider.getUriForFile(
                this, "$packageName.files", f)
            startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                type = "audio/wav"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }, getString(R.string.share)))
        }.onFailure { toast(it.message ?: "공유 실패") }
    }

    // ── developer settings ───────────────────────────────────────────────

    private fun buildDevControls() {
        val dp = resources.displayMetrics.density
        val stepRow = id<LinearLayout>(R.id.devSteps)
        fun rebuildSteps() {
            stepRow.removeAllViews()
            for ((v, rtf) in STEPS) {
                stepRow.addView(segment("$v", rtf, v == steps) {
                    steps = v; rebuildSteps(); updateEstimate()
                }, segParams())
            }
        }
        rebuildSteps()

        val threadRow = id<LinearLayout>(R.id.devThreads)
        fun rebuildThreads() {
            threadRow.removeAllViews()
            for ((v, rtf) in THREADS) {
                threadRow.addView(segment("$v", rtf, v == threads) {
                    threads = v; rebuildThreads()
                }, segParams())
            }
        }
        rebuildThreads()

        val backendBox = id<LinearLayout>(R.id.devBackends)
        fun rebuildBackends() {
            backendBox.removeAllViews()
            for ((b, rtf, note) in BACKENDS) {
                val active = b == backend
                backendBox.addView(LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    background = getDrawable(R.drawable.glass_card)
                    isSelected = active
                    setPadding((15 * dp).toInt(), (13 * dp).toInt(), (15 * dp).toInt(), (13 * dp).toInt())
                    addView(TextView(this@MainActivity).apply {
                        setText(b.name)
                        setTextAppearance(R.style.Voxl_Body)
                        setTextColor(getColor(if (active) R.color.accent_light else R.color.text_muted))
                        textSize = 14f
                        layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    })
                    addView(TextView(this@MainActivity).apply {
                        setText(rtf)
                        setTextAppearance(R.style.Voxl_Readout)
                        setTextColor(getColor(if (active) R.color.accent_light else R.color.text_faint))
                        textSize = 11f
                    })
                    if (note.isNotEmpty()) addView(TextView(this@MainActivity).apply {
                        setText("  $note")
                        setTextAppearance(R.style.Voxl_Readout)
                        textSize = 10.5f
                    })
                    setOnClickListener { backend = b; rebuildBackends() }
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT).apply { bottomMargin = (6 * dp).toInt() })
            }
        }
        rebuildBackends()

        val info = id<TableLayout>(R.id.devInfo)
        info.removeAllViews()
        for ((k, v) in listOf(
            "기기" to "${Build.MODEL} · ${Build.HARDWARE}",
            "모델" to if (modelsReady()) "설치됨 · int4" else "없음",
            "등록 모델" to if (OmniVoiceEnroller.available(File(filesDir, "models"))) "있음" else "없음",
            "네트워크" to "권한 없음",
        )) info.addView(TableRow(this).apply { addView(cell(k, false)); addView(cell(v, true)) })
    }

    private fun segment(label: String, sub: String, active: Boolean, onClick: () -> Unit) =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = android.view.Gravity.CENTER
            background = getDrawable(R.drawable.glass_card)
            isSelected = active
            val dp = resources.displayMetrics.density
            setPadding((8 * dp).toInt(), (11 * dp).toInt(), (8 * dp).toInt(), (11 * dp).toInt())
            addView(TextView(this@MainActivity).apply {
                setText(label)
                setTextAppearance(R.style.Voxl_Readout_Value)
                setTextColor(getColor(if (active) R.color.accent_light else R.color.text))
                textSize = 17f
            })
            addView(TextView(this@MainActivity).apply {
                setText(sub)
                setTextAppearance(R.style.Voxl_Readout)
                setTextColor(getColor(if (active) R.color.accent_light else R.color.text_faint))
                textSize = 10f
            })
            setOnClickListener { onClick() }
        }

    private fun segParams() = LinearLayout.LayoutParams(0,
        ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
        marginEnd = (7 * resources.displayMetrics.density).toInt()
    }

    private fun hideKeyboard() {
        val imm = getSystemService(android.view.inputmethod.InputMethodManager::class.java)
        imm?.hideSoftInputFromWindow(window.decorView.windowToken, 0)
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    override fun onStop() {
        super.onStop()
        if (recorder.isRecording) cancelRecording()
        player.pause()
        stopTicking()
    }

    override fun onDestroy() {
        stopTicking()
        player.release()
        runCatching { unbindService(connection) }
        super.onDestroy()
    }
}
