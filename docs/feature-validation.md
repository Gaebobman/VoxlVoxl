# Feature validation — VoxlVoxl Feature Definition vs. measurement

Every row is either measured or explicitly marked unverified. Where the spec says
"가정하지 않는다 / 검증한다", this is the verification.

---

## 1. The headline gap: RTF < 1.0

`F-F02` sets a first performance target of **RTF < 1.0**. Measured on the target
device (Galaxy S26 Ultra, SM8850, CPU, 6 threads, int4):

| utterance | audio | RTF @ 8 steps | RTF @ 16 steps | RTF @ 32 steps |
|---|---:|---:|---:|---:|
| short ("네, 알겠습니다.") | 1.26 s | ~7.2 | 15.08 | ~33 |
| typical | 1.88 s | 5.40 | 11.21 | 24.64 |
| long (3 sentences) | 10.57 s | ~3.6 | 7.48 | ~16 |

**Best measured is RTF 7.48**, and the gap to 1.0 is not closable on this device:

- every accelerator path is exhausted (`docs/benchmark.md` §7.4, §7.7) — NNAPI and
  XNNPACK claim 0 nodes, QNN cannot create a device because ORT's bundled SDK
  predates this chip
- 85 % of the work is int4 GEMM with no overhead left to reclaim
- `guidance_scale = 0` would nearly halve it but produces **silence**, so it is
  not a lever
- below 8 steps quality drops past the fp32 sampling noise floor

This is structural to OmniVoice, not to our port: bidirectional attention forbids
a KV cache, so all 32 (or 16) forwards re-read the whole sequence.

**Consequence for the roadmap.** RTF ≈ 7 is fine for Phase 1–2 (asynchronous
synthesis: type, generate, play). It is *disqualifying* for Phase 4 Conversation
and Phase 5 Telephony, where a reply must be spoken within a second or two.

So `§13 Module Architecture`'s instruction — *"앱 architecture가 OmniVoice 자체에
종속되지 않도록 한다"* — is not future-proofing. It is a hard requirement created
by this measurement, and `SpeechSynthesizer` needs a second implementation before
Phase 4 is attemptable. `k2-fsa/sherpa-onnx` + **ZipVoice** is the obvious
candidate: same organisation, Apache-2.0, already shipping offline Android voice
cloning, and autoregressive-with-cache so it can run near or below RTF 1.

---

## 2. Voice Identity

| ID | feature | status | evidence |
|---|---|---|---|
| F-A01 | Voice Enrollment | **PC done, device not yet** | `infer_onnx.py enroll` produces a profile in 1.7 s. The Android path needs the 3 encoder graphs (+654 MB → 1.17 GB total), `RECORD_AUDIO`, and a recording UI. |
| F-A02 | Voice Profile Generation | **done, matches the spec exactly** | `voice_prompt.bin` stores the encoded representation, not the audio: 8×T int16 codec codes + transcript + RMS. **1766 bytes** for a 4.12 s reference. Encoders are needed once; the generation path never loads them. |
| F-A03 | Multiple Voice Profiles | **architecturally free** | At ~1.8 kB per profile, hundreds cost nothing. See §4 — this turns out to be the answer to F-C04 as well. |

---

## 3. Text-to-Speech

| ID | feature | status | evidence |
|---|---|---|---|
| F-B01 | Text Input | done (engine) | |
| F-B02 | Voice Cloned TTS | **done on device** | 1.88 s cloned WAV on the S26 Ultra; output scores 2.7325 on the fp32 reference's own re-masking objective against PyTorch's own 2.7672 — interchangeable with the reference implementation |
| F-B03 | Audio Playback | not built | Play/Pause/Stop/Replay over `AudioTrack` |
| F-B04 | Save Audio (WAV) | done (engine) | `WavIo`, 16-bit PCM |
| F-B05 | Generation Parameters | partly | `num_steps` ✅ measured (8/16/32), `language` ✅ plumbed but hardcoded `ko`, `speed` ✅ in the duration estimator, not exposed |

One parameter the spec does not mention but which must be locked down:
**`guidance_scale` must never be exposed or defaulted to 0** — it yields 1.92 s of
output entirely below −50 dBFS. It is validated in `GenConfig`'s constructor.

---

## 4. Voice Style — the section the spec asked to verify

Measured with an objective acoustic probe (`scripts/style_probe.py`), not by
listening: F0 by autocorrelation, voiced-frame fraction, >4 kHz energy ratio,
zero-crossing rate.

### F-C03 Voice Design — ✅ works, and works strongly

Design mode (no clone prompt), same text, 16 steps:

| instruct | F0 | ΔF0 | voiced % |
|---|---:|---:|---:|
| (none) | 193.5 Hz | — | 72.4 % |
| `very low pitch` | 110.6 Hz | **−82.9** | 65.9 % |
| `elderly` | 113.7 Hz | **−79.8** | 62.0 % |
| `low pitch` | 226.4 Hz | +32.9 | 67.7 % |
| `female` | 216.2 Hz | +22.7 | 70.9 % |
| `child` | 235.3 Hz | +41.7 | 85.7 % |
| `high pitch` | 252.6 Hz | +59.1 | 70.8 % |
| `very high pitch` | 279.1 Hz | **+85.5** | 85.2 % |
| `whisper` | — | — | **20.3 %** |

`whisper` collapses the voiced fraction from 72 % to 20 % — the exact acoustic
signature of whispering (unvoiced excitation). The controls are real.

Note the vocabulary is a **closed set** and unknown items raise, so the UI must
offer a picker, not a text field. Port `_resolve_instruct`'s list.

### F-C04 My Voice + Style Control — ❌ does **not** work

The spec says not to assume clone and design compose. They do not. Same probe,
same styles, **with** a voice clone prompt:

| instruct | F0 | ΔF0 vs baseline | voiced % |
|---|---:|---:|---:|
| (none) | 102.8 Hz | — | 80.7 % |
| `whisper` | 96.8 Hz | −6.0 | 79.5 % |
| `very low pitch` | 96.8 Hz | −6.0 | 82.0 % |
| `very high pitch` | 97.2 Hz | −5.6 | 79.9 % |
| `child` | 94.1 Hz | −8.7 | 80.4 % |
| `female` | 100.4 Hz | −2.4 | 82.7 % |

Every style lands within ±9 Hz of baseline and all deltas share a sign — noise,
not control. `whisper` leaves the voiced fraction at 79.5 % where design mode
drove it to 20.3 %. **The clone conditioning completely dominates the instruct.**

The prompt *does* carry both (`<|instruct_start|>…<|instruct_end|>` and the
reference codes are independent fields in `_prepare_inference_inputs`), so this is
a training-distribution limit, not an API limit. No prompt-level workaround.

### F-C01 Reference Style Cloning — ✅ works, and it is the workaround

Speaking manner transfers through the reference recording. Enrolling each
design-mode output as a profile, then generating *different* text with it:

| enrolled from | reference F0 / voiced % | generated F0 / voiced % |
|---|---|---|
| baseline | 193.5 Hz / 72.4 % | 183.2 Hz / 74.0 % |
| `whisper` | — / **20.3 %** | — / **24.1 %** |
| `very low pitch` | 110.6 Hz / 65.9 % | **98.4 Hz** / 66.5 % |
| `very high pitch` | 279.1 Hz / 85.2 % | **269.7 Hz** / 87.7 % |

The voiced fraction transfers almost exactly (20.3→24.1, 72.4→74.0, 65.9→66.5,
85.2→87.7), and so does pitch. **Whisper survives as a manner.**

**So F-C04's product goal is reachable without the model feature that fails.**
Instead of "my voice + whisper instruct", it becomes "enroll yourself whispering
as a second profile" — which is exactly F-A03:

```
My Voice
My Voice — Whisper      ← a whispered enrollment recording
My Voice — Calm
My Voice — Bright
```

This reframes F-C02 "Explicit Style Control" too: for cloned voices, the style
picker selects **which profile**, not which instruct. Design-mode instructs stay
available for the no-reference case.

### F-C05 Cross-Speaker Style Transfer — ❌ unsupported, as the spec assumed

There is one reference slot in the prompt. Two references cannot be expressed.
Correctly scoped as experimental.

---

## 5. Expressive Speech

**F-D01 Non-verbal Expression — ✅ works.** The tags do not merely tokenize; they
produce audible events. Method: the same carrier sentence with and without a tag,
against four no-tag control runs at different seeds whose spread sets the
threshold (`scripts/nonverbal_probe.py`).

Control (`"네 알겠습니다"`): duration 1.067 s ± 0.135, voiced 78.8 % ± 2.7 pp,
HF 0.67 % ± 0.13 pp.

| tag | duration | Δ duration | voiced % | Δ voiced | verdict |
|---|---:|---:|---:|---:|---|
| `[laughter]` | 1.82 s | **+0.75 s (+5.5σ)** | 62.6 % | **−6.1σ** | audible |
| `[sigh]` | 1.42 s | +0.35 s (+2.6σ) | 63.3 % | **−5.9σ** | audible |
| `[confirmation-en]` | 1.74 s | +0.67 s (+5.0σ) | 84.3 % | +2.0σ | audible |
| `[surprise-oh]` | 1.70 s | +0.63 s (+4.6σ) | 83.8 % | +1.9σ | audible |
| `[question-en]` | 1.51 s | +0.44 s (+3.3σ) | 87.9 % | **+3.4σ** | audible |
| `[surprise-ah]` | 1.50 s | +0.43 s (+3.2σ) | 86.7 % | **+3.0σ** | audible |
| `[dissatisfaction-hnn]` | 1.37 s | +0.30 s (+2.2σ) | 82.3 % | +1.3σ | marginal |

Every tag adds duration — it is an extra utterance, not a modifier. The direction
of the voiced-fraction change is the giveaway that it is the *right* utterance:
`[laughter]` and `[sigh]` push it sharply **down** (breathy, aperiodic), while
`[surprise-ah]`, `[question-en]` and `[confirmation-en]` push it **up** (sustained
vowels). `[dissatisfaction-hnn]` is the one weak case and should be left out of a
user-facing picker until it is checked by ear.

Note the tags append an event to the utterance; they do not colour the sentence.
So the UI abstraction in the spec (`Laugh` / `Sigh` / `Surprise` /
`Dissatisfaction`) is best rendered as an insert-at-cursor button, not a mode.

---

## 6. Offline & Privacy

| ID | status | evidence |
|---|---|---|
| F-E01 Fully Offline | **✅ proven, more strongly than airplane mode** | The APK declares no `INTERNET` permission; an explicit socket attempt returns `EPERM` **while the device is on Wi-Fi**. Airplane mode shows "the radios were off this run"; this shows "this process can never reach the network". |
| F-E02 Voice Data Protection | **partly** | App-private internal storage ✅ (`filesDir`, forced by an SELinux finding — see `scripts/push_models.sh`). No upload possible ✅ by construction. Explicit deletion, no raw-voice logging, Keystore encryption: not built. |

---

## 7. Developer & Benchmark

| ID | status | evidence |
|---|---|---|
| F-F01 Developer Settings | engine-side done, no UI | Backend (CPU/XNNPACK/NNAPI/QNN), model variant, threads, steps are all instrumentation args today |
| F-F02 Performance Metrics | **✅ all of them** | load, encode, generate, decode, duration, peak PSS, RTF — emitted per generation. The one addition worth making: report **thermal state**, because a warm device is ~2× slower (11.30 → 20.89 at identical settings) and a metric without it is not comparable. |

---

## 8. What this changes in the Phase 1 plan

1. **On-device enrollment is in scope** (the DoD says 등록하고), so the encoder
   graphs must ship: 520 MB → **1.17 GB**, plus `RECORD_AUDIO`.
2. **Style in the UI means profile selection**, not an instruct dropdown — for
   cloned voices. That is cheaper to build *and* it is the thing that works.
3. **RTF < 1.0 must be re-stated as a Phase 4 gate, not a Phase 1 target**, and
   met by a second `SpeechSynthesizer` implementation rather than by optimising
   this one.
4. **Generation must be a background job.** 10 s of audio takes 79 s; a
   foreground-only button loses the work on app switch.
5. **Do not split text into sentences.** Short utterances cost ~2× per second of
   audio because the reference prefix is re-paid every forward.
