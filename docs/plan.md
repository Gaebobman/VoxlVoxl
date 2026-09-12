# Implementation Plan

Priority order is fixed by the brief and is not negotiable during execution:

```
Correctness > Android portability > Offline execution > Performance > NPU > UI
```

---

## 0. Decisions taken (rationale in `docs/onnx-reuse-audit.md`)

| decision | choice |
|---|---|
| backbone ONNX | **export ourselves**, one fused graph, 4-D bidirectional mask |
| codec ONNX | **reuse** `onnx-community/OmniVoice-Onnx/audio_tokenizer/`, after numeric verification |
| runtime | ONNX Runtime `1.22.0` Android AAR (`com.microsoft.onnxruntime:onnxruntime-android`) |
| tokenizer on device | Kotlin byte-level BPE reading `tokenizer.json` (fallback: `onnxruntime-extensions-android:0.13.0`) |
| duration / DSP on device | Kotlin ports of the Apache-2.0 Python |
| reference transcript | typed by the user — no Whisper in v1, per the brief |
| model delivery | first launch copies from a side-loaded file / OBB into app-private storage; **never** downloaded at inference time |

---

## Phase 1 — PC golden reference  *(blocker for everything else)*

Goal: a byte-reproducible PyTorch output to diff every later stage against.

- [ ] `uv` env: `torch 2.8 + cu128`, `transformers>=5.4`, `omnivoice==0.2.1`
- [ ] `scripts/download_models.py` → `models/omnivoice/` (2.45 GB) + `models/omnivoice/audio_tokenizer/` (806 MB)
- [ ] `scripts/reference_infer.py` — pin `torch.manual_seed`, dump **every** intermediate:
      `text_tokens`, `input_ids`, `audio_mask`, per-step `logits`, final `audio_codes`, waveform
- [ ] record `sample/reference.wav` + `sample/reference.txt` + `sample/target.txt` (Korean)
- [ ] fix `position_temperature`/Gumbel non-determinism: expose `--deterministic` that sets
      `position_temperature = 0` for bit-exact comparison, and run the stochastic default separately

**Exit:** `out/golden/{codes.npy, audio.wav, step_*.npz}` exists and sounds like the speaker.

---

## Phase 2 — ONNX export + PC validation

- [ ] `scripts/export_onnx.py`
  - wrapper module: `(input_ids, audio_mask, attention_mask_4d, position_ids) → logits`
  - bool mask → additive `-inf` bias inside the wrapper; `attn_implementation="sdpa"`
  - opset 20, legacy exporter, external data, dynamic axes `{batch, seq}`
  - `--precision fp32 | fp16 | int4` (int4 via `onnxruntime.quantization.matmul_4bits_quantizer`,
    block 128, RTN; keep `embed_tokens` and `audio_heads` at int8 or fp16 first, measure before
    pushing them to int4)
- [ ] `scripts/validate_onnx.py`
  - `--stage lm`    : per-step `logits` vs golden — report max-abs / cosine per codebook
  - `--stage codec` : reused Higgs graphs vs `HiggsAudioV2TokenizerModel` — encode, decode, round-trip
  - `--stage e2e`   : full ONNX pipeline vs golden — codes Hamming distance, waveform MCD/SNR
- [ ] `scripts/infer_onnx.py` — the **reference implementation of the Kotlin port**; pure
      `onnxruntime + numpy`, no torch. Every function here gets a 1:1 Kotlin counterpart.
- [ ] RTF + peak-RSS table on this PC for `fp32 / fp16 / int4 × num_step {32,16,8} × CFG {on,off}`

**Exit (= brief's Level 1):** `reference.wav + reference.txt + target.txt → cloned.wav`
from ONNX alone, perceptually matching the golden, with a numeric report.

**Risk:** `torch.onnx.export` of Qwen3 + SDPA + a 4-D mask may fold the mask into a
constant or drop the `Where`. Mitigation: export with `dynamo=False` first, verify the
graph has a live `attention_mask` input feeding every attention `Add`, and diff logits at
two different `S` values.

---

## Phase 3 — Voice prompt cache  *(designed in Phase 2, not bolted on later)*

The enrollment path (acoustic + semantic + quantizer, 654 MB fp32) and the generation
path (LM + decoder) are **separate sessions with separate lifetimes**.

```
enrollment (once per speaker)            generation (every utterance)
  reference.wav ──┐                        voice_prompt.bin ──┐
  reference.txt ──┤                        target text     ──┤
                  ▼                                          ▼
  acoustic+semantic+quantizer                        omnivoice_lm (×32 steps ×2)
                  ▼                                          ▼
        voice_prompt.bin  ────────────────────────►   higgs_decoder → WAV
```

`voice_prompt.bin` v1 format (little-endian):

```
magic "OVVP"  u32
version 1     u32
num_codebooks u32 = 8
num_frames    u32 = T_ref
ref_rms       f32
sample_rate   u32 = 24000
ref_text_len  u32
ref_text      utf8[ref_text_len]
codes         i16[8 * T_ref]        # row-major, codebook-major
```

~4 kB for a 10 s reference. Removes 654 MB of models and ~1–2 s of encoder work from the
hot path, and makes an "enrollment-only" APK variant possible.

**Exit:** `scripts/infer_onnx.py --voice-prompt my.bin` produces the same codes as
`--ref-audio ref.wav --ref-text "…"`.

---

## Phase 4 — Minimal Android app

`android/OmniVoicePoC/`, Kotlin, ARM64-only, `minSdk 31`, `targetSdk 36`.

```
MainActivity
 └── OmniVoiceEngine            orchestration, coroutines, cancellation
      ├── QwenBpeTokenizer      tokenizer.json → byte-level BPE (+ 7 special tokens)
      ├── DurationEstimator     port of RuleDurationEstimator
      ├── AudioPreprocessor     WAV decode, mono, 24k/16k resample, RMS, silence trim
      ├── OnnxModelRunner       OrtEnvironment/OrtSession lifecycle, EP selection, tensor reuse
      ├── OmniVoiceDecoder      prompt build + 32-step unmasking + CFG  (port of infer_onnx.py)
      ├── VoicePromptStore      voice_prompt.bin read/write
      └── AudioOutput           AudioTrack playback + 16-bit WAV writer
```

UI is deliberately four fields and three buttons. No Compose theming work.

Model placement: app-private `filesDir/models/`. `assets/` is rejected — a ~500 MB asset
inflates the APK past the install limit and is decompressed on every read.

Explicit error surface (each with its own exception type and Logcat tag):
`MODEL_MISSING`, `MODEL_LOAD_FAILED`, `UNSUPPORTED_OP`, `WAV_DECODE_FAILED`,
`BAD_SAMPLE_RATE`, `TOKENIZER_FAILED`, `OOM`, `GENERATION_FAILED`.

**Exit (= Level 2):** all sessions load on device and report their input/output specs.

---

## Phase 5 — CPU baseline on Galaxy S26+

- [ ] session options: `intra_op_num_threads` swept 1…8, `GraphOptimizationLevel.ALL_OPT`,
      `ExecutionMode.SEQUENTIAL`, arena allocator on
- [ ] measure per the brief: model load time, peak RAM (`Debug.MemoryInfo` + `dumpsys meminfo`),
      input text length, generated duration, inference time, **RTF**, and a per-stage breakdown
      (tokenize / encode / 32×LM / decode / post)
- [ ] thermal: 5 consecutive generations, log `RTF[n]` and thermal status

**Exit (= Level 3):** `reference.wav + transcripts → out.wav` on device, audibly the same
speaker as the PC golden.

**Exit (= Level 4):** repeat with airplane mode + Wi-Fi off + data off. Verified negatively
too: `INTERNET` permission removed from the manifest for the offline build variant, so a
stray network call is a crash, not a silent success.

---

## Phase 6 — XNNPACK

`OrtSession.SessionOptions.addXnnpack(mapOf("intra_op_num_threads" to "4"))`.
Compare latency / RTF / peak RAM / CPU utilisation / thermal against Phase 5.
Known issue: ORT Android XNNPACK registration has aborted in some 1.2x builds
(microsoft/onnxruntime#23826) — if it does, fall back to a custom AAR built with
`--use_xnnpack`, and record the version that works.

---

## Phase 7 — NNAPI / NPU  *(investigation, not a deliverable)*

Reality check before spending time here:

- **NNAPI is deprecated as of Android 15**; the S26+ runs Android 16. ORT's NNAPI EP still
  functions, but vendor coverage is frozen and `MatMulNBits` / `GroupQueryAttention` /
  `SimplifiedLayerNormalization` are contrib ops that NNAPI cannot take at all.
  Expect a heavily fragmented graph and a **slowdown**.
- Exynos 2600's 80-TOPS NPU is reached through Samsung **ENN SDK / Exynos AI Studio**,
  not through ORT. There is no Exynos EP in ONNX Runtime.
  (`onnxruntime-android-qnn` is Qualcomm-only and irrelevant to an Exynos S26+.)

So the deliverable for this phase is **evidence, not acceleration**:

- [ ] run with `NNAPIFlags.USE_FP16 | CPU_DISABLED` off, dump the partitioning
      (`ORT_LOGGING_LEVEL_VERBOSE`, count `NnapiExecutionProvider` vs `CPUExecutionProvider` nodes)
- [ ] write down exactly which ops fell back and why
- [ ] if NNAPI is slower than CPU — and it likely is — **say so and stop**, per the brief

Stretch, only if everything above is green: evaluate ENN SDK / Exynos AI Studio for the
28-layer transformer block. Scoped as research with an explicit time box.

---

## 8. Contingencies

| if | then |
|---|---|
| `torch.onnx.export` will not keep the 4-D mask | export with `dynamo=True` (ONNX IR 10 / opset 23), or hand-patch the graph to inject the mask `Add` |
| int4 destroys quality | fall back to int8 weight-only, then fp16; the phone has 12 GB |
| RTF stays > 30 even at `num_step=8` | (a) ship it as a PoC and report it — that *is* a valid answer to the brief's question 4; (b) re-open the ggml path from `docs/onnx-reuse-audit.md` §4 once licensing is clear |
| ORT Android cannot load a >2 GB-adjacent external-data model | already sized around it: int4 is ~390 MB in one `.onnx.data` |
| Kotlin BPE mismatches HF tokenizer | ship a fixture test: 500 strings tokenized on PC, asserted on device |

---

## 9. Deliverables checklist (from the brief)

- [ ] Android Studio project that builds and runs — `android/OmniVoicePoC/`
- [ ] PC-side ONNX validation script — `scripts/validate_onnx.py`
- [ ] model download / preparation scripts — `scripts/download_models.py`, `scripts/export_onnx.py`
- [ ] architecture documentation — `docs/architecture.md`, `docs/model-analysis.md`
- [ ] Galaxy S26+ benchmark — `docs/benchmark.md`
- [ ] generated sample WAVs — `sample/`
- [ ] README with exact reproduction steps — `README.md`
- [ ] final answers to the four closing questions — `docs/benchmark.md` §Conclusion
