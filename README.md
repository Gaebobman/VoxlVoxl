# OmniVoice Android On-Device PoC

Run [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) voice-cloning TTS **entirely
on a Samsung Galaxy S26+**, with no network access at inference time.

```
reference.wav + reference transcript + target text  ──►  cloned speech (24 kHz WAV)
```

Status: **Phase 1 — model analysis complete, export not yet run.**
Read [`docs/plan.md`](docs/plan.md) for where this is going and
[`docs/onnx-reuse-audit.md`](docs/onnx-reuse-audit.md) for why the obvious shortcut
(`onnx-community/OmniVoice-Onnx`) does not work.

---

## The one-paragraph version

OmniVoice is a **masked diffusion LM**, not an autoregressive codec LM: a Qwen3-0.6B
backbone with **bidirectional** attention, run 32 times over the whole sequence while
un-masking cells of an 8×T audio-token grid, then decoded to 24 kHz by the Higgs Audio V2
codec. Bidirectional attention means **no KV cache is possible**, so the cost is
`32 steps × 2 CFG branches × full forward` — about **13 TFLOP for 5 s of audio**. That
number, not operator support, is the real obstacle on a phone.

The published community ONNX conversion exports the backbone with
`com.microsoft::GroupQueryAttention`, which is causal — so it cannot reproduce the model
regardless of precision. We export the backbone ourselves as one fused bidirectional
graph and reuse only that repo's Higgs codec graphs.

---

## Setup

### 1. PC environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r scripts/requirements.txt
# for export + the PyTorch golden reference:
uv pip install --python .venv/bin/python \
    "torch>=2.8" "torchaudio>=2.8" "transformers>=5.4" "omnivoice==0.2.1"
```

### 2. Model download  (~4 GB)

```bash
.venv/bin/python scripts/download_models.py
```

Fetches `k2-fsa/OmniVoice` (PyTorch source of truth, 3.3 GB) and the reusable Higgs codec
ONNX graphs from `onnx-community/OmniVoice-Onnx` (740 MB fp32), then sanity-checks the
config against `docs/model-analysis.md` §1.

### 3. Golden reference *(not yet implemented — Phase 1)*

```bash
.venv/bin/python scripts/reference_infer.py \
    --ref-audio sample/reference.wav --ref-text "$(cat sample/reference.txt)" \
    --text "$(cat sample/target.txt)" --out out/golden/
```

### 4. ONNX export *(not yet implemented — Phase 2)*

```bash
.venv/bin/python scripts/export_onnx.py --precision int4 --out models/onnx/int4
.venv/bin/python scripts/validate_onnx.py --stage lm --stage codec --stage e2e
```

### 5. Model placement on device *(Phase 4)*

```bash
adb push models/android/. /sdcard/Android/data/<pkg>/files/models/
```

`assets/` is not used: a ~500 MB asset breaks the install size limit and is decompressed
on every read. The app copies to app-private `filesDir/models/` on first launch and
verifies `manifest.json` checksums. **No model is ever fetched over the network.**

### 6. Android build *(Phase 4)*

```bash
cd android/OmniVoicePoC && ./gradlew assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md). Short form:

```
Kotlin  →  ONNX Runtime 1.22 (Java API)  →  CPU / XNNPACK / NNAPI
```

No Python on device. No Termux, Chaquopy, embedded CPython or PyTorch-Android.

| on-device component | provenance |
|---|---|
| Qwen2 byte-level BPE | Kotlin port reading `tokenizer.json` |
| duration estimator | Kotlin port of `RuleDurationEstimator` (Apache-2.0) |
| prompt build + 32-step CFG unmasking | Kotlin port of `_generate_iterative` (Apache-2.0) |
| `omnivoice_lm.onnx` | exported by `scripts/export_onnx.py` |
| `higgs_*.onnx` | reused from `onnx-community/OmniVoice-Onnx` |
| silence-trim / re-gain / fade | Kotlin port of `_post_process_audio` (Apache-2.0) |

---

## Known limitations (current, honest)

- **Speed is the open question.** Bidirectional attention forbids a KV cache, so every one
  of the 32 steps re-reads the full sequence. The projection in
  `docs/model-analysis.md` §6 is RTF 18–52 on the S26+ CPU at default settings, and
  RTF 3–8 once int4 + `num_step=8` + no-CFG are applied. Real measurements go in
  `docs/benchmark.md`; the projection stays labelled as a projection until then.
- **NNAPI will probably not help.** It is deprecated as of Android 15, and
  `MatMulNBits` / `SimplifiedLayerNormalization` are contrib ops it cannot execute — expect
  a fragmented graph. Exynos's 80-TOPS NPU is reachable only via Samsung ENN SDK, for
  which ONNX Runtime has no execution provider.
- **The reference transcript is typed by the user.** No Whisper in v1, by design.
- **Long text is expensive**, quadratically so — attention is `O(S²)` and `S` includes the
  reference prefix. Chunked generation (upstream's `audio_chunk_threshold`) is the fix.
- ORT Python here is 1.30 while the Android AAR on Maven Central is 1.22 — exports are
  pinned to **opset 20** so the older runtime can load them.

## Repository layout

```
docs/
  model-analysis.md    Phase 1 — what OmniVoice is, exactly, with the cost model
  onnx-reuse-audit.md  Phase 2 — what already exists and what is actually usable
  plan.md              phased plan, contingencies, deliverables checklist
  architecture.md      on-device component and session design
  benchmark.md         result tables (empty until measured)
scripts/               PC-side: download, export, validate, ONNX reference inference
android/OmniVoicePoC/  the app
models/                gitignored — see models/README.md
sample/                reference.wav / reference.txt / target.txt
```

## Licences

OmniVoice code and weights, the ONNX conversions we reuse, and sherpa-onnx are Apache-2.0;
ONNX Runtime is MIT.

**The Higgs Audio V2 codec weights are not.** They ship under the *Boson Higgs Audio 2
Community License* (a Llama-3-derived licence, `models/omnivoice/audio_tokenizer/LICENSE`):
royalty-free, but it requires attribution ("Built with Higgs Materials licensed from Boson
AI USA, Inc." + the Llama 3 notice), binds you to the Llama 3 Acceptable Use Policy,
prefixes derivative model names with "Higgs Audio 2", and **requires an expanded licence
from Boson AI above 100,000 annual active users**. The codec is not optional — it *is*
OmniVoice's audio representation — so this term applies to any product built on OmniVoice.
See `docs/onnx-reuse-audit.md` §6.

The `ggml` community ports and `AFun9/Omnivoice-onnx` are **not** usable commercially —
see `docs/onnx-reuse-audit.md` §2 and §4.
