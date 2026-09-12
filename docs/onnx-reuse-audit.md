# Phase 2 — Audit of existing OmniVoice ONNX conversions

The seed brief says: *reuse what exists, verify before trusting it.* This is the
verification. Every claim below was checked against the artifact itself (ONNX graph
inspection / HF API file listing / repo source), not against its README.

---

## Candidates found

| # | artifact | license | verdict |
|---|---|---|---|
| 1 | `onnx-community/OmniVoice-Onnx` (HF) | Apache-2.0 | **backbone rejected, codec reused** |
| 2 | `Prince-1/OmniVoice-Onnx`, `gluschenko/omnivoice-onnx`, `OpenVoiceOS/phoonnx-omnivoice` | mixed | same lineage / partial; superseded by #1 |
| 3 | `AFun9/Omnivoice-onnx` (GitHub) | **no LICENSE = all rights reserved** | correct design, cannot copy — reference only |
| 4 | `FerrisMind/omnivoice-rs` (Candle, Rust) | Apache-2.0 | usable algorithm cross-check; not an Android path |
| 5 | `bluryar/omnivoice.cpp`, `rockerritesh/omnivoice-tts.cpp` (ggml) | NOASSERTION / "research, non-commercial" | **not usable commercially** — see §4 |
| 6 | `k2-fsa/sherpa-onnx` | Apache-2.0 | does **not** support OmniVoice (TTS families: VITS, Matcha, Kokoro, Kitten, ZipVoice, PocketTTS, Supertonic). Reused as Android/ORT packaging reference only. |
| 7 | official k2-fsa ONNX export | — | does not exist; issue [#151](https://github.com/k2-fsa/OmniVoice/issues/151) is open with no maintainer response |

---

## 1. `onnx-community/OmniVoice-Onnx` — the backbone is architecturally wrong

The repo splits the backbone into three graphs and drives them from `inference.py`.
Inspecting `int4/llm_decoder.onnx` (the 0.3 MB graph file; weights are in the sidecar
`.onnx.data`, so the topology is checkable without downloading 296 MB):

```
producer : onnxruntime-genai
opsets   : ai.onnx 21, com.microsoft 1
inputs   : attention_mask [batch, total_sequence_length]  (int64, 1-D per token)
           inputs_embeds  [batch, sequence_length, 1024]
           past_key_values.{0..27}.{key,value} [batch, 8, past_seq, 128]
outputs  : hidden_states, present.{0..27}.{key,value}
ops      : 140 com.microsoft::MatMulNBits
            28 com.microsoft::GroupQueryAttention   ← decisive
            56 com.microsoft::SkipSimplifiedLayerNormalization
            57 ai.onnx::SimplifiedLayerNormalization  ...
GQA attrs: num_heads=16, kv_num_heads=8, do_rotary=1, softcap=0.0
GQA input[5,6] = ReduceSum(attention_mask) / Gather(Shape(attention_mask))
```

`com.microsoft::GroupQueryAttention` is ORT's **autoregressive** attention kernel. It
derives `seqlens_k` / `total_sequence_length` from the 1-D mask and applies causal
masking internally. There is no 4-D mask input and no non-causal mode.

> **OmniVoice needs bidirectional attention (`docs/model-analysis.md` §1.1). This graph
> is causal. It cannot reproduce the model, at any precision.**

The repo's own `HANDOFF.md` even documents the symptom without naming the cause —
*"the 32-step loop feeds empty `past_key_values (B, heads, 0, head_dim)` and does a full-
sequence forward each step"*. Feeding an empty cache to a causal graph gives you a
causal full-sequence forward, not a bidirectional one.

### Additional defects in that repo's `inference.py`

Independent of the graph problem, the driver script does not implement OmniVoice:

| # | what the script does | what OmniVoice does | effect |
|---|---|---|---|
| a | `--ref_text` is *"used for logging"* only (`inference.py:357`) | `_combine_text()` prepends `ref_text` to the target text inside `<\|text_start\|>…<\|text_end\|>` | the codec prefix has no matching transcript → cloning cannot align |
| b | `audio_mask[0, T_text + T_ref:] = True` — reference positions get `audio_mask = False` | `audio_mask` is `True` from the start of the reference codes | reference codes (0…1023) are looked up in the **text** embedding table → the voice prompt is fed as garbage text |
| c | no style block | `<\|denoise\|><\|lang_start\|>…<\|instruct_start\|>…` prefix is always present | out-of-distribution prompt |
| d | no CFG | `guidance_scale = 2.0`, two branches per step | different distribution |
| e | unmasks whole frames across all 8 codebooks at once, confidence = weighted mean over codebooks | unmasks individual `(codebook, frame)` cells, with `layer_penalty_factor` and Gumbel position noise | different schedule |
| f | `--num_audio_tokens` fixed at 256 (≈10.24 s) | `RuleDurationEstimator` | wrong duration for every input |
| g | no `ref_rms` re-gain, no silence removal, no fade/pad | `_post_process_audio()` | level and edge artifacts |

Its "Verified ✅ / RTF 0.483" table therefore certifies that *a* WAV comes out, not that
OmniVoice came out — and the RTF is measured on the cheapest possible configuration
(no CFG, no reference prefix, causal graph).

### What we DO reuse from it

`audio_tokenizer/` — the four Higgs graphs, exported through Olive from plain
conv/transformer PyTorch modules, no genai involvement, with a verified
audio→codes→audio round-trip:

| file | fp32 | fp16 |
|---|---|---|
| `acoustic_encoder.onnx` | 205.5 MB | 102.6 MB |
| `semantic_encoder.onnx` | 436.7 MB | 218.2 MB |
| `quantizer_encoder.onnx` | 12.1 MB | 6.1 MB |
| `higgs_decoder.onnx` | 86.5 MB | 43.1 MB |

Reusing these skips re-exporting HuBERT + DAC (weight-norm hooks have to be stripped and
the DAC branches pre-traced before export — `optimize.py:_prepare_tok()` is where that
pain lives). **Still gated on our own numerical check against PyTorch** —
`scripts/validate_onnx.py --stage codec`. If they fail, we export them ourselves.

---

## 2. `AFun9/Omnivoice-onnx` — right idea, wrong licence

`export_lm.py` exports **one fused graph**:

```
input_names  = ["input_ids", "audio_mask", "attention_mask", "position_ids"]
output_names = ["logits"]
```

with a `_bool_mask_to_bias()` helper converting the 4-D boolean mask to an additive bias,
and CFG driven by a block-diagonal `[2B, 1, S, S]` mask. That is exactly the right shape
of solution, and it independently confirms our design.

**There is no LICENSE file in the repository**, which under GitHub's terms means all
rights reserved. We treat it as prior art we may read and must not copy. Our
`scripts/export_onnx.py` is written against the Apache-2.0 upstream
(`omnivoice/models/omnivoice.py`) instead.

---

## 3. `sherpa-onnx` — why the obvious shortcut is not available

`k2-fsa/sherpa-onnx` is the same organisation, is Apache-2.0, and already ships
offline Android TTS with a Kotlin API, prebuilt ORT ARM64 libraries and voice-cloning
support for **ZipVoice** and **PocketTTS**. It does not support OmniVoice, and adding a
model family to it means writing C++ inside their framework — the seed brief explicitly
rules out "대규모 framework 개발".

We reuse it as a **reference implementation for packaging**: its Android JNI layout,
its `csukuangfj/android-onnxruntime-libs` prebuilds, and its model-asset/download UX are
the patterns to copy.

> If the goal ever shifts from "OmniVoice specifically" to "offline voice cloning on
> Galaxy S26", sherpa-onnx + ZipVoice is a same-day result instead of a multi-week one.
> That trade-off is recorded here deliberately; the brief asks for OmniVoice, so we build
> OmniVoice.

---

## 4. The ggml option (and why it is parked, not dismissed)

`bluryar/omnivoice.cpp` and `rockerritesh/omnivoice-tts.cpp` run OmniVoice on ggml with
no Python. ggml's ARM64 int4 kernels (`Q4_0` with `i8mm`/`dotprod`) are the best-tuned
CPU matmuls on mobile and would very plausibly beat ORT by 2–3× on Exynos — and a
`.so` + JNI is fully compatible with the brief's target architecture (the forbidden list
is Termux / Chaquopy / CPython / PyTorch-Android, not native C++).

Blockers: `rockerritesh` is explicitly *"Research / non-commercial"*, `bluryar` is
`NOASSERTION`. Neither can be shipped commercially as-is.

**Parked as Plan B**, to be revisited only if (a) ORT CPU RTF turns out unusable *and*
(b) licensing is clarified upstream. Recorded in `docs/plan.md` §8.

---

## 5. Conclusion

```
backbone        →  export ourselves   (scripts/export_onnx.py)
Higgs codec     →  reuse onnx-community/OmniVoice-Onnx, after our own numeric check
tokenizer       →  tokenizer.json from k2-fsa/OmniVoice + Kotlin byte-level BPE
duration / DSP  →  port ~250 lines of Apache-2.0 Python to Kotlin
Android runtime →  com.microsoft.onnxruntime:onnxruntime-android (CPU → XNNPACK → NNAPI)
```

---

## 6. Post-download verification (executed)

`scripts/download_models.py` has been run; all 10 config assertions against
`docs/model-analysis.md` §1 pass. Two things the download turned up:

### 6.1 The reused Higgs graphs are clean — pure `ai.onnx` opset 20, zero contrib ops

| graph | nodes | input → output | notable ops |
|---|---:|---|---|
| `acoustic_encoder` | 1170 | `waveform_24k [B,1,T]` → `acoustic_features [B,256,T_a]` | 37 `Conv`, 36× `Sin`/`Pow`/`Reciprocal` (Snake activation) |
| `semantic_encoder` | 1014 | `waveform_16k [B,T]` → `semantic_features [B,768,T_a]` | 97 `MatMul` (HuBERT-base) |
| `quantizer_encoder` | 354 | `acoustic + semantic` → `codes [8,B,T_a]` | 24 `MatMul`, 8 `ArgMax` (RVQ) |
| `higgs_decoder` | 1250 | `codes [8,B,T]` → `waveform_24k [B,1,T*960]` | 32 `Conv`, 5 `ConvTranspose`, Snake |

No `com.microsoft::*` anywhere, so these load on any ORT build including reduced ones,
and they are the *only* part of the pipeline NNAPI has any chance with. (It still will not
take `Sin`, so the Snake activations will fall back.) Numerical verification against
`HiggsAudioV2TokenizerModel` is still required — `scripts/validate_onnx.py --stage codec`.

### 6.2 Licence correction — the codec weights are NOT Apache-2.0

`models/omnivoice/audio_tokenizer/LICENSE` is the **Boson Higgs Audio 2 Community
License**, derived from the Meta Llama 3 Community License. OmniVoice itself is
Apache-2.0, but it cannot produce audio without this codec, so these terms reach any
product built on it:

| clause | obligation |
|---|---|
| §1.b.i | ship a copy of this licence + the Llama 3 licence; display *"Built with Higgs Materials licensed from Boson AI USA, Inc. … and Meta Llama 3 licensed under the Meta Llama 3 Community License"* |
| §1.b.i | any derivative model's name must **begin with "Higgs Audio 2"** |
| §1.b.iii | include a `NOTICE` file with the Boson + Meta attribution lines |
| §1.b.iv | comply with the Llama 3 Acceptable Use Policy |
| §1.b.v | may not use outputs to improve any other LLM |
| **§2** | **> 100 000 annual active users ⇒ you must request an expanded licence from Boson AI, granted at their sole discretion** |

Royalty-free and fine for a PoC and for a product under 100 k AAU, with attribution. It is
not "Apache-2.0, do what you like", and the OmniVoice README does not mention it. Flagged
here because the brief asked specifically for commercially usable licences.

No alternative codec exists: OmniVoice's audio token space *is* Higgs Audio V2's RVQ
codebook, so swapping it means retraining the model.
