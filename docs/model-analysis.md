# Phase 1 — OmniVoice Model Analysis

> Source of truth: `k2-fsa/OmniVoice` @ master (Apache-2.0), `omnivoice` PyPI v0.2.1.
> All facts below were read out of the upstream source and the published HF configs,
> not from blog posts. File/line references are to the upstream repo.

---

## 1. What OmniVoice actually is

OmniVoice is **not** an autoregressive codec-LM (not VALL-E / not ZipVoice flow-matching).
It is a **masked diffusion language model** over an 8-codebook neural audio codec:

```
Qwen3-0.6B backbone (28 layers, hidden 1024, 16 Q heads / 8 KV heads, head_dim 128)
        ▲ bidirectional (NON-causal) full attention over the whole sequence
        │
  input_ids [B, 8, S]  +  audio_mask [B, S]
        │
        ▼
  audio_heads: Linear(1024 → 8 × 1025)  →  logits [B, 8, S, 1025]
        │
  32-step iterative un-masking over the (codebook × frame) grid
        ▼
  audio_codes [8, T]  →  Higgs Audio V2 Tokenizer decoder  →  24 kHz waveform
```

Constants (`OmniVoiceConfig`, `config.json` of `k2-fsa/OmniVoice`):

| name | value |
|---|---|
| `audio_vocab_size` | 1025 (1024 codes + MASK) |
| `audio_mask_id` | 1024 |
| `num_audio_codebook` | 8 |
| `audio_codebook_weights` | `[8, 8, 6, 6, 4, 4, 2, 2]` |
| `llm_config.hidden_size` | 1024 |
| `llm_config.num_hidden_layers` | 28 |
| `llm_config.num_attention_heads` / `num_key_value_heads` | 16 / 8 |
| `llm_config.head_dim` | 128 |
| `llm_config.intermediate_size` | 3072 |
| `llm_config.vocab_size` | 151676 |
| `rope_theta` | 1 000 000 |
| `tie_word_embeddings` | true |
| frame rate | **25 fps** (hop_length 960 @ 24 kHz) |
| output sample rate | 24 000 Hz |

### 1.1 The one fact that governs the whole port

`OmniVoice._generate_iterative()` builds a **4-D boolean attention mask**
`[2B, 1, S, S]` that is `True` everywhere inside the valid region, and hands it to
`self.llm(...)`. transformers uses a 4-D mask verbatim, bypassing causal-mask creation.

> **OmniVoice attention is bidirectional. Every unmasked step re-reads the entire
> sequence, including the not-yet-generated MASK positions.**

Consequences, all of which drive the deployment plan:

1. **No KV cache is possible.** Not even a prefix cache: with bidirectional attention the
   prefix's own K/V depend on the suffix, so they change at every step. The cost is
   `num_step × full_forward(S)`, and that is inherent to the algorithm, not an oversight.
2. **Any ONNX export built with an autoregressive LLM exporter is wrong** (see
   `docs/onnx-reuse-audit.md`) — those emit causal attention and cannot express this.
3. The upside: no cache management, no dynamic shapes across steps, static `S` per
   utterance. That is actually *friendlier* for NNAPI/NPU partitioning than a normal LLM.

---

## 2. Input pipeline (exact)

### 2.1 Prompt construction — `_prepare_inference_inputs()` (omnivoice.py:1198)

```text
style_text = ("<|denoise|>" if denoise and cloning)
           + "<|lang_start|>"     + (lang or "None")     + "<|lang_end|>"
           + "<|instruct_start|>" + (instruct or "None") + "<|instruct_end|>"

full_text  = _combine_text(text, ref_text)            #  ref_text + " " + text
text_part  = "<|text_start|>" + full_text + "<|text_end|>"

input_ids  = concat([ tok(style_text),                 # audio_mask = False
                      tok(text_part),                  # audio_mask = False
                      ref_audio_tokens (8, T_ref),     # audio_mask = TRUE
                      MASK × T_gen (8, T_gen) ])       # audio_mask = TRUE
                      → shape [1, 8, S], text rows replicated across all 8 codebooks

audio_mask[0, S - T_ref - T_gen :] = True
```

**The reference transcript is prepended to the target text in the *same* text field.**
This is the mechanism that lets the model align the reference codec prefix with the
reference speech; it is not optional, and it is why the PoC asks the user to type it.

### 2.2 Embedding fusion — `_prepare_embed_inputs()` (omnivoice.py:470)

```python
text_embeds  = llm.embed_tokens(input_ids[:, 0, :])                 # row 0 only
shifted      = input_ids * audio_mask.unsqueeze(1) + codebook_offsets   # offsets = arange(8)*1025
audio_embeds = audio_embeddings(shifted).sum(dim=1)                  # sum over 8 codebooks
inputs_embeds = where(audio_mask[..., None], audio_embeds, text_embeds)
```

`audio_embeddings` is a single `nn.Embedding(8 * 1025, 1024)` — codebook *c* occupies
rows `[c*1025, (c+1)*1025)`.

### 2.3 Tokenizer

Standard **Qwen2 byte-level BPE** (`tokenizer_class: Qwen2Tokenizer`, NFC normalizer,
GPT-2 split regex, ByteLevel pre-tokenizer, `add_prefix_space: false`), vocab 151643 +
33 added tokens. OmniVoice adds 7 of its own:

| id | token |
|---|---|
| 151669 | `<\|denoise\|>` |
| 151670 / 151671 | `<\|lang_start\|>` / `<\|lang_end\|>` |
| 151672 / 151673 | `<\|instruct_start\|>` / `<\|instruct_end\|>` |
| 151674 / 151675 | `<\|text_start\|>` / `<\|text_end\|>` |

Non-verbal tags (`[laughter]`, `[sigh]`, `[question-en]`, …) are tokenized standalone by
`_tokenize_with_nonverbal_tags()` so their ids do not depend on the surrounding language.

### 2.4 Reference audio preprocessing — `create_voice_clone_prompt()` (omnivoice.py:729)

1. load → mono → resample to **24 kHz**
2. `ref_rms = sqrt(mean(x²))`; if `0 < ref_rms < 0.1`, scale to RMS 0.1 (kept for output re-gain)
3. `remove_silence(mid_sil=200ms, lead_sil=100ms, trail_sil=200ms)` (skipped if `preprocess_prompt=False`)
4. truncate length to a multiple of `hop_length = 960`
5. `HiggsAudioV2TokenizerModel.encode()` → `audio_codes (8, T_ref)`
6. `add_punctuation(ref_text)`
7. warn if reference > 20 s (recommended 3–10 s)

### 2.5 Target length — `RuleDurationEstimator` (utils/duration.py)

A **pure rule-based, table-driven** function — no model, no learned parameters:

```
weight(char) from a Unicode-block table (cjk 3.0, hangul 2.5, kana 2.2, indic 1.8,
    latin/cyrillic/greek 1.0, digit 3.5, punctuation 0.5, space 0.2, combining mark 0.0)
speed_factor  = weight(ref_text) / num_ref_audio_tokens
T_gen         = weight(target_text) / speed_factor   (/ speed)
  with a power-curve boost below 50 frames: 50 * (est/50)^(1/3)
```

Fallback when no reference: `ref_text = "Nice to meet you."`, `ref_duration = 25` frames.

**This is ~120 lines of pure logic and ports to Kotlin verbatim. No ONNX needed.**

---

## 3. Decoding loop (exact) — `_generate_iterative()` (omnivoice.py:1275)

### 3.1 Classifier-free guidance

Two sequences are run **every step**:

| branch | contents | length |
|---|---|---|
| conditional | style + text + ref codes + MASK block | `S_c` |
| unconditional | **only** the trailing MASK block (`input_ids[..., -T_gen:]`) | `T_gen` |

```python
c_lp = log_softmax(c_logits); u_lp = log_softmax(u_logits)
log_probs = log_softmax(c_lp + guidance_scale * (c_lp - u_lp))   # guidance_scale = 2.0
log_probs[..., 1024] = -inf                                       # never emit MASK
```

### 3.2 Schedule

```python
t = linspace(0, 1, num_step + 1)
t = t_shift * t / (1 + (t_shift - 1) * t)         # t_shift = 0.1 → front-loaded
total = T_gen * 8
k[step] = min(ceil(total * (t[step+1] - t[step])), remaining)   # last step takes the rest
```

### 3.3 Position selection — over the flattened **(8 × T_gen)** grid, not per frame

```python
scores = log_probs.max(-1)                                   # confidence
scores -= layer_id * layer_penalty_factor                    # 5.0 → lower codebooks first
scores  = scores / position_temperature + gumbel_noise        # position_temperature = 5.0
scores[already_filled] = -inf
topk(scores.flatten(), k) → write pred_tokens at those (codebook, frame) cells
```

Token choice: `class_temperature = 0.0` → **greedy argmax**; if `> 0`, top-10 % filter +
Gumbel sampling.

Default generation config (`OmniVoiceGenerationConfig`):

| param | default |
|---|---|
| `num_step` | 32 |
| `guidance_scale` | 2.0 |
| `t_shift` | 0.1 |
| `layer_penalty_factor` | 5.0 |
| `position_temperature` | 5.0 |
| `class_temperature` | 0.0 |
| `denoise` | true |
| `audio_chunk_threshold` / `audio_chunk_duration` | 30 s / 15 s |

### 3.4 Output post-processing — `_post_process_audio()`

`remove_silence(mid=500, lead=100, trail=100)` → re-gain to `ref_rms` (or peak-normalize
to 0.5 when no reference) → `fade_and_pad_audio(pad 0.1 s, fade 0.1 s)`.

---

## 4. Audio codec — Higgs Audio V2 Tokenizer

`HiggsAudioV2TokenizerModel` (transformers ≥ 5.3), weights in
`k2-fsa/OmniVoice/audio_tokenizer/` (805.7 MB fp32). **Licensed under the Boson Higgs
Audio 2 Community License, not Apache-2.0** — see `docs/onnx-reuse-audit.md` §6.2. Two encoders + RVQ + one decoder:

| part | model | rate |
|---|---|---|
| acoustic encoder | DAC, `hop_length 960`, `downsampling_ratios [8,5,4,2,3]`, hidden 256 | 24 kHz in → 25 fps, 256-d |
| semantic encoder | **HuBERT-base** (12 layers, hidden 768) + pad(160,160), layer-averaged, ×½ downsample | 16 kHz in → 25 fps, 768-d |
| quantizer | RVQ, 8 codebooks × 1024 entries, `codebook_dim 64` | → codes `(8, T)` |
| decoder | DAC decoder, `upsampling_ratios [8,5,4,2,3]` | codes → 24 kHz waveform |

**Both sample rates are needed at enrollment time** (24 kHz for acoustic, 16 kHz for
semantic). Generation needs only the decoder.

---

## 5. Component table (ONNX deployment view)

`P` = per-utterance prompt length, `S` = full conditional sequence length,
`T_ref`/`T_gen` = reference / generated frames (25 fps).

| # | component | framework | params | ONNX input | ONNX output | dtype | dynamic | exportable | needed at generate time? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Qwen2 BPE tokenizer | tokenizers (Rust) | — (11.4 MB json) | UTF-8 string | `int64[N]` | — | yes | n/a — port or ORT-extensions | **yes** |
| 1 | duration estimator | pure Python | — | text, ref_text, T_ref | `int` | — | yes | n/a — port to Kotlin | **yes** |
| 2 | `omnivoice_lm` (fused: embeddings + Qwen3 + audio_heads) | PyTorch | ≈ 612 M | `input_ids int64[B,8,S]`, `audio_mask bool[B,S]`, `attention_mask bool[B,1,S,S]`, `position_ids int64[B,S]` | `logits float[B,8,S,1025]` | fp32→int4 | `B`, `S` dynamic | **yes — we export it** | **yes** (×32 steps ×2 CFG branches) |
| 3a | higgs acoustic encoder | DAC | ≈ 51 M | `waveform_24k float[1,1,T]` | `acoustic_features float[1,256,T_a]` | fp32/fp16 | `T` dynamic | yes (reuse) | enrollment only |
| 3b | higgs semantic encoder | HuBERT-base | ≈ 95 M | `waveform_16k float[1,T]` | `semantic_features float[1,768,T_a]` | fp32/fp16 | `T` dynamic | yes (reuse) | enrollment only |
| 3c | higgs quantizer encoder | RVQ | ≈ 3 M | `acoustic_features`, `semantic_features` | `codes int64[8,1,T_a]` | fp32/fp16 | `T_a` dynamic | yes (reuse) | enrollment only |
| 3d | higgs decoder | DAC decoder | ≈ 21 M | `codes int64[8,1,T]` | `waveform_24k float[1,1,T*960]` | fp32/fp16 | `T` dynamic | yes (reuse) | **yes** (once per utterance) |
| 4 | silence-removal / fade / re-gain | numpy | — | waveform | waveform | — | — | n/a — port to Kotlin | **yes** |

Splitting #2 into three graphs (as the community conversion does) buys nothing and costs
two extra tensor round-trips per step; we keep it fused.

---

## 6. Cost model — why this is the hard part

One conditional forward is `~2 × 440 M = 0.88 GFLOP` per token (transformer stack
only), plus `~0.017 GFLOP/token` for `audio_heads`, plus `O(S²)` attention. For the
repo's sample — 4.1 s reference, 1.9 s target, 37 text tokens — `S_c = 188`,
`S_u = 48`, and 32 steps costs ≈ 4.1 TFLOP.

**These are now measurements, not estimates** — see `docs/benchmark.md` for the full
tables. Ryzen 7 5700X, ORT CPU EP, int4 block-32:

| config | threads | RTF |
|---|---:|---:|
| 32 steps, guidance 2.0 | 16 | 8.39 |
| 16 steps, guidance 2.0 | 16 | 4.12 |
| 16 steps, guidance 2.0 | 4 | 5.44 |
| 16 steps, guidance 2.0 | 1 | 16.39 |
| 8 steps, guidance 2.0 | 16 | 2.18 |
| any steps, guidance 0 | — | **generation fails — silence** |

Thread scaling flattens past 8 threads (4.44× at 16 threads), so the loop is
bandwidth-bound on the int4 weight stream rather than compute-bound. Anchoring on
the 4-thread row and allowing 1.5–3× for an Exynos 2600 big core gives a projected
**RTF 8–16 at 16 steps on the S26+** — roughly 40–80 s for a 5 s sentence.

### Levers, with their measured cost

| lever | measured saving | measured quality cost |
|---|---|---|
| `num_step` 32 → 16 | **2.0×** | NLL 2.749 → 2.948, still inside the fp32 stochastic band (3.116) |
| `num_step` 16 → 8 | **1.9×** | NLL → 3.184 — worse than re-rolling fp32; not recommended |
| int4 block-32 vs fp32 | 5.8× smaller, ~1.15× faster on x86 | none measurable (NLL 2.749 vs 2.752) |
| int4 block-128 | a further 64 MB | NLL → 3.193 — not worth it |
| int4 including `audio_heads` | a further 34 MB | NLL → 3.274 — **never do this** |
| `guidance_scale = 0` | would be ~1.4× | **breaks generation entirely** |
| shorter reference | shrinks `S_c` linearly | mild speaker-similarity drop (unmeasured) |
| ORT thread count 1 → 4 | **3.05×** | none |
| chunked generation | bounds `O(S²)` for long text | cross-fade seams (unmeasured) |
| NNAPI / Exynos NPU | unknown — see `docs/plan.md` §7 | unknown |

**The honest summary: this is asynchronous synthesis, not interactive TTS.** The two
levers that looked most promising before measurement — dropping CFG and pushing int4
everywhere — are the two that do not work.

## 7. Memory budget (int4 backbone, fp32 codec)

| artifact | size |
|---|---|
| `omnivoice_lm` int4 (incl. 151676×1024 tied text embedding) | ~390 MB |
| `higgs_decoder` fp32 / fp16 | 86 MB / 43 MB |
| `tokenizer.json` | 11.4 MB |
| **generation-time total** | **~490 MB** |
| + acoustic + semantic + quantizer encoders (enrollment only) | +654 MB fp32 / +327 MB fp16 |

Peak runtime RAM ≈ weights + activations. Activations are dominated by
`logits [1, 8, S, 1025]` fp32 = **11 MB at S=335** (and the `[8, T_gen, 1025]` slice used
for scoring), plus attention `[1, 16, S, S]` = 7 MB. Comfortable inside 12 GB, and even
inside a 512 MB-per-process Android heap once the ORT arenas are native (off-heap).

**This is why the voice-prompt cache matters**: enrolling once and storing
`codes (8, T_ref) int16` (a few kB) lets the app ship *without* the 654 MB of encoders on
the generation path, and removes ~1.5 s of encoder work per utterance.
