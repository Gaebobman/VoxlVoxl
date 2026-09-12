# Benchmark

Measured numbers only. Empty cells mean "not measured yet"; projections are
labelled as projections and kept out of the measured tables.

## Test condition

```
reference     : sample/reference.wav  4.25 s, 24 kHz mono
                (synthesised by OmniVoice voice-design so the transcript is exact)
reference text: 안녕하세요. 이것은 제 목소리를 등록하기 위한 테스트 음성입니다.
target text   : 오늘 회의를 시작하겠습니다.
voice prompt  : (8, 103) codes = 4.12 s, 1766 bytes on disk
sequence      : S = 188  (37 text + 103 reference + 48 generated)
generated     : 48 frames = 1.92 s @ 24 kHz, 25 fps
RTF           = (decode loop + vocoder) / generated_audio_duration
PC            : Ryzen 7 5700X (8C/16T), 31 GB, ORT 1.30 CPU EP
GPU reference : RTX 5080, PyTorch 2.14 fp16
```

The duration estimator picked **48 frames on both the PyTorch and the ONNX path** —
the Kotlin port has to reproduce that exactly or every utterance is the wrong length.

---

## 1. Quantization — what survives

Two metrics, because the obvious ones lie.

*Single-forward argmax* compares one forward pass against cached PyTorch fp32
logits. *Re-masking NLL* takes each candidate's generated codes, re-masks
25/50/75 % of the (codebook, frame) cells and has the fp32 graph fill them back
in — OmniVoice's actual training objective, so the number is a real likelihood.
Comparing generated codes directly is worthless: the loop is chaotic, and int8
agrees with fp32 on 89.6 % of single-forward argmaxes but only **13.8 %** of
final codes.

| variant | int4 scope | block | head | MB | cos | argmax % | top-5 % | re-mask NLL |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fp32 (reference) | — | — | fp32 | 2451.6 | 1.000000 | 100.00 | 100.00 | **2.752** |
| `int8_mg` | MatMul+Gather, 8-bit | 128 | int8 | 1122.8 | 0.999998 | 89.57 | 99.67 | 2.718 |
| `int4_nohead_b32` | MatMul only | 32 | fp32 | 972.2 | 0.999837 | 63.25 | 95.78 | 2.746 |
| **`android_b32`** | **MatMul+Gather** | **32** | **fp32** | **422.2** | 0.999847 | 63.82 | 95.70 | **2.749** |
| `hqq_nohead` | MatMul, HQQ | 128 | fp32 | 937.8 | 0.999661 | 60.02 | 92.63 | — |
| `android_b128` | MatMul+Gather | 128 | fp32 | 358.5 | 0.999627 | 57.04 | 91.56 | 3.193 |
| `asym_b32_mg` | MatMul+Gather | 32 | int4 | 394.0 | 0.999717 | 30.88 | 69.29 | — |
| `asym_acc1_mg` | MatMul+Gather | 128 | int4 | 329.4 | 0.999423 | 26.49 | 58.36 | — |
| `sym_acc4_mg` | MatMul+Gather, symmetric | 128 | int4 | 327.0 | 0.998829 | 25.50 | 57.20 | 3.274 |

**The noise floor**: the same fp32 graph run with upstream's default stochastic
sampling (`position_temperature=5.0`) scores **3.116**. Anything at or below
~2.75 is indistinguishable from fp32; anything above 3.1 is worse than simply
re-rolling the dice.

Two conclusions, both non-obvious:

1. **`/audio_heads/MatMul` must not be int4.** It is `Linear(1024 → 8×1025)`
   sitting directly on the logits, so its error is not averaged away by anything
   downstream. Excluding it moves argmax agreement 25 % → 63 % and NLL
   3.27 → 2.75, at a cost of 34 MB.
2. **Block size 32, not 128.** `android_b128` (358 MB) scores 3.193 — worse than
   the stochastic fp32 run. `android_b32` (422 MB) scores 2.749. 64 MB buys the
   whole quality difference.

Shipping configuration: **int4 asymmetric, block 32, MatMul + Gather, audio head
at fp32 — 422 MB, 5.8× smaller than fp32, quality tied with fp32.**

---

## 2. Decoding steps and classifier-free guidance

int4 `android_b32`, 16 threads, deterministic (`position_temperature = 0`).

| num_step | guidance | LM calls | LM time | audio | RTF | re-mask NLL | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 2.0 | 64 | 14.52 s | 1.78 s | **8.39** | 2.749 | tied with fp32 |
| 16 | 2.0 | 32 | 7.37 s | 1.86 s | **4.12** | 2.948 | inside the stochastic-fp32 band |
| 8 | 2.0 | 16 | 3.71 s | 1.82 s | **2.18** | 3.184 | worse than re-rolling fp32 |
| any | 0.0 | — | — | **0.00 s** | — | — | **fails — pure silence** |

**`guidance_scale = 0` does not degrade OmniVoice, it breaks it.** The vocoder
output is 1.92 s entirely below −50 dBFS (peak 7.3e-04, rms 6.6e-05), so the
silence trimmer removes all of it. The unconditional branch is load-bearing, and
dropping it is not available as a performance lever. `infer_onnx.py` now detects
this and exits with a named error instead of dividing by zero.

**`num_step = 16` is the knee**: half the latency, and the quality drop is
smaller than the gap between deterministic and stochastic fp32.

---

## 3. Thread scaling — the basis for the phone projection

int4 `android_b32`, `num_step = 16`, guidance 2.0.

| intra_op_num_threads | LM time | RTF | speed-up |
|---:|---:|---:|---:|
| 1 | 30.09 s | 16.39 | 1.00× |
| 2 | 17.34 s | 9.49 | 1.74× |
| 4 | 9.88 s | 5.44 | 3.05× |
| 8 | 7.18 s | 4.00 | 4.19× |
| 16 | 6.78 s | 3.80 | 4.44× |

Scaling is good to 4 threads and flattens after 8 — the workload is
memory-bandwidth bound on the int4 weight stream, which is exactly the regime a
phone is worse at.

### Projection for the Galaxy S26+ *(projection, not measurement)*

The right anchor is the **4-thread** row (RTF 5.44 at 16 steps), since that is
roughly the S26+'s big-core count. An Exynos 2600 big core is plausibly 1.5–3×
slower than a Zen 3 core at int4/int8 GEMM once clocks and thermal limits are
accounted for:

| num_step | desktop 4-thread RTF | projected S26+ RTF |
|---:|---:|---:|
| 32 | 11.1 | **17 – 33** |
| 16 | 5.44 | **8 – 16** |
| 8 | 2.9 | **4 – 9** |

Read plainly: **a 5-second sentence will take roughly 40–80 seconds on the phone
at the recommended 16-step setting.** Useful for asynchronous synthesis — draft
a message, generate it, send it — and not useful for anything interactive. The
measurement that replaces this table is the first thing Phase 5 produces.

---

## 4. Component timings (PC, int4, 16 threads)

| stage | time | note |
|---|---:|---|
| model load (int4, 422 MB) | 1.2 s | mmap'd external data |
| model load (fp32, 2.45 GB) | 3.3 s | |
| enrollment (3 encoder graphs, 4.25 s reference) | 1.7 s | once per speaker, then cached |
| one conditional forward, S = 188 | 0.31 s | |
| decode loop, 16 steps × 2 branches | 7.4 s | 97 % of total |
| `higgs_decoder` | 0.21 s | |
| post-process | < 0.01 s | |

The decode loop is everything. The vocoder and the encoders are rounding error,
which is why the voice-prompt cache matters for *size*, not for speed.

---

## 5. Fidelity of the reused Higgs codec ONNX

| path | vs transformers `HiggsAudioV2TokenizerModel` |
|---|---|
| `higgs_decoder` (codes → waveform) | max\|Δ\| 1.10e-06, cos 1.0000 — numerically exact |
| encode, identical waveform in | 98.94 % code agreement |
| encode, full enrollment path | 92.11 % (per codebook 99/95/95/94/94/89/83/87) |

The enrollment gap is the 24 kHz → 16 kHz resampler feeding the semantic
(HuBERT) branch plus RVQ residual rounding. Resampler choice moves the total
only from 91.3 % (soxr) to 93.3 % (torchaudio), and codebook 0 — which carries
most of the speaker identity — stays at 99–100 % in every case. **A decent
Kotlin resampler is good enough; a bit-exact one is not required.**

The numpy ports of pydub's silence detection are bit-exact against pydub
(max\|Δ\| 0.00e+00). They have to be: the first, approximate version trimmed the
reference to 104 frames instead of 103 and code agreement collapsed to 2.55 %.

---

## 6. Galaxy S26+ *(pending — Phase 5)*

| Backend | Precision | num_step | Audio | Load | Latency | RTF | Peak RAM |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU | int4 | 16 | | | | | |
| CPU | int4 | 32 | | | | | |
| CPU | int4 | 8 | | | | | |
| XNNPACK | int4 | 16 | | | | | |
| NNAPI | int4 | 16 | | | | | |

### NNAPI partitioning *(pending)*

| metric | value |
|---|---|
| nodes on NNAPI / on CPU | |
| partitions | |
| top unsupported ops | |

---

## Conclusion — the four questions from the brief

1. **Did OmniVoice inference actually run on the Galaxy S26+?**
   *Pending.* It runs end to end from ONNX alone on the PC (Level 1 complete),
   with a graph whose only device dependency is ONNX Runtime.
2. **Is it fully offline?**
   *Pending on device.* By construction there is no network path: 422 MB
   backbone + 86 MB vocoder + 11 MB tokenizer, all local; the offline build
   variant drops the `INTERNET` permission so a stray call crashes rather than
   silently succeeding.
3. **Which backend is fastest?** *Pending.* CPU is the baseline; NNAPI is
   expected to lose (deprecated in Android 15, and `MatMulNBits` /
   `GatherBlockQuantized` / `SimplifiedLayerNormalization` are contrib ops it
   cannot execute).
4. **Is it fast enough for real use?**
   *Provisionally no, for interactive use.* Desktop measurement is RTF 4.12 at
   the recommended 16 steps with 16 threads, RTF 5.44 with 4; the phone will be
   slower, not faster. The levers that exist have been measured: steps 32→16
   halves it at a quality cost smaller than upstream's own sampling noise,
   steps→8 costs real quality, and dropping CFG is not a lever at all because it
   produces silence.
