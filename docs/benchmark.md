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

## 1b. End-to-end equivalence — the Level 1 acceptance evidence

Both pipelines run fully deterministic (`position_temperature = 0`,
`class_temperature = 0`, greedy) on the same voice prompt and text, then the
fp32 reference scores both outputs by re-masking:

| pipeline | frames | audio | re-mask NLL | top-1 @ 50 % |
|---|---:|---:|---:|---:|
| ONNX fp32, `infer_onnx.py` | 48 | 1.82 s | **2.7325** | 40.43 % |
| PyTorch fp16 CUDA, `reference_infer.py` | 48 | 1.83 s | 2.7672 | 44.03 % |

**Final-code agreement between them is 31.77 %** (per codebook
88/56/31/29/10/25/10/4) — and that is fine. Both sides run the same greedy loop,
but one tie broken differently in step 1 propagates through every later step; the
agreement decays with codebook depth exactly as trajectory divergence predicts,
not as a porting bug would. The metric that matters says the ONNX output is
**as good as, in fact marginally better than, PyTorch's own** — the two are
interchangeable samples of the same distribution.

The parts that *are* required to match exactly do:

| invariant | result |
|---|---|
| assembled `input_ids` / `audio_mask` | byte-identical, `(1,8,188)` |
| duration estimator | both choose 48 frames |
| single forward, ONNX fp32 vs PyTorch fp32 | rel 9.6e-07, cos 1.0000, argmax 100.000 % |
| `higgs_decoder` | max\|Δ\| 1.1e-06 |
| silence removal | bit-exact against pydub |

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

## 6. Runtime-version compatibility ✅

The PC exports and quantizes with **ORT 1.30**, but the Android AAR on Maven
Central is **1.22.0**. The shipping graphs were loaded and run under a pinned
`onnxruntime==1.22.0`:

| graph | ORT 1.22 | result |
|---|---|---|
| `omnivoice_lm.onnx` (int4, opset 21 + com.microsoft 1) | loads | `logits (1,8,188,1025)`, 318 ms, cos 0.999847 vs PyTorch fp32 — identical to 1.30 |
| `higgs_decoder.onnx` (opset 20, pure ai.onnx) | loads | `(1,1,46080)` from 48 frames |

So `com.microsoft::MatMulNBits` (4-bit, block 32) and
`com.microsoft::GatherBlockQuantized` are both present and numerically identical
in 1.22. The remaining Android-specific unknowns are the ARM64 kernels and the
external-data loader, not operator coverage.

---

## 7. On device — Galaxy S26 Ultra ✅

**The target device is not the one the brief assumed.** It is a Galaxy S26
**Ultra** (SM-S948N), and its SoC is **Qualcomm SM8850 — Snapdragon 8 Elite
Gen 5**, not an Exynos 2600:

```
ro.product.model       SM-S948N          ro.soc.manufacturer  QTI
ro.build.version.release 16              ro.soc.model         SM8850
ro.product.cpu.abi     arm64-v8a         ro.board.platform    canoe
MemTotal               11 389 624 kB     cores                8
CPU part 0x002 (Oryon) x8: 6 @ 3.63 GHz + 2 @ 4.74 GHz
```

That invalidates the brief's Phase 7 premise (Samsung ENN SDK / Exynos NPU) and
replaces it with a better one — Qualcomm ships a QNN execution provider for ONNX
Runtime (`onnxruntime-android-qnn`) targeting the Hexagon NPU. See §7.5.

### Level 2 — sessions load ✅

| | |
|---|---|
| backbone `omnivoice_lm.onnx` (422 MB int4 + external data) | **1 023 – 2 110 ms** |
| `higgs_decoder.onnx` (86 MB fp32) | **153 – 287 ms** |
| declared tensor signatures | all four inputs correct, `logits [-1,8,-1,1025]` out |
| Kotlin BPE on device | `<\|denoise\|>`→151669, `<\|text_start\|>`→151674, `<\|text_end\|>`→151675 |

ORT 1.22's ARM64 build accepts `MatMulNBits` (4-bit, block 32) and
`GatherBlockQuantized` exactly as the desktop 1.22 check predicted.

### Level 3 — voice cloning on device ✅

`sample/reference.wav` + its transcript + "오늘 회의를 시작하겠습니다." →
a 1.88 s WAV whose duration, RMS (0.0904) and peak (0.5692) sit right between the
desktop ONNX output (1.86 s / 0.0862 / 0.6696) and the PyTorch golden
(1.83 s / 0.0949 / 0.5887).

Best configuration, cold device: **RTF 11.30**, peak PSS **591 MB**
(native 439 MB) — comfortable inside 11.4 GB, and inside a normal Android heap
because the ORT arenas are native.

### Level 4 — offline ✅ (proven more strongly than airplane mode)

```
RESULT offline     socket_attempt=SocketException msg=socket failed: EPERM (Operation not permitted)
RESULT permissions ai.omnivoice.poc.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION
```

The app tried to open a TCP socket to 1.1.1.1:53 **while the device was fully
connected to Wi-Fi**, and the kernel refused: the manifest declares no `INTERNET`
permission, and the only entry in the permission list is one the build system
injects. Airplane mode would show "the radios were off during this run"; this
shows "this process cannot reach the network, ever".

### Level 5 — backend, threads, steps ✅

All at S = 188, 48 generated frames, guidance 2.0.

**Threads** (CPU EP, 16 steps):

| threads | LM time | RTF |
|---:|---:|---:|
| 1 | 69.3 s | 38.19 |
| 2 | 39.4 s | 22.19 |
| 4 | 23.6 s | 13.95 |
| **6** | **18.9 s** | **11.56** |
| 8 | 29.5 s | 17.31 |

8 threads is *worse* than 6. The SoC is 6 + 2, so thread 7 and 8 land on the two
prime cores and the whole step then waits on scheduler migration and thermal
budget sharing. **6 is the setting.**

**Execution providers** (6 threads, 16 steps):

| backend | LM time | RTF | vs CPU |
|---|---:|---:|---:|
| **CPU** | 18.2 s | **11.06** | — |
| XNNPACK | 20.5 s | 12.46 | +13 % slower |
| NNAPI | 21.1 s | 12.94 | +17 % slower |

**Steps** (CPU, 6 threads):

| num_step | LM calls | LM time | RTF |
|---:|---:|---:|---:|
| 8 | 16 | 7.6 s | **5.40** |
| 16 | 32 | 18.5 s | **11.30** |
| 32 | 64 | 40.5 s | **24.64** |

### 7.4 Why NNAPI and XNNPACK lose — the partitioning evidence

ORT's Android AAR does not route its own log to logcat, so the verbose
partitioning report is invisible on device. `SessionOptions.enableProfiling`
answers the question better anyway — it records the execution provider of every
node that actually ran (`scripts/analyze_profile.py`):

| backend | node executions | on the EP | on CPU |
|---|---:|---:|---:|
| CPU | 2785 | — | **2785 (100 %)** |
| XNNPACK | 2785 | **0** | **2785 (100 %)** |
| NNAPI | 2785 | **0** | **2785 (100 %)** |

**Neither accelerator claimed a single node.** The 13–17 % penalty is pure
registration and partitioning overhead with zero offload in return. The cause is
structural, not a misconfiguration: the graph's arithmetic is
`com.microsoft::MatMulNBits`, a contrib op neither backend implements.

Where the time actually goes (one forward, S = 188, CPU, 6 threads, 685.5 ms):

| op | time | share | count |
|---|---:|---:|---:|
| `MatMulNBits` | 583.5 ms | **85.1 %** | 196 |
| `MatMul` | 25.8 ms | 3.8 % | 30 |
| `FusedMatMul` | 16.4 ms | 2.4 % | 28 |
| `SimplifiedLayerNormalization` | 8.1 ms | 1.2 % | 113 |
| `GatherBlockQuantized` | 8.0 ms | 1.2 % | 2 |
| everything else (Unsqueeze ×456, Gather ×312, Concat ×227, …) | ~43 ms | 6.3 % | ~2400 |

The dynamic-shape export leaves ~2400 shape-manipulation nodes in the graph and
they cost 6 % combined. **There is no overhead to reclaim — the model is
genuinely int4-GEMM bound**, so the only real levers are fewer steps, shorter
sequences, or a faster int4 kernel.

### 7.5 The remaining acceleration option: QNN, not NNAPI

Because this is a Snapdragon, `com.microsoft.onnxruntime:onnxruntime-android-qnn`
is applicable — it targets the Hexagon NPU directly instead of going through the
deprecated NNAPI abstraction. The device carries `libSnpeHtpV81Stub.so` and
`libnspextensiongenericqnnservice.so`, so the runtime is present.

It is **not** a drop-in: QNN's HTP backend wants static shapes and its own
quantization (QDQ int8/int16, not `MatMulNBits`), so it needs a separate export
with fixed `S` and a QDQ quantization pass. That is the single highest-value
remaining experiment and it did not exist as an option under the Exynos premise.

### 7.6 Thermal — sustained throughput is about half of cold

Five consecutive generations in one process (16 steps, 6 threads), device on USB
power:

| run | total | RTF | thermal status |
|---:|---:|---:|---|
| 1 | 50.0 s | 26.59 | 1 (LIGHT) |
| 2 | 44.5 s | 23.67 | 1 |
| 3 | 36.4 s | 19.34 | 1 |
| 4 | 40.9 s | 21.75 | 1 |
| 5 | 44.3 s | 23.57 | 1 |

Battery temperature rose 32.4 °C → 39.0 °C over the session.

The confound worth naming: those runs were back-to-back *and* the phone was
charging over USB. So a controlled check — one isolated run, identical settings,
on a warm device:

| device state | RTF |
|---|---:|
| cold | **11.30** |
| warm (same isolated test) | **20.89** |

**Sustained performance is roughly half the cold-start number, and thermal state
dominates every other variable measured here** — it is a larger effect than
backend choice (17 %), and comparable to halving the step count. Any future
benchmark on this device has to state its thermal state or it is not comparable.
`thermal_status` never left 1 (LIGHT), so this is ordinary DVFS, not throttling.

---

## Conclusion — the four questions from the brief

**1. Did OmniVoice inference actually run on the Galaxy S26?**
**Yes.** Reference WAV + typed transcript + target text → a 1.88 s cloned WAV,
entirely on a Galaxy S26 Ultra (SM-S948N, Android 16), through ONNX Runtime with
no Python anywhere. Levels 2, 3, 4 and 5 are all reached. Note the device is an
**Ultra with Snapdragon 8 Elite Gen 5 (SM8850)**, not the Exynos 2600 the brief
assumed.

**2. Is it fully offline?**
**Yes, and enforced rather than asserted.** The APK declares no `INTERNET`
permission; an explicit socket attempt returns `EPERM` while the device is on
Wi-Fi. All 520 MB of model plus the 11 MB tokenizer are local, and nothing is
fetched at any point.

**3. Which backend is fastest?**
**Plain CPU.** RTF 11.06 against XNNPACK 12.46 and NNAPI 12.94, and the profiler
shows why: both accelerators claimed **0 of 2785 nodes**. 85 % of the time is
`com.microsoft::MatMulNBits`, which neither implements. The untried option is
**QNN / Hexagon**, which this Snapdragon supports and an Exynos would not — but
it needs a separate static-shape QDQ export, so it is the next experiment, not a
result.

**4. Is it fast enough for real use?**
**For asynchronous use, yes. For interactive use, no.**

| setting | cold RTF | 5 s sentence |
|---|---:|---:|
| 8 steps | 5.4 | ~27 s |
| 16 steps (recommended) | 11.3 | ~57 s |
| 32 steps (upstream default) | 24.6 | ~123 s |

and sustained/warm operation roughly doubles all of those. That is usable for
"type a message, generate it, send it" and unusable for anything conversational.

The ceiling is structural, not an implementation gap. OmniVoice's attention is
bidirectional, so no KV cache is possible and every step re-reads the whole
sequence; the graph is 85 % int4 GEMM with no overhead left to reclaim; and the
two levers that looked most promising before measurement both failed — dropping
CFG produces silence, and pushing int4 into the audio head destroys quality.
What remains is QNN/Hexagon offload, or a faster ARM int4 kernel than MLAS's.
