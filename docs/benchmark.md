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

### 7.4 Why hardware acceleration does not engage — the full chain

ORT's Android AAR does not route its own log to logcat, so the verbose
partitioning report is invisible on device. `SessionOptions.enableProfiling`
answers the question better anyway — it records the execution provider of every
node that actually ran (`scripts/analyze_profile.py`). Four experiments, each
isolating one suspect.

**(a) The shipping model — int4, dynamic shapes**

| backend | node executions | claimed by the EP | on CPU |
|---|---:|---:|---:|
| CPU | 2785 | — | 2785 (100 %) |
| XNNPACK | 2785 | **0** | 2785 (100 %) |
| NNAPI | 2785 | **0** | 2785 (100 %) |

**(b) Is it the int4 contrib op?** Same test with the **fp32** graph — plain
`MatMul`, no `MatMulNBits` anywhere:

| backend | claimed | on CPU |
|---|---:|---:|
| XNNPACK | **0** | 2785 (100 %) |
| NNAPI | **0** | 2785 (100 %) |

**No.** Removing the contrib op changes nothing. The quantization format was not
the blocker.

**(c) Is it the dynamic shapes?** Same int4 graph with `batch` pinned to 1 and
`seq` pinned to 188 (`onnxruntime.tools.make_dynamic_shape_fixed`):

| backend | total nodes | claimed by the EP | on CPU | EP time | CPU time |
|---|---:|---:|---:|---:|---:|
| CPU | 1285 | — | 1285 | — | 766 ms |
| XNNPACK | 1286 | **28 (2.2 %)** | 1258 | 30 ms | 910 ms |
| NNAPI | 466 | **142 (30.5 %)** | 324 | **4266 ms** | 760 ms |

**Yes — dynamic shapes were the blocker.** NNAPI goes from 0 nodes to 142 the
moment the shapes are fixed. NNAPI, like every NPU compiler, builds an operand
graph with concrete dimensions ahead of time; a symbolic `seq` makes every node
ineligible, and ORT's NNAPI EP therefore returns an empty capability set.

**And it does not help.** Those 142 NNAPI nodes cost **4266 ms** against
**766 ms for the entire graph on CPU** — roughly **5.6× slower**. Each partition
boundary forces a CPU↔NNAPI tensor copy, and 142 nodes interleaved with 324 CPU
nodes means many boundaries. NNAPI is not merely unable to help here; when it
does engage it is actively harmful.

Only 30.5 % is reachable at all because the arithmetic is
`com.microsoft::MatMulNBits` — a Microsoft contrib op for 4-bit weight-only
quantization that exists nowhere outside ORT's own CPU/CUDA kernels. NNAPI got
the elementwise, normalisation and shape nodes; it could not touch the GEMMs
that are 85 % of the work.

**(d) Do static shapes at least help the CPU?** Two rounds, alternating, same
thermal state:

| graph | nodes | forward (round 1) | forward (round 2) |
|---|---:|---:|---:|
| dynamic int4 | 2785 | 890.6 ms | 958.5 ms |
| static int4 | 1285 | 949.6 ms | 985.0 ms |

**No.** Fixing the shapes constant-folds away 1500 nodes and buys nothing,
because those nodes were never the cost. Where the time actually goes
(one forward, S = 188, CPU, 6 threads):

| op | time | share | count |
|---|---:|---:|---:|
| `MatMulNBits` | 583.5 ms | **85.1 %** | 196 |
| `MatMul` | 25.8 ms | 3.8 % | 30 |
| `FusedMatMul` | 16.4 ms | 2.4 % | 28 |
| `SimplifiedLayerNormalization` | 8.1 ms | 1.2 % | 113 |
| `GatherBlockQuantized` | 8.0 ms | 1.2 % | 2 |
| everything else (Unsqueeze ×456, Gather ×312, Concat ×227, …) | ~43 ms | 6.3 % | ~2400 |

**So: the model is int4-GEMM bound, and no accelerator on this device implements
int4 GEMM through ORT.** That is the whole story.

### 7.5 What a working NPU path would actually require

A static export is not a drop-in, for a reason the pipeline itself forces:

```
conditional branch    S = 188   (style + text + reference codes + MASK block)
unconditional branch  S =  48   (the MASK block alone)
```

Two different lengths every step, and `S` changes with every utterance (text
length + reference length + target length). A fixed-shape deployment therefore
needs **bucketed lengths with padding** — compile for, say, S ∈ {128, 256, 384,
512} and pad up, with the attention mask masking the padding out. Our graph takes
an explicit 4-D `attention_mask` input, so that is expressible without
re-exporting; upstream PyTorch does exactly this, padding the unconditional
branch up to `max_c_len`.

For QNN specifically (`com.microsoft.onnxruntime:onnxruntime-android-qnn`, which
applies because this is a Snapdragon — the device carries
`libSnpeHtpV81Stub.so` and `libnspextensiongenericqnnservice.so`):

| requirement | why | status |
|---|---|---|
| static shapes | HTP compiles ahead of time | doable — §7.4(c) proves the mechanics |
| **QDQ int8/int16, not weight-only int4** | HTP has no int4 GEMM; it wants quantized *activations* too, which needs calibration data | **not done — the real work** |
| context binary caching | HTP graph compilation takes seconds; recompiling per launch would dwarf inference | not done |
| bucketed S with padding | see above | designed, not built |
| 422 MB of weights vs HTP tightly-coupled memory | large models stream weights; this is what Qualcomm's Genie/QAIRT stack exists to manage | unknown |

That is a real project, not a flag. It is the single highest-value remaining
experiment, and it only exists because the device turned out to be a Snapdragon —
an Exynos 2600 would have offered NNAPI (deprecated) or Samsung's ENN SDK, for
which ONNX Runtime has no execution provider at all.

### 7.6b Utterance length — short sentences are the expensive case

RTF is not a constant. The prompt carries a fixed cost — 37 text tokens plus 103
reference frames — that is re-paid at every one of the 32 forwards regardless of
how much audio comes out, so it amortises over longer outputs. Measured, CPU,
6 threads, 16 steps:

| text | audio | S | LM time | total | **RTF** | peak PSS |
|---|---:|---:|---:|---:|---:|---:|
| "네, 알겠습니다." | 1.26 s | 178 | 16.9 s | 19.0 s | **15.08** | 629 MB |
| "오늘 회의를 시작하겠습니다." | 1.88 s | 188 | 18.7 s | 21.1 s | **11.21** | 594 MB |
| three sentences | 10.57 s | 444 | 68.3 s | 79.1 s | **7.48** | 882 MB |

A simple cost model — `(S_cond + T_gen) / T_gen` units per generated frame —
predicts 0.52× relative RTF at `T_gen = 250`; measured was 7.48 / 15.08 =
**0.50×**. The model holds, which makes it usable for the UI's time estimate.

Two consequences for the product:

1. **Generate whole messages, not sentence by sentence.** Splitting a paragraph
   into short utterances roughly doubles the total work, because each fragment
   re-pays the reference prefix. This is the opposite of the usual TTS intuition.
2. **Memory scales with `S`, and faster than the tensors alone suggest.** Peak
   PSS went 594 MB → 882 MB (native 616 MB) between S = 188 and S = 444. Upstream
   chunks above 30 s for this reason; by the cost model chunking 30 s into 2×15 s
   costs about 8 % more compute but bounds peak memory, so it is a memory guard
   rather than a speed feature.

### 7.7 QNN / Hexagon — attempted, and blocked by the runtime, not by us

§7.5 listed what a QNN path needs. All of it was built, and it still does not run
on this device. The reason is worth recording precisely.

**Building the QDQ model.** Calibration inputs were captured from real decoding
runs rather than synthesised — 50 samples over 5 texts and 5 decoding steps,
sequence lengths 48–211 (`scripts/make_calibration.py`). Activation statistics
change enormously between the first un-masking step and the last, so synthetic
tensors would mis-calibrate. Then `get_qnn_qdq_config` + `quantize_static`, per
tensor, no per-channel (`scripts/export_qnn.py`).

Two obstacles on the way, both from the model being 2.45 GB:

- `onnxruntime.tools.make_dynamic_shape_fixed` saves without external data, so it
  cannot round-trip anything over the 2 GB protobuf limit. Replaced with an
  external-data-aware `fix_shapes()`.
- uint16 activations need opset ≥ 21, and ORT's quantizer upgrades by calling
  `onnx.version_converter`, which serializes the whole model and dies for the
  same reason. Opset 20 → 21 changed no semantics for the ops in this graph, so
  the declared version is rewritten directly — and then verified rather than
  assumed: **max|Δ| 0.0** between the opset-20 and opset-21 graphs.

**Quality of the QDQ models** (fp32 reference re-masking judge, same as §1):

| build | size | meanNLL | top-1 @ 50 % | verdict |
|---|---:|---:|---:|---|
| int4 `android_b32` (shipping) | 422 MB | 2.749 | 39.8 % | tied with fp32 |
| *fp32 stochastic (noise floor)* | — | *3.116* | *34.4 %* | — |
| **QDQ a16w8** | 780 MB | **3.331** | 30.5 % | degraded but usable |
| QDQ a8w8 | 617 MB | **8.020** | **1.26 %** | destroyed |

uint8 activations annihilate the model — 1.26 % agreement is essentially random.
That is the familiar transformer activation-outlier problem: a few channels have
ranges orders of magnitude wider than the rest, and per-tensor 8-bit scaling
quantizes everything else to nothing. 16-bit activations recover most of it, at
780 MB and still outside the fp32 band.

**On the device it never gets to run.** With `libQnnHtp.so` loaded and *no* extra
provider options:

```
V  QnnDsp <V> Async property not supported. Skipping setup async threads
E  qnn_execution_provider.cc:767 GetCapability] QNN SetupBackend failed
   Failed to create device. Error: QNN_DEVICE_ERROR_INVALID_CONFIG: Invalid config values
V  session_state.cc:1263] All nodes placed on [CPUExecutionProvider]. Number of nodes: 6493
```

`QnnDevice_create` fails outright, so `GetCapability` returns nothing and all
6493 nodes fall to CPU. It is not a misconfiguration — `htp_arch=79`,
`htp_arch=75`, `soc_model=0`, `device_id=0` and the bare default all produce the
identical error. (An earlier run that appeared clean was a grep artifact; the
error is on every run.)

**Root cause — the shipped QNN SDK predates the chip:**

| | |
|---|---|
| device SoC | id **660**, machine **CANOE** = SM8850, Hexagon **V81** |
| HTP skels inside `onnxruntime-android-qnn:1.22.0` | V68, V69, V73, V75, **V79** — nothing newer |
| QNN skels on the device itself | **none** (`/vendor/lib64/rfsa/adsp/` has no `libQnn*`; only SNPE's `libSnpeHtpV81Stub.so`) |
| newest ORT Android artifact on Maven Central | **1.22.0, published 2025-05-09** |

So there is no combination of options that can work: the QNN SDK bundled with the
newest published ORT Android package was cut before Snapdragon 8 Elite Gen 5
existed, and the device provides no QNN backend of its own to fall back on.

**And the QDQ graph is worse on CPU anyway** — one forward at S = 188:

| graph | forward | composition |
|---|---:|---|
| static int4 | 766 ms | 85 % `MatMulNBits` |
| static QDQ a16w8 | 1209 ms | 68.5 % `MatMul` + 20.0 % `DequantizeLinear` (2830 of them) |

Without an execution provider that consumes QDQ pairs natively, ORT dequantizes
back to float and the 2830 `DequantizeLinear` nodes are pure overhead.

**Reverted.** `onnxruntime-android-qnn` costs +122 MB of APK (25.9 MB → 148 MB)
for skels this chip cannot use, so the dependency is back to
`onnxruntime-android`. The `Backend.QNN` code path, the calibration capture and
`export_qnn.py` are all kept: retrying is a one-line dependency swap plus
`-e backend QNN` once ORT ships a QNN SDK that knows V81. Requesting QNN without
that artifact now fails with a named `MODEL_LOAD_FAILED` rather than silently
falling back to CPU — which is precisely the trap that made the first NNAPI
reading look like a success.

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

### 7.8 The diffusion-LLM speedups, and why one of them does not transfer

The literature on masked diffusion LMs has two headline accelerations, and
OmniVoice is exactly that class of model, so both were worth trying.

**Confidence-aware parallel decoding** (Fast-dLLM, arXiv:2505.22618) is the
cheaper one and needs no re-export. Instead of un-masking a fixed `k` cells per
step, commit *every* cell whose predicted probability clears a threshold; on
LLaDA and Dream this collapses many steps into one and is most of the reported
27.6x. Implemented behind `--confidence-threshold` in `scripts/infer_onnx.py`
and measured on the PC:

| steps | threshold | LM calls | LM time | RTF |
|---:|---:|---:|---:|---:|
| 16 | 0.00 (off) | 32 | 8.39 s | 4.97 |
| 16 | 0.90 | 32 | 7.74 s | 4.59 |
| 16 | 0.95 | 32 | 7.30 s | 4.35 |
| 32 | 0.90 | 64 | 13.95 s | 8.18 |
| 8 | 0.00 (off) | 16 | 3.36 s | 1.89 |

**`LM calls` never drops.** The loop never finishes early, so the small time
differences are run-to-run noise, not a speedup.

`scripts/speed_probe.py` shows why. Per step, out of 384 masked cells, the number
whose CFG-mixed probability clears each threshold:

```
  step  masked  sched k    p>0.5    p>0.7    p>0.9   p>0.95    maxp
     1     384        3       15        1        1        0   0.900
     4     374        4       40       13        3        1   0.998
     8     353        8       46       16        3        3   0.965
    12     309       20       16        6        2        1   0.963
    16     145      145       11        2        0        0   0.821
```

At no point are there more high-confidence cells than the schedule was already
going to take — by step 8 the schedule takes 8 and only 3 cells exceed p > 0.9.
This is a property of the data, not of the implementation. A text diffusion LM
picking among 150k tokens is often overwhelmingly sure of the next word; an RVQ
audio cell is one of 1025 near-equivalent codes, and our own re-masking judge
measures the model's NLL at ~2.75 — a perplexity near 15. **A model that is never
confident cannot be accelerated by trusting its confidence.** The flag stays in
the script because the measurement is the useful artefact, but it is off by
default and there is nothing to gain by turning it on.

**Approximate prefix KV cache** is the other half of Fast-dLLM and is untried.
It is not blocked by the model the way a real KV cache is: §1.1's finding
forbids caching across the *generated* region, whose tokens change every step,
but the reference prefix does not change at all. At S = 188 the prefix is 140
tokens — **74 % of the sequence**, re-encoded 32 times for nothing. Trimming the
reference measures the size of that prize directly:

| reference frames | reference seconds | S | forward | speedup |
|---:|---:|---:|---:|---:|
| 103 (as enrolled) | 4.12 | 188 | 314 ms | 1.00x |
| 75 | 3.00 | 155 | 274 ms | 1.15x |
| 50 | 2.00 | 125 | 204 ms | 1.54x |
| 25 | 1.00 | 92 | 156 ms | 2.01x |

Halving the reference nearly halves the work, which is the cost of re-reading
the prefix stated in wall-clock. Caching it rather than truncating it would keep
the voice quality and needs a re-export carrying `past_key_values` I/O on the
prefix range only. That is the one identified, quantified, un-taken speedup left,
and it is a day of work on the export, not a flag.

### 7.8b The shipping model asks for fp32 compute, and §7.7 was reading a stale Maven

Two things this section previously got wrong, both found by reading the artifacts
rather than the documentation.

**`accuracy_level` was never actually swept.** It selects the *compute* type of an
int4 GEMM and touches no stored weight: level 1 is fp32 compute, level 4 is
`SQNBIT_CompInt8` — the quantised-activation path, and the only one ONNX Runtime's
ARM64 KleidiAI int4 kernels will dispatch to. Reading the attributes straight out
of the graphs:

```
models/onnx/int4/omnivoice_lm.onnx        (bits, block, accuracy_level) -> {(4, 32, 1): 196}
models/onnx/int4_kv/omnivoice_lm_kv.onnx                                -> {(4, 32, 4): 196}
```

The shipping export is at **level 1**, so the 85 % of device runtime that is
`MatMulNBits` runs on the generic MLAS fp32-compute path. §1's note that
"accuracy_level 4 cost 75 points of argmax agreement" is about `sym_acc4_mg`,
which was *also* symmetric, block-128 and int4-headed — the three things §1
independently proves are fatal. Level 4 on top of `android_b32` was never isolated.

`scripts/set_accuracy_level.py` rewrites the attribute and copies the external-data
blob verbatim, so the same weights can be judged both ways. Re-masking NLL against
the fp32 scorer, three seeds each:

| | seed 1234 | seed 7 | seed 99 | mean |
|---|---:|---:|---:|---:|
| acc1 (shipping) | 2.8722 | 2.9477 | **3.1080** | 2.976 |
| acc4 | 3.0443 | 2.9853 | 2.9833 | 3.004 |

**The distributions overlap completely** — acc1's worst seed scores worse than
acc4's worst. The seed-to-seed spread of 0.236 swamps the 0.028 difference in
means, and both sit under the 3.116 stochastic-fp32 noise floor from §1. A single
seed would have said level 4 costs 0.17 nats; it does not. There is no measurable
quality reason not to ship level 4.

**ORT Android is eight releases newer than §7.7 assumed.** `maven-metadata.xml`
for `com.microsoft.onnxruntime:onnxruntime-android` lists through **1.29.0**
(`lastUpdated 20260812`); 1.24.1, 1.25.0 and 1.29.0 all resolve. The Maven
*solrsearch* API still reports 1.22.0 as newest and is what produced the earlier
conclusion — it is stale, and `maven-metadata.xml` is the authority. KleidiAI
asymmetric-int4 kernels for `MatMulNBits` landed in 1.25.0 and 1.29.0, which means
the pairing that matters is a runtime bump *and* the attribute, since neither is
any use without the other.

**A caution before projecting the published KleidiAI numbers onto this device:
it has SME1, not SME2.** The full feature line:

```
… sve sve2 svei8mm svebf16 i8mm bf16 … sme smei8i32 smef16f32 smeb16f32 smef32f32 lrcpc3
```

No `sme2`. Most of the reported uplift in that work is attributed to SME2 kernels,
so only the **i8mm / dotprod** path can dispatch here. The device measurement is
pending and is the only one worth quoting: PC timings taken while other jobs
shared the machine varied 6.6–12.4 s for the same configuration and are not
evidence of anything.

### 7.9 Codebook ablation — which layers of the RVQ actually matter

The UI claims the voice's outline resolves before its detail, which is a claim
about the residual quantiser and therefore testable. `scripts/codebook_ablation.py`
randomises one codebook at a time and re-decodes (12 trials, LSD and SNR against
the intact decode):

| corrupted | LSD dB | SNR dB |
|---|---:|---:|
| **codebook 0** | **13.70** | **-1.13** |
| **codebook 1** | **9.14** | 2.48 |
| codebook 2 | 3.90 | 9.94 |
| codebook 3 | 4.40 | 5.18 |
| codebook 4 | 3.42 | 12.93 |
| codebook 5 | 3.57 | 13.85 |
| codebook 6 | 3.13 | 14.56 |
| codebook 7 | 3.07 | 13.53 |

and keeping only the first k, randomising the rest:

| kept | LSD dB | SNR dB |
|---|---:|---:|
| 0 only | 10.82 | -0.99 |
| 0–1 | 6.84 | 1.87 |
| 0–3 | 5.30 | 7.65 |
| 0–5 | 3.95 | 11.17 |
| 0–6 | 3.07 | 13.88 |
| 0–7 | 0.00 | 147.85 |

The ordering by damage is `[0, 1, 3, 2, 5, 4, 6, 7]` — **not** monotone in the
index. So "coarse to fine" is directionally right but overstated: it is not a
smooth eight-step gradient but a **cliff and a plateau**. Codebooks 0 and 1 carry
~4.5x the damage of any other, and 2 through 7 are near-interchangeable refinement
whose individual order barely matters.

The *temporal* half of the claim is on firmer ground and is a property of the
sampler rather than the codec: `layer_penalty_factor = 5.0` subtracts 5 per
codebook index from the selection score, so layer 0 is un-masked first by
construction. On device the ladder shows the resulting staircase directly —
83 / 56 / 27 / 10 / 3 / 2 / 1 / 0 percent filled at the same step.

---

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
**Plain CPU**, RTF 11.06 against XNNPACK 12.46 and NNAPI 12.94 — and §7.4 traces
exactly why, with four experiments rather than an assertion. Dynamic shapes make
both accelerators claim **0 of 2785 nodes** at any precision; pinning the shapes
gets NNAPI to 142 nodes, which then run **5.6× slower** than the whole graph does
on CPU; and the 85 % of the work that is `MatMulNBits` is unreachable for any of
them. The untried option is **QNN / Hexagon**, which this Snapdragon supports and
an Exynos would not — but it needs static bucketed shapes, a QDQ int8 requantize
with calibration, and context-binary caching, so it is a project, not a flag.

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
