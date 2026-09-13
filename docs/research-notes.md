# Research notes

A lab notebook for the inference-speed and quality work, kept because most of
the value here is in the experiments that **failed** and the reasons they
failed. `benchmark.md` is the result tables; this is the reasoning, the method,
and the dead ends.

Every number below was produced by a script in `scripts/` or by the
instrumented benchmark in `android/OmniVoicePoC/app/src/androidTest/`. Where a
number is quoted from a paper rather than measured here, it says so.

Device for all on-device figures:

```
Galaxy S26 Ultra · SM-S948N · SM8850 · Android 16 · arm64-v8a
8 cores (6 @ 3.63 GHz + 2 @ 4.74 GHz) · 11.4 GB RAM
CPU features: … sve sve2 svei8mm svebf16 i8mm bf16 … sme smei8i32 smef16f32 smeb16f32 smef32f32
```

Note the feature line: this part has **SME1, not SME2**. That matters in §5.1.

---

## 1. Method — why the obvious measurements are useless here

### 1.1 You cannot compare generated codes

OmniVoice's un-masking loop is chaotic. One different token at step 1 changes
everything downstream. Measured: an int8 variant agrees with fp32 on **89.6 %**
of single-forward argmaxes but on only **13.8 %** of final codes.

So neither "do the codes match" nor "do the waveforms match" can rank two
variants. Both answer *no* for changes that cost nothing and for changes that
destroy the model.

### 1.2 The re-masking NLL judge

The metric used throughout:

1. take the codes a candidate produced (8 codebooks × T frames)
2. randomly re-mask 25 % / 50 % / 75 % of the cells
3. ask the **fp32 reference** to fill them back in
4. report its negative log-likelihood of the codes that were actually there

Step 3 is *exactly* OmniVoice's training objective, so the number is a real
likelihood rather than an invented score. Units are nats per cell, and each
cell is one of 1025 codes:

| NLL | perplexity | meaning |
|---:|---:|---|
| 6.93 | 1025 | uniform guessing |
| **2.75** | ~16 | fp32 scoring its own output — effectively the ceiling |
| **3.12** | ~23 | fp32 re-run with different sampling noise — **the noise floor** |

The usable window is 2.75–3.12, not 0–∞. **Anything above 3.12 is worse than
simply re-rolling fp32's dice** and is disqualified; `android_b128` died exactly
there, at 3.193.

What this metric does **not** measure: intelligibility (WER) or speaker
similarity. It answers "is this something OmniVoice itself would have produced",
nothing more. §6.3 explains why that gap matters.

### 1.3 One seed is not a measurement

A single seed will happily report a 0.17-nat difference that three seeds show to
be nothing. See §5.1. Every quality claim here is at least three seeds, and the
seed-to-seed spread is reported next to the effect.

### 1.4 On-device A/B has to interleave, or it measures DVFS

The most expensive methodological lesson of the project. A first sequential
device sweep reported thread affinity at 1.05 s and `dynamic_block_base=4` at
0.95 s against a 2.14 s baseline — apparent 2× wins. Re-measured with both
sessions **live in one process, round-robin**, they are **1.03×** and **0.97×**:
noise.

A single forward moves **30 %** between adjacent measurements on this phone.
Any future A/B has to interleave inside one process, or it is reporting the
governor's mood.

The same applies to the desktop when it is shared. Timings taken while other
jobs ran on the same box swung **6.6 s and 12.4 s for the identical
configuration**. Those runs are excluded from every table here.

---

## 2. The architectural finding everything else follows from

OmniVoice is a **masked diffusion LM**, not an autoregressive codec LM: a
Qwen3-0.6B backbone with **bidirectional** attention, run once per un-masking
step over the whole sequence.

Bidirectional attention means **no KV cache is possible**. In an autoregressive
LM tokens are appended left-to-right and past K/V never change; here every
position can change at every step, so there is no fixed past to cache. The cost
is therefore `steps × 2 CFG branches × full forward` — about 13 TFLOP for 5 s of
audio at the upstream default of 32 steps. That number, not operator support, is
the real obstacle on a phone.

**This also invalidates the obvious reuse.** `onnx-community/OmniVoice-Onnx`
exports the backbone with `com.microsoft::GroupQueryAttention`, which is causal,
so it cannot reproduce the model at any precision. We export the backbone
ourselves as one fused bidirectional graph and reuse only that repo's Higgs
codec graphs. Details in `onnx-reuse-audit.md`.

**The one exception is the prefix.** The reference audio codes, the reference
transcript and the target text do not change from step to step. At S = 188 that
prefix is 140 tokens — **74 % of the sequence**, re-encoded every forward. It is
the only thing in the graph that a cache could legally hold, and it is the
largest un-taken lever (§7.1).

---

## 3. Quantization — what survives, and the two hard constraints

16 variants swept (`sweep_quant.py`). Two conclusions, neither obvious:

**`/audio_heads/MatMul` must stay fp32.** It is `Linear(1024 → 8×1025)` sitting
directly on the logits, so its error is not averaged away by anything
downstream. Excluding it from int4 moves argmax agreement 25 % → 63 % and NLL
3.27 → 2.75, at a cost of 34 MB.

**Block size 32, not 128.** `android_b128` (358 MB) scores 3.193 — past the
noise floor. `android_b32` (422 MB) scores 2.749. 64 MB buys the entire quality
difference.

Shipping: **int4 asymmetric, block 32, MatMul + Gather quantized, audio head
fp32 — 422 MB, 5.8× smaller than fp32, quality tied with fp32.**

---

## 4. Levers measured before the device arrived

| lever | effect | quality |
|---|---|---|
| `num_step` 32 → 16 | **2.2×** | NLL 2.749 → 2.948, inside the band |
| `num_step` 16 → 8 | 1.9× | NLL → 3.184 — **past the noise floor, rejected** |
| `guidance_scale` 2.0 → 0 | would be ~1.4× | **1.92 s entirely below −50 dBFS — silence** |
| shorter reference | linear in S | speaker similarity unmeasured |

The CFG result deserves care, because it is easy to state wrongly. In
`infer_onnx.py` `guidance_scale = 0` takes the **pure conditional** path, not the
unconditional one. So the silence means OmniVoice's *conditional distribution
alone is degenerate* — the model has learned to rely on the `c + w(c − u)`
extrapolation being the output. That is a stronger statement than "CFG is
required": it means no pure loop change recovers the full CFG cost.

---

## 5. On device

Levels 2–5 of the brief reached: sessions load, cloning runs, offline is
enforced (the APK declares no `INTERNET` permission and an explicit socket
attempt returns `EPERM` while the device is on Wi-Fi), and the backend / thread
/ step matrix is measured.

Two results from that matrix are worth restating:

**8 threads is worse than 6** (RTF 17.31 vs 11.56). The SoC is 6 + 2; threads 7
and 8 land on the prime cores and the step then waits on migration and shared
thermal budget.

**Plain CPU beats both accelerators** — CPU 11.06, XNNPACK 12.46 (+13 %), NNAPI
12.94 (+17 %). §7.4 of `benchmark.md` traces why with four experiments rather
than an assertion: with dynamic shapes both accelerators claim **0 of 2785
nodes**; pinning the shapes gets NNAPI to 142 nodes which then run **5.6×
slower** than the whole graph on CPU; and the 85 % of the work that is
`MatMulNBits` is unreachable for either of them.

### 5.1 The largest win, and it was a default nobody questioned

`MatMulNBits` has an `accuracy_level` attribute that selects the **compute**
type of the int4 GEMM. It touches no stored weight:

| level | what the kernel does | instructions |
|---|---|---|
| 1 | dequantize int4 → fp32, fp32 GEMM | ordinary FMA |
| **4** (`SQNBIT_CompInt8`) | quantize activations to int8, int8×int4 → int32 GEMM | `sdot` / `i8mm` |

Reading the attributes out of the shipping graph:

```
models/onnx/int4/omnivoice_lm.onnx   (bits, block, accuracy_level) -> {(4, 32, 1): 196}
```

**Level 1.** So the 85 % of device runtime that is `MatMulNBits` was on the
generic fp32-compute path, and ONNX Runtime's ARM64 KleidiAI int4 kernels —
which require int8 activations — could never dispatch, at any runtime version.

This had never been isolated. `sweep_quant.py` carried a note that accuracy
level 4 "cost 75 points of argmax agreement", but that observation is about a
variant which was **also** symmetric, block-128 and int4-headed — the three
things §3 independently proves are fatal. Level 4 on top of `android_b32` was
never tested on its own.

`set_accuracy_level.py` rewrites the attribute and reuses the weight blob
verbatim, so the same weights can be judged both ways. Three seeds each:

| | seed 1234 | seed 7 | seed 99 | mean |
|---|---:|---:|---:|---:|
| fp32 compute (level 1) | 2.8722 | 2.9477 | **3.1080** | 2.976 |
| int8 compute (level 4) | 3.0443 | 2.9853 | 2.9833 | 3.004 |

The distributions overlap completely — level 1's worst seed is worse than level
4's worst. The 0.236 seed spread swamps the 0.028 difference in means, and both
sit under the noise floor. **No measurable quality cost.** Logit agreement on a
single forward: argmax **384/384**, max |Δ| 0.113.

**The second half is the runtime.** `benchmark.md` §7.7 had concluded ORT 1.22.0
was the newest Android artifact. That came from Maven's `solrsearch` API, which
is stale and still reports 1.22.0 as newest. `maven-metadata.xml` — the
authority — lists through **1.29.0**. The KleidiAI asymmetric-int4 kernels for
`MatMulNBits` landed after 1.22.

Neither half works alone:

| ORT | compute | LM gen | RTF |
|---|---|---:|---:|
| 1.22.0 | fp32 (shipping) | 25 439 ms | 15.39 |
| 1.22.0 | int8 | 20 303 ms | 12.60 |
| 1.29.0 | fp32 | 23 927 ms | 14.63 |
| **1.29.0** | **int8** | **9 426 ms** | **6.65** |

Version bump alone 1.05×; attribute alone 1.25×; **together 2.3×**. Confirmed by
interleaved A/B (§1.4): int8 vs fp32 compute is **1.398×** on 1.22 and
**2.163×** on 1.29, faster in 8/8 rounds both times.

**Through the product**, same sentence, same device, same app:

```
before   6.05 s of audio in 38.1 s    RTF 6.3
after    5.35 s of audio in 19.4 s    RTF 3.63
```

**Caution against over-reading the published numbers.** Most of the reported
KleidiAI uplift in ORT's release notes is attributed to SME2 kernels. This
device has **SME1** — there is no `sme2` in `/proc/cpuinfo`. Only the
`i8mm`/dotprod path can dispatch here, and 2.3× is what that is worth on this
part.

**Cost: +90 MB peak PSS** at S = 188 (769 vs 680 MB) — the int8 path prepacks an
extra weight copy. See §5.2.

### 5.2 Memory and utterance length

Long utterances are dramatically cheaper *per second of audio*, because the
reference prefix is a fixed cost paid once per generation:

| S | audio | wall | RTF | peak PSS |
|---:|---:|---:|---:|---:|
| 188 | 1.88 s | — | 11.30 → (see above) | 591 MB → 769 MB |
| 370 | 5.35 s | 19.4 s | **3.63** | — |
| 757 | 19.28 s | 54.1 s | **2.81** | **1.45 GB** |

The last row is the ceiling observation. 1.45 GB is comfortable inside 11.4 GB
and inside a normal Android process because the ORT arenas are native, but it
scales with S and it is now ~90 MB higher than before §5.1. Anything that pushes
S much past 757 should be chunked, and `TextChunker` exists for that.

### 5.3 Everything else measured on device, and it is all noise

| lever | device delta |
|---|---:|
| offline-optimized graph (`optimized_model_filepath`) | **load 1.47×** (1279 → 870 ms) |
| CFG fusion, block-diagonal, one forward of S+T | 1.044× on old path, **0.98× on the new one** |
| thread affinity, 6 workers pinned to cpu0–5 | 1.03× (noise) |
| `dynamic_block_base=4` | 0.97× |
| 7 threads / 8 threads across all cores | 0.83× / 0.72× |
| `graph_optimization_level` below `ALL` | 0.89–0.90× |
| `enable_mem_pattern` off / `cpu_mem_arena` off | 0.93× / 0.91× |
| `allow_spinning=0` | 0.85× |
| `inter_op=2` + `ORT_PARALLEL` | 0.85× |
| `disable_prepacking` | **0.36×** |

The last row is worth keeping: prepacking is worth **2.8×** and is already on by
default. The existing configuration — 6 threads, CPU EP, `ORT_ENABLE_ALL`,
arena and prepacking on, spinning on — is the optimum; no session option beat
it on either machine.

The offline-optimized graph is real and shippable: 410 ms off every session
load. One trap found the hard way — the saved `.opt.onnx` is **693 KB** because
ORT keeps the external-data reference rather than copying the weights, so it
must be written *next to* `omnivoice_lm.onnx.data` or it cannot be reopened.

---

## 6. Experiments that failed, and why

### 6.1 Confidence-aware parallel decoding (Fast-dLLM, arXiv:2505.22618)

The headline acceleration for masked diffusion LMs: instead of un-masking a
fixed `k` cells per step, commit **every** cell whose probability clears a
threshold. Reported up to 27.6× on LLaDA and Dream. It needs no re-export, only
a loop change — so it was the first thing tried.

Implemented behind `--confidence-threshold`. **LM calls never dropped below 32**
at any threshold. `speed_probe.py` shows why — masked cells clearing each
threshold, per step:

```
  step  masked  sched k    p>0.5    p>0.7    p>0.9   p>0.95
     1     384        3       15        1        1        0
     4     374        4       40       13        3        1
     8     353        8       46       16        3        3
    12     309       20       16        6        2        1
    16     145      145       11        2        0        0
```

At no point are there more confident cells than the schedule was already going
to take — by step 8 the schedule takes 8 and **three** cells exceed p > 0.9.

This is a property of the data, not the implementation. A text diffusion LM
choosing among 150k tokens is frequently overwhelmingly sure of the next word;
an RVQ audio cell is one of 1025 near-equivalent codes, and the model's own NLL
here is ~2.75 — perplexity near 15. **A model that is never confident cannot be
accelerated by trusting its confidence.**

The audio field reached the same conclusion independently and earlier: MAGNeT
(ICLR 2024) abandoned per-token masking for **span** masking because "adjacent
audio tokens often share information due to the receptive field of the audio
encoder", and then still needed external model rescoring because its own
confidence ranking was inadequate. Di4C (ICML 2025) gives the theory: masked
diffusion factorises each step into independent per-dimension marginals, valid
only in the many-step limit — and an 8×T RVQ grid is saturated with exactly the
correlations that breaks.

The flag stays in the script because the measurement is the artefact. It is off
by default and there is nothing to gain by turning it on.

### 6.2 Batching the two CFG branches

The intuitive optimisation — pad the unconditional branch to S and run batch 2 —
is **0.64×**. It does `2·S` rows of work instead of `S + T`, because our two
branches are **188 and 48**, not equal.

The correct transform is concatenating them into one sequence of `S+T` with a
block-diagonal attention mask and per-branch `position_ids`. The exported graph
accepts it unchanged and it is **bit-identical** (max |Δ| 0.0, identical codes
over a full 16-step decode, byte-identical WAV). It is worth 1.04× on the old
GEMM path and **0.98×** on the new one: it saves per-call fixed cost and adds
`(S+T)²` attention instead of `S² + T²`. Once `MatMulNBits` got 2.3× faster the
trade flipped.

Implemented as `OnnxModelRunner.forwardFused` / `OmniVoiceEngine.fuseCfg` and
**left off**, with the numbers in the comment. An independent ggml port of
OmniVoice measured the same thing (+29 % compute) from the other direction.

### 6.3 Branch asymmetry also caps the CFG prize

The literature quotes CFG as a 2× tax because cond and uncond branches are the
same length. Ours are not. Under the linear-in-S cost model validated by the
reference-length table in §7.1:

| | relative cost | speedup |
|---|---:|---:|
| CFG on all 16 steps (today) | 1.00 | 1.00× |
| CFG on 20 % of steps | 0.83 | 1.20× |
| CFG removed entirely | 0.80 | **1.26× ceiling** |
| prefix KV cache, CFG unchanged | 0.44 | **2.25×** |

This inverts the natural priority. Guidance distillation is real and published,
but it costs a training run and buys 1.26× here, while the prefix cache buys
more and costs an export.

### 6.4 Sparse and windowed attention

From the device profile, attention is `MatMul` 3.8 % + `FusedMatMul` 2.4 % +
`Softmax` 0.4 % = **6.6 % of a forward**. Deleting attention entirely caps at
1.07×. Published sparse-attention gains for diffusion LMs are measured at 64k
context; our S is a few hundred. Dead end.

### 6.5 A lighter vocoder, or fewer codebooks

`higgs_decoder` is 0.21 s against a 7.4 s decode loop — **2.8 %**. `audio_heads`
is ~2 % of a forward and is not even in the `MatMulNBits` budget. Neither is
worth touching for speed.

The ablation (`codebook_ablation.py`, 12 trials) is still worth having, because
it corrects the app's own copy. Randomising one codebook at a time:

| corrupted | LSD dB | SNR dB |
|---|---:|---:|
| **codebook 0** | **13.70** | −1.13 |
| **codebook 1** | **9.14** | 2.48 |
| codebook 2 | 3.90 | 9.94 |
| codebook 3 | 4.40 | 5.18 |
| codebook 4 | 3.42 | 12.93 |
| codebook 5 | 3.57 | 13.85 |
| codebook 6 | 3.13 | 14.56 |
| codebook 7 | 3.07 | 13.53 |

Damage ordering is `[0, 1, 3, 2, 5, 4, 6, 7]` — **not monotone in the index**.
"Coarse to fine" is directionally right but overstated: it is a **cliff and a
plateau**, not a gradient. Codebooks 0 and 1 carry ~4.5× the damage of any
other; 2–7 are near-interchangeable refinement.

The *temporal* half of the claim is solid and is a property of the sampler, not
the codec: `layer_penalty_factor = 5.0` subtracts 5 per codebook index from the
selection score, so layer 0 un-masks first by construction. On device the ladder
shows 83/56/27/10/3/2/1/0 percent filled at the same step.

### 6.6 QNN / Hexagon

Attempted and blocked at the time by the runtime, not by us: ORT 1.22's bundled
QNN SDK predates this SoC and ships V79 HTP skels against the device's **V81**.
That specific blocker is gone in newer ORT.

The project is unchanged, though. The QNN EP still requires static shapes and a
quantized (uint8/uint16) model with no int4 / `MatMulNBits` path on HTP, and our
own QDQ a16w8 experiment measured **NLL 3.331** — past the noise floor — at
1209 ms against 766 ms on CPU. Static bucketed shapes, a calibrated int8
requantize, and context-binary caching: a project, not a flag.

---

## 7. Open, with prices attached

### 7.1 Approximate prefix KV cache — the largest remaining lever

**State: exported, exact where checkable, and it pays on device — up to 2.0×.**

Measured 2026-09-13 with `DeviceBenchmark#t12_levers -e lever kvcache`: the
shipping plain graph and the level-4 KV graph held open in one process, sharing
one weight blob, round-robin with the order alternated each round, 7 rounds,
medians, 6 threads, thermal 0.

| case | S | prefix | plain gen | cached gen | speedup |
|---|---:|---:|---:|---:|---:|
| short | 188 | 140 (74 %) | 5 888 ms | 3 050 ms | **1.93×** |
| long | 498 | 198 (40 %) | 29 424 ms | 22 550 ms | **1.30×** |

"cached gen" is `prefill + 15 × cached cond + 16 × uncond`, because step 1's
conditional forward *is* the prefill. Per forward, short case: plain cond 295 ms
→ cached cond **99 ms**; prefill 285 ms (≈ one plain cond, as it should be).

The speedup tracks the prefix share, which is the point: the cache removes a
fixed cost, so it is worth most on short sentences — exactly the utterances that
were most expensive per second before.

**Correctness at the state the cache was built from is exact**: max |Δ| 0.0,
argmax 384/384 (short) and 2400/2400 (long), for both the prefill and the cached
forward against the plain graph. That verifies the plumbing — positions, masks,
the sliced K/V. The approximation's own error (a cache built at the all-MASK
state reused as the grid fills) is what the judge measures below.

Running the unconditional branch through the KV session with an empty past costs
80 vs 73 ms (short) and 641 vs 622 ms (long) — about 3 %. So the cache ships as
**one graph, one session**: 1.93× instead of 2.00×, without a second prepacked
copy of the weights in memory.

The desktop could not have answered this. It has AVX2 without VNNI, so its
int8-compute path behaves nothing like the phone's `i8mm`; on the desktop,
level 4 even ran slower than level 1 on the long case.
`models/onnx/int4_kv/omnivoice_lm_kv.onnx` carries `past_key`/`past_value` in and
`present_key`/`present_value` out (28 layers × 8 heads × 128), opset 20, sharing
the same 422 MB weight blob. `scripts/kvcache_probe.py` has the harness
(`forward` / `parity` / `drift` / `wavdelta`).

The quality half is already settled by the judge run that produced §5.1's table.
Refreshing the cache **never** — the most aggressive setting — scores 2.8662
against an uncached baseline of 2.8722, which is inside the seed spread:

```
kvr4      2.8558     refresh every 4 steps
kvnever   2.8662     never refresh
kvoff     2.8722     cache off (baseline)
kvr2      2.9365
kvr8      2.9632
```

Bidirectional attention makes the cached prefix a genuine approximation, and the
approximation costs nothing measurable. So the open question is only whether it
**pays**.

**Two things to get right when measuring it.** The exported KV graph is at
`accuracy_level=1`; the shipping graph is now at 4, so it must be levelled up
first or the comparison is cache-vs-compute-path rather than cache-vs-no-cache.
And the 2.25× figure below comes from a linear-in-S cost model fitted **before**
§5.1. What a cache saves is weight and activation traffic, and the GEMM it
competes against just got 2.3× faster — the same reversal that turned CFG fusion
from 1.04× into 0.98× (§6.2). Expect less than the model says, and measure on
the current shipping configuration, not the old one.



§2 forbids a cache across the generated region. It does not forbid one over the
prefix, which does not change. Trimming the reference prices the prefix
directly:

| reference frames | reference s | S | forward | speedup |
|---:|---:|---:|---:|---:|
| 103 (as enrolled) | 4.12 | 188 | 314 ms | 1.00× |
| 75 | 3.00 | 155 | 274 ms | 1.15× |
| 50 | 2.00 | 125 | 204 ms | 1.54× |
| 25 | 1.00 | 92 | 156 ms | **2.01×** |

Halving the reference nearly halves the work. Caching it rather than truncating
it keeps the voice. Two honest cautions before believing the 2.25× model figure:
the published `dLLM-Cache` numbers show 5.0× FLOPs → 3.17× wall-clock on an
A100, and at M = 48 tokens against int4 weights the cached forward is
compute-bound rather than bandwidth-bound, so it will not be `48/188` of the
cost. Expect 1.5–2×, and measure.

### 7.2 Layer-wise step budget

Our §6.5 ablation and the audio literature agree, and our current schedule does
not implement it. We run a flat cosine top-k over all 384 cells. Published
schedules spend their steps on layer 0:

- **MaskGCT** (ICLR 2025): `[40,16,1,1,…]` over 12 layers, and explicitly that
  `[10,1,1,…]` works "with only a very slight performance loss".
- **SoundStorm**: 16 iterations on RVQ level 0, one greedy pass on each of the
  remaining 11 → 27 passes total.
- **MAGNeT**: "reducing the decoding steps for higher levels does not impact
  quality as much as for the first level."

Modelled: `[6,1,1,1,1,1,1,1]` = 13 steps → 1.23×; collapsing 2–7 into one or two
greedy sweeps → 9–10 steps → **1.6–1.8×**. Pure loop change;
`layer_penalty_factor` and `t_shift` are already knobs.

### 7.3 The canary for §7.2 is WER, not similarity

**DLLM-TTS** is a 0.6B Qwen2-initialised block-diffusion TTS — nearly our
configuration — and its step ablation is the warning:

> WER **2.25 % (T=32) → 14.58 % (T=8)**, while speaker similarity barely moved
> (0.746 → 0.765).

Cutting steps leaves the voice intact and destroys the pronunciation. Our
re-masking NLL judge (§1.2) does **not** see this. Before touching the step
schedule, the project needs a WER gate — which it does not yet have. That is the
next piece of infrastructure, ahead of the optimisation it protects.

### 7.4 Fuse `gate_proj` + `up_proj`

The one upstream `flashinfer` trick ORT has not taken: the profile shows
`gate_proj` 19.4 % and `up_proj` 19.3 % as separate `MatMulNBits`. Concatenating
the two int4 weight matrices into one GEMM plus a `Slice` is an export-time
rewrite worth a few percent of 38.7 %. Small, cheap, unglamorous.

### 7.5 Slice before the audio head

`/audio_heads/MatMul` runs over all 188 positions and **74 % of its output is
discarded** — 8.95 ms of a 392 ms forward, ~6.6 ms wasted, 1.7 %. It needs a
`Slice` before the head in the exported graph, not a flag. This is the last
mathematically-free compute found.

### 7.6 Training-based options, for completeness

Everything that attacks `num_step` itself needs a training run producing a
checkpoint we then re-export — ONNX Runtime is not the blocker, GPUs and data
are. Ranked by cost: **DiDi-Instruct** (one H100-hour), **dParallel**
(certainty-forcing distillation, LoRA on 24 GB, and notably its distillation set
is self-generated from the teacher's own trajectories, so it needs prompts
rather than ground-truth speech), **SDTT** (20+ GPU-hours). dParallel is the
honest answer to §6.1: you cannot decode around a model that is never confident,
you train the confidence in. Whether audio-token entropy is *reducible* — many
codes are perceptually equivalent — is unanswered anywhere in the literature.

---

## 8. State of the community

Checked because the brief says not to reinvent the wheel. Short version:
there is no wheel.

- **No OmniVoice in sherpa-onnx**, and no third-party Android port.
- The public ONNX conversions are worse than ours.
  `onnx-community/OmniVoice-Onnx` uses **block 128** — the setting §3 measures at
  NLL 3.193, past the noise floor — and exports causal attention (§2).
  `OpenVoiceOS/phoonnx-omnivoice` states that a community int4 build failed
  accuracy testing and publishes none.
- Upstream issues are user-side: a ggml port, a missing `torch.inference_mode()`,
  and an **unanswered** request for ONNX-on-Android.
- The OmniVoice paper itself states that no existing approach enables comparable
  inference acceleration for discrete-space NAR TTS, and lists it as future work.
- Upstream's `omnivoice_flashinfer.py` gets 2–2.9×, but against an unoptimised
  PyTorch eager baseline and from CUDA-only tricks (ragged attention, fused
  kernels, CUDA graphs). Its own header says `KV cache: disabled`, confirming §2.
  ORT's graph optimiser has already collected the portable part — fused RMSNorm
  is one `SimplifiedLayerNormalization` node at 1.2 % in our profile. §7.4 is the
  only item on its list we have not taken.

---

## 9. Summary

**Shipped, with the measured factor for each**

| | factor |
|---|---|
| int4 block-32, audio head fp32 | 2.45 GB → 422 MB |
| `num_step` 32 → 16 | 2.2× |
| 6 intra-op threads, not 8 | 1.5× |
| CPU EP over XNNPACK / NNAPI | 1.13× / 1.17× |
| **int8 compute (`accuracy_level=4`) + ORT 1.29** | **2.3×** |
| offline-optimized graph | load 1.47× |
| voice-prompt cache | the ~3 s encode happens once per voice |
| chunked long text | prefix paid once for many sentences |

**Rejected, with the number that killed it**

| | |
|---|---|
| confidence-aware parallel decoding | LM calls unchanged at 32 |
| `guidance_scale = 0` | 1.92 s of silence |
| CFG batch-2 with padding | 0.64× |
| CFG block-diagonal fusion | 0.98× on the current path |
| sparse / windowed attention | attention is 6.6 % of a forward |
| lighter vocoder, fewer codebooks | 2.8 % and ~2 % respectively |
| NNAPI with pinned shapes | 142 nodes, 5.6× slower than CPU |
| int4 in the audio head, or block 128 | NLL 3.274 / 3.193, past the floor |
| every ORT session option tried | ≤ 1.0× |

**Open, priced**

| | expected | effort |
|---|---|---|
| prefix KV cache | 1.5–2× | export (already built, unmeasured) |
| layer-wise step budget | 1.3–1.8× | loop — **needs a WER gate first** |
| CFG on a subset of steps | 1.1–1.2× | loop |
| fuse `gate_proj`+`up_proj`; slice before the head | ~1.05× | export |
| QNN / Hexagon | 2–5× if it lands | a project |
| step distillation | 2–8× | a training run |
