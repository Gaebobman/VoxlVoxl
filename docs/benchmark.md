# Benchmark

All numbers to be filled from real runs. Empty cells mean "not measured yet" — never
estimated. Projections live in `docs/model-analysis.md` §6 and are labelled as such.

## Test condition

```
device        : Samsung Galaxy S26+ / Android 16 / Exynos 2600 / 12 GB
reference     : sample/reference.wav   (   s,     kHz)
reference text: sample/reference.txt
target text   : sample/target.txt
generated     :    frames →    s @ 24 kHz
RTF           = inference_time / generated_audio_duration
```

## PC reference (Ryzen 7 5700X / RTX 5080)

| runtime | precision | num_step | CFG | load (s) | infer (s) | RTF | peak RSS |
|---|---|---:|---|---:|---:|---:|---:|
| PyTorch CUDA | fp16 | 32 | on | | | | |
| PyTorch CPU | fp32 | 32 | on | | | | |
| ORT CPU | fp32 | 32 | on | | | | |
| ORT CPU | fp16 | 32 | on | | | | |
| ORT CPU | int4 | 32 | on | | | | |
| ORT CPU | int4 | 16 | on | | | | |
| ORT CPU | int4 | 8 | on | | | | |
| ORT CPU | int4 | 32 | off | | | | |

## Galaxy S26+

| Backend | Precision | num_step | CFG | Audio | Load | Latency | RTF | Peak RAM |
|---|---|---:|---|---:|---:|---:|---:|---:|
| CPU | int4 | 32 | on | 10 s | | | | |
| CPU | int4 | 16 | on | 10 s | | | | |
| CPU | int4 | 8 | on | 10 s | | | | |
| XNNPACK | int4 | 32 | on | 10 s | | | | |
| NNAPI | int4 | 32 | on | 10 s | | | | |

### Per-stage breakdown (best configuration)

| stage | time (ms) | share |
|---|---:|---:|
| tokenize | | |
| voice prompt load | | |
| LM — 32 steps × 2 branches | | |
| higgs_decoder | | |
| post-process | | |

### Thread sweep (CPU, int4)

| intra_op_num_threads | latency (s) | RTF |
|---:|---:|---:|
| 1 | | |
| 2 | | |
| 4 | | |
| 6 | | |
| 8 | | |

### Thermal (5 consecutive generations)

| run | latency (s) | RTF | thermal status |
|---:|---:|---:|---|
| 1 | | | |
| 5 | | | |

## NNAPI partitioning

| metric | value |
|---|---|
| nodes assigned to NNAPI | |
| nodes on CPU fallback | |
| number of partitions | |
| unsupported ops (top offenders) | |

## Quality

| config | speaker similarity (SIM-o) | WER/CER vs target | subjective |
|---|---:|---:|---|
| PyTorch fp16, 32 steps (golden) | — | | reference |
| ORT int4, 32 steps | | | |
| ORT int4, 16 steps | | | |
| ORT int4, 8 steps | | | |
| ORT int4, 32 steps, CFG off | | | |

---

## Conclusion — the four questions from the brief

1. **Did OmniVoice inference actually run on the Galaxy S26+?** _(pending)_
2. **Is it fully offline?** _(pending — airplane mode + no `INTERNET` permission)_
3. **Which backend is fastest?** _(pending)_
4. **Is it fast enough for real use?** _(pending — see §RTF; the projection in
   `docs/model-analysis.md` §6 says no at 32 steps, and that projection is itself a
   finding to be confirmed or refuted, not hidden)_
