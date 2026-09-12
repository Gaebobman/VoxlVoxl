# Target Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ Android App (Kotlin, arm64-v8a, minSdk 31)                           │
│                                                                      │
│  MainActivity ──► OmniVoiceEngine                                    │
│                      │                                               │
│   ┌──────────────────┼────────────────────────────────────────────┐  │
│   │ enrollment (once per speaker)                                 │  │
│   │  ref.wav ─► AudioPreprocessor ─► 24 kHz f32 ─► acoustic_enc ─┐ │  │
│   │                              └► 16 kHz f32 ─► semantic_enc ─┤ │  │
│   │                                        quantizer_enc ◄──────┘ │  │
│   │                                             │ codes (8,T_ref) │  │
│   │                                             ▼                 │  │
│   │                              VoicePromptStore → voice_prompt.bin │
│   └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐  │
│   │ generation (per utterance)                                    │  │
│   │  target text ─► QwenBpeTokenizer ─┐                           │  │
│   │  ref text    ─────────────────────┤                           │  │
│   │  voice_prompt.bin ────────────────┤                           │  │
│   │                                   ▼                           │  │
│   │                          DurationEstimator → T_gen            │  │
│   │                                   ▼                           │  │
│   │                          OmniVoiceDecoder                     │  │
│   │                   ┌───────────────┴───────────────┐           │  │
│   │                   │  32 × { cond fwd, uncond fwd, │           │  │
│   │                   │        CFG, top-k unmask }    │           │  │
│   │                   └───────────────┬───────────────┘           │  │
│   │                        omnivoice_lm.onnx (int4)               │  │
│   │                                   ▼ codes (8,T_gen)           │  │
│   │                          higgs_decoder.onnx                   │  │
│   │                                   ▼ 24 kHz f32                │  │
│   │                   post: silence-trim → re-gain → fade/pad     │  │
│   │                                   ▼                           │  │
│   │                       AudioOutput: AudioTrack / WAV           │  │
│   └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  OnnxModelRunner ──► ONNX Runtime 1.22 Java API                      │
└───────────────────────────┬──────────────────────────────────────────┘
                            ▼
            CPU (MLAS/NEON) → XNNPACK → NNAPI (investigated)
```

## Sessions and their lifetimes

| session | file | precision | loaded when | freed when |
|---|---|---|---|---|
| `omnivoice_lm` | `omnivoice_lm.onnx(+.data)` | int4 | first generate | app background (configurable) |
| `higgs_decoder` | `higgs_decoder.onnx` | fp32 | first generate | with the LM |
| `acoustic_encoder` | `acoustic_encoder.onnx` | fp32 | enrollment only | immediately after enrollment |
| `semantic_encoder` | `semantic_encoder.onnx` | fp32 | enrollment only | immediately after enrollment |
| `quantizer_encoder` | `quantizer_encoder.onnx` | fp32 | enrollment only | immediately after enrollment |

Enrollment sessions are opened, used and closed inside one `use {}` block so their
~650 MB never coexists with the generation sessions.

## Files on device

```
filesDir/
  models/
    omnivoice_lm.onnx
    omnivoice_lm.onnx.data
    higgs_decoder.onnx
    acoustic_encoder.onnx        (optional — enrollment build only)
    semantic_encoder.onnx        (optional)
    quantizer_encoder.onnx       (optional)
    tokenizer.json
    manifest.json                 sha256 + sizes, checked on first launch
  voices/
    <name>.bin                    voice_prompt.bin, see docs/plan.md §3
```

`.onnx.data` must sit next to its `.onnx` — ORT resolves the external-data path relative
to the model file, which is why models live in `filesDir`, not in `assets`.

## Threading

- One `OrtEnvironment` per process.
- `intra_op_num_threads` pinned to the number of big cores (swept in Phase 5).
- Generation runs on `Dispatchers.Default` inside a cancellable coroutine; each of the 32
  steps checks `ensureActive()` so Cancel is responsive.
- Input tensors for the LM are allocated once per utterance and rewritten in place across
  steps (`OnnxTensor.createTensor` on a reused `LongBuffer`/`ByteBuffer`) — `S` is constant
  within one utterance, which is the whole reason the no-KV-cache design is tolerable here.
