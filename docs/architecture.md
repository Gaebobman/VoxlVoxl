# Target Architecture

## Enrollment — once per speaker

```mermaid
flowchart LR
    WAV["ref.wav"] --> PRE["AudioPreprocessor"]
    PRE -->|"24 kHz f32"| AE["acoustic_encoder.onnx"]
    PRE -->|"16 kHz f32"| SE["semantic_encoder.onnx"]
    AE --> QE["quantizer_encoder.onnx"]
    SE --> QE
    QE -->|"codes (8, T_ref)"| VP["voice_prompt.bin<br/>1 766 bytes"]

    style VP fill:#3a2a10,stroke:#eda13f,color:#f6f1ea
```

The recording itself is never kept. What survives enrollment is 1 766 bytes of
codec codes, which is why a leaked profile is not a leaked recording.

## Generation — once per utterance

```mermaid
flowchart TB
    TXT["target text"] --> TOK["QwenBpeTokenizer"]
    REF["ref text"] --> TOK
    VP["voice_prompt.bin"] --> PROMPT
    TOK --> DUR["DurationEstimator"]
    DUR -->|"T_gen"| PROMPT["prompt build"]

    PROMPT --> LOOP

    subgraph LOOP["OmniVoiceDecoder — 16 un-masking steps"]
        direction TB
        C["conditional forward"] --> CFG["CFG mix<br/>c + w(c − u)"]
        U["unconditional forward"] --> CFG
        CFG --> TOPK["top-k unmask<br/>layer_penalty_factor 5.0"]
        TOPK -->|"next step"| C
        TOPK -->|"next step"| U
    end

    LOOP -.->|"each forward"| LM["omnivoice_lm.onnx<br/>int4, accuracy_level 4"]
    LOOP -->|"codes (8, T_gen)"| DEC["higgs_decoder.onnx"]
    DEC -->|"24 kHz f32"| POST["silence-trim → re-gain → fade/pad"]
    POST --> OUT["AudioTrack / WAV"]

    style LM fill:#2a1a30,stroke:#9b84df,color:#f6f1ea
    style OUT fill:#3a2a10,stroke:#eda13f,color:#f6f1ea
```

Both branches of the CFG mix are full forwards over the whole sequence, and
there are two of them per step, because the backbone's attention is
bidirectional and no KV cache is possible. That loop is the entire cost of the
system — see [`research-notes.md`](research-notes.md) §2.

## Stack

```mermaid
flowchart TB
    UI["MainActivity — 8 screens"] --> SVC["SynthesisService<br/>foreground, cancellable"]
    SVC --> ENG["OmniVoiceEngine"]
    ENG --> RUN["OnnxModelRunner"]
    RUN --> ORT["ONNX Runtime 1.29 — Java API"]
    ORT --> CPU["CPU · MLAS/NEON · 6 threads"]
    ORT -.->|"measured slower"| XNN["XNNPACK"]
    ORT -.->|"measured slower"| NNAPI["NNAPI"]
    ORT -.->|"not yet"| QNN["QNN / Hexagon"]

    style CPU fill:#1a3020,stroke:#8fd6a0,color:#f6f1ea
```

No Python on device. No Termux, Chaquopy, embedded CPython or PyTorch-Android.

## Sessions and their lifetimes

| session | file | precision | loaded when | freed when |
|---|---|---|---|---|
| `omnivoice_lm` | `omnivoice_lm.onnx(+.data)` | int4, int8 compute | first generate | app background (configurable) |
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
    omnivoice_lm.onnx             graph only, 1.4 MB
    omnivoice_lm.onnx.data        weights, 421 MB
    higgs_decoder.onnx
    acoustic_encoder.onnx         (optional — enrollment build only)
    semantic_encoder.onnx         (optional)
    quantizer_encoder.onnx        (optional)
    tokenizer.json
    manifest.json                 sha256 + sizes, checked on first launch
  voices/
    <id>.bin                      the codec codes
    <id>.name                     the display name, so a rename never rewrites codes
```

`.onnx.data` must sit next to its `.onnx` — ORT resolves the external-data path relative
to the model file, which is why models live in `filesDir`, not in `assets`.

## Threading

- One `OrtEnvironment` per process.
- `intra_op_num_threads` = **6**. Measured: 8 is *worse* than 6, because threads 7
  and 8 land on the two prime cores and the step then waits on migration.
- Generation runs on `Dispatchers.Default` inside a cancellable coroutine; every step
  checks `ensureActive()` so Cancel is responsive.
- Input tensors for the LM are allocated once per utterance and rewritten in place across
  steps (`OnnxTensor.createTensor` on a reused `LongBuffer`/`ByteBuffer`) — `S` is constant
  within one utterance, which is the whole reason the no-KV-cache design is tolerable here.
