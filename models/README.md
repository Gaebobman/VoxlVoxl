# models/

Model files are **never committed**. `python scripts/download_models.py` populates this
directory; `scripts/export_onnx.py` writes the ONNX artifacts.

```
models/
  omnivoice/                     ← k2-fsa/OmniVoice   (Apache-2.0, 2.45 GB + 806 MB)
    config.json  tokenizer.json  model.safetensors
    audio_tokenizer/config.json  audio_tokenizer/model.safetensors  ...
  higgs-onnx/                    ← onnx-community/OmniVoice-Onnx/audio_tokenizer (Apache-2.0)
    acoustic_encoder.onnx  semantic_encoder.onnx
    quantizer_encoder.onnx higgs_decoder.onnx
  onnx/
    fp32/  fp16/  int4/          ← our exports: omnivoice_lm.onnx (+ .onnx.data)
  android/                       ← exactly what gets pushed to the phone + manifest.json
```

## Licences

| artifact | licence |
|---|---|
| `k2-fsa/OmniVoice` (code + weights) | Apache-2.0 |
| Higgs Audio V2 Tokenizer weights | **Boson Higgs Audio 2 Community License** (Llama-3-derived) — attribution required, >100k AAU needs Boson's consent. See `docs/onnx-reuse-audit.md` §6.2 |
| `onnx-community/OmniVoice-Onnx` | Apache-2.0 |
| ONNX Runtime | MIT |
