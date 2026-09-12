#!/usr/bin/env python3
"""Assemble models/android/ — exactly the bytes that go on the phone.

  python scripts/prepare_android_models.py                # generate-only bundle
  python scripts/prepare_android_models.py --with-encoders # + enrollment on device

Then:
  adb push models/android/. /sdcard/Android/data/<pkg>/files/models/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AUDIO_MASK_ID, AUDIO_VOCAB_SIZE, FRAME_RATE, HIGGS_ONNX_DIR, HIDDEN_SIZE,
    MODELS, NUM_CODEBOOKS, SR_16K, SR_24K, UPSTREAM_DIR,
)

DEST = MODELS / "android"

# generation path — always shipped
CORE = [
    (MODELS / "onnx" / "int4" / "omnivoice_lm.onnx", "omnivoice_lm.onnx"),
    (MODELS / "onnx" / "int4" / "omnivoice_lm.onnx.data", "omnivoice_lm.onnx.data"),
    (HIGGS_ONNX_DIR / "higgs_decoder.onnx", "higgs_decoder.onnx"),
    (UPSTREAM_DIR / "tokenizer.json", "tokenizer.json"),
]

# enrollment path — optional, 654 MB, only needed to create a voice_prompt.bin
ENCODERS = [
    (HIGGS_ONNX_DIR / "acoustic_encoder.onnx", "acoustic_encoder.onnx"),
    (HIGGS_ONNX_DIR / "semantic_encoder.onnx", "semantic_encoder.onnx"),
    (HIGGS_ONNX_DIR / "quantizer_encoder.onnx", "quantizer_encoder.onnx"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-encoders", action="store_true",
                    help="also ship the 654 MB enrollment graphs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dest = Path(args.out or DEST)
    dest.mkdir(parents=True, exist_ok=True)

    items = CORE + (ENCODERS if args.with_encoders else [])
    missing = [str(s) for s, _ in items if not s.exists()]
    if missing:
        sys.exit("missing inputs:\n  " + "\n  ".join(missing)
                 + "\n\nrun download_models.py, export_onnx.py and sweep_quant.py first.")

    files, total = {}, 0
    for src, name in items:
        dst = dest / name
        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            shutil.copy2(src, dst)
        size = dst.stat().st_size
        total += size
        files[name] = {"bytes": size, "sha256": sha256(dst)}
        print(f"  {size / 1e6:9.1f} MB  {name}")

    variant = (MODELS / "onnx" / "int4" / "quant_variant.txt")
    manifest = {
        "schema": 1,
        "model": "k2-fsa/OmniVoice",
        "quant_variant": variant.read_text().strip() if variant.exists() else "unknown",
        "includes_encoders": args.with_encoders,
        "constants": {
            "num_codebooks": NUM_CODEBOOKS,
            "audio_vocab_size": AUDIO_VOCAB_SIZE,
            "audio_mask_id": AUDIO_MASK_ID,
            "hidden_size": HIDDEN_SIZE,
            "frame_rate": FRAME_RATE,
            "sample_rate": SR_24K,
            "semantic_sample_rate": SR_16K,
            "codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
            "special_tokens": {
                "denoise": 151669, "lang_start": 151670, "lang_end": 151671,
                "instruct_start": 151672, "instruct_end": 151673,
                "text_start": 151674, "text_end": 151675,
            },
        },
        "defaults": {
            "num_step": 16,
            "guidance_scale": 2.0,
            "t_shift": 0.1,
            "layer_penalty_factor": 5.0,
            "position_temperature": 5.0,
            "class_temperature": 0.0,
            "_note": "guidance_scale must not be 0 — it produces silence, see docs/benchmark.md §2",
        },
        "graphs": {
            "omnivoice_lm.onnx": {
                "inputs": {
                    "input_ids": "int64 [batch, 8, seq]",
                    "audio_mask": "bool [batch, seq]",
                    "attention_mask": "bool [batch, 1, seq, seq] (True = attend)",
                    "position_ids": "int64 [batch, seq]",
                },
                "outputs": {"logits": "float32 [batch, 8, seq, 1025]"},
            },
            "higgs_decoder.onnx": {
                "inputs": {"codes": "int64 [8, batch, frames]"},
                "outputs": {"waveform_24k": "float32 [batch, 1, frames*960]"},
            },
        },
        "files": files,
        "total_bytes": total,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  {'-' * 9}")
    print(f"  {total / 1e6:9.1f} MB  total  →  {dest}")
    print(f"\n  adb push {dest}/. /sdcard/Android/data/<pkg>/files/models/")


if __name__ == "__main__":
    main()
