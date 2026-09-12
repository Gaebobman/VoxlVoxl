#!/usr/bin/env python3
"""Fetch every model artifact the PoC needs into ``models/``.

Nothing here runs on the device — this is the PC-side preparation step.

  python scripts/download_models.py                 # everything
  python scripts/download_models.py --only upstream # PyTorch source only
  python scripts/download_models.py --only higgs    # reusable codec ONNX only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# k2-fsa/OmniVoice — Apache-2.0. The PyTorch source of truth: we export from this and
# validate against it.
UPSTREAM_REPO = "k2-fsa/OmniVoice"
UPSTREAM_DEST = MODELS / "omnivoice"

# onnx-community/OmniVoice-Onnx — Apache-2.0. We reuse ONLY audio_tokenizer/; the
# backbone graphs in that repo are causal and unusable (docs/onnx-reuse-audit.md §1).
HIGGS_REPO = "onnx-community/OmniVoice-Onnx"
HIGGS_DEST = MODELS / "higgs-onnx"
HIGGS_FILES_FP32 = [
    "audio_tokenizer/acoustic_encoder.onnx",
    "audio_tokenizer/semantic_encoder.onnx",
    "audio_tokenizer/quantizer_encoder.onnx",
    "audio_tokenizer/higgs_decoder.onnx",
    "audio_tokenizer/model_config.json",
]
HIGGS_FILES_FP16 = [
    "audio_tokenizer/fp16/acoustic_encoder.onnx",
    "audio_tokenizer/fp16/acoustic_encoder.onnx.data",
    "audio_tokenizer/fp16/semantic_encoder.onnx",
    "audio_tokenizer/fp16/semantic_encoder.onnx.data",
    "audio_tokenizer/fp16/quantizer_encoder.onnx",
    "audio_tokenizer/fp16/higgs_decoder.onnx",
    "audio_tokenizer/fp16/higgs_decoder.onnx.data",
    "audio_tokenizer/fp16/model_config.json",
]


def _hub():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        sys.exit(
            "huggingface_hub is missing.\n"
            "  pip install -r scripts/requirements.txt"
        )
    from huggingface_hub import snapshot_download

    return snapshot_download


def fetch_upstream(force: bool) -> None:
    snapshot_download = _hub()
    print(f"[1/2] {UPSTREAM_REPO}  →  {UPSTREAM_DEST}   (~3.3 GB)")
    snapshot_download(
        repo_id=UPSTREAM_REPO,
        local_dir=str(UPSTREAM_DEST),
        force_download=force,
        max_workers=4,
    )


def fetch_higgs(force: bool, precision: str) -> None:
    snapshot_download = _hub()
    files = HIGGS_FILES_FP32 if precision in ("fp32", "both") else []
    if precision in ("fp16", "both"):
        files += HIGGS_FILES_FP16
    print(f"[2/2] {HIGGS_REPO}:audio_tokenizer/ ({precision})  →  {HIGGS_DEST}")
    snapshot_download(
        repo_id=HIGGS_REPO,
        local_dir=str(HIGGS_DEST),
        allow_patterns=files,
        force_download=force,
        max_workers=4,
    )


def report() -> None:
    print("\n--- downloaded ---")
    total = 0
    for base in (UPSTREAM_DEST, HIGGS_DEST):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and ".cache" not in p.parts:
                size = p.stat().st_size
                total += size
                if size > 1_000_000:
                    print(f"  {size / 1e6:9.1f} MB  {p.relative_to(MODELS)}")
    print(f"  {'-' * 9}")
    print(f"  {total / 1e6:9.1f} MB  total")

    cfg = UPSTREAM_DEST / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        llm = c.get("llm_config", {})
        print("\n--- sanity check (expected values from docs/model-analysis.md §1) ---")
        checks = [
            ("architectures", c.get("architectures"), ["OmniVoice"]),
            ("audio_vocab_size", c.get("audio_vocab_size"), 1025),
            ("audio_mask_id", c.get("audio_mask_id"), 1024),
            ("num_audio_codebook", c.get("num_audio_codebook"), 8),
            ("audio_codebook_weights", c.get("audio_codebook_weights"), [8, 8, 6, 6, 4, 4, 2, 2]),
            ("llm.hidden_size", llm.get("hidden_size"), 1024),
            ("llm.num_hidden_layers", llm.get("num_hidden_layers"), 28),
            ("llm.num_attention_heads", llm.get("num_attention_heads"), 16),
            ("llm.num_key_value_heads", llm.get("num_key_value_heads"), 8),
            ("llm.vocab_size", llm.get("vocab_size"), 151676),
        ]
        ok = True
        for name, got, want in checks:
            good = got == want
            ok &= good
            print(f"  {'OK ' if good else 'BAD'}  {name:26s} {got!r}"
                  + ("" if good else f"   expected {want!r}"))
        if not ok:
            print("\n  Upstream config drifted from the analysis. Re-read "
                  "docs/model-analysis.md before exporting.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["upstream", "higgs"], default=None)
    ap.add_argument("--higgs-precision", choices=["fp32", "fp16", "both"], default="fp32",
                    help="fp32 is the validation target; fp16 halves on-device size")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "upstream"):
        fetch_upstream(args.force)
    if args.only in (None, "higgs"):
        fetch_higgs(args.force, args.higgs_precision)
    report()


if __name__ == "__main__":
    main()
