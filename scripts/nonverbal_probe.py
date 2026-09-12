#!/usr/bin/env python3
"""F-D01 — do the non-verbal tags actually produce sound, or only tokenize?

The tokenizer treats `[laughter]`, `[sigh]` and 11 others as standalone tokens
(pinned by android/fixtures/tokenizer.json). That proves the ids are right; it
proves nothing about the acoustics. This measures the acoustics.

Method: synthesise the same carrier sentence with and without a tag appended, and
compare. A tag that does something should lengthen the audio (it adds an
utterance) and change the voiced fraction and spectral balance; a tag that is
ignored will land inside the run-to-run spread, so a no-tag control is generated
too and its spread sets the threshold.

  python scripts/nonverbal_probe.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, ROOT, read_wav  # noqa: E402
from style_probe import f0_stats, spectral  # noqa: E402

TAGS = ["laughter", "sigh", "surprise-ah", "surprise-oh",
        "question-en", "dissatisfaction-hnn", "confirmation-en"]
CARRIER = "네 알겠습니다"


def gen(text: str, out: Path, args) -> bool:
    cmd = [str(ROOT / ".venv" / "bin" / "python"),
           str(ROOT / "scripts" / "infer_onnx.py"), "generate",
           "--voice-prompt", args.voice_prompt, "--text", text,
           "--lm", args.lm, "--num-step", str(args.num_step),
           "--seed", str(args.seed), "--out", str(out)]
    if args.deterministic:
        cmd.append("--deterministic")
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def measure(p: Path) -> dict:
    x, sr = read_wav(p)
    f0, voiced = f0_stats(x, sr)
    hf, zcr = spectral(x, sr)
    return {"dur": len(x) / sr, "f0": f0, "voiced": voiced, "hf": hf, "zcr": zcr,
            "rms": float(np.sqrt((x ** 2).mean()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", default="models/onnx/int4/omnivoice_lm.onnx")
    ap.add_argument("--voice-prompt", default="out/golden_det/voice_prompt.bin")
    ap.add_argument("--num-step", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--controls", type=int, default=4,
                    help="no-tag runs with different seeds, to size the noise floor")
    ap.add_argument("--outdir", default=str(OUT / "nonverbal"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- controls: same text, different seeds -> the natural spread ---
    ctrl = []
    for i in range(args.controls):
        p = outdir / f"control_{i}.wav"
        a = argparse.Namespace(**{**vars(args), "seed": args.seed + i})
        if not p.exists():
            gen(CARRIER + ".", p, a)
        if p.exists():
            ctrl.append(measure(p))
    if not ctrl:
        sys.exit("no control runs succeeded")

    def spread(key: str) -> tuple[float, float]:
        v = [c[key] for c in ctrl if not np.isnan(c[key])]
        return float(np.mean(v)), float(np.std(v))

    print(f"\n  control ({len(ctrl)} seeds, text={CARRIER!r}):")
    for k in ("dur", "voiced", "hf", "zcr"):
        m, s = spread(k)
        print(f"    {k:8s} mean {m:8.4f}  sd {s:8.4f}")

    print(f"\n  {'tag':22s} {'dur s':>7} {'Δdur':>8} {'voiced%':>8} {'Δvc(sd)':>9} "
          f"{'HF%':>7} {'ΔHF(sd)':>9}  verdict")
    print("  " + "-" * 92)
    for tag in TAGS:
        p = outdir / f"{tag}.wav"
        if not p.exists():
            gen(f"{CARRIER} [{tag}]", p, args)
        if not p.exists():
            print(f"  {tag:22s} FAILED to generate")
            continue
        m = measure(p)
        out = []
        for key, label in (("dur", "dur"), ("voiced", "vc"), ("hf", "hf")):
            mu, sd = spread(key)
            out.append((m[key] - mu) / sd if sd > 1e-9 else 0.0)
        z_dur, z_vc, z_hf = out
        moved = max(abs(z_dur), abs(z_vc), abs(z_hf))
        verdict = "AUDIBLE EFFECT" if moved > 3 else ("marginal" if moved > 2 else "no effect")
        print(f"  [{tag}]{'':<{max(0, 20 - len(tag))}} {m['dur']:7.2f} "
              f"{m['dur'] - spread('dur')[0]:+8.2f} {m['voiced'] * 100:8.1f} "
              f"{z_vc:+8.1f}σ {m['hf'] * 100:6.2f}% {z_hf:+8.1f}σ  {verdict}")


if __name__ == "__main__":
    main()
