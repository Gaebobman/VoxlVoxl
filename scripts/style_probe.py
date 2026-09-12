#!/usr/bin/env python3
"""F-C02 / F-C04 — which voice-design styles actually control the output, and do
they survive being combined with a voice clone?

The feature spec says to expose only styles the model controls reliably, and not
to assume clone + design compose. Both are empirical questions, so they get
measured rather than assumed. Judged on objective acoustics, not on listening:

  F0        median pitch over voiced frames (autocorrelation)
  HF ratio  energy above 4 kHz / total — whisper is breathy, so this rises
  ZCR       zero-crossing rate — rises with noise-like (unvoiced) excitation
  voiced%   fraction of frames with a detectable period — whisper drops this
  RMS       loudness after the pipeline's own re-gain

  python scripts/style_probe.py --lm models/onnx/int4/omnivoice_lm.onnx
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, ROOT, SR_24K, read_wav  # noqa: E402

STYLES = [
    ("baseline", None),
    ("whisper", "whisper"),
    ("low_pitch", "low pitch"),
    ("very_low_pitch", "very low pitch"),
    ("high_pitch", "high pitch"),
    ("very_high_pitch", "very high pitch"),
    ("elderly", "elderly"),
    ("child", "child"),
    ("female", "female"),
]


def f0_stats(x: np.ndarray, sr: int = SR_24K, frame: int = 1024, hop: int = 256):
    """Autocorrelation F0 over 60-400 Hz, plus the voiced fraction."""
    lo, hi = int(sr / 400), int(sr / 60)
    f0s = []
    voiced = 0
    total = 0
    for i in range(0, max(0, len(x) - frame), hop):
        seg = x[i:i + frame].astype(np.float64)
        if np.sqrt((seg ** 2).mean()) < 5e-3:
            continue
        total += 1
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        window = ac[lo:hi]
        if window.size == 0:
            continue
        k = int(np.argmax(window)) + lo
        if ac[k] > 0.35:          # periodic enough to call voiced
            voiced += 1
            f0s.append(sr / k)
    return (float(np.median(f0s)) if f0s else float("nan"),
            voiced / total if total else 0.0)


def spectral(x: np.ndarray, sr: int = SR_24K):
    n = (len(x) // 1024) * 1024
    if n == 0:
        return float("nan"), float("nan")
    frames = x[:n].reshape(-1, 1024)
    mag = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(1024, 1 / sr)
    total = mag.sum()
    hf = mag[:, freqs > 4000].sum() / max(total, 1e-12)
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    return float(hf), zcr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", default="models/onnx/int4/omnivoice_lm.onnx")
    ap.add_argument("--voice-prompt", default="out/golden_det/voice_prompt.bin")
    ap.add_argument("--text", default="조용히 말씀드릴게요. 지금 바로 확인해 주세요.")
    ap.add_argument("--num-step", type=int, default=16)
    ap.add_argument("--outdir", default=str(OUT / "style"))
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--no-voice-prompt", action="store_true",
                    help="voice-design mode: the positive control for this probe")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    py = str(ROOT / ".venv" / "bin" / "python")

    for name, instruct in STYLES:
        wav = outdir / f"{name}.wav"
        if wav.exists() and args.skip_generate:
            continue
        cmd = [py, str(ROOT / "scripts" / "infer_onnx.py"), "generate",
               "--text", args.text, "--lm", args.lm,
               "--num-step", str(args.num_step),
               "--deterministic", "--out", str(wav)]
        if not args.no_voice_prompt:
            cmd += ["--voice-prompt", args.voice_prompt]
        if instruct:
            cmd += ["--instruct", instruct]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {name:16s} FAILED: {r.stdout.strip().splitlines()[-1:] or r.stderr[-200:]}")

    print(f"\n  {'style':16s} {'instruct':18s} {'dur s':>6} {'F0 Hz':>7} "
          f"{'voiced%':>8} {'HF>4k':>7} {'ZCR':>6} {'RMS':>7}")
    print("  " + "-" * 82)
    base = None
    for name, instruct in STYLES:
        wav = outdir / f"{name}.wav"
        if not wav.exists():
            print(f"  {name:16s} (not generated)")
            continue
        x, sr = read_wav(wav)
        f0, voiced = f0_stats(x, sr)
        hf, zcr = spectral(x, sr)
        rms = float(np.sqrt((x ** 2).mean()))
        row = (f0, voiced, hf, zcr, rms)
        base = base or row
        d = "" if name == "baseline" else \
            f"  ΔF0 {(f0 - base[0]):+6.1f}Hz  ΔHF {(hf - base[2]) * 100:+5.1f}pp"
        print(f"  {name:16s} {str(instruct or '-'):18s} {len(x) / sr:6.2f} "
              f"{f0:7.1f} {voiced * 100:8.1f} {hf * 100:6.2f}% {zcr:6.3f} {rms:7.4f}{d}")


if __name__ == "__main__":
    main()
