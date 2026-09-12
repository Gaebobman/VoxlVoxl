#!/usr/bin/env python3
"""Is the Higgs codec actually coarse-to-fine across its eight codebooks?

The UI tells the user "the outline of the voice is formed first, the detail fills
in later". Half of that is observed (lower codebooks resolve first — the
un-masking loop subtracts layer_penalty_factor per index). The other half — that
a LOW codebook carries coarse structure and a HIGH one carries detail — is a
claim about the quantiser, and it deserves a measurement rather than an appeal to
"RVQ is residual".

Method: corrupt one codebook at a time and decode. If the stack is residual and
ordered, corrupting codebook 0 should wreck the audio and corrupting codebook 7
should barely register.

  python scripts/codebook_ablation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HIGGS_ONNX_DIR, NUM_CODEBOOKS, OUT, SR_24K, write_wav  # noqa: E402


def decoder(threads: int = 0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(HIGGS_ONNX_DIR / "higgs_decoder.onnx"),
                                sess_options=so, providers=["CPUExecutionProvider"])


def decode(sess, codes: np.ndarray) -> np.ndarray:
    return sess.run(["waveform_24k"], {"codes": codes[:, None, :].astype(np.int64)})[0].squeeze()


def logmel_like(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """A plain log-magnitude spectrogram; enough to compare timbre damage."""
    n = max(0, (len(x) - n_fft) // hop + 1)
    if n <= 0:
        return np.zeros((1, n_fft // 2 + 1))
    win = np.hanning(n_fft)
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n)])
    return np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-6)


def damage(ref: np.ndarray, got: np.ndarray) -> tuple[float, float]:
    """Log-spectral distance in dB, and broadband SNR in dB."""
    m = min(len(ref), len(got))
    a, b = ref[:m], got[:m]
    A, B = logmel_like(a), logmel_like(b)
    k = min(len(A), len(B))
    lsd = float(np.sqrt(((A[:k] - B[:k]) * 20) ** 2).mean())
    noise = a - b
    snr = 10 * np.log10(float((a ** 2).sum()) / max(float((noise ** 2).sum()), 1e-12))
    return lsd, snr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", default=str(OUT / "golden_det" / "codes.npy"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--save", action="store_true", help="write a wav per ablation")
    args = ap.parse_args()

    codes = np.load(args.codes)
    if codes.shape[0] != NUM_CODEBOOKS:
        sys.exit(f"expected {NUM_CODEBOOKS} codebooks, got {codes.shape}")
    sess = decoder()
    ref = decode(sess, codes)
    print(f"reference: {codes.shape[1]} frames, {len(ref) / SR_24K:.2f}s\n")

    rng = np.random.default_rng(args.seed)
    print(f"  {'corrupted':>12}  {'LSD dB':>8} {'SNR dB':>8}   {'damage':<28}")
    print("  " + "-" * 64)
    rows = []
    for k in range(NUM_CODEBOOKS):
        lsds, snrs = [], []
        for _ in range(args.trials):
            bad = codes.copy()
            # replace this codebook with uniform random codes — the strongest
            # possible corruption of exactly one layer, nothing else touched
            bad[k] = rng.integers(0, 1024, size=codes.shape[1])
            got = decode(sess, bad)
            l, s = damage(ref, got)
            lsds.append(l); snrs.append(s)
        lsd, snr = float(np.mean(lsds)), float(np.mean(snrs))
        rows.append((k, lsd, snr))
        bar = "#" * int(round(lsd / max(1e-9, 1) / 4))
        print(f"  {('codebook ' + str(k)):>12}  {lsd:8.2f} {snr:8.2f}   {bar[:28]}")
        if args.save:
            bad = codes.copy()
            bad[k] = rng.integers(0, 1024, size=codes.shape[1])
            write_wav(OUT / "ablation" / f"corrupt_cb{k}.wav", decode(sess, bad), SR_24K)

    print()
    first, last = rows[0], rows[-1]
    print(f"  codebook 0 vs codebook 7:  LSD {first[1]:.2f} vs {last[1]:.2f} dB "
          f"({first[1] / max(last[1], 1e-9):.2f}x)   SNR {first[2]:.2f} vs {last[2]:.2f} dB")
    order = [k for k, _, _ in sorted(rows, key=lambda r: -r[1])]
    print(f"  most damaging first: {order}")
    mono = all(rows[i][1] >= rows[i + 1][1] for i in range(len(rows) - 1))
    print(f"  strictly decreasing with index: {mono}")
    # The cumulative view is the one that actually tests "coarse to fine":
    # keep only codebooks 0..k-1 and corrupt the rest. If the stack is residual
    # and ordered, quality should climb steeply and then saturate.
    print(f"\n  keeping only the first k codebooks (rest randomised):")
    print(f"  {'kept':>6}  {'LSD dB':>8} {'SNR dB':>8}")
    print("  " + "-" * 28)
    for k in range(1, NUM_CODEBOOKS + 1):
        lsds, snrs = [], []
        for _ in range(args.trials):
            bad = codes.copy()
            if k < NUM_CODEBOOKS:
                bad[k:] = rng.integers(0, 1024, size=(NUM_CODEBOOKS - k, codes.shape[1]))
            got = decode(sess, bad)
            l, sn = damage(ref, got)
            lsds.append(l); snrs.append(sn)
        print(f"  {('0..' + str(k - 1)):>6}  {np.mean(lsds):8.2f} {np.mean(snrs):8.2f}")
        if args.save:
            bad = codes.copy()
            if k < NUM_CODEBOOKS:
                bad[k:] = rng.integers(0, 1024, size=(NUM_CODEBOOKS - k, codes.shape[1]))
            write_wav(OUT / "ablation" / f"keep_first_{k}.wav", decode(sess, bad), SR_24K)

    if args.save:
        write_wav(OUT / "ablation" / "reference.wav", ref, SR_24K)
        print(f"\n  wavs → {OUT / 'ablation'}")


if __name__ == "__main__":
    main()
