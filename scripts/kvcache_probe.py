#!/usr/bin/env python3
"""Approximate prefix KV cache — does it pay, and what does it cost?

OmniVoice's attention is bidirectional (docs/model-analysis.md §1.1), so the
reference prefix's K/V really do depend on the generated region and caching them
is an APPROXIMATION (Fast-dLLM, arXiv:2505.22618 §3.2). But the prefix is 74 %
of the sequence at S=188 and is re-encoded on all 32 forwards, so the prize is
large enough to be worth measuring rather than reasoning about.

Needs `export_onnx.py --kv-cache` (and its int4 quantization).

  python scripts/kvcache_probe.py forward      # per-forward timing, cache vs not
  python scripts/kvcache_probe.py parity       # cached graph, P=0, vs the plain graph
  python scripts/kvcache_probe.py drift        # logit error vs how full the gen region is
  python scripts/kvcache_probe.py wavdelta --a A.codes.npy --b B.codes.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AUDIO_MASK_ID, MODELS, NUM_CODEBOOKS, OUT, UPSTREAM_DIR, VoicePrompt  # noqa: E402
from infer_onnx import (  # noqa: E402
    Backbone, CachedBackbone, QwenTokenizer, build_prompt, estimate_target_frames,
)

PLAIN = MODELS / "onnx" / "int4" / "omnivoice_lm.onnx"
KV = MODELS / "onnx" / "int4_kv" / "omnivoice_lm_kv.onnx"
TEXTS = {
    "short": "오늘 회의를 시작하겠습니다.",
    "long": ("오늘 회의를 시작하겠습니다. 먼저 지난주에 정리한 실험 결과를 간단히 "
             "공유드리겠습니다. 그다음에 다음 분기 일정과 남은 과제를 함께 정리하면 "
             "좋겠습니다. 질문은 중간에 편하게 말씀해 주세요."),
}


def _prompt(text: str, vp: VoicePrompt, tok: QwenTokenizer):
    t_gen = estimate_target_frames(text, vp.ref_text, vp.num_frames)
    return build_prompt(tok, text, vp.ref_text, vp.codes.astype(np.int64),
                        t_gen, "ko", None, True)


def _bench(fn, reps: int) -> tuple[float, float]:
    """Median and min of individual calls — a mean over a handful of reps on a
    desktop is mostly a measurement of what else the machine was doing."""
    for _ in range(2):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.min(ts))


def cmd_forward(args) -> None:
    """One session per process by default (`--which`): holding both a plain and a
    cached int4 session in one process inflates every number by ~35 % through
    memory pressure, which would flatter whichever one is measured second."""
    vp = VoicePrompt.load(args.voice_prompt)
    tok = QwenTokenizer(UPSTREAM_DIR / "tokenizer.json")
    lm = (Backbone(Path(args.plain), args.threads) if args.which == "plain"
          else CachedBackbone(Path(args.kv), args.threads))

    print(f"\n  per-forward, int4 `{args.which}`, threads={args.threads or 'auto'}, "
          f"{args.reps} reps")
    print(f"  {'case':>6} {'S':>5} {'P':>5} {'Tgen':>5} {'cond ms':>9} {'(min)':>9} "
          f"{'uncond ms':>10} {'(min)':>10} {'step ms':>9}")
    print("  " + "-" * 74)
    for name, text in TEXTS.items():
        if args.only and name != args.only:
            continue
        ids, am, gs = _prompt(text, vp, tok)
        S, T = ids.shape[2], ids.shape[2] - gs
        u_ids, u_mask = ids[:, :, gs:].copy(), am[:, gs:].copy()
        if args.which == "plain":
            c, cm = _bench(lambda: lm(ids, am), args.reps)
        else:
            lm.past_k = None
            lm.cond(ids, am, gs, 0)                       # prefill fills the cache
            c, cm = _bench(lambda: lm.cond(ids, am, gs, 1), args.reps)
        # the unconditional branch is the gen block alone: no prefix, nothing to
        # cache, and both variants pay it identically
        u, um = _bench(lambda: (lm(u_ids, u_mask) if args.which == "plain"
                                else lm.uncond(u_ids, u_mask)), args.reps)
        print(f"  {name:>6} {S:5d} {gs:5d} {T:5d} {c:9.1f} {cm:9.1f} {u:10.1f} "
              f"{um:10.1f} {c + u:9.1f}")


def cmd_parity(args) -> None:
    """With an empty past the cached graph is not an approximation of anything —
    it must reproduce the plain graph. This separates plumbing bugs from the
    approximation's own error."""
    vp = VoicePrompt.load(args.voice_prompt)
    tok = QwenTokenizer(UPSTREAM_DIR / "tokenizer.json")
    ids, am, gs = _prompt(TEXTS["short"], vp, tok)
    plain = Backbone(Path(args.plain), args.threads)
    cached = CachedBackbone(Path(args.kv), args.threads)
    a = plain(ids, am)[0].astype(np.float64)
    b = cached.cond(ids, am, gs, 0).astype(np.float64)     # prefill path, P=0
    a_gen = a[:, gs:, :]
    mad = float(np.abs(a_gen - b).max())
    cos = float((a_gen.ravel() @ b.ravel()) /
                (np.linalg.norm(a_gen.ravel()) * np.linalg.norm(b.ravel())))
    agree = float((a_gen.argmax(-1) == b.argmax(-1)).mean())
    print(f"\n  [parity] cached graph with P=0 vs the plain graph")
    print(f"    max|Δ| {mad:.4e}   cos {cos:.9f}   argmax {agree * 100:.2f} %")
    ok = cos > 0.9999
    print(f"    {'PASS' if ok else 'FAIL'}  — int4 quantizes the two graphs "
          f"independently, so argmax ties move; fp32 is bit-exact.")


def cmd_drift(args) -> None:
    """The approximation's own error: build the cache from the all-MASK state,
    then keep using it as the generated region fills up."""
    vp = VoicePrompt.load(args.voice_prompt)
    tok = QwenTokenizer(UPSTREAM_DIR / "tokenizer.json")
    ids, am, gs = _prompt(TEXTS["short"], vp, tok)
    T = ids.shape[2] - gs
    codes = np.load(args.codes).astype(np.int64)[:, :T]
    plain = Backbone(Path(args.plain), args.threads)
    cached = CachedBackbone(Path(args.kv), args.threads)
    cached.cond(ids, am, gs, 0)                            # cache from all-MASK

    rng = np.random.default_rng(0)
    print(f"\n  [drift] prefix K/V frozen at the all-MASK state, S={ids.shape[2]}, "
          f"prefix={gs}, gen={T}")
    print(f"  {'filled':>7} {'max|Δ|':>10} {'cos':>11} {'argmax':>8} {'mean KL':>9}")
    print("  " + "-" * 50)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        sel = rng.random((NUM_CODEBOOKS, T)) < frac
        ids2 = ids.copy()
        ids2[0, :, gs:] = np.where(sel, codes, AUDIO_MASK_ID)
        want = plain(ids2, am)[0, :, gs:, :].astype(np.float64)
        got = cached.cond(ids2, am, gs, 1).astype(np.float64)
        lw = want - want.max(-1, keepdims=True)
        lw -= np.log(np.exp(lw).sum(-1, keepdims=True))
        lg = got - got.max(-1, keepdims=True)
        lg -= np.log(np.exp(lg).sum(-1, keepdims=True))
        kl = float((np.exp(lw) * (lw - lg)).sum(-1).mean())
        cos = float((want.ravel() @ got.ravel()) /
                    (np.linalg.norm(want.ravel()) * np.linalg.norm(got.ravel())))
        print(f"  {frac:7.2f} {np.abs(want - got).max():10.3e} {cos:11.8f} "
              f"{(want.argmax(-1) == got.argmax(-1)).mean() * 100:7.2f}% {kl:9.5f}")


def cmd_wavdelta(args) -> None:
    """LSD/SNR of one decoded candidate against another. Decodes the codes rather
    than diffing the written WAVs: post-processing trims silence, so a WAV diff
    would measure an alignment shift instead of the audio."""
    from codebook_ablation import damage, decode, decoder
    sess = decoder(args.threads)
    a = decode(sess, np.load(args.a))
    b = decode(sess, np.load(args.b))
    lsd, snr = damage(a, b)
    print(f"\n  [wavdelta] {Path(args.a).name} vs {Path(args.b).name}")
    print(f"    LSD {lsd:.2f} dB   SNR {snr:.2f} dB   "
          f"({len(a) / 24000:.2f}s vs {len(b) / 24000:.2f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["forward", "parity", "drift", "wavdelta"])
    ap.add_argument("--plain", default=str(PLAIN))
    ap.add_argument("--kv", default=str(KV))
    ap.add_argument("--voice-prompt", default=str(OUT / "golden_det" / "voice_prompt.bin"))
    ap.add_argument("--codes", default=str(OUT / "golden_det" / "codes.npy"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--which", choices=["plain", "cache"], default="plain")
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    args = ap.parse_args()
    {"forward": cmd_forward, "parity": cmd_parity, "drift": cmd_drift,
     "wavdelta": cmd_wavdelta}[args.cmd](args)


if __name__ == "__main__":
    main()
