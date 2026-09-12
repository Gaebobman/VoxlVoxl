#!/usr/bin/env python3
"""Why the standard diffusion-LLM speedups do or do not transfer to OmniVoice.

Fast-dLLM (arXiv:2505.22618) reports large gains on LLaDA/Dream from committing
every token whose probability clears a threshold instead of a fixed number per
step. Wired into our loop it changed nothing. This measures the reason: how
confident the model actually is about an audio cell.

Also measures the one lever that needs no re-export — the reference prefix is
over half the conditional sequence, and it is re-read at every forward.

  python scripts/speed_probe.py
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
    Backbone, GenConfig, QwenTokenizer, _log_softmax, build_prompt,
    estimate_target_frames, unmask_schedule,
)


def confidence_profile(args) -> None:
    """Per step: how many masked cells clear each probability threshold."""
    vp = VoicePrompt.load(args.voice_prompt)
    tok = QwenTokenizer(UPSTREAM_DIR / "tokenizer.json")
    cfg = GenConfig(num_step=args.num_step)
    t_gen = estimate_target_frames(args.text, vp.ref_text, vp.num_frames)
    ids, am, gen_start = build_prompt(
        tok, args.text, vp.ref_text, vp.codes.astype(np.int64), t_gen, "ko", None, True)
    S = ids.shape[2]
    lm = Backbone(Path(args.lm), args.threads)

    tokens = np.full((NUM_CODEBOOKS, t_gen), AUDIO_MASK_ID, dtype=np.int64)
    sched = unmask_schedule(cfg.num_step, cfg.t_shift, t_gen * NUM_CODEBOOKS)
    u_ids = ids[:, :, gen_start:].copy()
    u_mask = am[:, gen_start:].copy()
    rng = np.random.default_rng(0)
    thresholds = (0.5, 0.7, 0.9, 0.95)

    print(f"\n  S={S} (prefix {gen_start} + gen {t_gen}), cells {t_gen * NUM_CODEBOOKS}")
    print(f"  {'step':>4} {'masked':>7} {'sched k':>8} " +
          " ".join(f"{'p>' + str(t):>8}" for t in thresholds) + f" {'maxp':>7}")
    print("  " + "-" * 62)

    for step, k in enumerate(sched):
        c = lm(ids, am)[0, :, gen_start:, :].astype(np.float32)
        u = lm(u_ids, u_mask)[0].astype(np.float32)
        c_lp, u_lp = _log_softmax(c), _log_softmax(u)
        lp = _log_softmax(c_lp + cfg.guidance_scale * (c_lp - u_lp))
        lp[..., AUDIO_MASK_ID] = -np.inf
        conf = np.exp(lp.max(-1))
        masked = tokens == AUDIO_MASK_ID
        n_masked = int(masked.sum())
        counts = [int(((conf > t) & masked).sum()) for t in thresholds]
        print(f"  {step + 1:4d} {n_masked:7d} {k:8d} " +
              " ".join(f"{c_:8d}" for c_ in counts) +
              f" {float(conf[masked].max()) if n_masked else 0:7.3f}")

        scores = lp.max(-1) - np.arange(NUM_CODEBOOKS, dtype=np.float32)[:, None] * cfg.layer_penalty_factor
        u_ = rng.random(scores.shape).astype(np.float32)
        scores = scores / cfg.position_temperature + (-np.log(-np.log(u_ + 1e-10) + 1e-10))
        scores = np.where(~masked, -np.inf, scores)
        kk = min(k, int(np.isfinite(scores).sum()))
        if kk <= 0:
            continue
        top = np.argpartition(-scores.ravel(), kk - 1)[:kk]
        tokens.ravel()[top] = lp.argmax(-1).ravel()[top]
        ids[0, :, gen_start:] = tokens
        u_ids[0] = tokens
        if not (tokens == AUDIO_MASK_ID).any():
            break

    print(f"\n  {lm.calls} forwards, {lm.compute_seconds:.1f}s")


def reference_length(args) -> None:
    """The reference prefix is re-read at every forward. What does trimming buy?"""
    vp = VoicePrompt.load(args.voice_prompt)
    tok = QwenTokenizer(UPSTREAM_DIR / "tokenizer.json")
    cfg = GenConfig(num_step=args.num_step)
    print(f"\n  {'ref frames':>11} {'ref s':>7} {'S':>5} {'fwd ms':>8} {'vs full':>8}")
    print("  " + "-" * 46)
    base = None
    for frames in (vp.num_frames, 100, 75, 50, 25):
        if frames > vp.num_frames:
            continue
        codes = vp.codes[:, :frames].astype(np.int64)
        t_gen = estimate_target_frames(args.text, vp.ref_text, frames)
        ids, am, _ = build_prompt(tok, args.text, vp.ref_text, codes, t_gen, "ko", None, True)
        lm = Backbone(Path(args.lm), args.threads)
        lm(ids, am)                       # warm
        t0 = time.perf_counter()
        for _ in range(3):
            lm(ids, am)
        ms = (time.perf_counter() - t0) / 3 * 1000
        base = base or ms
        print(f"  {frames:11d} {frames / 25:7.2f} {ids.shape[2]:5d} {ms:8.0f} {base / ms:7.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", default=str(MODELS / "onnx" / "int4" / "omnivoice_lm.onnx"))
    ap.add_argument("--voice-prompt", default=str(OUT / "golden_det" / "voice_prompt.bin"))
    ap.add_argument("--text", default="오늘 회의를 시작하겠습니다.")
    ap.add_argument("--num-step", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", choices=["confidence", "reference"], default=None)
    args = ap.parse_args()
    if args.only in (None, "confidence"):
        confidence_profile(args)
    if args.only in (None, "reference"):
        reference_length(args)


if __name__ == "__main__":
    main()
