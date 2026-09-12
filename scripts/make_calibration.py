#!/usr/bin/env python3
"""Capture real backbone inputs for static quantization calibration.

Static (QDQ) quantization needs activation ranges, which need representative
inputs. Synthetic tensors would mis-calibrate badly here: the sequence is a
mixture of text token ids, reference codec codes and MASK cells, and the
activation statistics change a lot between the first un-masking step (almost
everything masked) and the last (almost nothing masked). So the inputs are
captured from actual runs, across several texts and several steps.

  python scripts/make_calibration.py --out out/calib/calib.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODELS, OUT, UPSTREAM_DIR, VoicePrompt  # noqa: E402
from infer_onnx import (  # noqa: E402
    Backbone, GenConfig, QwenTokenizer, build_prompt, estimate_target_frames,
    generate_codes,
)

TEXTS = [
    "오늘 회의를 시작하겠습니다.",
    "내일 오후 세 시에 다시 연락드리겠습니다.",
    "This is a short English sentence for calibration.",
    "안녕하세요, 반갑습니다. 잘 부탁드립니다.",
    "지금 바로 확인해 주시면 감사하겠습니다.",
]


class CapturingBackbone(Backbone):
    """Records every (input_ids, audio_mask) the decoding loop actually feeds."""

    def __init__(self, path, threads=0, keep_steps=(0, 3, 7, 11, 15)):
        super().__init__(path, threads)
        self.samples: list[dict] = []
        self.keep_steps = set(keep_steps)
        self._call = 0

    def __call__(self, input_ids, audio_mask):
        # two calls per step: conditional then unconditional
        step = self._call // 2
        if step in self.keep_steps:
            b, _, s = input_ids.shape
            self.samples.append({
                "input_ids": input_ids.copy(),
                "audio_mask": audio_mask.copy(),
                "attention_mask": np.ones((b, 1, s, s), dtype=bool),
                "position_ids": np.arange(s, dtype=np.int64)[None, :].repeat(b, 0),
            })
        self._call += 1
        return super().__call__(input_ids, audio_mask)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", default=str(MODELS / "onnx" / "int4" / "omnivoice_lm.onnx"),
                    help="model used only to DRIVE the loop; int4 is fine and fast")
    ap.add_argument("--voice-prompt", default=str(OUT / "golden_det" / "voice_prompt.bin"))
    ap.add_argument("--num-step", type=int, default=16)
    ap.add_argument("--texts", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "calib" / "calib.npz"))
    args = ap.parse_args()

    vp = VoicePrompt.load(args.voice_prompt)
    tok = QwenTokenizer(UPSTREAM_DIR / "tokenizer.json")
    cfg = GenConfig(num_step=args.num_step)
    rng = np.random.default_rng(0)

    all_samples: list[dict] = []
    for text in (args.texts or TEXTS):
        t_gen = estimate_target_frames(text, vp.ref_text, vp.num_frames)
        ids, am, gen_start = build_prompt(
            tok, text, vp.ref_text, vp.codes.astype(np.int64), t_gen, "ko", None, True)
        lm = CapturingBackbone(Path(args.lm), args.threads)
        generate_codes(lm, ids, am, gen_start, cfg, rng, progress=False)
        print(f"  {text[:34]:36s} S={ids.shape[2]:4d} T_gen={t_gen:3d} "
              f"→ {len(lm.samples)} samples")
        all_samples += lm.samples

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for i, smp in enumerate(all_samples):
        for k, v in smp.items():
            flat[f"{i}__{k}"] = v
    np.savez_compressed(out, count=np.array(len(all_samples)), **flat)
    shapes = sorted({s["input_ids"].shape[2] for s in all_samples})
    print(f"\n  {len(all_samples)} samples, sequence lengths {shapes}")
    print(f"  → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
