#!/usr/bin/env python3
"""Phase 1 — PyTorch golden reference.

Runs upstream OmniVoice and dumps every intermediate the ONNX port has to
reproduce: the built prompt, the reference codec codes, per-step logits, the
final audio codes, and the waveform.

  # 1. bootstrap a reference clip (auto-voice mode, no reference needed)
  python scripts/reference_infer.py bootstrap

  # 2. the actual golden voice-cloning run
  python scripts/reference_infer.py clone

Determinism: OmniVoice adds Gumbel noise to the position scores
(position_temperature=5.0). ``--deterministic`` sets it to 0 so ONNX output can
be compared cell by cell; the default keeps upstream behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT, SAMPLE, SR_24K, UPSTREAM_DIR, VoicePrompt, write_wav,
)


def load_model(device: str, dtype: str):
    from omnivoice.models.omnivoice import OmniVoice

    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32,
                   "bf16": torch.bfloat16}[dtype]
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(str(UPSTREAM_DIR), device_map=device,
                                      dtype=torch_dtype)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s "
          f"({device}, {dtype}, sr={model.sampling_rate})")
    return model


class Recorder:
    """Wraps OmniVoice so every forward and prompt build is captured."""

    def __init__(self, model):
        self.model = model
        self.prompts: list[dict] = []
        self.steps: list[dict] = []
        self.final_codes = None
        self._orig_prepare = model._prepare_inference_inputs
        self._orig_forward = model.forward
        self._orig_decode = model._decode_and_post_process

    def __enter__(self):
        m = self.model

        def prepare(*a, **kw):
            out = self._orig_prepare(*a, **kw)
            self.prompts.append({
                "input_ids": out["input_ids"].detach().cpu().numpy(),
                "audio_mask": out["audio_mask"].detach().cpu().numpy(),
            })
            return out

        def forward(*a, **kw):
            out = self._orig_forward(*a, **kw)
            ids = kw.get("input_ids", a[0] if a else None)
            self.steps.append({
                "input_ids": ids.detach().cpu().numpy().astype(np.int32),
                "logits": out.logits.detach().float().cpu().numpy(),
            })
            return out

        def decode(tokens, *a, **kw):
            # tokens is the FINAL (C, T) code tensor; the last recorded forward
            # still has masked cells because it happens before the last update
            t = tokens[0] if isinstance(tokens, list) else tokens
            self.final_codes = t.detach().cpu().numpy().astype(np.int64)
            return self._orig_decode(tokens, *a, **kw)

        m._prepare_inference_inputs = prepare
        m.forward = forward
        m._decode_and_post_process = decode
        return self

    def __exit__(self, *exc):
        self.model._prepare_inference_inputs = self._orig_prepare
        self.model.forward = self._orig_forward
        self.model._decode_and_post_process = self._orig_decode
        return False


def gen_kwargs(args) -> dict:
    kw = dict(num_step=args.num_step, guidance_scale=args.guidance_scale,
              t_shift=args.t_shift, denoise=args.denoise,
              layer_penalty_factor=5.0, position_temperature=5.0,
              class_temperature=0.0, postprocess_output=args.postprocess)
    if args.deterministic:
        kw["position_temperature"] = 0.0
    return kw


def cmd_bootstrap(args) -> None:
    """Create sample/reference.wav with OmniVoice's own auto/design voice, so the
    repo has a reference clip whose transcript is known exactly."""
    model = load_model(args.device, args.dtype)
    text = args.text or (SAMPLE / "reference.txt").read_text(encoding="utf-8").strip()
    print(f"bootstrap text: {text!r}")
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    audio = model.generate(text=text, language=args.language,
                           instruct=args.instruct, **gen_kwargs(args))[0]
    dt = time.perf_counter() - t0
    dur = len(audio) / model.sampling_rate
    out = Path(args.out or SAMPLE / "reference.wav")
    write_wav(out, audio, model.sampling_rate)
    print(f"wrote {out}  {dur:.2f}s  (generated in {dt:.1f}s, RTF {dt / dur:.2f})")


def cmd_clone(args) -> None:
    model = load_model(args.device, args.dtype)
    ref_wav = Path(args.ref_audio or SAMPLE / "reference.wav")
    ref_text = args.ref_text or (SAMPLE / "reference.txt").read_text(encoding="utf-8").strip()
    text = args.text or (SAMPLE / "target.txt").read_text(encoding="utf-8").strip()
    outdir = Path(args.out or OUT / "golden")
    outdir.mkdir(parents=True, exist_ok=True)

    if not ref_wav.exists():
        sys.exit(f"{ref_wav} not found — run `reference_infer.py bootstrap` first.")

    print(f"ref audio : {ref_wav}")
    print(f"ref text  : {ref_text!r}")
    print(f"target    : {text!r}")

    # --- enrollment -------------------------------------------------------
    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(str(ref_wav), ref_text=ref_text)
    t_enroll = time.perf_counter() - t0
    codes = prompt.ref_audio_tokens.cpu().numpy()
    print(f"enrolled in {t_enroll:.2f}s → codes {codes.shape} "
          f"({codes.shape[1] / 25:.2f}s), ref_rms={prompt.ref_rms:.4f}")

    vp = VoicePrompt(codes=codes.astype(np.int16), ref_text=prompt.ref_text,
                     ref_rms=float(prompt.ref_rms))
    vp.save(outdir / "voice_prompt.bin")
    prompt.save(str(outdir / "voice_prompt.pt"))

    # --- generation -------------------------------------------------------
    torch.manual_seed(args.seed)
    with Recorder(model) as rec:
        t0 = time.perf_counter()
        audio = model.generate(text=text, language=args.language,
                               voice_clone_prompt=prompt, **gen_kwargs(args))[0]
        t_gen = time.perf_counter() - t0

    dur = len(audio) / model.sampling_rate
    write_wav(outdir / "golden.wav", audio, model.sampling_rate)
    print(f"generated {dur:.2f}s in {t_gen:.1f}s  →  RTF {t_gen / dur:.3f}")

    # --- dump -------------------------------------------------------------
    p = rec.prompts[0]
    S = p["input_ids"].shape[2]
    T_ref = codes.shape[1]
    T_gen = int(p["audio_mask"].sum()) - T_ref
    np.savez_compressed(
        outdir / "prompt.npz",
        input_ids=p["input_ids"].astype(np.int32),
        audio_mask=p["audio_mask"],
        ref_codes=codes.astype(np.int32),
    )
    np.save(outdir / "waveform.npy", audio.astype(np.float32))
    # the final audio codes (8, T_gen). Must come from _decode_and_post_process:
    # the last recorded forward runs BEFORE the final un-masking update, so its
    # input_ids still contain MASK (1024) cells.
    if rec.final_codes is None:
        raise RuntimeError("final codes were not captured")
    if int((rec.final_codes == 1024).sum()):
        raise RuntimeError("final codes still contain MASK cells")
    np.save(outdir / "codes.npy", rec.final_codes)
    keep = sorted({0, 1, len(rec.steps) // 2, len(rec.steps) - 1})
    np.savez_compressed(
        outdir / "steps.npz",
        **{f"input_ids_{i}": rec.steps[i]["input_ids"] for i in keep},
        **{f"logits_{i}": rec.steps[i]["logits"].astype(np.float16) for i in keep},
        kept_steps=np.array(keep),
    )

    meta = {
        "ref_audio": str(ref_wav), "ref_text": prompt.ref_text, "text": text,
        "language": args.language, "seed": args.seed,
        "deterministic": args.deterministic, "dtype": args.dtype,
        "device": args.device, "gen_config": gen_kwargs(args),
        "S": int(S), "T_ref": int(T_ref), "T_gen": int(T_gen),
        "ref_rms": float(prompt.ref_rms),
        "num_forward_calls": len(rec.steps),
        "kept_steps": keep,
        "audio_samples": int(len(audio)), "audio_seconds": dur,
        "enroll_seconds": t_enroll, "generate_seconds": t_gen,
        "rtf": t_gen / dur,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nS={S}  T_ref={T_ref}  T_gen={T_gen}  forwards={len(rec.steps)}")
    print(f"dumped → {outdir}/  " + ", ".join(sorted(q.name for q in outdir.iterdir())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bootstrap", "clone"):
        s = sub.add_parser(name)
        s.add_argument("--text", default=None)
        s.add_argument("--language", default="Korean")
        s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        s.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
        s.add_argument("--num-step", type=int, default=32)
        s.add_argument("--guidance-scale", type=float, default=2.0)
        s.add_argument("--t-shift", type=float, default=0.1)
        s.add_argument("--denoise", action="store_true", default=True)
        s.add_argument("--postprocess", action="store_true", default=True)
        s.add_argument("--deterministic", action="store_true")
        s.add_argument("--seed", type=int, default=1234)
        s.add_argument("--out", default=None)
        if name == "bootstrap":
            s.add_argument("--instruct", default="male, young adult, moderate pitch, korean accent")
        else:
            s.add_argument("--ref-audio", default=None)
            s.add_argument("--ref-text", default=None)
    args = ap.parse_args()
    {"bootstrap": cmd_bootstrap, "clone": cmd_clone}[args.cmd](args)


if __name__ == "__main__":
    main()
