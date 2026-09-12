#!/usr/bin/env python3
"""Phase 2 — weight-only quantization sweep for the backbone.

Symmetric int4 over MatMul+Gather at accuracy_level 4 turned out to cost 75
points of argmax agreement, so the settings are not a detail. This measures
size / speed / fidelity for each combination against cached PyTorch fp32 logits.

  python scripts/sweep_quant.py                       # full sweep
  python scripts/sweep_quant.py --keep asym_acc1_mg   # also write models/onnx/int4
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODELS, OUT, UPSTREAM_DIR  # noqa: E402

FP32 = MODELS / "onnx" / "fp32" / "omnivoice_lm.onnx"
REF = OUT / "golden" / "lm_fp32_logits.npy"
SCRATCH = OUT / "quant_sweep"

# The audio head is Linear(1024 -> 8*1025) sitting directly on the logits, so
# quantization error there is not averaged away by anything downstream.
HEAD = ["/audio_heads/MatMul"]
EMBEDS = ["/embed_tokens/Gather", "/audio_embeddings/Gather"]

# name -> dict(bits, block, symmetric, accuracy_level, ops, exclude, algo)
def V(bits=4, block=128, sym=False, acc=1, ops=("MatMul", "Gather"),
      exclude=None, algo="rtn"):
    return dict(bits=bits, block=block, sym=sym, acc=acc, ops=ops,
                exclude=exclude or [], algo=algo)


VARIANTS = {
    # --- round 1: does plain RTN int4 work at all? ---
    "sym_acc4_mg":    V(sym=True, acc=4),
    "asym_acc4_mg":   V(acc=4),
    "asym_acc1_mg":   V(),
    "asym_acc4_m":    V(acc=4, ops=("MatMul",)),
    "asym_acc1_m":    V(ops=("MatMul",)),
    "asym_b32_mg":    V(block=32),
    "int8_acc1_m":    V(bits=8, ops=("MatMul",)),
    # --- round 2: protect the sensitive layers / better algorithms ---
    "int4_nohead":    V(ops=("MatMul",), exclude=HEAD),
    "int4_nohead_b32": V(block=32, ops=("MatMul",), exclude=HEAD),
    "int4_noembed_nohead": V(exclude=HEAD + EMBEDS),
    # --- round 3: the Android candidates -- int4 everything except the head ---
    "android_b128":   V(exclude=HEAD),
    "android_b32":    V(block=32, exclude=HEAD),
    "android_b32_h8": V(block=32, exclude=[]),   # head at int8 instead of fp32

    "hqq_mg":         V(algo="hqq", ops=("MatMul",)),
    "hqq_nohead":     V(algo="hqq", ops=("MatMul",), exclude=HEAD),
    "kquant_mg":      V(algo="kquant", ops=("MatMul",)),
    "int8_mg":        V(bits=8),
    "int8_nohead":    V(bits=8, ops=("MatMul",), exclude=HEAD),
}


def _prompt():
    d = np.load(OUT / "golden" / "prompt.npz")
    ids, am = d["input_ids"].astype(np.int64), d["audio_mask"].astype(bool)
    b, _, s = ids.shape
    return ids, am, {
        "input_ids": ids, "audio_mask": am,
        "attention_mask": np.ones((b, 1, s, s), dtype=bool),
        "position_ids": np.arange(s, dtype=np.int64)[None, :].repeat(b, 0),
    }


def reference_logits() -> np.ndarray:
    if REF.exists():
        return np.load(REF)
    print("computing PyTorch fp32 reference logits (once) ...")
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    ids, am, _ = _prompt()
    S = ids.shape[2]
    model = OmniVoice.from_pretrained(str(UPSTREAM_DIR), device_map="cpu",
                                      dtype=torch.float32, train=True).eval()
    model.llm.set_attn_implementation("sdpa")
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids), audio_mask=torch.from_numpy(am),
                    attention_mask=torch.ones(1, 1, S, S, dtype=torch.bool)).logits.numpy()
    REF.parent.mkdir(parents=True, exist_ok=True)
    np.save(REF, out)
    return out


def evaluate(path: Path, want: np.ndarray, threads: int, repeats: int) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0

    _, am, feed = _prompt()
    got = sess.run(["logits"], feed)[0]              # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run(["logits"], feed)
        times.append(time.perf_counter() - t0)

    gen = am[0]
    a, b = got.astype(np.float64), want.astype(np.float64)
    cos = float((a.ravel() @ b.ravel()) /
                (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12))
    agree = float((got[0, :, gen].argmax(-1) == want[0, :, gen].argmax(-1)).mean())
    # top-5 containment: does the fp32 winner survive in the int4 top 5?
    top5 = np.argpartition(-got[0, :, gen], 5, axis=-1)[..., :5]
    want_arg = want[0, :, gen].argmax(-1)
    in5 = float((top5 == want_arg[..., None]).any(-1).mean())

    size = sum(p.stat().st_size for p in
               [path, path.with_suffix(".onnx.data")] if p.exists())
    return {"mb": round(size / 1e6, 1), "load_s": round(load_s, 2),
            "ms": round(float(np.median(times)) * 1000, 1),
            "cos": round(cos, 6), "argmax": round(agree * 100, 2),
            "top5": round(in5 * 100, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--keep", default=None, help="variant to install as models/onnx/int4")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not FP32.exists():
        sys.exit(f"{FP32} not found — run `export_onnx.py --precision fp32` first.")

    import onnx
    from onnxruntime.quantization import matmul_nbits_quantizer as qn

    want = reference_logits()
    base = evaluate(FP32, want, args.threads, args.repeats)
    print(f"\n  {'variant':16s} {'MB':>7} {'load s':>7} {'ms':>7} {'cos':>9} "
          f"{'argmax%':>8} {'top5%':>7}")
    print("  " + "-" * 66)
    print(f"  {'fp32 (baseline)':16s} {base['mb']:7.1f} {base['load_s']:7.2f} "
          f"{base['ms']:7.1f} {base['cos']:9.6f} {base['argmax']:8.2f} {base['top5']:7.2f}")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    results = {"fp32": base}
    names = args.only or list(VARIANTS)
    for name in names:
        v = VARIANTS[name]
        target = SCRATCH / name / "omnivoice_lm.onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        # the fp32 graph is >2 GB, so protobuf cannot round-trip it in memory;
        # reload from disk for each variant instead of deep-copying
        model_in = onnx.load(str(FP32), load_external_data=True)
        common = dict(op_types_to_quantize=tuple(v["ops"]),
                      quant_axes=(("Gather", 1),))
        if v["algo"] == "hqq":
            algo = qn.HQQWeightOnlyQuantConfig(block_size=v["block"], bits=v["bits"], **common)
        elif v["algo"] == "kquant":
            algo = qn.KQuantWeightOnlyQuantConfig(
                op_types_to_quantize=tuple(v["ops"]))   # no quant_axes kwarg
        else:
            algo = None
        q = qn.MatMulNBitsQuantizer(
            model_in, bits=v["bits"], block_size=v["block"], is_symmetric=v["sym"],
            accuracy_level=v["acc"], nodes_to_exclude=v["exclude"] or None,
            algo_config=algo, **({} if algo else common))
        q.process()
        model = q.model.model if hasattr(q.model, "model") else q.model
        # onnx.save_model APPENDS to an existing external-data file, so a re-run
        # of the same variant silently doubles it. Always start clean.
        for f in (target, target.with_suffix(".onnx.data")):
            f.unlink(missing_ok=True)
        onnx.save_model(model, str(target), save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=target.name + ".data", size_threshold=1024)
        r = evaluate(target, want, args.threads, args.repeats)
        results[name] = r
        print(f"  {name:16s} {r['mb']:7.1f} {r['load_s']:7.2f} {r['ms']:7.1f} "
              f"{r['cos']:9.6f} {r['argmax']:8.2f} {r['top5']:7.2f}", flush=True)

    out = Path(args.out or OUT / "quant_sweep" / "results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  → {out}")

    if args.keep:
        dst = MODELS / "onnx" / "int4"
        dst.mkdir(parents=True, exist_ok=True)
        for suffix in (".onnx", ".onnx.data"):
            s = SCRATCH / args.keep / ("omnivoice_lm" + suffix)
            if s.exists():
                shutil.copy2(s, dst / s.name)
        (dst / "quant_variant.txt").write_text(args.keep + "\n")
        print(f"  installed `{args.keep}` → {dst}")


if __name__ == "__main__":
    main()
