#!/usr/bin/env python3
"""Rewrite the `accuracy_level` attribute of every MatMulNBits node.

`accuracy_level` selects the COMPUTE type of an int4 GEMM; it does not touch a
single stored weight. Level 1 is fp32 compute, level 4 is `SQNBIT_CompInt8` --
the quantised-activation path, and the only one ONNX Runtime's ARM64 KleidiAI
kernels (dotprod / i8mm) will dispatch to. The shipping export is at level 1,
so 85 % of device runtime is on the generic MLAS path and cannot reach them.

Because this is metadata, the rewrite is seconds and reversible, and quality
can be judged against the very same weights:

  python scripts/set_accuracy_level.py --in models/onnx/int4 --out models/onnx/int4_acc4 --level 4
  python scripts/set_accuracy_level.py --in models/onnx/int4_kv --out models/onnx/int4_kv_acc4 \
      --graph omnivoice_lm_kv.onnx
"""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import sys
from pathlib import Path

import onnx
from onnx import AttributeProto

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODELS  # noqa: E402

LM = "omnivoice_lm.onnx"


def place(src: Path, dst: Path) -> None:
    """Hard-link where the filesystem allows it, copy where it does not. The
    weight blob is identical by construction, so there is no reason to spend
    420 MB of disk on a second copy of it."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def levels(model: onnx.ModelProto) -> dict:
    c = collections.Counter()
    for n in model.graph.node:
        if n.op_type == "MatMulNBits":
            a = {x.name: x.i for x in n.attribute if x.type == AttributeProto.INT}
            c[(a.get("bits"), a.get("block_size"), a.get("accuracy_level"))] += 1
    return dict(c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default=str(MODELS / "onnx" / "int4"))
    ap.add_argument("--out", dest="dst", default=str(MODELS / "onnx" / "int4_acc4"))
    ap.add_argument("--level", type=int, default=4, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--graph", default=LM, help="graph file name inside --in")
    args = ap.parse_args()
    graph = args.graph

    src, dst = Path(args.src), Path(args.dst)
    model = onnx.load(str(src / graph), load_external_data=False)
    print(f"  in   {src}  {levels(model)}")

    touched = 0
    for n in model.graph.node:
        if n.op_type != "MatMulNBits":
            continue
        for a in n.attribute:
            if a.name == "accuracy_level":
                a.i = args.level
                break
        else:
            n.attribute.append(onnx.helper.make_attribute("accuracy_level", args.level))
        touched += 1

    dst.mkdir(parents=True, exist_ok=True)
    # The weights are untouched, so the external-data blob is linked verbatim
    # rather than re-serialised -- and `onnx.save_model` APPENDS to an existing
    # external file, so anything already there has to go first.
    for f in src.iterdir():
        if f.name == graph:
            continue
        place(f, dst / f.name)
    (dst / graph).unlink(missing_ok=True)
    onnx.save(model, str(dst / graph))

    check = onnx.load(str(dst / graph), load_external_data=False)
    print(f"  out  {dst}  {levels(check)}   ({touched} nodes)")


if __name__ == "__main__":
    main()
