#!/usr/bin/env python3
"""Phase 7 — build a QNN/HTP-targeted model.

The device is a Snapdragon 8 Elite Gen 5, so the Hexagon NPU is reachable through
ONNX Runtime's QNN execution provider. Three things have to change versus the CPU
build, and none of them is a flag (docs/benchmark.md §7.5):

  1. static shapes      — HTP compiles the graph ahead of time, so a symbolic
                          `seq` makes every node ineligible (measured: NNAPI goes
                          from 0 to 142 claimed nodes the moment shapes are fixed)
  2. QDQ, not int4      — HTP has no 4-bit GEMM. It wants quantized *activations*
                          too, which is why calibration data is needed at all
  3. per-tensor, uint8  — HTP dislikes per-channel for most ops

  python scripts/export_qnn.py --activation uint8  --out models/onnx/qnn_a8w8
  python scripts/export_qnn.py --activation uint16 --out models/onnx/qnn_a16w8
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODELS, OUT  # noqa: E402


class NpzCalibrationReader:
    """Feeds the captured real inputs (scripts/make_calibration.py) to ORT."""

    def __init__(self, path: Path, limit: int | None = None):
        self.data = np.load(path)
        self.count = int(self.data["count"])
        if limit:
            self.count = min(self.count, limit)
        self.i = 0

    def get_next(self):
        if self.i >= self.count:
            return None
        i, self.i = self.i, self.i + 1
        return {
            "input_ids": self.data[f"{i}__input_ids"],
            "audio_mask": self.data[f"{i}__audio_mask"],
            "attention_mask": self.data[f"{i}__attention_mask"],
            "position_ids": self.data[f"{i}__position_ids"],
        }

    def rewind(self):
        self.i = 0


def bump_opset(src: Path, dst: Path, target: int = 21) -> None:
    """uint16 QDQ needs opset >= 21, and ORT's quantizer upgrades by calling
    onnx.version_converter, which serializes the whole model and therefore dies
    on anything over 2 GB. Opset 20 -> 21 changed no semantics for the ops this
    graph uses (it added type support to Cast/QuantizeLinear/DequantizeLinear and
    friends), so the declared version is rewritten directly instead. The caller
    verifies numerics afterwards rather than trusting that claim.
    """
    import onnx

    m = onnx.load(str(src), load_external_data=True)
    for o in m.opset_import:
        if (o.domain or "ai.onnx") == "ai.onnx" and o.version < target:
            print(f"  opset ai.onnx {o.version} -> {target}")
            o.version = target
    for f in (dst, dst.with_suffix(".onnx.data")):
        f.unlink(missing_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(m, str(dst), save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=dst.name + ".data", size_threshold=1024)


def fix_shapes(src: Path, dst: Path, seq: int, batch: int = 1) -> None:
    """`make_dynamic_shape_fixed` cannot round-trip a >2 GB model because it
    saves without external data; do the same job, external-data aware."""
    import onnx
    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

    m = onnx.load(str(src), load_external_data=True)
    make_dim_param_fixed(m.graph, "batch", batch)
    make_dim_param_fixed(m.graph, "seq", seq)
    for f in (dst, dst.with_suffix(".onnx.data")):
        f.unlink(missing_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(m, str(dst), save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=dst.name + ".data", size_threshold=1024)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(MODELS / "onnx" / "fp32" / "omnivoice_lm.onnx"))
    ap.add_argument("--calib", default=str(OUT / "calib" / "calib.npz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--activation", choices=["uint8", "uint16"], default="uint8")
    ap.add_argument("--weight", choices=["uint8", "int8"], default="uint8")
    ap.add_argument("--seq", type=int, default=188,
                    help="fixed sequence length for the static variant")
    ap.add_argument("--limit", type=int, default=None, help="calibration samples to use")
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

    src = Path(args.source)
    if not src.exists():
        sys.exit(f"{src} not found — run export_onnx.py --precision fp32 first")
    calib = Path(args.calib)
    if not calib.exists():
        sys.exit(f"{calib} not found — run make_calibration.py first")

    tag = f"a{args.activation[4:]}w{args.weight[-1] if args.weight[-1].isdigit() else '8'}"
    outdir = Path(args.out or MODELS / "onnx" / f"qnn_{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    qdq = outdir / "omnivoice_lm.onnx"

    act = {"uint8": QuantType.QUInt8, "uint16": QuantType.QUInt16}[args.activation]
    wt = {"uint8": QuantType.QUInt8, "int8": QuantType.QInt8}[args.weight]

    if args.activation == "uint16":
        bumped = OUT / "qnn_tmp" / "omnivoice_lm_opset21.onnx"
        if not bumped.exists():
            print("uint16 activations need opset >= 21; bumping the source ...")
            bump_opset(src, bumped)
        src = bumped

    reader = NpzCalibrationReader(calib, args.limit)
    print(f"calibrating on {reader.count} captured samples "
          f"(activation={args.activation}, weight={args.weight}) ...")

    t0 = time.perf_counter()
    cfg = get_qnn_qdq_config(
        str(src), reader,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=act, weight_type=wt,
        per_channel=False,        # HTP dislikes per-channel for most ops
    )
    cfg.use_external_data_format = True
    print(f"  config built in {time.perf_counter() - t0:.1f}s; quantizing ...")

    t0 = time.perf_counter()
    quantize(str(src), str(qdq), cfg)
    print(f"  quantized in {time.perf_counter() - t0:.1f}s")
    _report(qdq)

    if not args.skip_static:
        static = outdir / f"omnivoice_lm_s{args.seq}.onnx"
        print(f"\nfixing shapes → batch=1, seq={args.seq} ...")
        t0 = time.perf_counter()
        fix_shapes(qdq, static, args.seq)
        print(f"  done in {time.perf_counter() - t0:.1f}s")
        _report(static)


def _report(path: Path) -> None:
    import collections

    import onnx

    m = onnx.load(str(path), load_external_data=False)
    hist = collections.Counter((n.domain or "ai.onnx") + "::" + n.op_type for n in m.graph.node)
    size = sum(p.stat().st_size for p in
               [path, path.with_suffix(".onnx.data")] if p.exists())
    print(f"  {path.name}  {size / 1e6:.1f} MB, {len(m.graph.node)} nodes")
    for k, v in hist.most_common(8):
        print(f"    {v:6d}  {k}")


if __name__ == "__main__":
    main()
