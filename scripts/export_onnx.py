#!/usr/bin/env python3
"""Phase 2 — export the OmniVoice backbone as ONE fused ONNX graph.

Why not reuse onnx-community/OmniVoice-Onnx: its llm_decoder is built by
onnxruntime-genai and uses com.microsoft::GroupQueryAttention, which is causal.
OmniVoice's attention is bidirectional (docs/onnx-reuse-audit.md §1), so the
mask has to be a real graph input we control.

  graph: omnivoice_lm.onnx
    in  input_ids      int64 [B, 8, S]
        audio_mask     bool  [B, S]
        attention_mask bool  [B, 1, S, S]      True = attend
        position_ids   int64 [B, S]
    out logits         float [B, 8, S, 1025]

  python scripts/export_onnx.py --precision fp32
  python scripts/export_onnx.py --precision int4      # needs fp32 exported first
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AUDIO_MASK_ID, AUDIO_VOCAB_SIZE, HIDDEN_SIZE, MODELS, NUM_CODEBOOKS,
    UPSTREAM_DIR,
)

OPSET = 20          # ORT 1.22 on Android must be able to load this


class OmniVoiceLMWrapper(nn.Module):
    """Flattens OmniVoice.forward into something torch.onnx.export can trace.

    The only non-obvious part is the mask: OmniVoice hands transformers a 4-D
    *boolean* mask, which SDPA wants as an additive float bias. We do that
    conversion inside the graph so the bool mask stays a live input and the
    exporter cannot constant-fold the attention pattern away.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.llm = model.llm
        self.audio_embeddings = model.audio_embeddings
        self.audio_heads = model.audio_heads
        self.register_buffer("codebook_layer_offsets",
                             model.codebook_layer_offsets.clone(), persistent=False)
        self.num_codebooks = model.config.num_audio_codebook
        self.audio_vocab_size = model.config.audio_vocab_size

    def forward(self, input_ids, audio_mask, attention_mask, position_ids):
        # --- embedding fusion (omnivoice.py:_prepare_embed_inputs) ---
        text_embeds = self.llm.get_input_embeddings()(input_ids[:, 0, :])
        shifted = (input_ids * audio_mask.unsqueeze(1).to(input_ids.dtype)) \
            + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted).sum(dim=1)
        inputs_embeds = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

        # --- bool mask -> additive bias, in-graph ---
        dtype = inputs_embeds.dtype
        bias = torch.zeros_like(attention_mask, dtype=dtype)
        bias = bias.masked_fill(~attention_mask, torch.finfo(dtype).min)

        hidden = self.llm(inputs_embeds=inputs_embeds, attention_mask=bias,
                          position_ids=position_ids, use_cache=False,
                          return_dict=True)[0]

        # --- audio heads (omnivoice.py:forward) ---
        b, s, _ = hidden.shape
        logits = self.audio_heads(hidden)
        return logits.view(b, s, self.num_codebooks, self.audio_vocab_size) \
                     .permute(0, 2, 1, 3)


def dummy_inputs(batch: int = 1, seq: int = 24, device="cpu"):
    ids = torch.randint(0, 1000, (batch, NUM_CODEBOOKS, seq), dtype=torch.long, device=device)
    am = torch.zeros(batch, seq, dtype=torch.bool, device=device)
    am[:, seq // 2:] = True
    ids[:, :, seq // 2:] = AUDIO_MASK_ID
    attn = torch.ones(batch, 1, seq, seq, dtype=torch.bool, device=device)
    pos = torch.arange(seq, device=device).unsqueeze(0).expand(batch, seq).contiguous()
    return ids, am, attn, pos


def export_fp32(args) -> Path:
    from omnivoice.models.omnivoice import OmniVoice

    outdir = Path(args.out or MODELS / "onnx" / "fp32")
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "omnivoice_lm.onnx"

    print(f"loading {UPSTREAM_DIR} (fp32, cpu) ...")
    model = OmniVoice.from_pretrained(str(UPSTREAM_DIR), device_map="cpu",
                                      dtype=torch.float32, train=True)
    model.eval()
    model.llm.set_attn_implementation("sdpa")
    wrapper = OmniVoiceLMWrapper(model).eval()

    args_t = dummy_inputs(seq=args.trace_seq)
    with torch.no_grad():
        ref = wrapper(*args_t)
    print(f"traced forward OK → logits {tuple(ref.shape)}")

    # torch.onnx.export writes one sidecar per tensor; stage it in a scratch dir
    # and re-save with a single .onnx.data so the phone copies two files, not 400.
    staging = outdir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper, args_t, str(staging / "model.onnx"),
        input_names=["input_ids", "audio_mask", "attention_mask", "position_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 2: "seq"},
            "audio_mask": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 2: "seq", 3: "seq"},
            "position_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 2: "seq"},
        },
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )
    print(f"exported in {time.perf_counter() - t0:.1f}s, consolidating external data ...")

    import onnx
    m = onnx.load(str(staging / "model.onnx"))
    for f in (target, target.with_suffix(".onnx.data")):
        f.unlink(missing_ok=True)
    onnx.save_model(m, str(target), save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=target.name + ".data", size_threshold=1024)
    shutil.rmtree(staging)

    _report(target)
    _write_manifest(outdir, "fp32")
    return target


def quantize_int4(args) -> Path:
    """Weight-only int4, block-wise RTN.

    MatMul -> com.microsoft::MatMulNBits and Gather -> GatherBlockQuantized. The
    Gather part is not optional: the tied text embedding is 151676 x 1024, i.e.
    621 MB of the 2.45 GB fp32 graph, so leaving it alone would cap the model at
    ~840 MB no matter what happens to the transformer stack.
    """
    import onnx
    from onnxruntime.quantization import matmul_nbits_quantizer as qn

    src = Path(args.source or MODELS / "onnx" / "fp32" / "omnivoice_lm.onnx")
    if not src.exists():
        sys.exit(f"{src} not found — run `export_onnx.py --precision fp32` first.")
    outdir = Path(args.out or MODELS / "onnx" / "int4")
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "omnivoice_lm.onnx"

    op_types = tuple(args.op_types)
    print(f"quantizing {src} -> int4 (block {args.block_size}, "
          f"{'symmetric' if args.symmetric else 'asymmetric'}, "
          f"accuracy_level={args.accuracy_level}, ops={op_types}, "
          f"excluded={args.exclude}) ...")
    m = onnx.load(str(src), load_external_data=True)
    quant = qn.MatMulNBitsQuantizer(
        m, bits=4, block_size=args.block_size, is_symmetric=args.symmetric,
        accuracy_level=args.accuracy_level,
        op_types_to_quantize=op_types,
        quant_axes=(("Gather", 1),),
        nodes_to_exclude=args.exclude or None,
    )
    t0 = time.perf_counter()
    quant.process()
    print(f"quantized in {time.perf_counter() - t0:.1f}s")

    out_model = quant.model.model if hasattr(quant.model, "model") else quant.model
    if not isinstance(out_model, onnx.ModelProto):
        out_model = onnx.load(str(out_model)) if isinstance(out_model, (str, Path)) else out_model
    for f in (target, target.with_suffix(".onnx.data")):
        f.unlink(missing_ok=True)
    onnx.save_model(out_model, str(target), save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=target.name + ".data", size_threshold=1024)
    _report(target)
    _write_manifest(outdir, "int4")
    return target


def _report(path: Path) -> None:
    import collections
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    hist = collections.Counter((n.domain or "ai.onnx") + "::" + n.op_type for n in m.graph.node)
    total = sum(p.stat().st_size for p in
                [path, path.with_suffix(".onnx.data")] if p.exists())
    print(f"\n  {path}  ({total / 1e6:.1f} MB, {len(m.graph.node)} nodes, opset "
          f"{[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]})")
    for k, v in hist.most_common(10):
        print(f"    {v:6d}  {k}")
    live = {i.name for i in m.graph.input}
    for need in ("input_ids", "audio_mask", "attention_mask", "position_ids"):
        mark = "OK " if need in live else "BAD"
        print(f"    {mark} input `{need}` present")
    # the mask must still reach the attention subgraphs
    consumers = sum(1 for n in m.graph.node if "attention_mask" in n.input)
    print(f"    attention_mask has {consumers} direct consumer node(s) "
          f"{'OK' if consumers else '— BAD: the exporter folded the mask away'}")


def _write_manifest(outdir: Path, precision: str) -> None:
    import hashlib

    files = {}
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.name != "manifest.json":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            files[p.name] = {"bytes": p.stat().st_size, "sha256": h}
    (outdir / "manifest.json").write_text(json.dumps({
        "model": "k2-fsa/OmniVoice",
        "graph": "omnivoice_lm",
        "precision": precision,
        "opset": OPSET,
        "num_codebooks": NUM_CODEBOOKS,
        "audio_vocab_size": AUDIO_VOCAB_SIZE,
        "audio_mask_id": AUDIO_MASK_ID,
        "hidden_size": HIDDEN_SIZE,
        "inputs": {
            "input_ids": "int64 [batch, 8, seq]",
            "audio_mask": "bool [batch, seq]",
            "attention_mask": "bool [batch, 1, seq, seq] (True = attend)",
            "position_ids": "int64 [batch, seq]",
        },
        "outputs": {"logits": "float32 [batch, 8, seq, 1025]"},
        "files": files,
    }, indent=2))
    print(f"    manifest → {outdir / 'manifest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", choices=["fp32", "int4"], default="fp32")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-seq", type=int, default=24)
    ap.add_argument("--source", default=None, help="int4: fp32 graph to quantize")
    ap.add_argument("--block-size", type=int, default=32,
                    help="32 is the measured sweet spot; 128 falls outside the "
                         "fp32 run-to-run band (docs/benchmark.md)")
    ap.add_argument("--accuracy-level", type=int, default=4,
                    help="MatMulNBits accuracy_level; 4 = int8 compute (fastest on ARM)")
    ap.add_argument("--op-types", nargs="*", default=["MatMul", "Gather"],
                    help="op types to quantize; Gather covers the tied text embedding")
    ap.add_argument("--exclude", nargs="*", default=["/audio_heads/MatMul"],
                    help="node names to keep at full precision; the audio head "
                         "sits directly on the logits and must not be int4")
    ap.add_argument("--symmetric", action="store_true",
                    help="symmetric RTN (measurably worse; kept for reproduction)")
    args = ap.parse_args()
    (export_fp32 if args.precision == "fp32" else quantize_int4)(args)


if __name__ == "__main__":
    main()
