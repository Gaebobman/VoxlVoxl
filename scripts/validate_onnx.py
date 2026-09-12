#!/usr/bin/env python3
"""Phase 2 — verify the ONNX artifacts against PyTorch.

  python scripts/validate_onnx.py --stage bidir   # the mask is live and non-causal
  python scripts/validate_onnx.py --stage lm      # backbone logits vs PyTorch fp32
  python scripts/validate_onnx.py --stage codec   # reused Higgs graphs vs transformers
  python scripts/validate_onnx.py --stage all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AUDIO_MASK_ID, HIGGS_ONNX_DIR, MODELS, NUM_CODEBOOKS, OUT, SR_16K, SR_24K,
    UPSTREAM_DIR, read_wav, write_wav,
)

GOLDEN = OUT / "golden"


def _ort(path: Path, threads: int = 0):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _golden_prompt():
    if not (GOLDEN / "prompt.npz").exists():
        sys.exit("out/golden/prompt.npz missing — run `reference_infer.py clone` first.")
    d = np.load(GOLDEN / "prompt.npz")
    return (d["input_ids"].astype(np.int64), d["audio_mask"].astype(bool),
            d["ref_codes"].astype(np.int64))


def _feed(ids, am):
    b, _, s = ids.shape
    return {
        "input_ids": ids,
        "audio_mask": am,
        "attention_mask": np.ones((b, 1, s, s), dtype=bool),
        "position_ids": np.arange(s, dtype=np.int64)[None, :].repeat(b, 0),
    }


def _stats(name, a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mad = float(np.abs(a - b).max())
    rel = mad / max(float(np.abs(b).max()), 1e-9)
    cos = float((a.ravel() @ b.ravel()) /
                (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12))
    print(f"    {name:28s} max|Δ| {mad:.4e}   rel {rel:.2e}   cos {cos:.8f}")
    return mad, rel, cos


# ---------------------------------------------------------------------------

def stage_bidir(args) -> bool:
    """The whole port rests on the exported attention being non-causal. Prove it:
    change a token near the END of the sequence and check that the logits at the
    START move. Under causal attention they cannot."""
    print("\n[bidir] exported attention is bidirectional")
    path = Path(args.lm or MODELS / "onnx" / "fp32" / "omnivoice_lm.onnx")
    sess = _ort(path, args.threads)
    ids, am, _ = _golden_prompt()
    S = ids.shape[2]

    base = sess.run(["logits"], _feed(ids, am))[0]
    ids2 = ids.copy()
    ids2[0, :, S - 1] = (ids2[0, :, S - 1] + 7) % 1024      # perturb the last frame
    pert = sess.run(["logits"], _feed(ids2, am))[0]

    d = np.abs(base - pert).max(axis=(0, 1, 3))              # per position
    early = float(d[:S // 4].max())
    late = float(d[-1])
    print(f"    perturbed position {S - 1}; max|Δ| over first {S // 4} positions = {early:.4e}")
    print(f"    (same metric at the perturbed position itself      = {late:.4e})")
    ok = early > 1e-3
    print(f"    {'PASS' if ok else 'FAIL'} — earlier positions "
          f"{'do' if ok else 'do NOT'} see the later token")

    # and that the mask input is actually honoured: mask out the second half and
    # the first half must change too (it loses the keys it was attending to).
    feed = _feed(ids, am)
    feed["attention_mask"] = feed["attention_mask"].copy()
    feed["attention_mask"][:, :, :, S // 2:] = False
    masked = sess.run(["logits"], feed)[0]
    dm = float(np.abs(base[:, :, :S // 2] - masked[:, :, :S // 2]).max())
    ok2 = dm > 1e-3
    print(f"    masking keys [{S // 2}:] changes earlier logits by {dm:.4e} — "
          f"{'PASS' if ok2 else 'FAIL: attention_mask is being ignored'}")
    return ok and ok2


def stage_lm(args) -> bool:
    """ONNX logits vs PyTorch fp32 on the exact golden prompt."""
    print("\n[lm] backbone logits: ONNX fp32 vs PyTorch fp32")
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    ids, am, _ = _golden_prompt()
    S = ids.shape[2]

    path = Path(args.lm or MODELS / "onnx" / "fp32" / "omnivoice_lm.onnx")
    sess = _ort(path, args.threads)
    t0 = time.perf_counter()
    got = sess.run(["logits"], _feed(ids, am))[0]
    t_onnx = time.perf_counter() - t0
    print(f"    ONNX forward(S={S}) {t_onnx * 1000:.0f} ms  → {got.shape}")

    model = OmniVoice.from_pretrained(str(UPSTREAM_DIR), device_map="cpu",
                                      dtype=torch.float32, train=True).eval()
    model.llm.set_attn_implementation("sdpa")
    with torch.no_grad():
        t0 = time.perf_counter()
        want = model(
            input_ids=torch.from_numpy(ids),
            audio_mask=torch.from_numpy(am),
            attention_mask=torch.ones(1, 1, S, S, dtype=torch.bool),
        ).logits.numpy()
        t_pt = time.perf_counter() - t0
    print(f"    PyTorch forward(S={S}) {t_pt * 1000:.0f} ms  → {want.shape}")

    mad, rel, cos = _stats("logits", got, want)
    gen = am[0]
    agree = float((got[0, :, gen].argmax(-1) == want[0, :, gen].argmax(-1)).mean())
    print(f"    {'argmax agreement (audio positions)':28s} {agree * 100:.3f} %")
    ok = rel < 1e-3 and cos > 0.9999 and agree > 0.999
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def stage_codec(args) -> bool:
    """The reused Higgs graphs vs transformers' HiggsAudioV2TokenizerModel."""
    print("\n[codec] Higgs Audio V2: reused ONNX vs transformers")
    import torch
    from transformers import HiggsAudioV2TokenizerModel

    wav_path = args.wav or "sample/reference.wav"
    x24, sr = read_wav(wav_path)
    assert sr == SR_24K, f"{wav_path} is {sr} Hz, expected {SR_24K}"
    import torchaudio
    x16 = torchaudio.functional.resample(torch.from_numpy(x24), SR_24K, SR_16K).numpy()

    tok = HiggsAudioV2TokenizerModel.from_pretrained(
        str(UPSTREAM_DIR / "audio_tokenizer"), device_map="cpu").eval()
    hop = tok.config.acoustic_model_config.hop_length
    n = (len(x24) // hop) * hop
    x24 = x24[:n]
    with torch.no_grad():
        want_codes = tok.encode(torch.from_numpy(x24)[None, None, :]).audio_codes[0].numpy()
        want_wav = tok.decode(torch.from_numpy(want_codes)[None]).audio_values[0].numpy()
    print(f"    torch  codes {want_codes.shape}  wav {want_wav.shape}")

    enc_a = _ort(HIGGS_ONNX_DIR / "acoustic_encoder.onnx", args.threads)
    enc_s = _ort(HIGGS_ONNX_DIR / "semantic_encoder.onnx", args.threads)
    quant = _ort(HIGGS_ONNX_DIR / "quantizer_encoder.onnx", args.threads)
    dec = _ort(HIGGS_ONNX_DIR / "higgs_decoder.onnx", args.threads)

    af = enc_a.run(["acoustic_features"], {"waveform_24k": x24[None, None, :]})[0]
    sf = enc_s.run(["semantic_features"], {"waveform_16k": x16[None, :].astype(np.float32)})[0]
    T = min(af.shape[2], sf.shape[2])
    got_codes = quant.run(["codes"], {"acoustic_features": af[:, :, :T],
                                      "semantic_features": sf[:, :, :T]})[0]
    got_codes = got_codes[:, 0, :]
    print(f"    onnx   codes {got_codes.shape}  (acoustic T={af.shape[2]}, semantic T={sf.shape[2]})")

    Tm = min(got_codes.shape[1], want_codes.shape[1])
    match = float((got_codes[:, :Tm] == want_codes[:, :Tm]).mean())
    per_cb = [(got_codes[c, :Tm] == want_codes[c, :Tm]).mean() for c in range(NUM_CODEBOOKS)]
    print(f"    code agreement {match * 100:.2f} %   per-codebook "
          + " ".join(f"{v * 100:.0f}" for v in per_cb))

    got_wav = dec.run(["waveform_24k"], {"codes": want_codes[:, None, :].astype(np.int64)})[0].squeeze()
    m = min(len(got_wav), len(want_wav.squeeze()))
    mad, rel, cos = _stats("decoder waveform", got_wav[:m], want_wav.squeeze()[:m])
    write_wav(OUT / "codec_roundtrip_onnx.wav", got_wav, SR_24K)
    write_wav(OUT / "codec_roundtrip_torch.wav", want_wav.squeeze(), SR_24K)

    ok = match > 0.98 and cos > 0.99
    print(f"    {'PASS' if ok else 'FAIL'}  (wrote out/codec_roundtrip_{{onnx,torch}}.wav)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append",
                    choices=["bidir", "lm", "codec", "all"], default=None)
    ap.add_argument("--lm", default=None, help="path to omnivoice_lm.onnx")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["bidir", "lm", "codec"]

    results = {}
    for s in stages:
        results[s] = {"bidir": stage_bidir, "lm": stage_lm, "codec": stage_codec}[s](args)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
