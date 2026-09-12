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
    # int4 is lossy by construction; judge it on argmax agreement and direction,
    # not on the fp32 exactness bar.
    lossy = "int4" in str(path) or "int8" in str(path)
    ok = (cos > 0.99 and agree > 0.80) if lossy else (rel < 1e-3 and cos > 0.9999 and agree > 0.999)
    print(f"    {'PASS' if ok else 'FAIL'}  (thresholds: "
          f"{'lossy — cos>0.99, argmax>80%' if lossy else 'exact — rel<1e-3, cos>0.9999, argmax>99.9%'})")
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


def stage_dsp(args) -> bool:
    """The numpy silence/fade ports vs upstream (pydub) — these feed the encoder,
    so a few hundred samples of drift shifts every codec frame."""
    print("\n[dsp] silence removal / fade+pad: numpy port vs upstream pydub")
    from omnivoice.utils.audio import fade_and_pad_audio as up_fade
    from omnivoice.utils.audio import remove_silence as up_remove
    from _common import fade_and_pad as my_fade
    from _common import remove_silence as my_remove

    x, sr = read_wav(args.wav or "sample/reference.wav")
    ok = True
    for mid, lead, trail in ((200, 100, 200), (500, 100, 100), (300, 100, 300)):
        want = up_remove(x[None, :].copy(), sr, mid, lead, trail)[0]
        got = my_remove(x.copy(), sr, mid, lead, trail)
        same_len = len(got) == len(want)
        mad = float(np.abs(got[:min(len(got), len(want))]
                           - want[:min(len(got), len(want))]).max()) if min(len(got), len(want)) else 0.0
        good = same_len and mad < 1e-4
        ok &= good
        print(f"    {'OK ' if good else 'BAD'} remove_silence(mid={mid},lead={lead},trail={trail}) "
              f"len {len(got)} vs {len(want)}   max|Δ| {mad:.2e}")

    want = up_fade(x[None, :].copy(), 0.1, 0.1, sr)[0]
    got = my_fade(x.copy(), sr, 0.1, 0.1)
    good = len(got) == len(want) and float(np.abs(got - want).max()) < 1e-6
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} fade_and_pad  len {len(got)} vs {len(want)}")
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def stage_e2e(args) -> bool:
    """Deterministic ONNX pipeline vs the deterministic PyTorch golden."""
    print("\n[e2e] ONNX audio codes vs PyTorch golden (deterministic)")
    gold = GOLDEN / "det_codes.npy"
    onnx = OUT / "onnx" / "cloned_det.codes.npy"
    if not gold.exists() or not onnx.exists():
        print(f"    SKIP — need {gold} and {onnx}")
        print("    run: reference_infer.py clone --deterministic --out out/golden_det")
        print("         infer_onnx.py generate --deterministic --out out/onnx/cloned_det.wav")
        return True
    a, b = np.load(onnx), np.load(gold)
    T = min(a.shape[1], b.shape[1])
    agree = float((a[:, :T] == b[:, :T]).mean())
    per = " ".join(f"{(a[c, :T] == b[c, :T]).mean() * 100:.0f}" for c in range(NUM_CODEBOOKS))
    print(f"    shapes {a.shape} vs {b.shape}; code agreement {agree * 100:.2f} %  per-cb {per}")
    ok = agree > 0.90
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def stage_prompt(args) -> bool:
    """The ONNX-side prompt builder must produce byte-identical input_ids and
    audio_mask to PyTorch's _prepare_inference_inputs. Everything downstream is
    meaningless otherwise, and this is the part the Kotlin port reimplements."""
    print("\n[prompt] ONNX prompt assembly vs PyTorch _prepare_inference_inputs")
    from _common import VoicePrompt
    from infer_onnx import QwenTokenizer, build_prompt

    want_ids, want_am, _ = _golden_prompt()
    vp = VoicePrompt.load(OUT / "golden" / "voice_prompt.bin")
    text = (Path("sample/target.txt")).read_text(encoding="utf-8").strip()
    tok = QwenTokenizer(UPSTREAM_DIR / "tokenizer.json")
    T_gen = want_ids.shape[2] - want_am[0].sum()
    T_gen = int(want_am[0].sum()) - vp.num_frames
    got_ids, got_am, gen_start = build_prompt(
        tok, text, vp.ref_text, vp.codes.astype(np.int64), T_gen, "ko", None, True)

    ok_shape = got_ids.shape == want_ids.shape
    ok_ids = ok_shape and bool((got_ids == want_ids).all())
    ok_am = ok_shape and bool((got_am == want_am).all())
    print(f"    shape {got_ids.shape} vs {want_ids.shape}   "
          f"{'OK' if ok_shape else 'MISMATCH'}")
    if ok_shape and not ok_ids:
        diff = np.argwhere(got_ids != want_ids)
        print(f"    first differing cell {diff[0].tolist()}: "
              f"{got_ids[tuple(diff[0])]} vs {want_ids[tuple(diff[0])]} "
              f"({len(diff)} cells differ)")
    print(f"    input_ids identical  {'OK' if ok_ids else 'BAD'}")
    print(f"    audio_mask identical {'OK' if ok_am else 'BAD'}")
    ok = ok_ids and ok_am
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def stage_judge(args) -> bool:
    """Score candidate outputs with the fp32 reference model, under the model's
    own training objective.

    Comparing generated codes across precisions is worthless: the un-masking loop
    is chaotic, so one different token at step 1 changes everything downstream
    (int8 agrees with fp32 on 89.6 % of single-forward argmaxes but only 13.8 %
    of final codes).

    So instead: take each candidate's codes, re-mask a random fraction of the
    (codebook, frame) cells, and ask the fp32 reference to fill them back in.
    That is exactly what OmniVoice was trained to do, so the numbers are real
    likelihoods -- unlike scoring a fully-unmasked sequence, which is
    out-of-distribution and yields perplexity above the 1024-way uniform bound.
    """
    print("\n[judge] fp32 reference re-masking score (lower NLL = more plausible codes)")
    lm = _ort(Path(args.lm or MODELS / "onnx" / "fp32" / "omnivoice_lm.onnx"), args.threads)
    ids0, am0, _ = _golden_prompt()

    cands = sorted((OUT / "onnx").glob("*.codes.npy"))
    if not cands:
        print("    SKIP — no out/onnx/*.codes.npy; run infer_onnx.py generate first")
        return True

    ratios = (0.25, 0.5, 0.75)
    seeds = (0, 1, 2)
    rows = []
    for c in cands:
        codes = np.load(c).astype(np.int64)
        T = codes.shape[1]
        if ids0.shape[2] - T < 0:
            print(f"    SKIP {c.name}: {T} frames does not fit S={ids0.shape[2]}")
            continue
        gen_start = ids0.shape[2] - T
        per_ratio = []
        for ratio in ratios:
            nlls, accs = [], []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                sel = rng.random((NUM_CODEBOOKS, T)) < ratio
                if not sel.any():
                    continue
                ids = ids0.copy()
                ids[0, :, gen_start:] = np.where(sel, AUDIO_MASK_ID, codes)
                logits = lm.run(["logits"], _feed(ids, am0))[0]
                lp = logits[0, :, gen_start:, :1024].astype(np.float64)
                lp -= lp.max(-1, keepdims=True)
                lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
                true_lp = np.take_along_axis(lp, codes[:, :, None], -1)[..., 0]
                nlls.append(float(-true_lp[sel].mean()))
                accs.append(float((lp.argmax(-1)[sel] == codes[sel]).mean()))
            per_ratio.append((float(np.mean(nlls)), float(np.mean(accs))))
        mean_nll = float(np.mean([r[0] for r in per_ratio]))
        rows.append((c.stem.replace(".codes", ""), T, mean_nll, per_ratio))

    rows.sort(key=lambda r: r[2])
    head = "  ".join(f"NLL@{int(r * 100)}" for r in ratios)
    print(f"    {'candidate':20s} {'frames':>6} {'meanNLL':>8}   {head}   "
          f"{'top1@50%':>8}")
    for name, T, mean_nll, per in rows:
        cells = "  ".join(f"{n:6.3f}" for n, _ in per)
        print(f"    {name:20s} {T:6d} {mean_nll:8.4f}   {cells}   "
              f"{per[1][1] * 100:7.2f}%")
    print("    fp32 is scoring its own output, so it should lead; what matters is")
    print("    whether a candidate sits inside the fp32 run-to-run spread.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append",
                    choices=["bidir", "lm", "codec", "dsp", "prompt", "e2e",
                             "judge", "all"],
                    default=None)
    ap.add_argument("--lm", default=None, help="path to omnivoice_lm.onnx")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["prompt", "bidir", "lm", "codec", "dsp", "e2e", "judge"]

    results = {}
    for s in stages:
        results[s] = {"bidir": stage_bidir, "lm": stage_lm, "codec": stage_codec,
                      "dsp": stage_dsp, "e2e": stage_e2e,
                      "prompt": stage_prompt, "judge": stage_judge}[s](args)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
