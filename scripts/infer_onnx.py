#!/usr/bin/env python3
"""Phase 2 — the ONNX reference pipeline. No torch, no transformers.

This file is the SPECIFICATION for the Kotlin port: every function here gets a
1:1 counterpart in android/OmniVoicePoC. Keep it dependency-free beyond
onnxruntime + numpy (+ `tokenizers` for BPE, which the device side reimplements).

  # enrollment — reference WAV -> voice_prompt.bin (encoders needed)
  python scripts/infer_onnx.py enroll --ref-audio sample/reference.wav \
      --ref-text "$(cat sample/reference.txt)" --out out/voices/me.bin

  # generation — voice_prompt.bin + text -> WAV (no encoders needed)
  python scripts/infer_onnx.py generate --voice-prompt out/voices/me.bin \
      --text "$(cat sample/target.txt)" --out out/onnx/cloned.wav
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from bisect import bisect_left
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AUDIO_MASK_ID, AUDIO_VOCAB_SIZE, FRAME_RATE, HIGGS_ONNX_DIR, MODELS,
    NUM_CODEBOOKS, OUT, SR_16K, SR_24K, TOK_DENOISE, TOK_INSTRUCT_END,
    TOK_INSTRUCT_START, TOK_LANG_END, TOK_LANG_START, TOK_TEXT_END,
    TOK_TEXT_START, UPSTREAM_DIR, VoicePrompt, fade_and_pad, read_wav,
    remove_silence, write_wav,
)

CODEBOOK_WEIGHTS = [8, 8, 6, 6, 4, 4, 2, 2]


# ===========================================================================
# Generation config  (upstream OmniVoiceGenerationConfig defaults)
# ===========================================================================

class GenConfig:
    def __init__(self, **kw):
        self.num_step = kw.get("num_step", 32)
        self.guidance_scale = kw.get("guidance_scale", 2.0)
        self.t_shift = kw.get("t_shift", 0.1)
        self.layer_penalty_factor = kw.get("layer_penalty_factor", 5.0)
        self.position_temperature = kw.get("position_temperature", 5.0)
        self.class_temperature = kw.get("class_temperature", 0.0)
        # Confidence-aware parallel decoding (Fast-dLLM, arXiv:2505.22618):
        # instead of committing exactly k cells per step, commit every cell the
        # model is already sure about. The fixed schedule is kept as a FLOOR so
        # the loop can never stall, and the loop exits as soon as nothing is
        # masked — which is the actual saving.
        self.confidence_threshold = kw.get("confidence_threshold", 0.0)
        self.denoise = kw.get("denoise", True)
        self.postprocess_output = kw.get("postprocess_output", True)
        self.pad_duration = kw.get("pad_duration", 0.1)
        self.fade_duration = kw.get("fade_duration", 0.1)


# ===========================================================================
# Duration estimator — port of omnivoice/utils/duration.py (Apache-2.0)
# ===========================================================================

_W = {"cjk": 3.0, "hangul": 2.5, "kana": 2.2, "ethiopic": 3.0, "yi": 3.0,
      "indic": 1.8, "thai_lao": 1.5, "khmer_myanmar": 1.8, "arabic": 1.5,
      "hebrew": 1.5, "latin": 1.0, "cyrillic": 1.0, "greek": 1.0,
      "armenian": 1.0, "georgian": 1.0, "punctuation": 0.5, "space": 0.2,
      "digit": 3.5, "mark": 0.0, "default": 1.0}

_RANGES = [
    (0x02AF, "latin"), (0x03FF, "greek"), (0x052F, "cyrillic"), (0x058F, "armenian"),
    (0x05FF, "hebrew"), (0x077F, "arabic"), (0x089F, "arabic"), (0x08FF, "arabic"),
    (0x097F, "indic"), (0x09FF, "indic"), (0x0A7F, "indic"), (0x0AFF, "indic"),
    (0x0B7F, "indic"), (0x0BFF, "indic"), (0x0C7F, "indic"), (0x0CFF, "indic"),
    (0x0D7F, "indic"), (0x0DFF, "indic"), (0x0EFF, "thai_lao"), (0x0FFF, "indic"),
    (0x109F, "khmer_myanmar"), (0x10FF, "georgian"), (0x11FF, "hangul"),
    (0x137F, "ethiopic"), (0x139F, "ethiopic"), (0x13FF, "default"), (0x167F, "default"),
    (0x169F, "default"), (0x16FF, "default"), (0x171F, "default"), (0x173F, "default"),
    (0x175F, "default"), (0x177F, "default"), (0x17FF, "khmer_myanmar"),
    (0x18AF, "default"), (0x18FF, "default"), (0x194F, "indic"), (0x19DF, "indic"),
    (0x19FF, "khmer_myanmar"), (0x1A1F, "indic"), (0x1AAF, "indic"), (0x1B7F, "indic"),
    (0x1BBF, "indic"), (0x1BFF, "indic"), (0x1C4F, "indic"), (0x1C7F, "indic"),
    (0x1C8F, "cyrillic"), (0x1CBF, "georgian"), (0x1CCF, "indic"), (0x1CFF, "indic"),
    (0x1D7F, "latin"), (0x1DBF, "latin"), (0x1DFF, "default"), (0x1EFF, "latin"),
    (0x309F, "kana"), (0x30FF, "kana"), (0x312F, "cjk"), (0x318F, "hangul"),
    (0x9FFF, "cjk"), (0xA4CF, "yi"), (0xA4FF, "default"), (0xA63F, "default"),
    (0xA69F, "cyrillic"), (0xA6FF, "default"), (0xA7FF, "latin"), (0xA82F, "indic"),
    (0xA87F, "default"), (0xA8DF, "indic"), (0xA8FF, "indic"), (0xA92F, "indic"),
    (0xA95F, "indic"), (0xA97F, "hangul"), (0xA9DF, "indic"), (0xA9FF, "khmer_myanmar"),
    (0xAA5F, "indic"), (0xAA7F, "khmer_myanmar"), (0xAADF, "indic"), (0xAAFF, "indic"),
    (0xAB2F, "ethiopic"), (0xAB6F, "latin"), (0xABBF, "default"), (0xABFF, "indic"),
    (0xD7AF, "hangul"), (0xFAFF, "cjk"), (0xFDFF, "arabic"), (0xFE6F, "default"),
    (0xFEFF, "arabic"), (0xFFEF, "latin"),
]
_BREAKS = [r[0] for r in _RANGES]


def _char_weight(ch: str) -> float:
    code = ord(ch)
    if (65 <= code <= 90) or (97 <= code <= 122):
        return _W["latin"]
    if code == 32:
        return _W["space"]
    if code == 0x0640:
        return _W["mark"]
    cat = unicodedata.category(ch)
    if cat.startswith("M"):
        return _W["mark"]
    if cat.startswith("P") or cat.startswith("S"):
        return _W["punctuation"]
    if cat.startswith("Z"):
        return _W["space"]
    if cat.startswith("N"):
        return _W["digit"]
    i = bisect_left(_BREAKS, code)
    if i < len(_RANGES):
        return _W.get(_RANGES[i][1], _W["default"])
    if code > 0x20000:
        return _W["cjk"]
    return _W["default"]


def text_weight(s: str) -> float:
    return sum(_char_weight(c) for c in s)


def estimate_target_frames(text: str, ref_text: str, ref_frames: int,
                           speed: float = 1.0, low_threshold: float = 50.0,
                           boost_strength: float = 3.0) -> int:
    if not ref_text or ref_frames <= 0:
        ref_text, ref_frames = "Nice to meet you.", 25
    rw = text_weight(ref_text)
    if rw == 0:
        return 1
    est = text_weight(text) / (rw / ref_frames)
    if speed > 0 and speed != 1.0:
        est /= speed
    if est < low_threshold:
        est = low_threshold * (est / low_threshold) ** (1.0 / boost_strength)
    return max(1, int(est))


# ===========================================================================
# Text assembly — port of _combine_text / _prepare_inference_inputs
# ===========================================================================

import re  # noqa: E402

_CJK = r"[一-鿿]"
_NONVERBAL = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]")


def combine_text(text: str, ref_text: str | None = None) -> str:
    full = (ref_text.strip() + " " + text.strip()) if ref_text else text.strip()
    full = re.sub(r"[\r\n]+", "", full)
    full = full.replace("（", "(").replace("）", ")")
    full = re.sub(r"[ \t]+", " ", full)
    return re.sub(rf"(?<={_CJK})\s+|\s+(?={_CJK})", "", full)


def add_punctuation(s: str) -> str:
    s = s.strip()
    return s if (s and s[-1] in ".。!！?？,，;；:：") else s + "."


class QwenTokenizer:
    """PC side uses HF `tokenizers`; the device side reimplements byte-level BPE
    from the same tokenizer.json. Fixtures generated here pin the two together."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s, add_special_tokens=False).ids

    def encode_with_nonverbal(self, text: str) -> list[int]:
        out, last = [], 0
        for m in _NONVERBAL.finditer(text):
            if m.start() > last:
                out += self.encode(text[last:m.start()])
            out += self.encode(m.group())
            last = m.end()
        if last < len(text):
            out += self.encode(text[last:])
        return out or self.encode(text)


def build_prompt(tok: QwenTokenizer, text: str, ref_text: str | None,
                 ref_codes: np.ndarray | None, num_target: int,
                 lang: str | None, instruct: str | None, denoise: bool):
    """Returns (input_ids [1,8,S] int64, audio_mask [1,S] bool, gen_start int)."""
    style = (TOK_DENOISE if (denoise and ref_codes is not None) else "")
    style += f"{TOK_LANG_START}{lang or 'None'}{TOK_LANG_END}"
    style += f"{TOK_INSTRUCT_START}{instruct or 'None'}{TOK_INSTRUCT_END}"
    ids = tok.encode(style)
    ids += tok.encode_with_nonverbal(
        TOK_TEXT_START + combine_text(text, ref_text) + TOK_TEXT_END)

    n_text = len(ids)
    t_ref = ref_codes.shape[1] if ref_codes is not None else 0
    S = n_text + t_ref + num_target

    input_ids = np.zeros((1, NUM_CODEBOOKS, S), dtype=np.int64)
    input_ids[0, :, :n_text] = np.asarray(ids, dtype=np.int64)[None, :]
    if t_ref:
        input_ids[0, :, n_text:n_text + t_ref] = ref_codes.astype(np.int64)
    input_ids[0, :, n_text + t_ref:] = AUDIO_MASK_ID

    audio_mask = np.zeros((1, S), dtype=bool)
    audio_mask[0, n_text:] = True          # reference codes included — upstream does this
    return input_ids, audio_mask, n_text + t_ref


# ===========================================================================
# Decoding — port of _generate_iterative / _predict_tokens_with_scoring
# ===========================================================================

def time_steps(num_step: int, t_shift: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_step + 1)
    return t_shift * t / (1.0 + (t_shift - 1.0) * t)


def unmask_schedule(num_step: int, t_shift: float, total: int) -> list[int]:
    t = time_steps(num_step, t_shift)
    sched, rem = [], total
    for i in range(num_step):
        n = rem if i == num_step - 1 else min(int(np.ceil(total * (t[i + 1] - t[i]))), rem)
        sched.append(int(n))
        rem -= int(n)
    return sched


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _gumbel(x: np.ndarray, temperature: float, rng) -> np.ndarray:
    u = rng.random(x.shape).astype(np.float32)
    return x / temperature + (-np.log(-np.log(u + 1e-10) + 1e-10))


def _filter_top_k(logits: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    k = int(np.ceil(ratio * logits.shape[-1]))
    idx = np.argpartition(-logits, k - 1, axis=-1)[..., :k]
    out = np.full_like(logits, -np.inf)
    np.put_along_axis(out, idx, np.take_along_axis(logits, idx, -1), -1)
    return out


class Backbone:
    def __init__(self, path: Path, threads: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(str(path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.load_seconds = time.perf_counter() - t0
        self.calls = 0
        self.compute_seconds = 0.0

    def __call__(self, input_ids: np.ndarray, audio_mask: np.ndarray) -> np.ndarray:
        b, _, s = input_ids.shape
        feed = {
            "input_ids": input_ids,
            "audio_mask": audio_mask,
            "attention_mask": np.ones((b, 1, s, s), dtype=bool),
            "position_ids": np.arange(s, dtype=np.int64)[None, :].repeat(b, 0),
        }
        t0 = time.perf_counter()
        out = self.sess.run(["logits"], feed)[0]
        self.compute_seconds += time.perf_counter() - t0
        self.calls += 1
        return out


def generate_codes(lm: Backbone, input_ids: np.ndarray, audio_mask: np.ndarray,
                   gen_start: int, cfg: GenConfig, rng, progress=True) -> np.ndarray:
    """32-step confidence-ordered un-masking with classifier-free guidance.

    The conditional branch is the whole sequence; the unconditional branch is the
    trailing MASK block alone. Upstream batches them and pads the short one to the
    long one's length; we run them as two batch-1 calls instead, which is the same
    maths for less compute (S_c + T_gen instead of 2*S_c).
    """
    S = input_ids.shape[2]
    T = S - gen_start
    tokens = np.full((NUM_CODEBOOKS, T), AUDIO_MASK_ID, dtype=np.int64)
    sched = unmask_schedule(cfg.num_step, cfg.t_shift, T * NUM_CODEBOOKS)
    layer_ids = np.arange(NUM_CODEBOOKS, dtype=np.float32)[:, None]

    u_ids = input_ids[:, :, gen_start:].copy()
    u_mask = audio_mask[:, gen_start:].copy()

    for step, k in enumerate(sched):
        if k <= 0:
            continue
        c_logits = lm(input_ids, audio_mask)[0, :, gen_start:, :].astype(np.float32)
        if cfg.guidance_scale != 0:
            u_logits = lm(u_ids, u_mask)[0].astype(np.float32)
            c_lp, u_lp = _log_softmax(c_logits), _log_softmax(u_logits)
            log_probs = _log_softmax(c_lp + cfg.guidance_scale * (c_lp - u_lp))
        else:
            log_probs = _log_softmax(c_logits)
        log_probs[..., AUDIO_MASK_ID] = -np.inf

        if cfg.class_temperature > 0:
            pred = _gumbel(_filter_top_k(log_probs), cfg.class_temperature, rng).argmax(-1)
        else:
            pred = log_probs.argmax(-1)
        scores = log_probs.max(-1) - layer_ids * cfg.layer_penalty_factor
        if cfg.position_temperature > 0:
            scores = _gumbel(scores, cfg.position_temperature, rng)
        scores = np.where(tokens != AUDIO_MASK_ID, -np.inf, scores)

        flat = scores.ravel()
        available = int(np.isfinite(flat).sum())
        k = min(k, available)
        if k <= 0:
            continue
        if cfg.confidence_threshold > 0:
            # probability of the chosen token, per cell, masked cells only
            conf = np.exp(log_probs.max(-1))
            conf = np.where(tokens != AUDIO_MASK_ID, -1.0, conf).ravel()
            confident = int((conf > cfg.confidence_threshold).sum())
            k = min(max(k, confident), available)
        top = np.argpartition(-flat, k - 1)[:k]
        tokens.ravel()[top] = pred.ravel()[top]

        input_ids[0, :, gen_start:] = tokens
        u_ids[0] = tokens
        left = int((tokens == AUDIO_MASK_ID).sum())
        if progress and ((step + 1) % 8 == 0 or step == 0):
            print(f"    step {step + 1:3d}/{cfg.num_step}  {left:5d} cells masked", flush=True)
        if left == 0:
            if progress:
                print(f"    finished early at step {step + 1}/{cfg.num_step}", flush=True)
            break

    leftover = int((tokens == AUDIO_MASK_ID).sum())
    if leftover:
        print(f"    warning: {leftover} cells still masked — clamping to 0")
        tokens[tokens == AUDIO_MASK_ID] = 0
    return tokens


# ===========================================================================
# Audio post-processing — port of _post_process_audio
# ===========================================================================

def post_process(x: np.ndarray, ref_rms: float | None, cfg: GenConfig) -> np.ndarray:
    if cfg.postprocess_output:
        x = remove_silence(x, SR_24K, mid_sil=500, lead_sil=100, trail_sil=100)
    if ref_rms is not None and ref_rms < 0.1:
        x = x * ref_rms / 0.1
    elif ref_rms is None:
        peak = float(np.abs(x).max())
        if peak > 1e-6:
            x = x / peak * 0.5
    return fade_and_pad(x, SR_24K, cfg.pad_duration, cfg.fade_duration)


# ===========================================================================
# Commands
# ===========================================================================

def cmd_enroll(args) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3

    def sess(name):
        return ort.InferenceSession(str(HIGGS_ONNX_DIR / name), sess_options=so,
                                    providers=["CPUExecutionProvider"])

    x24, sr = read_wav(args.ref_audio)
    if sr != SR_24K:
        sys.exit(f"{args.ref_audio} is {sr} Hz; resample to {SR_24K} first")
    ref_rms = float(np.sqrt((x24 ** 2).mean()))
    if 0 < ref_rms < 0.1:
        x24 = x24 * 0.1 / ref_rms
    if args.preprocess:
        # parity with create_voice_clone_prompt(preprocess_prompt=True)
        x24 = remove_silence(x24, SR_24K, mid_sil=200, lead_sil=100, trail_sil=200)
        if len(x24) == 0:
            sys.exit("reference audio is empty after silence removal — pass --no-preprocess")
    n = (len(x24) // 960) * 960
    x24 = x24[:n]

    try:
        import soxr
        x16 = soxr.resample(x24, SR_24K, SR_16K).astype(np.float32)
    except ImportError:
        from _common import resample_linear
        x16 = resample_linear(x24, SR_24K, SR_16K)

    t0 = time.perf_counter()
    af = sess("acoustic_encoder.onnx").run(
        ["acoustic_features"], {"waveform_24k": x24[None, None, :]})[0]
    sf = sess("semantic_encoder.onnx").run(
        ["semantic_features"], {"waveform_16k": x16[None, :]})[0]
    T = min(af.shape[2], sf.shape[2])
    codes = sess("quantizer_encoder.onnx").run(
        ["codes"], {"acoustic_features": af[:, :, :T],
                    "semantic_features": sf[:, :, :T]})[0][:, 0, :]
    dt = time.perf_counter() - t0

    ref_text = add_punctuation(args.ref_text)
    vp = VoicePrompt(codes=codes.astype(np.int16), ref_text=ref_text, ref_rms=ref_rms)
    out = Path(args.out or OUT / "voices" / "me.bin")
    vp.save(out)
    print(f"enrolled in {dt:.2f}s → {out}  codes {codes.shape} "
          f"({vp.duration_s:.2f}s) rms {ref_rms:.4f}  {out.stat().st_size} bytes")


def cmd_generate(args) -> None:
    import onnxruntime as ort

    cfg = GenConfig(num_step=args.num_step, guidance_scale=args.guidance_scale,
                    t_shift=args.t_shift, denoise=args.denoise,
                    postprocess_output=args.postprocess,
                    confidence_threshold=args.confidence_threshold)
    if args.deterministic:
        cfg.position_temperature = 0.0
    rng = np.random.default_rng(args.seed)

    vp = VoicePrompt.load(args.voice_prompt) if args.voice_prompt else None
    ref_text = vp.ref_text if vp else None
    ref_codes = vp.codes.astype(np.int64) if vp else None
    ref_rms = vp.ref_rms if vp else None
    text = args.text

    tok = QwenTokenizer(Path(args.tokenizer or UPSTREAM_DIR / "tokenizer.json"))
    t_gen = args.frames or estimate_target_frames(
        text, ref_text or "", vp.num_frames if vp else 0, speed=args.speed)
    input_ids, audio_mask, gen_start = build_prompt(
        tok, text, ref_text, ref_codes, t_gen, args.language, args.instruct, cfg.denoise)
    S = input_ids.shape[2]
    print(f"S={S} (text {gen_start - (vp.num_frames if vp else 0)} + "
          f"ref {vp.num_frames if vp else 0} + gen {t_gen})  "
          f"≈ {t_gen / FRAME_RATE:.2f}s target")

    lm_path = Path(args.lm or MODELS / "onnx" / "fp32" / "omnivoice_lm.onnx")
    lm = Backbone(lm_path, args.threads)
    print(f"backbone loaded in {lm.load_seconds:.1f}s  ({lm_path})")

    t0 = time.perf_counter()
    codes = generate_codes(lm, input_ids, audio_mask, gen_start, cfg, rng)
    t_decode_loop = time.perf_counter() - t0

    so = ort.SessionOptions()
    so.log_severity_level = 3
    if args.threads:
        so.intra_op_num_threads = args.threads
    dec = ort.InferenceSession(str(HIGGS_ONNX_DIR / "higgs_decoder.onnx"),
                               sess_options=so, providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    wav = dec.run(["waveform_24k"], {"codes": codes[:, None, :]})[0].squeeze()
    t_vocoder = time.perf_counter() - t0

    raw = wav.astype(np.float32)
    wav = post_process(raw, ref_rms, cfg)
    dur = len(wav) / SR_24K
    if dur <= 0.0:
        print(f"\n  GENERATION FAILED: {len(raw) / SR_24K:.2f}s of vocoder output was "
              f"entirely below the -50 dBFS silence floor.\n"
              f"  The model produced silence, not a short clip. Most common cause: "
              f"guidance_scale=0 — classifier-free guidance is not optional for "
              f"OmniVoice.\n  peak={float(np.abs(raw).max()):.2e} "
              f"rms={float(np.sqrt((raw ** 2).mean())):.2e}")
        write_wav(Path(args.out or OUT / "onnx" / "cloned.wav").with_suffix(".raw.wav"),
                  raw, SR_24K)
        sys.exit(2)
    out = Path(args.out or OUT / "onnx" / "cloned.wav")
    write_wav(out, wav, SR_24K)

    total = t_decode_loop + t_vocoder
    stats = {
        "S": int(S), "T_gen": int(t_gen), "num_step": cfg.num_step,
        "confidence_threshold": cfg.confidence_threshold,
        "guidance_scale": cfg.guidance_scale, "lm_calls": lm.calls,
        "lm_seconds": round(lm.compute_seconds, 3),
        "loop_seconds": round(t_decode_loop, 3),
        "vocoder_seconds": round(t_vocoder, 3),
        "load_seconds": round(lm.load_seconds, 3),
        "audio_seconds": round(dur, 3), "rtf": round(total / dur, 3),
        "model": str(lm_path), "threads": args.threads,
    }
    print(f"\n  {out}  {dur:.2f}s")
    print(f"  lm {lm.calls} calls / {lm.compute_seconds:.1f}s, "
          f"vocoder {t_vocoder:.2f}s, total {total:.1f}s  →  RTF {total / dur:.2f}")
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(stats, indent=2))
    np.save(out.with_suffix(".codes.npy"), codes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll")
    e.add_argument("--ref-audio", required=True)
    e.add_argument("--ref-text", required=True)
    e.add_argument("--out", default=None)
    e.add_argument("--no-preprocess", dest="preprocess", action="store_false",
                   default=True, help="skip silence removal on the reference")

    g = sub.add_parser("generate")
    g.add_argument("--text", required=True)
    g.add_argument("--voice-prompt", default=None)
    g.add_argument("--language", default="ko")
    g.add_argument("--instruct", default=None)
    g.add_argument("--lm", default=None)
    g.add_argument("--tokenizer", default=None)
    g.add_argument("--frames", type=int, default=None)
    g.add_argument("--speed", type=float, default=1.0)
    g.add_argument("--num-step", type=int, default=32)
    g.add_argument("--guidance-scale", type=float, default=2.0)
    g.add_argument("--t-shift", type=float, default=0.1)
    g.add_argument("--confidence-threshold", type=float, default=0.0,
                   help="commit every cell above this probability, not just k per step")
    g.add_argument("--denoise", action="store_true", default=True)
    g.add_argument("--postprocess", action="store_true", default=True)
    g.add_argument("--deterministic", action="store_true")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--threads", type=int, default=0)
    g.add_argument("--stats", default=None)
    g.add_argument("--out", default=None)

    args = ap.parse_args()
    {"enroll": cmd_enroll, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    main()
