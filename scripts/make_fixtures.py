#!/usr/bin/env python3
"""Generate the fixtures the Kotlin unit tests assert against.

Every pure-logic component that gets reimplemented on the device — the Qwen2
byte-level BPE tokenizer, the duration estimator, prompt assembly, the pydub
silence ports, the voice_prompt.bin container — is pinned here by input/output
pairs produced from the Python side. A Kotlin port that drifts fails a test
instead of producing subtly wrong audio.

  python scripts/make_fixtures.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ROOT, SR_24K, UPSTREAM_DIR, VoicePrompt, read_wav, remove_silence,
)
from infer_onnx import (  # noqa: E402
    QwenTokenizer, add_punctuation, build_prompt, combine_text,
    estimate_target_frames, text_weight, time_steps, unmask_schedule,
)

DEST = ROOT / "android" / "fixtures"

TEXTS = [
    "오늘 회의를 시작하겠습니다.",
    "안녕하세요. 이것은 제 목소리를 등록하기 위한 테스트 음성입니다.",
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "你好，世界！今天天气很好。",
    "こんにちは、元気ですか？",
    "Привет, как дела?",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "Mixed 한국어 and English 123 numbers.",
    "Emoji test 🌍 and punctuation!?",
    "[laughter] That was funny [sigh] but also sad.",
    "   leading and trailing   spaces   ",
    "line\nbreaks\r\nand\ttabs",
    "（중국식 괄호）and (english ones)",
    "",
    "a",
    "숫자 2345 와 3.14 그리고 50%",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dest = Path(args.out or DEST)
    dest.mkdir(parents=True, exist_ok=True)

    tok = QwenTokenizer(UPSTREAM_DIR / "tokenizer.json")

    # --- 1. tokenizer ----------------------------------------------------
    tokenizer_cases = []
    for t in TEXTS:
        tokenizer_cases.append({
            "text": t,
            "ids": tok.encode(t),
            "ids_nonverbal": tok.encode_with_nonverbal(t),
        })
    # every special token must round-trip on its own
    for name, s in [("denoise", "<|denoise|>"), ("lang_start", "<|lang_start|>"),
                    ("lang_end", "<|lang_end|>"), ("instruct_start", "<|instruct_start|>"),
                    ("instruct_end", "<|instruct_end|>"), ("text_start", "<|text_start|>"),
                    ("text_end", "<|text_end|>")]:
        tokenizer_cases.append({"text": s, "ids": tok.encode(s),
                                "ids_nonverbal": tok.encode_with_nonverbal(s),
                                "special": name})
    (dest / "tokenizer.json").write_text(json.dumps(
        {"tokenizer": "k2-fsa/OmniVoice tokenizer.json", "cases": tokenizer_cases},
        ensure_ascii=False, indent=1))

    # --- 2. text normalisation -------------------------------------------
    text_cases = [{"text": t, "ref_text": r,
                   "combined": combine_text(t, r),
                   "punctuated": add_punctuation(t)}
                  for t in TEXTS
                  for r in (None, "안녕하세요. 테스트입니다.", "Reference sentence")]
    (dest / "text.json").write_text(json.dumps(
        {"cases": text_cases}, ensure_ascii=False, indent=1))

    # --- 3. duration estimator -------------------------------------------
    dur_cases = []
    for t in TEXTS:
        dur_cases.append({"text": t, "weight": round(text_weight(t), 6)})
    for t in TEXTS:
        for ref, frames in (("안녕하세요. 이것은 제 목소리를 등록하기 위한 테스트 음성입니다.", 103),
                            ("Hello, world.", 40), ("", 0)):
            for speed in (1.0, 1.2, 0.8):
                dur_cases.append({
                    "text": t, "ref_text": ref, "ref_frames": frames, "speed": speed,
                    "frames": estimate_target_frames(t, ref, frames, speed=speed),
                })
    (dest / "duration.json").write_text(json.dumps(
        {"cases": dur_cases}, ensure_ascii=False, indent=1))

    # --- 4. unmask schedule ----------------------------------------------
    sched_cases = []
    for num_step in (8, 16, 32):
        for t_shift in (0.1, 0.5, 1.0):
            for total in (48 * 8, 125 * 8, 7):
                sched = unmask_schedule(num_step, t_shift, total)
                sched_cases.append({
                    "num_step": num_step, "t_shift": t_shift, "total": total,
                    "timesteps": time_steps(num_step, t_shift).tolist(),   # full float64
                    "schedule": sched, "sum": int(sum(sched)),
                })
    (dest / "schedule.json").write_text(json.dumps({"cases": sched_cases}, indent=1))

    # --- 5. prompt assembly ----------------------------------------------
    vp = VoicePrompt.load(ROOT / "out" / "golden" / "voice_prompt.bin")
    ids, am, gen_start = build_prompt(
        tok, "오늘 회의를 시작하겠습니다.", vp.ref_text, vp.codes.astype(np.int64),
        48, "ko", None, True)
    (dest / "prompt.json").write_text(json.dumps({
        "text": "오늘 회의를 시작하겠습니다.",
        "ref_text": vp.ref_text, "ref_frames": vp.num_frames,
        "num_target": 48, "language": "ko", "instruct": None, "denoise": True,
        "S": int(ids.shape[2]), "gen_start": int(gen_start),
        "text_token_count": int(gen_start - vp.num_frames),
        "text_row_ids": ids[0, 0, :gen_start - vp.num_frames].tolist(),
        "audio_mask_first_true": int(np.argmax(am[0])),
        "audio_mask_sum": int(am[0].sum()),
        "tail_unique": np.unique(ids[0, :, gen_start:]).tolist(),
    }, ensure_ascii=False, indent=1))

    # --- 6. silence removal ----------------------------------------------
    x, sr = read_wav(ROOT / "sample" / "reference.wav")
    sil_cases = []
    for mid, lead, trail in ((200, 100, 200), (500, 100, 100), (300, 100, 300)):
        y = remove_silence(x, sr, mid, lead, trail)
        sil_cases.append({
            "mid_sil": mid, "lead_sil": lead, "trail_sil": trail,
            "in_samples": int(len(x)), "out_samples": int(len(y)),
            "out_sha_head": [round(float(v), 6) for v in y[:8]],
            "out_sha_tail": [round(float(v), 6) for v in y[-8:]],
            "out_rms": round(float(np.sqrt((y ** 2).mean())), 8),
        })
    (dest / "silence.json").write_text(json.dumps(
        {"wav": "sample/reference.wav", "sample_rate": sr, "cases": sil_cases}, indent=1))

    # --- 7. voice_prompt.bin ---------------------------------------------
    (dest / "voice_prompt.json").write_text(json.dumps({
        "file": "out/golden/voice_prompt.bin",
        "magic": "OVVP", "version": 1,
        "num_codebooks": 8, "num_frames": vp.num_frames,
        "ref_rms": round(vp.ref_rms, 8), "sample_rate": vp.sample_rate,
        "ref_text": vp.ref_text,
        "codes_head": vp.codes[:, :4].tolist(),
        "codes_tail": vp.codes[:, -4:].tolist(),
        "codes_sum": int(vp.codes.astype(np.int64).sum()),
    }, ensure_ascii=False, indent=1))

    for f in sorted(dest.iterdir()):
        n = len(json.loads(f.read_text()).get("cases", [])) if "cases" in f.read_text()[:200] else 1
        print(f"  {f.stat().st_size / 1024:8.1f} KB  {f.relative_to(ROOT)}  ({n} case(s))")


if __name__ == "__main__":
    main()
