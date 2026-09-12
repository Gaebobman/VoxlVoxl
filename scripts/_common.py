"""Shared PC-side helpers: paths, WAV I/O, and the voice_prompt.bin format.

Everything here has a 1:1 Kotlin counterpart on the device side, so keep it
dependency-light (numpy only) and keep the binary layout in sync with
``docs/plan.md`` §3.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SAMPLE = ROOT / "sample"
OUT = ROOT / "out"

UPSTREAM_DIR = MODELS / "omnivoice"
HIGGS_ONNX_DIR = MODELS / "higgs-onnx" / "audio_tokenizer"

SR_24K = 24_000
SR_16K = 16_000
FRAME_RATE = 25          # 24000 / hop_length 960
NUM_CODEBOOKS = 8
AUDIO_VOCAB_SIZE = 1025
AUDIO_MASK_ID = 1024
HIDDEN_SIZE = 1024

# OmniVoice special tokens (k2-fsa/OmniVoice tokenizer.json)
TOK_DENOISE = "<|denoise|>"
TOK_LANG_START, TOK_LANG_END = "<|lang_start|>", "<|lang_end|>"
TOK_INSTRUCT_START, TOK_INSTRUCT_END = "<|instruct_start|>", "<|instruct_end|>"
TOK_TEXT_START, TOK_TEXT_END = "<|text_start|>", "<|text_end|>"


# ---------------------------------------------------------------------------
# WAV I/O (16-bit PCM only — same subset the Android side will implement)
# ---------------------------------------------------------------------------

def write_wav(path: str | Path, wav: np.ndarray, sr: int = SR_24K) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    data = pcm16.tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16
    ) + b"data" + struct.pack("<I", len(data))
    path.write_bytes(hdr + data)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Minimal RIFF/PCM16 reader. Returns (mono float32 in [-1,1], sample_rate)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    audio_format, channels, sr, _, _, bits = fmt
    if audio_format != 1 or bits != 16:
        raise ValueError(f"{path}: only 16-bit PCM is supported (got format={audio_format}, bits={bits})")
    x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, sr


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Deliberately naive resampler — only used as a fallback and as the spec for
    the Kotlin port. Prefer torchaudio/soxr on the PC side."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    idx = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    i0 = np.floor(idx).astype(np.int64).clip(0, len(x) - 1)
    i1 = (i0 + 1).clip(0, len(x) - 1)
    frac = (idx - i0).astype(np.float32)
    return ((1.0 - frac) * x[i0] + frac * x[i1]).astype(np.float32)


# ---------------------------------------------------------------------------
# voice_prompt.bin  —  see docs/plan.md §3
# ---------------------------------------------------------------------------

VOICE_PROMPT_MAGIC = b"OVVP"
VOICE_PROMPT_VERSION = 1


@dataclass
class VoicePrompt:
    codes: np.ndarray      # (8, T) int16, values 0..1023
    ref_text: str
    ref_rms: float
    sample_rate: int = SR_24K

    @property
    def num_frames(self) -> int:
        return int(self.codes.shape[1])

    @property
    def duration_s(self) -> float:
        return self.num_frames / FRAME_RATE

    def save(self, path: str | Path) -> None:
        codes = np.ascontiguousarray(self.codes, dtype="<i2")
        if codes.ndim != 2 or codes.shape[0] != NUM_CODEBOOKS:
            raise ValueError(f"codes must be ({NUM_CODEBOOKS}, T), got {codes.shape}")
        if codes.min() < 0 or codes.max() >= AUDIO_MASK_ID:
            raise ValueError("codes outside the valid codebook range 0..1023")
        text = self.ref_text.encode("utf-8")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(VOICE_PROMPT_MAGIC)
            f.write(struct.pack("<IIIfII", VOICE_PROMPT_VERSION, NUM_CODEBOOKS,
                                codes.shape[1], float(self.ref_rms),
                                int(self.sample_rate), len(text)))
            f.write(text)
            f.write(codes.tobytes())

    @classmethod
    def load(cls, path: str | Path) -> "VoicePrompt":
        raw = Path(path).read_bytes()
        if raw[:4] != VOICE_PROMPT_MAGIC:
            raise ValueError(f"{path}: bad magic {raw[:4]!r}")
        version, ncb, nframes, rms, sr, tlen = struct.unpack("<IIIfII", raw[4:28])
        if version != VOICE_PROMPT_VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
        text = raw[28:28 + tlen].decode("utf-8")
        body = raw[28 + tlen:]
        codes = np.frombuffer(body, dtype="<i2", count=ncb * nframes).reshape(ncb, nframes)
        return cls(codes=codes.copy(), ref_text=text, ref_rms=float(rms), sample_rate=int(sr))
