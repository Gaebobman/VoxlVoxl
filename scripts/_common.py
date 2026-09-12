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


# ---------------------------------------------------------------------------
# Silence handling — numpy port of omnivoice/utils/audio.py, which delegates to
# pydub (MIT). Reimplemented rather than wrapped because the Android side has no
# pydub; `validate_onnx.py --stage dsp` asserts this matches pydub sample-exactly.
#
# pydub works on int16 samples at 1 ms granularity, so everything below is in
# milliseconds and on the int16 scale. max_possible_amplitude = 32768.
# ---------------------------------------------------------------------------

_MAX_AMP = 32768.0


def _ms_rms(x_i16: np.ndarray, sr: int, start_ms: int, len_ms: int) -> float:
    a = int(start_ms * sr / 1000)
    b = int((start_ms + len_ms) * sr / 1000)
    seg = x_i16[a:b]
    if seg.size == 0:
        return 0.0
    # audioop.rms truncates to int; match that so thresholds tie-break identically
    return float(int(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))


def detect_silence(x_i16: np.ndarray, sr: int, min_silence_len: int,
                   silence_thresh_db: float, seek_step: int = 1) -> list[list[int]]:
    """Port of pydub.silence.detect_silence. Returns [start_ms, end_ms] ranges."""
    seg_len = int(len(x_i16) * 1000 // sr)
    if seg_len < min_silence_len:
        return []
    thresh = (10.0 ** (silence_thresh_db / 20.0)) * _MAX_AMP

    last_start = seg_len - min_silence_len
    starts = list(range(0, last_start + 1, seek_step))
    if last_start % seek_step:
        starts.append(last_start)

    silent = [i for i in starts if _ms_rms(x_i16, sr, i, min_silence_len) <= thresh]
    if not silent:
        return []

    ranges, prev = [], silent[0]
    cur = prev
    for i in silent[1:]:
        contiguous = (i == prev + seek_step)
        has_gap = i > (prev + min_silence_len)
        if not contiguous and has_gap:
            ranges.append([cur, prev + min_silence_len])
            cur = i
        prev = i
    ranges.append([cur, prev + min_silence_len])
    return ranges


def detect_nonsilent(x_i16: np.ndarray, sr: int, min_silence_len: int,
                     silence_thresh_db: float, seek_step: int = 1) -> list[list[int]]:
    """Port of pydub.silence.detect_nonsilent."""
    silent = detect_silence(x_i16, sr, min_silence_len, silence_thresh_db, seek_step)
    length = int(len(x_i16) * 1000 // sr)
    if not silent:
        return [[0, length]]
    if silent[0][0] == 0 and silent[0][1] == length:
        return []

    out, prev_end = [], 0
    if silent[0][0] == 0:
        prev_end = silent[0][1]
        silent = silent[1:]
    for start, end in silent:
        out.append([prev_end, start])
        prev_end = end
    if prev_end < length:
        out.append([prev_end, length])
    return out


def _detect_leading_silence(x_i16: np.ndarray, sr: int, thresh_db: float = -50.0,
                            chunk_ms: int = 10) -> int:
    """Port of pydub.silence.detect_leading_silence (dBFS, not rms)."""
    length = int(len(x_i16) * 1000 // sr)
    trim = 0
    while trim < length:
        rms = _ms_rms(x_i16, sr, trim, chunk_ms)
        dbfs = -np.inf if rms <= 0 else 20.0 * np.log10(rms / _MAX_AMP)
        if dbfs >= thresh_db:
            break
        trim += chunk_ms
    return trim


def remove_silence(x: np.ndarray, sr: int, mid_sil: int = 300, lead_sil: int = 100,
                   trail_sil: int = 300, silence_thresh_db: float = -50.0) -> np.ndarray:
    """Port of omnivoice.utils.audio.remove_silence (float32 in, float32 out)."""
    if x.size == 0:
        return x
    # upstream numpy_to_audiosegment: *32768 then clip to int16 range
    xi = (np.asarray(x, np.float32) * 32768.0).clip(-32768, 32767).astype(np.int16)

    def ms(a: int) -> int:
        return int(a * sr / 1000)

    if mid_sil > 0:
        # pydub.split_on_silence(keep_silence=mid_sil, seek_step=10) then concatenate
        nonsilent = detect_nonsilent(xi, sr, mid_sil, silence_thresh_db, seek_step=10)
        length = int(len(xi) * 1000 // sr)
        pieces = []
        for start, end in nonsilent:
            pieces.append(xi[ms(max(0, start - mid_sil)):ms(min(length, end + mid_sil))])
        xi = np.concatenate(pieces) if pieces else xi[:0]

    if xi.size:
        head = max(0, _detect_leading_silence(xi, sr, silence_thresh_db) - lead_sil)
        xi = xi[ms(head):]
    if xi.size:
        rev = xi[::-1].copy()
        tail = max(0, _detect_leading_silence(rev, sr, silence_thresh_db) - trail_sil)
        xi = rev[ms(tail):][::-1].copy()

    return (xi.astype(np.float32) / 32768.0)


def fade_and_pad(x: np.ndarray, sr: int = SR_24K, pad_s: float = 0.1,
                 fade_s: float = 0.1) -> np.ndarray:
    """Port of omnivoice.utils.audio.fade_and_pad_audio."""
    if x.size == 0:
        return x
    out = np.asarray(x, np.float32).copy()
    k = min(int(fade_s * sr), out.shape[-1] // 2)
    if k > 0:
        out[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        out[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    p = int(pad_s * sr)
    if p > 0:
        out = np.concatenate([np.zeros(p, np.float32), out, np.zeros(p, np.float32)])
    return out
