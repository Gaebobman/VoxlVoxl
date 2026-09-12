#!/usr/bin/env python3
"""Build the VoxlVoxl design canvas.

The .dc.html sources carry a `/*@FONT@*/` placeholder rather than the font
itself: Pretendard is not on Google Fonts and the canvas admits no other font
host, so it has to ride inline as base64 — and ~95 KB of it per artboard is not
something to keep in git. This collects the glyphs the artboards actually use,
subsets the variable font to them, injects it, and seeds the canvas.

  python design/build.py                 # subset + inject + seed
  python design/build.py --seed-only     # skip the font step
"""
from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONT_SRC = HERE / "PretendardVariable.woff2"
SUBSET = HERE / "build" / "pretendard-subset.woff2"
OUT = HERE / "voxlvoxl-app.html"
PLACEHOLDER = "/*@FONT@*/"

# order matters: Main is the entry artboard
ARTBOARDS = [
    "Main.dc.html", "FirstRun.dc.html", "EnrollRead.dc.html", "EnrollVerify.dc.html",
    "Library.dc.html", "Compose.dc.html", "Result.dc.html", "DevSettings.dc.html",
    "AltInstrument.dc.html", "AltCalm.dc.html", "AltDense.dc.html",
]

FACE = """    /* Pretendard v1.3.9 by Kil Hyung-jin — SIL Open Font License 1.1.
       Subset to the glyphs these artboards use, variable wght axis kept so one
       face covers 400-700. Not on Google Fonts, and the canvas admits no other
       font host, so it rides inline. Regenerate with design/build.py. */
    @font-face {{
      font-family: Pretendard;
      src: url(data:font/woff2;base64,{b64}) format('woff2');
      font-weight: 100 900;
      font-style: normal;
      font-display: block;
    }}"""


def collect_glyphs() -> str:
    chars: set[str] = set()
    for name in ARTBOARDS:
        t = (HERE / name).read_text(encoding="utf-8")
        t = re.sub(r"<style>.*?</style>", " ", t, flags=re.S)   # CSS is not content
        chars |= set(t)
    chars |= {chr(c) for c in range(0x20, 0x7F)}
    chars |= set("—–·…“”‘’×≈°%")
    chars = {c for c in chars if c.isprintable() and not c.isspace()} | {" "}
    return "".join(sorted(chars))


def _python() -> str:
    """fontTools lives in the project venv, not necessarily in the interpreter
    that launched this script."""
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def subset(glyphs: str) -> None:
    if not FONT_SRC.exists():
        sys.exit(f"{FONT_SRC} missing — fetch PretendardVariable.woff2 from\n"
                 "  https://github.com/orioncactus/pretendard/releases (OFL 1.1)\n"
                 "  web/variable/woff2/PretendardVariable.woff2")
    SUBSET.parent.mkdir(parents=True, exist_ok=True)
    txt = SUBSET.with_suffix(".txt")
    txt.write_text(glyphs, encoding="utf-8")
    subprocess.run([_python(), "-m", "fontTools.subset", str(FONT_SRC),
                    f"--text-file={txt}", "--flavor=woff2",
                    "--layout-features=kern,liga,calt,tnum",
                    "--no-hinting", "--desubroutinize",
                    "--name-IDs=*", "--name-legacy",
                    f"--output-file={SUBSET}"], check=True)
    print(f"  subset {SUBSET.stat().st_size / 1024:.1f} KB from {len(glyphs)} glyphs")


def inject() -> list[Path]:
    b64 = base64.b64encode(SUBSET.read_bytes()).decode()
    face = FACE.format(b64=b64)
    staged = []
    stage = HERE / "build"
    stage.mkdir(parents=True, exist_ok=True)
    for name in ARTBOARDS:
        src = (HERE / name).read_text(encoding="utf-8")
        dst = stage / name
        dst.write_text(src.replace(PLACEHOLDER, face), encoding="utf-8")
        staged.append(dst)
    return staged


def seed(files: list[Path], skill_dir: Path) -> None:
    cmd = ["node", str(skill_dir / "seed-canvas.mjs"),
           "--template", str(skill_dir / "payload.template.html"),
           "--out", str(OUT), "--title", "VoxlVoxl App"]
    for f in files:
        cmd += ["--artboard", str(f)]
    cmd += ["--canvas", str(HERE / "canvas.json")]
    subprocess.run(cmd, check=True, cwd=HERE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-dir", default=None,
                    help="the design skill's base directory (holds seed-canvas.mjs)")
    ap.add_argument("--seed-only", action="store_true")
    args = ap.parse_args()

    if not args.seed_only:
        g = collect_glyphs()
        subset(g)
    files = inject()
    if not args.skill_dir:
        print(f"  staged {len(files)} artboards in {HERE / 'build'}; "
              "pass --skill-dir to seed")
        return
    seed(files, Path(args.skill_dir))


if __name__ == "__main__":
    main()
