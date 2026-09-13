# design/

The VoxlVoxl design canvas — eight screens plus three directions not taken.

```bash
# needs fonttools + brotli (in the project venv) and node
python design/build.py --skill-dir "<the /design skill's base directory>"
```

`build.py` collects the glyphs the artboards actually use, subsets
`PretendardVariable.woff2` to them, injects the result into a staged copy of each
`.dc.html`, and seeds the canvas.

The sources keep a `/*@FONT@*/` placeholder instead of the font. Pretendard is
not on Google Fonts and the canvas admits no other font host, so it has to ride
inline as base64 — about 95 KB per artboard, which is not something to keep in
git. `design/build/` and the seeded `voxlvoxl-app.html` are both derived and
gitignored.

## What the screens encode

Every non-obvious choice here came out of a measurement, not a preference:

| screen | the measurement behind it |
|---|---|
| Enroll — read | the transcript is concatenated ahead of the target text and aligns the reference, so a wrong one degrades the voice **with no error**. Handing the user a sentence removes the error class. |
| Enroll — verify | decoding the enrolled codes back through the vocoder costs ~0.5 s warm (2.4 s cold) against ~20 s for a test synthesis, and is more diagnostic: that IS what the model will hear. |
| Voices | a voice-design instruct is ignored whenever a clone prompt is present (every style within ±9 Hz of baseline F0), but manner transfers through the recording (whisper: voiced 20.3 % → 24.1 %). So style is a **profile**, not a setting. |
| Generating | the eight bars are the model's real state — the un-masking loop penalises higher codebooks, so layer 0 resolves first and 7 last. |
| Generating | the composer stays reachable and lines queue, because short utterances cost ~2× per second of audio (the reference prefix is re-paid at every forward). |
| Compose | language comes from the script of the text, using the same Unicode tables that weight speaking time. |
| Developer | every number shown is measured on the target device. |

## Licences

Pretendard v1.3.9 © Kil Hyung-jin, SIL Open Font License 1.1 — see
`android/OmniVoicePoC/app/src/main/assets/licenses/OFL-1.1-pretendard.txt`.
