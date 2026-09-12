# OmniVoicePoC — Android app

Kotlin, ARM64 only, `minSdk 31` / `targetSdk 36`, ONNX Runtime 1.22.0.

## Status

| component | state |
|---|---|
| `QwenBpeTokenizer` | ✅ ported, matches HuggingFace on all 25 fixture cases |
| `DurationEstimator` | ✅ ported, matches Python on all 180 fixture cases |
| `TextUtils` (`combine` / `addPunctuation`) | ✅ ported, 54 cases |
| `UnmaskSchedule` | ✅ ported, 27 cases |
| `VoicePrompt` (`voice_prompt.bin`) | ✅ reads the file the Python side writes |
| `WavIo` (RIFF + resampler) | ✅ round-trips within one int16 LSB |
| `SilenceUtils` | ⬜ next |
| `OnnxModelRunner` / `OmniVoiceEngine` | ⬜ next |
| `MainActivity` UI | ⬜ last, per the brief |

`./gradlew :app:assembleDebug` produces a 25.9 MB arm64-v8a APK carrying
`libonnxruntime.so` + `libonnxruntime4j_jni.so`, with **no `INTERNET`
permission** — an accidental network call crashes instead of silently
succeeding, which is how the offline requirement is enforced rather than
merely asserted.

## Toolchain (no root required)

```bash
# JDK 17
curl -L -o /tmp/jdk.tgz \
  "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
mkdir -p ~/toolchains && tar -C ~/toolchains -xzf /tmp/jdk.tgz

# Android SDK
mkdir -p ~/Android/Sdk/cmdline-tools && cd ~/Android/Sdk/cmdline-tools
curl -LO https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip -q commandlinetools-linux-*.zip && mv cmdline-tools latest
export JAVA_HOME=$(echo ~/toolchains/jdk-17*) ANDROID_HOME=~/Android/Sdk
export PATH=$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH
yes | sdkmanager --licenses
sdkmanager "platform-tools" "platforms;android-36" "build-tools;36.0.0"
```

## Build and test

```bash
export JAVA_HOME=$(echo ~/toolchains/jdk-17*) ANDROID_HOME=~/Android/Sdk
cd android/OmniVoicePoC
./gradlew :app:testDebugUnitTest     # the fixture tests — no device needed
./gradlew :app:assembleDebug
```

## Fixtures

`app/build.gradle.kts` points the unit-test resource path at `android/fixtures/`,
which `scripts/make_fixtures.py` generates from the Python reference. Regenerate
the fixtures and the Kotlin ports are immediately re-tested against the new
truth — there is no second copy to drift.

Two bugs the fixtures caught on the very first run:

- `UnmaskSchedule` took `tShift` as a `Float`. `0.1f.toDouble()` is
  `0.10000000149`, which moves the schedule boundaries by ~1e-9 — the same order
  as the comparison tolerance. It is a `Double` now, matching the reference.
- `File.createTempFile("vp", …)` threw `IllegalArgumentException`; the JDK
  requires a prefix of at least three characters.

Neither would have been visible without running against generated truth, and
both would have produced subtly wrong audio rather than an obvious failure.
