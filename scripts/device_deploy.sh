#!/usr/bin/env bash
# Build and install BOTH APKs. The engine classes live in the app APK while the
# benchmark lives in the test APK, so installing only the test APK after
# changing the engine gives a NoSuchMethodError at runtime rather than a build
# error -- install them together, always.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export JAVA_HOME="${JAVA_HOME:-$(echo "$HOME"/toolchains/jdk-17*)}"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
ADB="${ADB:-$ANDROID_HOME/platform-tools/adb}"

cd "$ROOT/android/OmniVoicePoC"
./gradlew --no-daemon :app:assembleDebug :app:assembleDebugAndroidTest "$@"
"$ADB" install -r app/build/outputs/apk/debug/app-debug.apk
"$ADB" install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
echo "deployed"
