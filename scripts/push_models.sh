#!/usr/bin/env bash
# Stage the model bundle into the app's INTERNAL files dir.
#
# `adb push` can write the app-specific external dir, but the app process cannot
# then read what `shell` created there — the DAC bits say yes and SELinux says
# no (run-as succeeds only because it runs under a different context). So push
# to /data/local/tmp, then copy in through run-as, which requires a debuggable
# build. A release build would ship the models differently; for the PoC this is
# the shortest path that actually works.
set -euo pipefail

PKG="${PKG:-ai.omnivoice.poc}"
ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
SRC="${1:-$(cd "$(dirname "$0")/.." && pwd)/models/android}"
PROMPT="${2:-$(cd "$(dirname "$0")/.." && pwd)/out/golden_det/voice_prompt.bin}"
STAGE=/data/local/tmp/ovmodels

echo "staging $SRC → $STAGE"
"$ADB" shell rm -rf "$STAGE"
"$ADB" shell mkdir -p "$STAGE"
"$ADB" push "$SRC"/. "$STAGE"/
[ -f "$PROMPT" ] && "$ADB" push "$PROMPT" "$STAGE"/voice_prompt.bin

echo "copying into $PKG internal files/models (run-as)"
"$ADB" shell run-as "$PKG" mkdir -p files/models
"$ADB" shell "run-as $PKG sh -c 'for f in $STAGE/*; do cp \"\$f\" files/models/; done'"
"$ADB" shell run-as "$PKG" ls -la files/models

echo "clearing the staging copy"
"$ADB" shell rm -rf "$STAGE"
echo "done"
