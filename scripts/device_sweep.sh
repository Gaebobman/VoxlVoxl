#!/usr/bin/env bash
# Drive the on-device benchmark across backends / threads / step counts and
# collect the RESULT lines. Everything is an instrumentation argument, so a
# sweep needs no rebuild.
#
#   ./scripts/device_sweep.sh threads 1 2 4 6 8
#   ./scripts/device_sweep.sh backend CPU XNNPACK NNAPI
#   ./scripts/device_sweep.sh steps 8 16 32
set -uo pipefail

ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
RUNNER=ai.omnivoice.poc.test/androidx.test.runner.AndroidJUnitRunner
CLASS=ai.omnivoice.poc.DeviceBenchmark#t04_generate

BACKEND="${BACKEND:-CPU}"
THREADS="${THREADS:-0}"
STEPS="${STEPS:-16}"

dim="${1:?usage: $0 <threads|backend|steps> v1 v2 ...}"; shift
OUT="${OUT:-out/device/sweep_${dim}.log}"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

for v in "$@"; do
  case "$dim" in
    threads) THREADS="$v" ;;
    backend) BACKEND="$v" ;;
    steps)   STEPS="$v"   ;;
    *) echo "unknown dimension: $dim"; exit 2 ;;
  esac
  echo "--- backend=$BACKEND threads=$THREADS steps=$STEPS"
  "$ADB" logcat -c
  "$ADB" shell am instrument -w -r \
      -e class "$CLASS" -e steps "$STEPS" -e backend "$BACKEND" -e threads "$THREADS" \
      "$RUNNER" > /dev/null 2>&1
  line="$("$ADB" logcat -d -s OmniVoice.Bench:I | tr -d '\r' | grep "RESULT generate" | tail -1)"
  if [ -z "$line" ]; then
    # a backend that fails to register shows up here rather than as a silent pass
    err="$("$ADB" logcat -d | tr -d '\r' | grep -iE "OmniVoiceException|OrtException|FAILURES" | tail -3)"
    echo "  FAILED  ${err:-no RESULT line and no diagnosable error}" | tee -a "$OUT"
    continue
  fi
  echo "${line#*RESULT generate }" | tee -a "$OUT"
done

echo
echo "→ $OUT"
