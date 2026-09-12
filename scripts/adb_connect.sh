#!/usr/bin/env bash
# Connect this WSL2 shell to a physical Android device.
#
# WSL2 gets no USB passthrough, so `adb devices` over the cable is always empty
# here even when Windows sees the phone. WSL2 *can* reach the LAN directly
# (verified: 192.168.0.0/24 is routable from eth0), so wireless debugging is the
# path of least resistance — no usbipd-win, no Windows admin rights.
#
#   ./scripts/adb_connect.sh pair    <ip>:<pair-port> <6-digit-code>
#   ./scripts/adb_connect.sh connect <ip>:<connect-port>
#   ./scripts/adb_connect.sh scan    [subnet]          # look for an open 5555
#   ./scripts/adb_connect.sh status
#
# On the phone: Settings -> Developer options -> Wireless debugging -> ON.
# "Pair device with pairing code" shows the PAIRING port; the main Wireless
# debugging screen shows the CONNECT port. They are different numbers.
set -euo pipefail

ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
SUBNET="${SUBNET:-192.168.0}"

case "${1:-status}" in
  pair)
    [ $# -eq 3 ] || { echo "usage: $0 pair <ip>:<pair-port> <code>"; exit 2; }
    "$ADB" pair "$2" "$3"
    ;;
  connect)
    [ $# -eq 2 ] || { echo "usage: $0 connect <ip>:<connect-port>"; exit 2; }
    "$ADB" connect "$2"
    "$ADB" devices -l
    ;;
  scan)
    net="${2:-$SUBNET}"
    echo "scanning ${net}.1-254 for an open adb port 5555 ..."
    for i in $(seq 1 254); do
      ( timeout 0.6 bash -c "echo > /dev/tcp/${net}.$i/5555" 2>/dev/null \
          && echo "  ${net}.$i:5555 OPEN" ) &
    done
    wait
    echo "scan complete"
    ;;
  status)
    "$ADB" devices -l
    if "$ADB" get-state >/dev/null 2>&1; then
      echo
      for p in ro.product.model ro.product.manufacturer ro.build.version.release \
               ro.product.cpu.abi ro.board.platform ro.hardware.chipname; do
        printf '  %-32s %s\n' "$p" "$("$ADB" shell getprop $p | tr -d '\r')"
      done
      printf '  %-32s %s kB\n' MemTotal \
        "$("$ADB" shell grep MemTotal /proc/meminfo | tr -dc '0-9')"
      printf '  %-32s %s\n' cores "$("$ADB" shell nproc | tr -d '\r')"
    fi
    ;;
  *)
    echo "unknown command: $1"; exit 2 ;;
esac
