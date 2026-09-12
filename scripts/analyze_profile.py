#!/usr/bin/env python3
"""Count execution-provider node assignments in an ONNX Runtime profile.

ORT's Android AAR does not route its own log to logcat, so the verbose
partitioning report is invisible on device. The profile JSON is better evidence
anyway: it names the execution provider of every node that actually ran.

  python scripts/analyze_profile.py out/device/profile/*.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def analyze(path: Path) -> None:
    try:
        events = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"{path.name}: not valid JSON (truncated profile?)")
        return
    if not isinstance(events, list):
        return

    by_ep = collections.Counter()
    time_by_ep = collections.Counter()
    ops_by_ep = collections.defaultdict(collections.Counter)
    time_by_op = collections.Counter()
    count_by_op = collections.Counter()
    for e in events:
        args = e.get("args") or {}
        ep = args.get("provider")
        if not ep or e.get("cat") != "Node":
            continue
        if not e.get("name", "").endswith("_kernel_time"):
            continue
        by_ep[ep] += 1
        time_by_ep[ep] += e.get("dur", 0)
        op = args.get("op_name", "?")
        ops_by_ep[ep][op] += 1
        time_by_op[op] += e.get("dur", 0)
        count_by_op[op] += 1

    if not by_ep:
        print(f"{path.name}: no node events (only {len(events)} events)")
        return

    total_nodes = sum(by_ep.values())
    total_time = sum(time_by_ep.values())
    print(f"\n=== {path.name}")
    print(f"  {total_nodes} node executions, {total_time / 1000:.1f} ms total")
    for ep, n in by_ep.most_common():
        print(f"  {ep:28s} {n:6d} nodes ({n / total_nodes * 100:5.1f} %)  "
              f"{time_by_ep[ep] / 1000:8.1f} ms ({time_by_ep[ep] / max(total_time, 1) * 100:5.1f} %)")
        for op, c in ops_by_ep[ep].most_common(6):
            print(f"      {c:6d}  {op}")

    print("  time by op:")
    for op, us in time_by_op.most_common(12):
        print(f"    {us / 1000:8.1f} ms ({us / max(total_time, 1) * 100:5.1f} %)  "
              f"{count_by_op[op]:5d}x  {op}")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    for p in sorted(paths):
        if p.stat().st_size > 1000:
            analyze(p)


if __name__ == "__main__":
    main()
