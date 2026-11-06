#!/usr/bin/env python3
"""Table of episode summaries under a directory, at any depth (default /opt/arena_ws/data/bench3).

  python3 tools/summarize.py [dir] [--csv out.csv]
  python3 tools/summarize.py ~/work/bench3/results/offline      # HOST, one stage
  python3 tools/summarize.py ~/work/bench3/results              # everything

One row per episode: conditions, calls (by reason), latency p50/p95, stall time and
share, stale and unusable answers, safety stops, cost, and whether the model said arrived.
Success/collisions come from Arena's own evaluation, not from here.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

COLS = ["run", "episode", "model", "conditions", "calls", "ok", "failed", "stale", "unusable", "latency_p50", "latency_p95", "stall_s", "stall_frac", "safety_stop_s", "dist_m", "elapsed_s", "cost_usd", "arrived_signal", "reasons"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", nargs="?", default="/opt/arena_ws/data/bench3")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    rows = []
    for f in sorted(pathlib.Path(a.log_dir).rglob("summary.json")):
        s = json.loads(f.read_text())
        run_dir = f.parent.parent.parent if f.parent.parent.name == "calls" else f.parent.parent
        s["run"], s["episode"] = run_dir.name, f.parent.name
        try:
            cfg = json.loads((f.parent / "meta.jsonl").read_text().splitlines()[0]).get("config", {})
            s["model"] = cfg.get("model") or cfg.get("provider", "")
        except (OSError, IndexError, json.JSONDecodeError):
            s["model"] = ""
        rows.append(s)
    if not rows:
        print(f"no summary.json under {a.log_dir}")
        return
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    print(" | ".join(COLS))
    for r in rows:
        print(" | ".join(fmt(r.get(c, "")) for c in COLS))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
