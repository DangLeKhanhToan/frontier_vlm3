#!/usr/bin/env python3
"""Check the frontier_vlm3 scenarios without launching Arena (CPU, no ROS graph, no GPU).

  python3 tools/check_scenarios.py                 # every scenarios/fvlm3_*/
  python3 tools/check_scenarios.py fvlm3_follow_slow

Per scenario:
  1. Arena's own format-2 parser (arena_simulation_setup ... tree/World/Scenario.py parse_scenario),
     which rejects unknown keys and stimuli naming a non-human;
  2. every path agent type loads and samples in humansim, and declares each need a stimulus raises;
  3. every robot start, human pose, waypoint and go_to target_pose lies on free floor of the
     hospital_1 map (0.3 m clearance).
What it cannot check: how the people actually move and react in humansim. That needs a launch.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def _poses_of_agent_type(at: dict) -> list[tuple[str, float, float]]:
    out = []
    for sname, seq in (at.get("sequences") or {}).items():
        for stname, st in (seq.get("steps") or {}).items():
            tp = st.get("target_pose") if isinstance(st, dict) else None
            if tp:
                out.append((f"{sname}.{stname}", float(tp["x"]), float(tp["y"])))
    return out


def check(sdir: pathlib.Path, sm) -> list[str]:
    from arena_simulation_setup.tree.World.Scenario import parse_scenario  # noqa: PLC0415
    from arena_humansim.core.agents.loader import load_agent_type_from_file  # noqa: PLC0415
    from arena_humansim.core.agents.types import sample_agent_type  # noqa: PLC0415

    problems: list[str] = []
    raw = yaml.safe_load((sdir / "scenario.yaml").read_text())
    try:
        parse_scenario(raw)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"parse: {exc}")

    def free(label: str, x: float, y: float) -> None:
        i = int((x - sm.origin[0]) / sm.res)
        j = int((y - sm.origin[1]) / sm.res)
        r = max(1, int(0.3 / sm.res))
        h, w = sm.occ.shape
        if not (r <= i < w - r and r <= j < h - r):
            problems.append(f"{label} ({x:.1f}, {y:.1f}) is off the map")
        elif sm.occ[j - r : j + r + 1, i - r : i + r + 1].any():
            problems.append(f"{label} ({x:.1f}, {y:.1f}) is within 0.3 m of a wall")

    for k, rb in enumerate(raw.get("robots") or []):
        free(f"robots[{k}].start", *rb["start"][:2])
        for ph in rb.get("phases") or []:
            if isinstance(ph.get("pose"), list):
                free(f"robots[{k}] goal pose", *ph["pose"][:2])
    needs_of: dict[str, set[str]] = {}
    for hm in raw.get("humans") or []:
        name = hm["name"]
        free(f"{name} pose", *hm["pose"][:2])
        for n, wp in enumerate(hm.get("waypoints") or []):
            if isinstance(wp, list):
                free(f"{name} waypoint {n}", *wp[:2])
        at = (hm.get("agent") or {}).get("agent_type", "adult")
        needs_of[name] = set()
        if isinstance(at, str) and at.startswith("./"):
            p = sdir / at
            try:
                t = load_agent_type_from_file(p)
                sample_agent_type(t, np.random.default_rng(0))
                needs_of[name] = set((t.needs or {}).keys())
                if t.mode != "behavior_tree":
                    problems.append(f"{name}: {at} is mode {t.mode}, not behavior_tree")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}: agent type {at}: {type(exc).__name__}: {exc}")
                continue
            at_raw = yaml.safe_load(p.read_text())
            seqs = set((at_raw.get("sequences") or {}).keys())
            for sname, seq in (at_raw.get("sequences") or {}).items():
                refs = [seq.get("then"), seq.get("on_failure")] + [tr.get("goto") for tr in seq.get("transitions") or []]
                for ref in refs:
                    if ref is not None and ref not in seqs:
                        problems.append(f"{name}: {at} sequence {sname} refers to unknown sequence {ref!r}")
            if at_raw.get("initial_sequence") and at_raw["initial_sequence"] not in seqs:
                problems.append(f"{name}: {at} initial_sequence {at_raw['initial_sequence']!r} unknown")
            for label, x, y in _poses_of_agent_type(yaml.safe_load(p.read_text())):
                free(f"{name} {label}", x, y)
            if hm.get("waypoints"):
                problems.append(f"{name}: behavior_tree agents ignore waypoints:, use go_to steps")
    for rname, reg in (raw.get("regions") or {}).items():
        for n, pt in enumerate(reg.get("polygon") or []):
            free(f"region {rname} corner {n}", float(pt["x"]), float(pt["y"]))
    for k, ent in enumerate(raw.get("timeline") or []):
        for st in ent.get("stimulus") or []:
            for ag in st.get("agents") or []:
                if st["need"] not in needs_of.get(ag, set()):
                    problems.append(f"timeline[{k}]: {ag} does not declare the need {st['need']!r}")
    return problems


def main() -> int:
    from fvlm3 import floorplan as fp  # noqa: PLC0415

    map_yaml, world_yaml = fp.find_world_files("hospital_1")
    sm = fp.load_static_map(map_yaml, world_yaml)
    names = sys.argv[1:] or sorted(p.name for p in (HERE / "scenarios").glob("fvlm3_*") if p.is_dir())
    bad = 0
    for n in names:
        d = pathlib.Path(n) if pathlib.Path(n).is_dir() else HERE / "scenarios" / n
        probs = check(d, sm)
        print(f"{'ok  ' if not probs else 'FAIL'} {d.name}")
        for p in probs:
            print(f"     - {p}")
        bad += bool(probs)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
