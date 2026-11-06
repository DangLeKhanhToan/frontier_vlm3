"""Kinematic replay of a scenario's people for the offline sim (tools/hospital_sim.py --scenario).

Reads scenarios/<name>/scenario.yaml and its path agent types and plays a subset of what
humansim does, enough to preview an episode before any GPU time:

- plain humans: start at `pose` after `spawn_delay`, walk the `waypoints` in order at
  `desired_velocity`, `dwell` at a waypoint, `waypoint_mode` repeat | reverse | once | random;
- behaviour-tree humans (mode: behavior_tree): sequences of `kind: go_to` with `target_pose`
  and pure-wait steps (`duration.mean`), `then`, `transitions` on `{need: {above|below: v}}`,
  a sequence `tune` and type `tunes` on desired_velocity / max_velocity (mul, add, set);
- the timeline: `at`, `every`, and `when` with the atoms `<a> within <r> of <b>` and
  `not <atom>` joined by `and` (a, b = robot, a human name, or a point x,y), firing on the
  false-to-true edge; `stimulus` sets the need to 100 x intensity after 0.4 s.

The people are blind: they walk straight lines and neither avoid each other nor the robot
(humansim's social force model does both). So the preview shows the timing of each event
relative to the robot, not how people would really make room. Contacts are counted, not resolved.
"""

from __future__ import annotations

import math
import pathlib
import random
import re

import numpy as np
import yaml

REACTION_S = 0.4
_ATOM = re.compile(r"^(not\s+)?(\S+)\s+within\s+([0-9.]+)\s+of\s+(\S+)$")


def _tuned(base: float, tune: dict | None, key: str) -> float:
    if not tune or key not in tune:
        return base
    t = tune[key]
    if isinstance(t, (int, float)):
        return float(t)
    if "poly" in t:
        return sum(c * base**k for k, c in enumerate(t["poly"]))
    return float(t.get("add", 0.0)) + float(t.get("mul", 1.0)) * base


def _need_ok(cond: dict, needs: dict) -> bool:
    for n, c in cond.items():
        v = needs.get(n, 0.0)
        if "above" in c and not v > c["above"]:
            return False
        if "below" in c and not v < c["below"]:
            return False
    return True


class Person:
    def __init__(self, idx: int, spec: dict, sdir: pathlib.Path, rng: random.Random):
        self.id = idx + 1
        self.name = spec["name"]
        self.pos = np.array(spec["pose"][:2], float)
        self.vel = np.zeros(2)
        self.spawn = float(spec.get("spawn_delay", 0.0))
        agent = spec.get("agent") or {}
        self.v0 = float(agent.get("desired_velocity", spec.get("velocity", 1.0)))
        self.rng = rng
        self.needs: dict[str, float] = {}
        self.pending: list[tuple[float, str, float]] = []
        at = agent.get("agent_type", "adult")
        self.bt = None
        if isinstance(at, str) and at.startswith("./"):
            t = yaml.safe_load((sdir / at).read_text())
            if t.get("mode") == "behavior_tree":
                self.bt = t
                self.needs = {n: float((d.get("initial") or {}).get("mean", 0.0)) for n, d in (t.get("needs") or {}).items()}
                self._enter(t.get("initial_sequence", "default"))
        self.wps = [np.array(w[:2], float) if isinstance(w, list) else np.array([w["x"], w["y"]], float) for w in spec.get("waypoints") or []]
        self.dwells = [0.0 if isinstance(w, list) else float(w.get("dwell", 0.0)) for w in spec.get("waypoints") or []]
        self.mode = spec.get("waypoint_mode", "repeat")
        self.k, self.dirn, self.hold = 0, 1, 0.0

    # behaviour tree subset
    def _enter(self, seq: str) -> None:
        self.seq = seq
        self.steps = list((self.bt["sequences"][seq].get("steps") or {}).items())
        self.si = 0
        self.wait = None

    def _speed(self) -> float:
        v, vmax = self.v0, math.inf
        if self.bt:
            for tu in self.bt.get("tunes") or []:
                if _need_ok(tu["when"], self.needs):
                    v = _tuned(v, tu["tune"], "desired_velocity")
                    vmax = _tuned(vmax, tu["tune"], "max_velocity")
            st = self.bt["sequences"][self.seq].get("tune")
            v = _tuned(v, st, "desired_velocity")
            vmax = _tuned(vmax, st, "max_velocity") if st else vmax
        return min(v, vmax)

    def stimulus(self, t: float, need: str, intensity: float) -> None:
        self.pending.append((t + REACTION_S, need, 100.0 * intensity))

    def _walk(self, target: np.ndarray, dt: float) -> bool:
        d = target - self.pos
        L = float(np.linalg.norm(d))
        step = self._speed() * dt
        if L <= max(step, 0.05):
            self.vel = (d / dt) if dt > 0 else np.zeros(2)
            self.pos = target.copy()
            return True
        self.vel = d / L * self._speed()
        self.pos = self.pos + d / L * step
        return False

    def update(self, t: float, dt: float) -> None:
        for p in [p for p in self.pending if p[0] <= t]:
            self.needs[p[1]] = p[2]
            self.pending.remove(p)
        if t < self.spawn:
            return
        self.vel = np.zeros(2)
        if self.bt:
            seq = self.bt["sequences"][self.seq]
            for tr in seq.get("transitions") or []:
                if tr["goto"] != self.seq and _need_ok(tr["when"], self.needs):
                    self._enter(tr["goto"])
                    break
            if self.si >= len(self.steps):
                self._enter(seq.get("then", self.seq))
            _, st = self.steps[self.si]
            if st.get("kind") == "go_to" and "target_pose" in st:
                tp = st["target_pose"]
                if self._walk(np.array([tp["x"], tp["y"]], float), dt):
                    self.si += 1
            else:
                if self.wait is None:
                    self.wait = float((st.get("duration") or {}).get("mean", 0.0))
                self.wait -= dt
                if self.wait <= 0:
                    self.wait = None
                    self.si += 1
            return
        if not self.wps or self.v0 <= 0:
            return
        if self.hold > 0:
            self.hold -= dt
            return
        if self._walk(self.wps[self.k], dt):
            self.hold = self.dwells[self.k]
            n = len(self.wps)
            if self.mode == "once":
                if self.k < n - 1:
                    self.k += 1
                else:
                    self.v0 = 0.0  # last waypoint reached: stand there
            elif self.mode == "reverse":
                if not 0 <= self.k + self.dirn < n:
                    self.dirn = -self.dirn
                self.k += self.dirn
            elif self.mode == "random":
                self.k = self.rng.randrange(n)
            else:
                self.k = (self.k + 1) % n


class ScenarioPeds:
    """people.update(t, robot_xy) advances to time t; .array() gives (id, x, y, vx, vy) rows."""

    def __init__(self, sdir: pathlib.Path, seed: int = 0):
        self.sdir = sdir
        self.raw = yaml.safe_load((sdir / "scenario.yaml").read_text())
        rng = random.Random(seed)
        self.people = [Person(k, h, sdir, rng) for k, h in enumerate(self.raw.get("humans") or [])]
        self.by_name = {p.name: p for p in self.people}
        self.timeline = []
        for k, e in enumerate(self.raw.get("timeline") or []):
            self.timeline.append({"k": k, **e, "_last": False, "_next": e.get("offset", e.get("every"))})
        self.t = 0.0
        self.events: list[tuple[float, str]] = []
        rb = (self.raw.get("robots") or [{}])[0]
        self.start = [float(v) for v in rb.get("start", [10.0, 6.0, 1.5708])]
        if len(self.start) == 2:
            self.start.append(0.0)
        self.phase = (rb.get("phases") or [{}])[0]
        self.instruction = self.phase.get("text") or self.phase.get("instruction") or "Go to the laboratory."

    def _xy(self, s: str, robot: np.ndarray) -> np.ndarray | None:
        if s == "robot" or s == "jackal":
            return robot
        if s in self.by_name:
            p = self.by_name[s]
            return p.pos if self.t >= p.spawn else None
        if "," in s:
            return np.array([float(v) for v in s.split(",")[:2]])
        raise ValueError(f"unknown subject {s!r}")

    def _holds(self, cond: str, robot: np.ndarray) -> bool:
        for atom in cond.split(" and "):
            m = _ATOM.match(atom.strip())
            if not m:
                raise ValueError(f"offline preview cannot judge {atom!r}")
            a, b = self._xy(m.group(2), robot), self._xy(m.group(4), robot)
            ok = a is not None and b is not None and float(np.linalg.norm(a - b)) <= float(m.group(3))
            if bool(m.group(1)) == ok:
                return False
        return True

    def _fire(self, e: dict, why: str) -> None:
        for st in e.get("stimulus") or []:
            for ag in st["agents"]:
                self.by_name[ag].stimulus(self.t, st["need"], float(st.get("intensity", 1.0)))
            self.events.append((self.t, f"{why}: stimulus {st['need']} -> {', '.join(st['agents'])}"))

    def update(self, t: float, robot_xy) -> None:
        robot = np.asarray(robot_xy[:2], float)
        while self.t < t - 1e-9:
            dt = min(0.05, t - self.t)
            self.t += dt
            for e in self.timeline:
                if "at" in e and not e["_last"] and self.t >= float(e["at"]):
                    e["_last"] = True
                    self._fire(e, f"at {e['at']}")
                elif "every" in e and e["_next"] is not None and self.t >= e["_next"]:
                    e["_next"] += float(e["every"])
                    self._fire(e, f"every {e['every']}")
                elif "when" in e and isinstance(e["when"], str):
                    now = self._holds(e["when"], robot)
                    if now and not e["_last"]:
                        self._fire(e, f"when {e['when']!r}")
                    e["_last"] = now
            for p in self.people:
                p.update(self.t, dt)

    def array(self) -> np.ndarray:
        rows = [[p.id, p.pos[0], p.pos[1], p.vel[0], p.vel[1]] for p in self.people if self.t >= p.spawn]
        return np.array(rows, np.float32).reshape(-1, 5)

    def names(self) -> dict[int, str]:
        return {p.id: p.name for p in self.people}

    def judge(self, pose: np.ndarray, zones: list[dict]) -> tuple[bool, str]:
        """Would Arena count the goto as met where the robot stands now (tolerance_radius)?"""
        tol = float(self.phase.get("tolerance_radius", 0.5))
        goal = self.phase.get("goto")
        xy = np.asarray(pose[:2], float)
        if isinstance(self.phase.get("pose"), list):
            d = float(np.linalg.norm(xy - np.array(self.phase["pose"][:2])))
            return d <= tol, f"{d:.1f} m from the goal pose (tolerance {tol})"
        if isinstance(goal, list):
            d = float(np.linalg.norm(xy - np.array(goal[:2])))
            return d <= tol, f"{d:.1f} m from the goal pose (tolerance {tol})"
        if goal in self.by_name:
            d = float(np.linalg.norm(xy - self.by_name[goal].pos))
            return d <= tol, f"{d:.1f} m from {goal} (tolerance {tol})"
        import cv2  # noqa: PLC0415

        for z in zones:
            if z.get("name") == goal:
                inside = cv2.pointPolygonTest(np.asarray(z["polygon"], np.float32), (float(xy[0]), float(xy[1])), True)
                return inside >= -tol, f"{'inside' if inside >= 0 else f'{-inside:.1f} m outside'} zone {goal} (tolerance {tol})"
        return False, f"goal {goal!r} not judged offline"
