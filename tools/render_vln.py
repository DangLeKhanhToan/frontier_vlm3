#!/usr/bin/env python3
"""Render one episode of ANY bridge planner (CMA, NaVid, NaVILA, StreamVLN, frontier_vlm3) as a
1920x1080 video from the SDK decision trace. CONTAINER or any Python with numpy, opencv, pillow, pyyaml.

  python3 tools/render_vln.py <trace_episode_dir> --run <run_dir> [--world hospital_1] [--out x.mp4]
  python3 tools/render_vln.py <trace_episode_dir> --run <run_dir> --still 20     # one PNG at t = 20 s
  python3 tools/render_vln.py --all <run_dir>                                     # every trace under <run_dir>/trace

<trace_episode_dir> = <ARENA_PLANNER_TRACE_DIR>/<planner>/<episode_id>/ (trace.jsonl, frames/), see
arena_planners/trace.py. <run_dir> holds the recordings of the run (all optional):
  rec/   cam_recorder.py of <ns>/vln_camera/rec_image   (robot view, 30 Hz)
  chase/ cam_recorder.py of <ns>/vln_camera/chase_image (third-person view, 30 Hz)
  peds.csv  peds_recorder.py (people positions)
  run.json  scenario, planner, model (shown in the header)

Layout
  header  planner, scenario, instruction, simulation time, what the robot does now
  strip   timeline: one tick per model call (blue), colour = command executed (green forward,
          orange turn, red stop/arrived, grey hold); white cursor = now
  top     floor plan (trail red, people green, model calls = blue dots) | third-person chase
          camera | robot camera (30 Hz) with the current command underneath
  bottom  INPUT: the frame the model saw at its last call + the instruction it was given |
          OUTPUT: the model's raw answer, latency, and the commands parsed from it |
          EXECUTION: who drives the robot and how (Arena bridge primitives / pure pursuit),
          the command queue, measured speed
Time axis: the observation stamps of the trace (simulation time), the recordings' header stamps.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import pathlib
import sys

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fvlm3 import floorplan as fp  # noqa: E402
from tools.render_episode import F_BODY, F_BOLD, F_HEAD, F_MONO, F_SMALL, F_TITLE, INK, MUTED, PANEL, PAPER, PERSON, ROBOT, TRAIL, WALL, Writer, Xf, load_jsonl, wrap_text  # noqa: E402

W, H = 1920, 1080
HEADER_H, STRIP_H = 50, 26
TOP_Y = HEADER_H + STRIP_H
TOP_H = 600
BOT_Y = TOP_Y + TOP_H
BOT_H = H - BOT_Y
MAP_W, CHASE_W = 470, 800
CAM_X = MAP_W + CHASE_W
CAM_W = W - CAM_X
CMD_COLORS = {"forward": (70, 160, 60), "left": (0, 140, 255), "right": (0, 140, 255), "stop": (40, 40, 210), "chunk": (70, 160, 60), "hold": (150, 150, 150)}
CALL = (210, 110, 20)
PLANNERS = pathlib.Path(__file__).resolve().parents[2]


def _put(img: np.ndarray, draw_fn) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw_fn(ImageDraw.Draw(pil))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _rgb(c):  # BGR -> RGB for PIL
    return (c[2], c[1], c[0])


class Recording:
    """cam_recorder output: frames/ + index.csv (frame, stamp_s, wall_s)."""

    def __init__(self, d: pathlib.Path | None):
        self.t: list[float] = []
        self.p: list[pathlib.Path] = []
        if d is None or not (d / "index.csv").is_file():
            return
        rows = []
        with open(d / "index.csv") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append((float(r["stamp_s"]), d / "frames" / f"{int(r['frame']):06d}.jpg"))
                except (KeyError, ValueError):
                    continue
        rows.sort()
        self.t, self.p = [r[0] for r in rows], [r[1] for r in rows]
        self._cache: tuple[int, np.ndarray] | None = None

    def __bool__(self) -> bool:
        return bool(self.t)

    def at(self, T: float) -> np.ndarray | None:
        if not self.t:
            return None
        i = max(0, bisect.bisect_right(self.t, T) - 1)
        if self._cache and self._cache[0] == i:
            return self._cache[1]
        img = cv2.imread(str(self.p[i]))
        self._cache = (i, img)
        return img


def _primitives(planner: str) -> dict:
    m = PLANNERS / planner / "planner.yaml"
    if m.is_file():
        man = yaml.safe_load(m.read_text()) or {}
        return {"action_type": man.get("action_type", "?"), **{"forward_m": 0.25, "turn_deg": 15.0}, **(man.get("primitives") or {}), "chunk": man.get("chunk") or {}}
    return {"action_type": "?", "forward_m": 0.25, "turn_deg": 15.0, "chunk": {}}


def describe(step: dict, prim: dict) -> str:
    if step.get("signal"):
        return f"signal {step['signal']} (model said stop)"
    c, a = step.get("command") or "", float(step.get("amount") or 0.0)
    if c == "forward":
        return f"forward {a if a > 0 else prim['forward_m']:.2f} m"
    if c in ("left", "right"):
        return f"turn {c} {a if a > 0 else prim['turn_deg']:.0f} deg"
    if step.get("chunk"):
        ch = np.asarray(step["chunk"], dtype=float)
        L = float(np.linalg.norm(np.diff(np.vstack([[0, 0], ch[:, :2]]), axis=0), axis=1).sum()) if ch.ndim == 2 and ch.shape[1] >= 2 else 0.0
        return f"waypoints: {len(ch)} pts, {L:.2f} m path"
    return "hold (no command)"


def kind(step: dict) -> str:
    if step.get("signal"):
        return "stop"
    if step.get("command"):
        return step["command"]
    if step.get("chunk"):
        return "chunk"
    return "hold"


def load_frontier(tdir: pathlib.Path, walls: list[float]) -> tuple[list[dict], list[dict], pathlib.Path | None]:
    """frontier_vlm3 keeps its decisions in the call log, not in trace logs: return (calls, ticks, log dir)."""
    if tdir.parent.name != "frontier_vlm3":
        return [], [], None
    dirs = sorted(pathlib.Path("/opt/arena_ws/data/vlm_calls").glob(f"*/ep*_{tdir.name}"))
    if not dirs:
        return [], [], None
    recs = load_jsonl(dirs[-1] / "calls.jsonl")
    sent = {r["call"]: r for r in recs if r.get("phase") == "sent"}
    calls = []
    for r in recs:
        if r.get("phase") == "answer":
            i = min(bisect.bisect_left(walls, float(r["wall"])), len(walls) - 1)
            calls.append({**r, "i": i, "sent": sent.get(r["call"], {})})
    ticks = load_jsonl(dirs[-1] / "ticks.jsonl") if (dirs[-1] / "ticks.jsonl").is_file() else []
    return calls, ticks, dirs[-1]


class Episode:
    def __init__(self, tdir: pathlib.Path, run: pathlib.Path | None, world: str):
        self.dir = tdir
        recs = load_jsonl(tdir / "trace.jsonl")
        if not recs:
            raise SystemExit(f"{tdir}: no trace.jsonl")
        self.reset = next((r for r in recs if r["ev"] == "reset"), {"initial_state": {}})
        self.steps = [r for r in recs if r["ev"] == "step"]
        if not self.steps:
            raise SystemExit(f"{tdir}: no steps in the trace")
        self.planner = tdir.parent.name
        self.prim = _primitives(self.planner)
        self.instruction = (self.reset.get("initial_state") or {}).get("instruction", "")
        self.st = [float(s.get("stamp") or 0.0) for s in self.steps]
        if self.st[-1] - self.st[0] < 1e-6:  # no stamps: fall back to the host clock
            self.st = [float(s["wall"]) - float(self.steps[0]["wall"]) for s in self.steps]
        self.T0, self.T1 = self.st[0], self.st[-1]
        self.calls = [i for i, s in enumerate(self.steps) if s.get("logs")]
        self.frontier, self.ticks, self.call_dir = load_frontier(tdir, [float(s["wall"]) for s in self.steps])
        if self.frontier:
            self.calls = [c["i"] for c in self.frontier]
        self.tick_w = [float(t["wall"]) for t in self.ticks]
        self.conditions = (load_jsonl(self.call_dir / "meta.jsonl")[0].get("conditions") or {}) if self.call_dir else {}
        self.run = run
        self.meta = json.loads((run / "run.json").read_text()) if run and (run / "run.json").is_file() else {}
        self.rec = Recording(run / "rec" if run and (run / "rec").is_dir() else run if run and (run / "index.csv").is_file() else None)
        self.chase = Recording(run / "chase" if run else None)
        self.peds: dict[float, list] = {}
        self.ped_t: list[float] = []
        pc = run / "peds.csv" if run else None
        if pc and pc.is_file():
            with open(pc) as fh:
                for r in csv.DictReader(fh):
                    t = float(r.get("stamp_s") or r.get("t"))
                    self.peds.setdefault(t, []).append((float(r["x"]), float(r["y"]), r.get("name", "")))
            self.ped_t = sorted(self.peds)
        self.world = world or self.meta.get("world", "")
        self.static = None
        if self.world:
            my, wy = fp.find_world_files(self.world)
            if my:
                self.static = fp.load_static_map(my, wy)
        poses = [s.get("pose") for s in self.steps]
        self.poses = np.array([p if p is not None else [np.nan] * 3 for p in poses], dtype=float)
        # Gazebo poses and people carry Arena's env offset; the floor plan is in authored coordinates
        dx, dy = self._env_offset(wy if self.world and my else None)
        self.poses[:, :2] -= (dx, dy)
        self.peds = {t: [(x - dx, y - dy, n) for x, y, n in v] for t, v in self.peds.items()}

    def _env_offset(self, world_yaml) -> tuple[float, float]:
        """Env offset of this run: the first recorded pose minus the scenario's robot start, (0, 0) when unknown."""
        name = pathlib.Path(str(self.meta.get("scenario", ""))).name
        valid = self.poses[~np.isnan(self.poses[:, 0])]
        if not (world_yaml and name and len(valid)):
            return 0.0, 0.0
        path = pathlib.Path(world_yaml).parent / "scenarios" / name / "scenario.yaml"
        if not path.is_file():
            return 0.0, 0.0
        import yaml  # noqa: PLC0415

        robots = yaml.safe_load(path.read_text()).get("robots") or []
        if not robots or "start" not in robots[0]:
            return 0.0, 0.0
        sx, sy = robots[0]["start"][:2]
        return round(float(valid[0, 0] - sx), 2), round(float(valid[0, 1] - sy), 2)

    def idx(self, T: float) -> int:
        return max(0, bisect.bisect_right(self.st, T) - 1)

    def people(self, T: float) -> list:
        if not self.ped_t:
            return []
        i = max(0, bisect.bisect_right(self.ped_t, T) - 1)
        return self.peds[self.ped_t[i]]


class Renderer:
    def __init__(self, ep: Episode):
        self.ep = ep
        self.base, self.xf = self._build_map()

    def _build_map(self):
        ep = self.ep
        img = np.full((TOP_H, MAP_W, 3), PANEL, np.uint8)
        valid = self.ep.poses[~np.isnan(self.ep.poses[:, 0])]
        if ep.static is None:
            if len(valid):
                cx, cy = valid[:, 0].mean(), valid[:, 1].mean()
            else:
                cx = cy = 0.0
            k = (MAP_W - 40) / 30.0
            return img, Xf(k, cx - 15.0, cy + TOP_H / k / 2, 20, 0)
        sm = ep.static
        h, w = sm.occ.shape
        wx, wy = w * sm.res, h * sm.res
        k = min((MAP_W - 30) / wx, (TOP_H - 70) / wy)
        xf = Xf(k, sm.origin[0], sm.origin[1] + wy, (MAP_W - wx * k) / 2, 56)
        for z in sm.zones:
            pts = xf.many(z["polygon"])
            cv2.fillPoly(img, [pts], (238, 232, 226))
        occ = np.flipud(sm.occ).astype(np.uint8) * 255
        occ = cv2.resize(occ, (int(round(wx * k)), int(round(wy * k))), interpolation=cv2.INTER_AREA)
        x0, y0 = xf(sm.origin[0], sm.origin[1] + wy)
        sub = img[y0 : y0 + occ.shape[0], x0 : x0 + occ.shape[1]]
        m = occ[: sub.shape[0], : sub.shape[1]] > 60
        sub[m] = WALL
        return img, xf

    # ------------------------------------------------------------- panels
    def _map(self, T: float) -> np.ndarray:
        ep, xf = self.ep, self.xf
        img = self.base.copy()
        i = ep.idx(T)
        tr = ep.poses[: i + 1]
        tr = tr[~np.isnan(tr[:, 0])]
        if len(tr) > 1:
            cv2.polylines(img, [xf.many(tr[:, :2])], False, TRAIL, 2, cv2.LINE_AA)
        for c in ep.calls:
            if c <= i and not np.isnan(ep.poses[c, 0]):
                cv2.circle(img, xf(*ep.poses[c, :2]), 3, CALL, -1, cv2.LINE_AA)
        for x, y, _n in ep.people(T):
            cv2.circle(img, xf(x, y), max(3, int(0.25 * xf.k)), PERSON, -1, cv2.LINE_AA)
        if len(tr):
            x, y, th = tr[-1]
            r = max(4, int(0.3 * xf.k))
            cv2.circle(img, xf(x, y), r, ROBOT, -1, cv2.LINE_AA)
            cv2.line(img, xf(x, y), xf(x + 0.8 * math.cos(th), y + 0.8 * math.sin(th)), ROBOT, 2, cv2.LINE_AA)
        img = _put(img, lambda d: (d.text((12, 8), "Floor plan (not given to the VLN)", font=F_TITLE, fill=_rgb(INK)), d.text((12, 30), "red = driven   green = people   blue dots = model calls", font=F_SMALL, fill=_rgb(MUTED))))
        return img

    @staticmethod
    def _fit(src: np.ndarray | None, w: int, h: int, label: str) -> np.ndarray:
        out = np.full((h, w, 3), (60, 60, 60), np.uint8)
        if src is None:
            return _put(out, lambda d: d.text((20, h // 2 - 10), f"{label}: no recording", font=F_BODY, fill=(220, 220, 220)))
        sh, sw = src.shape[:2]
        k = min(w / sw, h / sh)
        im = cv2.resize(src, (int(sw * k), int(sh * k)), interpolation=cv2.INTER_AREA)
        y0, x0 = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
        out[y0 : y0 + im.shape[0], x0 : x0 + im.shape[1]] = im
        return out

    def _cam(self, T: float) -> np.ndarray:
        ep = self.ep
        cam_h = int(CAM_W * 3 / 4)
        panel = np.full((TOP_H, CAM_W, 3), PANEL, np.uint8)
        panel[:cam_h] = self._fit(ep.rec.at(T), CAM_W, cam_h, "robot camera")
        s = ep.steps[ep.idx(T)]
        k = kind(s)
        col = CMD_COLORS.get(k, MUTED)
        txt = describe(s, ep.prim)

        def d(dr):
            dr.rectangle([0, cam_h, CAM_W, TOP_H], fill=_rgb(PAPER))
            dr.text((14, cam_h + 10), "Executing now", font=F_SMALL, fill=_rgb(MUTED))
            dr.text((14, cam_h + 30), txt, font=font_big, fill=_rgb(col))
            dr.text((14, cam_h + 72), f"robot camera, 30 Hz (the planner itself sees its own camera at its own rate)", font=F_SMALL, fill=_rgb(MUTED))

        return _put(panel, d)

    def _chase(self, T: float) -> np.ndarray:
        img = self._fit(self.ep.chase.at(T), CHASE_W, TOP_H, "third-person camera")
        return _put(img, lambda d: d.text((12, 8), "Third-person view", font=F_TITLE, fill=(255, 255, 255)))

    def _call_img(self, call: int, suffix: str, size: int) -> np.ndarray:
        key = (call, suffix, size)
        if not hasattr(self, "_img_cache"):
            self._img_cache = {}
        if key not in self._img_cache:
            src = cv2.imread(str(self.ep.call_dir / "images" / f"c{call:03d}_{suffix}.jpg"))
            self._img_cache[key] = self._fit(None if src is None else cv2.resize(src, (size, size)), size, size, "image")
        return self._img_cache[key]

    def _bottom_frontier(self, T: float) -> np.ndarray:
        """frontier_vlm3: the image the model saw, the sub-goals it chose, its reasoning, and what the harness did."""
        ep = self.ep
        img = np.full((BOT_H, W, 3), PAPER, np.uint8)
        i = ep.idx(T)
        done = [c for c in ep.frontier if c["i"] <= i]
        c = done[-1] if done else None
        aw, ow, tile = 500, 860, 300
        if c:
            img[40:40 + tile, 14:14 + tile] = self._call_img(c["call"], "in0", tile)
            img[40:40 + tile, aw + 10:aw + 10 + tile] = self._call_img(c["call"], "answer", tile)
        tick = None
        if ep.ticks:
            tick = ep.ticks[min(bisect.bisect_left(ep.tick_w, float(ep.steps[i]["wall"])), len(ep.ticks) - 1)]
        cost = sum(float(x.get("cost_usd") or 0) for x in done)

        def d(dr):
            dr.text((14, 10), "MODEL INPUT", font=F_HEAD, fill=_rgb(INK))
            tx = 14 + tile + 10
            if not c:
                dr.text((14, 44), "no model call yet", font=F_BODY, fill=_rgb(MUTED))
            else:
                snt = c["sent"]
                hist = str(snt.get("text", "")).count("\n  - t=")
                y = 44
                cond = ep.conditions
                for ln in [f"decision {c['call']}", f"called: {snt.get('reason', '?')}", f"previous plan left: {float(snt.get('plan_left_m') or 0):.1f} m", f"recent decisions sent: {hist}", f"floor plan: {'sent' if cond.get('floorplan') else 'not sent'}", f"people list: {'sent' if cond.get('people') else 'not sent'}"]:
                    dr.text((tx, y), ln, font=F_SMALL, fill=_rgb(INK))
                    y += 18
                y += 8
                for ln in wrap_text("Also sent: system prompt with the social rules. Depth is used only to place the points, the model does not see it.", F_SMALL, aw - tx - 8)[:7]:
                    dr.text((tx, y), ln, font=F_SMALL, fill=_rgb(MUTED))
                    y += 16
            y = 40 + tile + 8
            for ln in wrap_text(f'Instruction: "{ep.instruction}"', F_SMALL, aw - 28)[:3]:
                dr.text((14, y), ln, font=F_SMALL, fill=_rgb(INK))
                y += 16
            x = aw + 10
            dr.line([(aw, 8), (aw, BOT_H - 8)], fill=(210, 205, 200), width=1)
            dr.text((x, 10), "MODEL OUTPUT", font=F_HEAD, fill=_rgb(INK))
            if c:
                dr.text((x, 40 + tile + 8), "red: sub-goals chosen by the model", font=F_SMALL, fill=_rgb(MUTED))
                rep = c.get("reply") or {}
                tx, tw = x + tile + 10, ow - tile - 30
                y = 12
                dr.text((tx, y), f"call {c['call']} at t = {ep.st[c['i']] - ep.T0:.1f} s, model took {float(c.get('latency_s') or 0):.1f} s", font=F_BOLD, fill=_rgb(CALL))
                y = 34

                def block(text, font, fill, limit):
                    nonlocal y
                    for ln in wrap_text(text, font, tw)[:limit]:
                        if y < BOT_H - 18:
                            dr.text((tx, y), ln, font=font, fill=_rgb(fill))
                            y += 15

                block(f"action: {rep.get('action', c.get('outcome') or 'no usable reply')}   progress: {rep.get('progress', '-')}   outcome: {c.get('outcome')}", F_BOLD, INK, 2)
                y += 3
                block("Scene: " + str(rep.get("scene", "")), F_SMALL, INK, 5)
                y += 3
                proj = c.get("projected") or []
                for n, g in enumerate(rep.get("subgoals") or []):
                    src = f", {proj[n]['range_m']:.1f} m by {proj[n]['source']}" if n < len(proj) and proj[n].get("range_m") else ""
                    block(f"{n + 1}. ({g.get('u')}, {g.get('v')}){src}: {g.get('why', '')}", F_SMALL, INK, 3)
                y += 3
                block("Reasoning: " + str(rep.get("path_reasoning", "")), F_SMALL, INK, 6)
                if rep.get("rules_applied"):
                    block("Rules: " + ", ".join(rep["rules_applied"]), F_SMALL, MUTED, 1)
            x = aw + ow + 20
            dr.line([(aw + ow + 10, 8), (aw + ow + 10, BOT_H - 8)], fill=(210, 205, 200), width=1)
            dr.text((x, 10), "EXECUTION", font=F_HEAD, fill=_rgb(INK))
            dwb = ep.meta.get("executor") == "dwb"
            how = "Pixels + depth -> map points -> A* path -> " + ("Nav2 DWB drives it (FollowPath)." if dwb else "pure pursuit on the waypoint chunk.")
            y = 44
            for ln in wrap_text(how, F_SMALL, W - x - 16):
                dr.text((x, y), ln, font=F_SMALL, fill=_rgb(INK))
                y += 16
            y += 12
            if tick:
                dr.text((x, y), "Harness now", font=F_SMALL, fill=_rgb(MUTED))
                stuck = tick.get("why") == "safety_stop"
                dr.text((x, y + 18), f"{tick.get('motion')}: {tick.get('why')}", font=F_BOLD, fill=(190, 60, 30) if stuck else _rgb(INK))
                dr.text((x, y + 40), f"plan left: {float(tick.get('plan_left_m') or 0):.1f} m", font=F_SMALL, fill=_rgb(INK))
                y += 70
            dr.text((x, y), f"model calls: {len(done)}   cost so far: ${cost:.2f}", font=F_SMALL, fill=_rgb(INK))
            v, w_ = self._speed(i)
            dr.text((x, BOT_H - 30), f"measured: v = {v:.2f} m/s, omega = {w_:+.2f} rad/s", font=F_SMALL, fill=_rgb(INK))

        return _put(img, d)

    def _bottom(self, T: float) -> np.ndarray:
        if self.ep.frontier:
            return self._bottom_frontier(T)
        ep = self.ep
        img = np.full((BOT_H, W, 3), PAPER, np.uint8)
        i = ep.idx(T)
        last = max([c for c in ep.calls if c <= i], default=None)
        aw, ow = 560, 700
        # INPUT
        if last is not None and ep.steps[last].get("frame"):
            fr = cv2.imread(str(ep.dir / ep.steps[last]["frame"]))
            if fr is not None:
                tile = self._fit(fr, 400, 300, "input")
                img[40:340, 14:414] = tile
        n_calls = sum(1 for c in ep.calls if c <= i)

        def d(dr):
            dr.text((14, 10), "MODEL INPUT", font=F_HEAD, fill=_rgb(INK))
            if last is None:
                dr.text((14, 44), "no model call yet", font=F_BODY, fill=_rgb(MUTED))
            else:
                dr.text((424, 44), f"frame of call {n_calls}", font=F_SMALL, fill=_rgb(MUTED))
                dr.text((424, 62), f"t = {ep.st[last] - ep.T0:.1f} s", font=F_SMALL, fill=_rgb(MUTED))
            y = 350
            for ln in wrap_text(f'Instruction: "{ep.instruction}"', F_BODY, aw - 28)[:3]:
                dr.text((14, y), ln, font=F_BODY, fill=_rgb(INK))
                y += 18
            # OUTPUT
            x = aw + 10
            dr.line([(aw, 8), (aw, BOT_H - 8)], fill=(210, 205, 200), width=1)
            dr.text((x, 10), "MODEL OUTPUT", font=F_HEAD, fill=_rgb(INK))
            y = 44
            if last is not None:
                s = ep.steps[last]
                dr.text((x, y), f"call {n_calls} at t = {ep.st[last] - ep.T0:.1f} s, step latency {float(s.get('latency_s', 0)):.2f} s", font=F_BOLD, fill=_rgb(CALL))
                y += 24
                for msg in s.get("logs", [])[:4]:
                    for ln in wrap_text(msg, F_MONO, ow - 30)[:5]:
                        dr.text((x, y), ln, font=F_MONO, fill=_rgb(INK))
                        y += 17
                y += 8
                dr.text((x, y), "-> parsed into: " + describe(s, ep.prim), font=F_BOLD, fill=_rgb(CMD_COLORS.get(kind(s), MUTED)))
                y += 30
            dr.text((x, y), "Earlier calls", font=F_BOLD, fill=_rgb(MUTED))
            y += 20
            for c in [c for c in ep.calls if c <= i][-7:-1][::-1]:
                msg = (ep.steps[c].get("logs") or [""])[0]
                msg = msg.split(": ", 1)[-1]
                ln = wrap_text(f"{ep.st[c] - ep.T0:5.1f} s  {msg}", F_SMALL, ow - 30)[0]
                dr.text((x, y), ln, font=F_SMALL, fill=_rgb(MUTED))
                y += 16
            # EXECUTION
            x = aw + ow + 20
            dr.line([(aw + ow + 10, 8), (aw + ow + 10, BOT_H - 8)], fill=(210, 205, 200), width=1)
            dr.text((x, 10), "EXECUTION", font=F_HEAD, fill=_rgb(INK))
            at = ep.prim["action_type"]
            if at == "discrete":
                how = f"Arena bridge, discrete primitives, closed loop on robot_pose: forward {ep.prim['forward_m']} m, turn {ep.prim['turn_deg']} deg, at up to 0.5 m/s / 1 rad/s. Next observation after each move. No DWB, no map."
            elif at == "waypoints":
                ch = ep.prim.get("chunk") or {}
                how = f"Arena bridge, pure pursuit on the waypoint chunk (lookahead {ch.get('lookahead_m', 0.3)} m, {ch.get('speed_mps', 0.5)} m/s, {ch.get('execute_s', 1.0)} s per chunk). No DWB."
            else:
                how = f"action_type {at}"
            y = 44
            for ln in wrap_text(how, F_BODY, W - x - 20):
                dr.text((x, y), ln, font=F_BODY, fill=_rgb(INK))
                y += 18
            y += 10
            dr.text((x, y), "Commands sent (newest first)", font=F_BOLD, fill=_rgb(MUTED))
            y += 20
            shown, j = 0, i
            prev = None
            while j >= 0 and shown < 9:
                s = ep.steps[j]
                dsc = describe(s, ep.prim)
                if kind(s) != "hold" and (dsc, ep.st[j]) != prev:
                    dr.text((x, y), f"{ep.st[j] - ep.T0:5.1f} s  {dsc}", font=F_MONO if shown else F_BOLD, fill=_rgb(CMD_COLORS.get(kind(s), MUTED)) if shown == 0 else _rgb(INK))
                    y += 17
                    shown += 1
                    prev = (dsc, ep.st[j])
                j -= 1
            v, w_ = self._speed(i)
            dr.text((x, BOT_H - 30), f"measured: v = {v:.2f} m/s, omega = {w_:+.2f} rad/s", font=F_BODY, fill=_rgb(INK))

        return _put(img, d)

    def _speed(self, i: int) -> tuple[float, float]:
        ep = self.ep
        j = max(0, i - 5)
        a, b = ep.poses[j], ep.poses[i]
        dt = ep.st[i] - ep.st[j]
        if dt <= 0 or np.isnan(a[0]) or np.isnan(b[0]):
            return 0.0, 0.0
        dth = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
        return float(np.hypot(b[0] - a[0], b[1] - a[1]) / dt), float(dth / dt)

    def _header_strip(self, img: np.ndarray, T: float) -> np.ndarray:
        ep = self.ep
        meta = ep.meta
        x0, x1 = 14, W - 14
        span = max(1e-6, ep.T1 - ep.T0)
        px = lambda t: int(x0 + (t - ep.T0) / span * (x1 - x0))  # noqa: E731
        for j in range(len(ep.steps) - 1):
            cv2.rectangle(img, (px(ep.st[j]), HEADER_H + 6), (px(ep.st[j + 1]) + 1, HEADER_H + STRIP_H - 4), CMD_COLORS.get(kind(ep.steps[j]), MUTED), -1)
        for c in ep.calls:
            cv2.line(img, (px(ep.st[c]), HEADER_H + 1), (px(ep.st[c]), HEADER_H + 7), CALL, 2)
        cv2.line(img, (px(T), HEADER_H + 2), (px(T), HEADER_H + STRIP_H - 1), (255, 255, 255), 3)
        cv2.line(img, (px(T), HEADER_H + 2), (px(T), HEADER_H + STRIP_H - 1), INK, 1)
        model = meta.get("model") or ep.planner
        scen = pathlib.Path(str(meta.get("scenario", ""))).name or "?"
        i = ep.idx(T)
        n_calls = sum(1 for c in ep.calls if c <= i)

        def d(dr):
            dr.text((14, 6), f"{ep.planner}  ({model})", font=F_HEAD, fill=_rgb(INK))
            dr.text((14, 28), f"scenario {scen}   run {meta.get('run_id', ep.dir.parent.parent.name)}   episode {ep.dir.name}", font=F_SMALL, fill=_rgb(MUTED))
            dr.text((640, 6), wrap_text(f'"{ep.instruction}"', F_BODY, 900)[0], font=F_BODY, fill=_rgb(INK))
            dr.text((W - 330, 6), f"t = {T - ep.T0:6.1f} s / {span:.0f} s", font=F_HEAD, fill=_rgb(INK))
            dr.text((W - 330, 28), f"model calls so far: {n_calls}", font=F_SMALL, fill=_rgb(MUTED))

        return _put(img, d)

    def frame(self, T: float) -> np.ndarray:
        img = np.full((H, W, 3), PAPER, np.uint8)
        img[TOP_Y : TOP_Y + TOP_H, :MAP_W] = self._map(T)
        img[TOP_Y : TOP_Y + TOP_H, MAP_W : MAP_W + CHASE_W] = self._chase(T)
        img[TOP_Y : TOP_Y + TOP_H, CAM_X:] = self._cam(T)
        img[BOT_Y:] = self._bottom(T)
        cv2.line(img, (0, BOT_Y), (W, BOT_Y), (210, 205, 200), 1)
        return self._header_strip(img, T)


from tools.render_episode import font  # noqa: E402

font_big = font(26, bold=True)


def render(tdir: pathlib.Path, a: argparse.Namespace, run: pathlib.Path | None) -> pathlib.Path:
    ep = Episode(tdir, run, a.world)
    r = Renderer(ep)
    print(f"{tdir}: {len(ep.steps)} steps, {len(ep.calls)} model calls, rec {len(ep.rec.t)} / chase {len(ep.chase.t)} frames, {len(ep.ped_t)} people samples, {ep.T1 - ep.T0:.1f} s")
    if a.still is not None:
        out = pathlib.Path(a.out) if a.out else tdir / f"still_{a.still:.1f}.png"
        cv2.imwrite(str(out), r.frame(ep.T0 + a.still))
        print(f"wrote {out}")
        return out
    out = pathlib.Path(a.out) if a.out and not a.all else tdir / "episode.mp4"
    wr = Writer(out, a.fps)
    n = int((ep.T1 - ep.T0) / (a.speed / a.fps)) + 1
    for k in range(n):
        wr.write(r.frame(ep.T0 + k * a.speed / a.fps))
    wr.close()
    print(f"wrote {out} ({n} frames)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="trace episode dir, or a run dir with --all")
    ap.add_argument("--run", default="", help="run dir with rec/, chase/, peds.csv, run.json (default: 3 levels above the trace episode)")
    ap.add_argument("--world", default="", help="world for the floor plan (default: run.json world)")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    p = pathlib.Path(a.path)
    if a.all:
        for t in sorted(p.glob("trace/*/*/trace.jsonl")):
            render(t.parent, a, p)
        return
    run = pathlib.Path(a.run) if a.run else p.parents[2] if len(p.parents) > 2 else None
    render(p, a, run)


if __name__ == "__main__":
    main()
