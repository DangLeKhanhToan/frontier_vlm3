#!/usr/bin/env python3
"""Render one frontier_vlm3 episode as a 1920x1080 video (default 30 fps). CONTAINER or any
Python with numpy, opencv, pillow, pyyaml (+ ffmpeg for H.264).

  python3 tools/render_episode.py <episode_dir> [--frames <rec_dir>] [--out episode.mp4]
  python3 tools/render_episode.py <episode_dir> --still 42.0          # one PNG at t = 42 s
  python3 tools/render_episode.py <run_dir> --all                      # every episode of a run

<episode_dir> is a planner log dir: /opt/arena_ws/data/vlm_calls/<run_label>/ep001_<id>/.
<rec_dir> is a cam_recorder.py output; default /opt/arena_ws/data/vlm_video/<run_label>
(i.e. <log_dir>/../vlm_video/<run_label>). Without it the camera panel shows the last image
sent to the model instead of the 30 Hz stream.

Layout
  header   run, conditions, instruction, time, what the robot is doing, calls, cost
  strip    timeline: blue = a model call in flight, green = driving, red = standing and
           waiting for the model, orange = safety stop; white cursor = now
  top      full floor plan | 12 m close-up around the robot | 30 Hz robot camera
           red line = path driven, blue = path being followed, numbered teal dots =
           sub-goals of the last answer after projection (grey x = dropped),
           small dots on the trail = where a call was sent, green discs = people
           (yellow ring + p1/p2 = listed to the model in the current call, under the
           label the model saw; #n = simulator id of a person not in the call)
  bottom   the current call: the images sent (with the model's points once answered),
           everything in the text input, and the full structured answer

Time axis: simulation time. Planner logs carry the host clock; the recorder's
(wall, header stamp) pairs map it to simulation time. The simulator does not pause
while the model thinks, so the robot keeps moving during a call, as in the run.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fvlm3 import floorplan as fp  # noqa: E402
from fvlm3.geometry import Camera, map_to_robot, wrap  # noqa: E402

W, H = 1920, 1080
HEADER_H, STRIP_H = 46, 24
TOP_Y = HEADER_H + STRIP_H
TOP_H = 600
BOT_Y = TOP_Y + TOP_H
BOT_H = H - BOT_Y
MAP_W, ZOOM_W = 470, 600
CAM_X = MAP_W + ZOOM_W
CAM_W = W - CAM_X
ZOOM_M = 12.0  # metres shown vertically in the close-up

# BGR
INK = (40, 36, 33)
MUTED = (120, 112, 106)
PAPER = (247, 245, 243)
PANEL = (255, 255, 255)
WALL = (70, 64, 60)
TRAIL = (40, 40, 210)
PLAN = (210, 110, 20)
SUBGOAL = (150, 150, 0)
DROPPED = (150, 150, 150)
PERSON = (70, 160, 60)
SEEN = (0, 200, 255)
ROBOT = (30, 30, 220)
GOAL = (60, 150, 40)
REASON_COLORS = {"idle": (90, 90, 90), "prefetch": (210, 110, 20), "blocked": (0, 120, 230), "person": (70, 160, 60), "refresh": (160, 90, 160)}
FONT_DIR = pathlib.Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    name += "-Bold" if bold else ""
    try:
        return ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), size)
    except OSError:
        return ImageFont.load_default()


F_HEAD = font(18, bold=True)
F_BODY = font(14)
F_BOLD = font(14, bold=True)
F_SMALL = font(12)
F_TITLE = font(16, bold=True)
F_MONO = font(13, mono=True)


def load_jsonl(p: pathlib.Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a killed run
    return out


def wrap_text(text: str, f: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for para in str(text).split("\n"):
        indent = len(para) - len(para.lstrip(" "))
        words = para.split(" ")
        cur = ""
        for w_ in words:
            cand = w_ if not cur else cur + " " + w_
            if f.getlength(cand) <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = " " * (indent + 2) + w_
        lines.append(cur)
    return lines


def zone_color(name: str) -> tuple[int, int, int]:
    pal = [(214, 230, 247), (226, 214, 245), (205, 236, 220), (240, 226, 205), (205, 228, 240), (236, 212, 226), (222, 236, 205), (232, 232, 214)]
    return pal[int(hashlib.md5(name.encode()).hexdigest(), 16) % len(pal)]


class Xf:
    """World metres -> panel pixels (north up)."""

    def __init__(self, k: float, x0: float, y_top: float, du: float = 0.0, dv: float = 0.0):
        self.k, self.x0, self.y_top, self.du, self.dv = k, x0, y_top, du, dv

    def __call__(self, x: float, y: float) -> tuple[int, int]:
        return int(round((x - self.x0) * self.k + self.du)), int(round((self.y_top - y) * self.k + self.dv))

    def many(self, pts) -> np.ndarray:
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return np.stack([(a[:, 0] - self.x0) * self.k + self.du, (self.y_top - a[:, 1]) * self.k + self.dv], axis=1).round().astype(np.int32)


# ------------------------------------------------------------------ episode data

class Episode:
    def __init__(self, ep_dir: pathlib.Path, frames_dir: pathlib.Path | None, world: str = ""):
        self.dir = ep_dir
        meta = load_jsonl(ep_dir / "meta.jsonl")
        if not meta:
            raise SystemExit(f"{ep_dir}: no meta.jsonl (not an episode dir?)")
        self.meta = meta[0]
        self.cfg = self.meta.get("config") or {}
        self.cond = self.meta.get("conditions") or {}
        self.instruction = self.meta.get("instruction", "")
        self.summary = json.loads((ep_dir / "summary.json").read_text()) if (ep_dir / "summary.json").is_file() else {}
        self.ticks = load_jsonl(ep_dir / "ticks.jsonl")
        if not self.ticks:
            raise SystemExit(f"{ep_dir}: no ticks.jsonl")
        calls = load_jsonl(ep_dir / "calls.jsonl")
        self.sent = {c["call"]: c for c in calls if c.get("phase") == "sent"}
        self.answers = {c["call"]: c for c in calls if c.get("phase") == "answer"}
        wall_t0 = self.meta.get("wall_t0")

        def wall(rec: dict) -> float:
            if "wall" in rec:
                return float(rec["wall"])
            return float(rec["t"]) + (float(wall_t0) if wall_t0 else 0.0)

        # recording: frames + host-clock -> simulation-time map
        self.frames: list[tuple[float, pathlib.Path]] = []
        pairs = []
        if frames_dir is not None and (frames_dir / "index.csv").is_file():
            with open(frames_dir / "index.csv") as fh:
                for r in csv.DictReader(fh):
                    try:
                        pairs.append((float(r["wall_s"]), float(r["stamp_s"]), frames_dir / "frames" / f"{int(r['frame']):06d}.jpg"))
                    except (KeyError, ValueError):
                        continue
        has_wall = "wall" in self.ticks[0] or bool(wall_t0)
        if pairs and has_wall:
            pairs.sort()
            wv = np.array([p[0] for p in pairs])
            sv = np.array([p[1] for p in pairs])
            slope = float(np.polyfit(wv - wv[0], sv, 1)[0]) if len(pairs) > 10 and wv[-1] > wv[0] else 1.0

            def to_t(w: float) -> float:
                if w < wv[0]:
                    return float(sv[0] + (w - wv[0]) * slope)
                if w > wv[-1]:
                    return float(sv[-1] + (w - wv[-1]) * slope)
                return float(np.interp(w, wv, sv))

            self.time_axis = "simulation time (from the recording)"
        else:
            if pairs and not has_wall:
                print("warning: logs have no host clock (older harness); ignoring the recording", file=sys.stderr)
            pairs = []
            w0 = wall(self.ticks[0])

            def to_t(w: float) -> float:
                return w - w0

            self.time_axis = "host time since episode start (no recording)"

        for r in self.ticks:
            r["T"] = to_t(wall(r))
        for d in (self.sent, self.answers):
            for r in d.values():
                r["T"] = to_t(wall(r))
        self.T0 = self.ticks[0]["T"]
        self.T1 = self.ticks[-1]["T"]
        self.tT = np.array([r["T"] for r in self.ticks])
        self.poses = np.array([r["pose"][:3] for r in self.ticks], dtype=np.float64)
        if pairs:
            self.frames = [(p[1], p[2]) for p in pairs if self.T0 - 0.5 <= p[1] <= self.T1 + 1.0]
        self.fT = [f[0] for f in self.frames]
        self.order = sorted(self.sent)
        self.sent_T = [self.sent[c]["T"] for c in self.order]
        self.ans_list = sorted(self.answers.values(), key=lambda r: r["T"])
        self.ans_T = [a["T"] for a in self.ans_list]

        # floor plan
        map_yaml, world_yaml = fp.find_world_files(world or str(self.cfg.get("world", "")), str(self.cfg.get("map_yaml", "")), str(self.cfg.get("world_yaml", "")))
        self.static = fp.load_static_map(map_yaml, world_yaml) if map_yaml else None
        if self.static is None:
            print("warning: no floor plan found (pass --world); drawing on a blank grid", file=sys.stderr)
        self.goal_zones = []
        if self.static is not None:
            low = self.instruction.lower()
            self.goal_zones = [z["name"] for z in self.static.zones if z["name"].lower() in low or z["name"].lower().replace("_", " ") in low]

    # -- state at time T
    def tick_idx(self, T: float) -> int:
        return max(0, int(np.searchsorted(self.tT, T, side="right")) - 1)

    def pose_at(self, T: float) -> np.ndarray:
        i = self.tick_idx(T)
        if i + 1 >= len(self.tT) or T <= self.tT[i]:
            return self.poses[i].copy()
        a = (T - self.tT[i]) / max(self.tT[i + 1] - self.tT[i], 1e-6)
        p, q = self.poses[i], self.poses[i + 1]
        return np.array([p[0] + a * (q[0] - p[0]), p[1] + a * (q[1] - p[1]), p[2] + a * wrap(q[2] - p[2])])

    def peds_at(self, T: float) -> list[tuple[int, float, float]]:
        i = self.tick_idx(T)
        cur = {int(p[0]): p for p in self.ticks[i].get("peds") or []}
        if i + 1 < len(self.ticks) and T > self.tT[i]:
            nxt = {int(p[0]): p for p in self.ticks[i + 1].get("peds") or []}
            a = (T - self.tT[i]) / max(self.tT[i + 1] - self.tT[i], 1e-6)
            return [(k, p[1] + a * (nxt[k][1] - p[1]), p[2] + a * (nxt[k][2] - p[2])) if k in nxt else (k, p[1], p[2]) for k, p in cur.items()]
        return [(k, p[1], p[2]) for k, p in cur.items()]

    def labels_at(self, T: float) -> dict[int, str]:
        """ped id -> the label the model saw in the current call (p1, p2, ... by distance)."""
        call = self.current_call(T)
        if call is None or not self.cond.get("people"):
            return {}
        return {int(p.get("ped_id", -1)): str(p.get("label", "")) for p in call.get("people") or []}

    def line_of_sight(self, a, b) -> bool:
        if self.static is None:
            return True
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.1))
        return not any(self.static.occupied(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) for t in np.linspace(0.05, 0.95, n))

    def current_call(self, T: float) -> dict | None:
        i = bisect.bisect_right(self.sent_T, T) - 1
        return self.sent[self.order[i]] if i >= 0 else None

    def answer_of(self, call_no: int, T: float) -> dict | None:
        a = self.answers.get(call_no)
        return a if a is not None and a["T"] <= T else None

    def active_plan(self, T: float) -> tuple[np.ndarray | None, dict | None]:
        """Plan the harness is following at T (same rules as Harness._apply)."""
        plan, src = None, None
        for a in self.ans_list[: bisect.bisect_right(self.ans_T, T)]:
            out = a.get("outcome")
            if a.get("plan"):
                plan, src = np.asarray(a["plan"], dtype=np.float64), a
            elif out in ("turn_left", "turn_right", "wait", "arrived"):
                plan, src = None, a
        tick = self.ticks[self.tick_idx(T)]
        if plan is None or float(tick.get("plan_left_m", 0.0)) <= 0.0:
            return None, src
        pose = self.pose_at(T)
        k = int(np.argmin(np.hypot(plan[:, 0] - pose[0], plan[:, 1] - pose[1])))
        rest = np.vstack([pose[:2], plan[k + 1 :]]) if k + 1 < len(plan) else None
        return rest, src

    def last_answer(self, T: float) -> dict | None:
        i = bisect.bisect_right(self.ans_T, T) - 1
        return self.ans_list[i] if i >= 0 else None

    def cost_until(self, T: float) -> float:
        return sum(float(a.get("cost_usd", 0.0)) for a in self.ans_list[: bisect.bisect_right(self.ans_T, T)])

    def mode(self, T: float) -> tuple[str, tuple[int, int, int]]:
        t = self.ticks[self.tick_idx(T)]
        kind, why, fly = t.get("motion"), t.get("why", ""), t.get("in_flight")
        if kind == "chunk":
            return "driving", (60, 150, 40)
        if kind == "turn":
            return "turning on the spot", (60, 150, 40)
        if kind == "arrived":
            return "model says arrived", GOAL
        if why == "safety_stop":
            return "safety stop (laser)", (0, 130, 240)
        if why == "model wait":
            return "waiting (model asked to wait)", MUTED
        if fly:
            return "standing, waiting for the model", (40, 40, 210)
        return "standing", MUTED


# ------------------------------------------------------------------ renderer

class Renderer:
    def __init__(self, ep: Episode):
        self.ep = ep
        self.label = f"{ep.dir.parent.name} / {ep.dir.name}"
        c = ep.cond
        self.cond_tag = "".join(f"{k[0].upper()}{int(bool(c.get(k)))}" for k in ("people", "floorplan", "rules")) if c else "?"
        self.cond_text = " · ".join(f"{n} {'on' if c.get(k) else 'off'}" for k, n in (("people", "people"), ("floorplan", "floor plan"), ("rules", "rules")))
        cfg = ep.cfg
        self.cam_kw = dict(hfov_deg=float(cfg.get("cam_hfov_deg", 90)), x=float(cfg.get("cam_x", 0.03)), y=float(cfg.get("cam_y", 0.0)), z=float(cfg.get("cam_z", 1.19)), pitch_deg=float(cfg.get("cam_pitch_deg", 0.0)))
        self._build_map()
        self._build_strip()
        self._img_cache: dict[str, np.ndarray | None] = {}
        self._frame_cache: tuple[int, np.ndarray | None] = (-1, None)
        self._bottom_cache: dict = {}

    # -- static map layers
    def _build_map(self):
        ep = self.ep
        sm = ep.static
        if sm is not None:
            h, w = sm.occ.shape
            self.wx0, self.wy0 = sm.origin
            self.ww, self.wh = w * sm.res, h * sm.res
        else:
            xy = ep.poses[:, :2]
            lo, hi = xy.min(0) - 5, xy.max(0) + 5
            self.wx0, self.wy0, self.ww, self.wh = lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]
        self.k_hi = min(40.0, 3000.0 / max(self.ww, self.wh))
        Wh, Hh = int(math.ceil(self.ww * self.k_hi)), int(math.ceil(self.wh * self.k_hi))
        self.xf_hi = Xf(self.k_hi, self.wx0, self.wy0 + self.wh)
        base = np.full((Hh, Wh, 3), 236, np.uint8)
        if sm is not None:
            free = cv2.resize(np.flipud(~sm.occ).astype(np.uint8), (Wh, Hh), interpolation=cv2.INTER_NEAREST).astype(bool)
            base[free] = PANEL
            col = base.copy()
            for z in sm.zones:
                cv2.fillPoly(col, [self.xf_hi.many(z["polygon"])], zone_color(z["name"]))
            base[free] = col[free]
            for z in sm.zones:
                if z["name"] in ep.goal_zones:
                    cv2.polylines(base, [self.xf_hi.many(z["polygon"])], True, GOAL, max(3, int(self.k_hi * 0.12)), cv2.LINE_AA)
            base[~free] = WALL
            for d in sm.doors:
                a, b = d["span"][0], d["span"][-1]
                cv2.line(base, self.xf_hi(*a), self.xf_hi(*b), (40, 150, 230), max(2, int(self.k_hi * 0.08)), cv2.LINE_AA)
        else:
            step = int(self.k_hi)
            base[::step, :] = 225
            base[:, ::step] = 225
        self.base_hi = base
        # full-map panel
        f = min((MAP_W - 16) / Wh, (TOP_H - 40) / Hh)
        self.full_size = (int(Wh * f), int(Hh * f))
        self.full_base = cv2.resize(base, self.full_size, interpolation=cv2.INTER_AREA)
        self.full_off = ((MAP_W - self.full_size[0]) // 2, 30 + (TOP_H - 30 - self.full_size[1]) // 2)
        self.xf_full = Xf(self.k_hi * f, self.wx0, self.wy0 + self.wh)
        if sm is not None:
            for z in sm.zones:
                cx, cy = np.asarray(z["polygon"]).mean(axis=0)
                u, v = self.xf_full(cx, cy)
                name = z["name"].replace("_", " ")
                s = 0.33
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
                cv2.putText(self.full_base, name, (u - tw // 2, v + th // 2), cv2.FONT_HERSHEY_SIMPLEX, s, GOAL if z["name"] in ep.goal_zones else (90, 80, 75), 1, cv2.LINE_AA)
        self.k_zoom = TOP_H / ZOOM_M

    def _build_strip(self):
        ep = self.ep
        x0, x1 = 12, W - 12
        self.strip_x = (x0, x1)
        span = max(ep.T1 - ep.T0, 1e-3)
        self.t2x = lambda T: int(x0 + (x1 - x0) * (T - ep.T0) / span)
        s = np.full((STRIP_H, W, 3), PAPER, np.uint8)
        cv2.rectangle(s, (x0, 3), (x1, STRIP_H - 3), (225, 222, 219), -1)
        for i, t in enumerate(ep.ticks[:-1]):
            a, b = self.t2x(t["T"]), self.t2x(ep.ticks[i + 1]["T"])
            kind, why = t.get("motion"), t.get("why", "")
            c = None
            if kind in ("chunk", "turn"):
                c = (110, 190, 90)
            elif why == "safety_stop":
                c = (0, 150, 245)
            elif kind == "hold" and t.get("in_flight"):
                c = (60, 60, 220)
            if c is not None:
                cv2.rectangle(s, (a, 13), (max(a + 1, b), STRIP_H - 4), c, -1)
        for cno, rec in ep.sent.items():
            a = self.t2x(rec["T"])
            ans = ep.answers.get(cno)
            b = self.t2x(ans["T"]) if ans else self.t2x(ep.T1)
            cv2.rectangle(s, (a, 4), (max(a + 1, b), 10), (200, 120, 40), -1)
            cv2.line(s, (a, 3), (a, 11), (120, 60, 20), 1)
        self.strip = s

    # -- helpers
    def _img(self, name: str | None) -> np.ndarray | None:
        if not name:
            return None
        if name not in self._img_cache:
            p = self.ep.dir / "images" / name
            self._img_cache[name] = cv2.imread(str(p)) if p.is_file() else None
        return self._img_cache[name]

    def _frame_at(self, T: float) -> tuple[np.ndarray | None, float]:
        ep = self.ep
        i = bisect.bisect_right(ep.fT, T) - 1
        if i < 0:
            return None, 0.0
        if i != self._frame_cache[0]:
            self._frame_cache = (i, cv2.imread(str(ep.frames[i][1])))
        return self._frame_cache[1], ep.fT[i]

    # -- overlays shared by both map panels
    def _draw_world(self, img: np.ndarray, xf: Xf, T: float, close: bool) -> None:
        ep = self.ep
        k = xf.k
        i = ep.tick_idx(T)
        pose = ep.pose_at(T)
        trail = np.vstack([ep.poses[: i + 1, :2], pose[None, :2]])
        if len(trail) > 1:
            cv2.polylines(img, [xf.many(trail)], False, TRAIL, 3 if close else 2, cv2.LINE_AA)
        for cno in ep.order:
            r = ep.sent[cno]
            if r["T"] > T:
                break
            u, v = xf(*r["pose"][:2])
            cv2.circle(img, (u, v), 5 if close else 3, REASON_COLORS.get(r.get("reason", ""), MUTED), -1, cv2.LINE_AA)
            if close:
                cv2.putText(img, str(cno), (u + 6, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, INK, 1, cv2.LINE_AA)
        plan, _ = ep.active_plan(T)
        if plan is not None and len(plan) > 1:
            cv2.polylines(img, [xf.many(plan)], False, PLAN, 4 if close else 2, cv2.LINE_AA)
        last = ep.last_answer(T)
        if last is not None and last.get("projected"):
            for n, pr in enumerate(last["projected"], 1):
                raw = xf(*pr["map_xy"])
                if "snapped_xy" in pr:
                    s = xf(*pr["snapped_xy"])
                    if close and raw != s:
                        cv2.line(img, raw, s, SUBGOAL, 1, cv2.LINE_AA)
                        cv2.circle(img, raw, 3, SUBGOAL, 1, cv2.LINE_AA)
                    rad = 10 if close else 5
                    cv2.circle(img, s, rad, SUBGOAL, -1, cv2.LINE_AA)
                    if close:
                        cv2.putText(img, str(n), (s[0] - 4, s[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    d = 7 if close else 4
                    cv2.line(img, (raw[0] - d, raw[1] - d), (raw[0] + d, raw[1] + d), DROPPED, 2, cv2.LINE_AA)
                    cv2.line(img, (raw[0] - d, raw[1] + d), (raw[0] + d, raw[1] - d), DROPPED, 2, cv2.LINE_AA)
                    if close:
                        cv2.putText(img, f"{n} {pr.get('dropped', '')}", (raw[0] + 9, raw[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, MUTED, 1, cv2.LINE_AA)
        call = ep.current_call(T)
        labels = ep.labels_at(T)
        if call is not None and ep.answer_of(call["call"], T) is None and call["T"] <= T:
            u, v = xf(*call["pose"][:2])
            th = call["pose"][2]
            cv2.circle(img, (u, v), max(4, int(0.35 * k)), ROBOT, 1, cv2.LINE_AA)
            if close:
                tip = xf(call["pose"][0] + 0.6 * math.cos(th), call["pose"][1] + 0.6 * math.sin(th))
                cv2.line(img, (u, v), tip, ROBOT, 1, cv2.LINE_AA)
                cv2.putText(img, f"photo for call {call['call']}", (u - 125, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ROBOT, 1, cv2.LINE_AA)
        for pid, x, y in ep.peds_at(T):
            u, v = xf(x, y)
            r = max(3, int(0.28 * k))
            cv2.circle(img, (u, v), r, PERSON, -1, cv2.LINE_AA)
            if pid in labels:
                cv2.circle(img, (u, v), r + 3, SEEN, 2, cv2.LINE_AA)
            if close:
                cv2.putText(img, labels.get(pid, f"#{pid}"), (u + r + 2, v - r), cv2.FONT_HERSHEY_SIMPLEX, 0.42 if pid in labels else 0.36, (0, 140, 200) if pid in labels else (40, 110, 30), 1, cv2.LINE_AA)
        # robot + camera wedge
        x, y, th = pose
        fov = math.radians(self.cam_kw["hfov_deg"]) / 2
        reach = 3.0 if close else 4.0
        wedge = xf.many([(x, y), (x + reach * math.cos(th + fov), y + reach * math.sin(th + fov)), (x + reach * math.cos(th - fov), y + reach * math.sin(th - fov))])
        ov = img.copy()
        cv2.fillPoly(ov, [wedge], (120, 120, 235))
        cv2.addWeighted(ov, 0.22, img, 0.78, 0, dst=img)
        c = xf(x, y)
        cv2.circle(img, c, max(4, int(0.35 * k)), ROBOT, 2, cv2.LINE_AA)
        cv2.arrowedLine(img, c, xf(x + 0.8 * math.cos(th), y + 0.8 * math.sin(th)), ROBOT, 3 if close else 2, cv2.LINE_AA, tipLength=0.4)

    def _full_panel(self, T: float) -> np.ndarray:
        p = np.full((TOP_H, MAP_W, 3), PAPER, np.uint8)
        img = self.full_base.copy()
        self._draw_world(img, self.xf_full, T, close=False)
        ox, oy = self.full_off
        p[oy : oy + img.shape[0], ox : ox + img.shape[1]] = img
        # close-up frame on the full map
        pose = self.ep.pose_at(T)
        hw, hh = ZOOM_W / self.k_zoom / 2, ZOOM_M / 2
        a = self.xf_full(pose[0] - hw, pose[1] + hh)
        b = self.xf_full(pose[0] + hw, pose[1] - hh)
        cv2.rectangle(p, (a[0] + ox, a[1] + oy), (b[0] + ox, b[1] + oy), (150, 140, 135), 1, cv2.LINE_AA)
        return p

    def _zoom_panel(self, T: float) -> np.ndarray:
        pose = self.ep.pose_at(T)
        k = self.k_zoom
        x_left = pose[0] - ZOOM_W / k / 2
        y_top = pose[1] + ZOOM_M / 2
        f = k / self.k_hi
        u0 = (x_left - self.wx0) * self.k_hi
        v0 = (self.wy0 + self.wh - y_top) * self.k_hi
        M = np.array([[f, 0, -u0 * f], [0, f, -v0 * f]], dtype=np.float64)
        img = cv2.warpAffine(self.base_hi, M, (ZOOM_W, TOP_H), flags=cv2.INTER_LINEAR, borderValue=WALL)
        xf = Xf(k, x_left, y_top)
        sm = self.ep.static
        if sm is not None:
            for z in sm.zones:
                cx, cy = np.asarray(z["polygon"]).mean(axis=0)
                u, v = xf(cx, cy)
                if 0 <= u < ZOOM_W and 0 <= v < TOP_H:
                    name = z["name"].replace("_", " ")
                    (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.putText(img, name, (u - tw // 2, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GOAL if z["name"] in self.ep.goal_zones else (90, 80, 75), 1, cv2.LINE_AA)
        self._draw_world(img, xf, T, close=True)
        # 1 m scale bar
        cv2.line(img, (14, TOP_H - 16), (14 + int(k), TOP_H - 16), INK, 2)
        cv2.putText(img, "1 m", (18 + int(k), TOP_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1, cv2.LINE_AA)
        cv2.line(img, (0, 0), (0, TOP_H), (200, 195, 190), 1)
        return img

    def _camera_panel(self, T: float) -> np.ndarray:
        ep = self.ep
        p = np.full((TOP_H, CAM_W, 3), (30, 28, 26), np.uint8)
        frame, ft = self._frame_at(T)
        live = frame is not None
        caption = ""
        if frame is None:
            call = ep.current_call(T)
            frame = self._img(call["images"][0]) if call and call.get("images") else None
            caption = f"no 30 Hz recording: last image sent to the model (call {call['call']})" if frame is not None else "no camera image"
        if frame is not None:
            fh, fw = frame.shape[:2]
            s = min(CAM_W / fw, TOP_H / fh)
            dw, dh = int(fw * s), int(fh * s)
            img = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            if live:
                cam = Camera(width=dw, height=dh, **self.cam_kw)
                pose = ep.pose_at(ft)
                plan, _ = ep.active_plan(T)
                if plan is not None:
                    rel = map_to_robot(pose, plan)
                    pts = [cam.robot_to_pixel_unclipped(x, y, 0.0) for x, y in rel if x > 0.35]
                    pts = [q for q in pts if q is not None and -dw < q[0] < 2 * dw and -dh < q[1] < 2 * dh]
                    if len(pts) > 1:
                        cv2.polylines(img, [np.array(pts, np.int32)], False, PLAN, 4, cv2.LINE_AA)
                last = ep.last_answer(T)
                if last is not None and plan is not None:
                    for n, pr in enumerate(last.get("projected") or [], 1):
                        if "snapped_xy" not in pr:
                            continue
                        rx, ry = map_to_robot(pose, np.array([pr["snapped_xy"]]))[0]
                        q = cam.robot_to_pixel(rx, ry, 0.0)
                        if q is not None:
                            c = (int(q[0]), int(q[1]))
                            cv2.circle(img, c, 12, SUBGOAL, -1, cv2.LINE_AA)
                            cv2.putText(img, str(n), (c[0] - 5, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                labels = ep.labels_at(T)
                for pid, x, y in ep.peds_at(ft):
                    rx, ry = map_to_robot(pose, np.array([[x, y]]))[0]
                    q = cam.robot_to_pixel(rx, ry, 1.95)
                    if q is not None and ep.line_of_sight(pose[:2], (x, y)):
                        txt = labels.get(pid, f"#{pid}")
                        cv2.putText(img, txt, (int(q[0]) - 10, int(q[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(img, txt, (int(q[0]) - 10, int(q[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255) if pid in labels else (120, 235, 120), 2, cv2.LINE_AA)
                caption = f"robot camera, {fw}x{fh} recording (the model gets 224x224 frames)"
            ox, oy = (CAM_W - dw) // 2, (TOP_H - dh) // 2
            p[oy : oy + dh, ox : ox + dw] = img
        cv2.putText(p, caption, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return p

    # -- header and call panel (PIL)
    def _header(self, T: float) -> np.ndarray:
        ep = self.ep
        im = Image.new("RGB", (W, HEADER_H), (PAPER[2], PAPER[1], PAPER[0]))
        d = ImageDraw.Draw(im)
        d.text((14, 4), self.label, font=F_SMALL, fill=(106, 112, 120))
        d.text((14, 20), f"{self.cond_tag}  ({self.cond_text})", font=F_BOLD, fill=(33, 36, 40))
        text, room = f"“{ep.instruction}”", 1590 - 470  # keep clear of the clock on the right
        if d.textlength(text, font=F_HEAD) > room:
            font_i = F_BOLD
            words, lines = text.split(), [""]
            for w_ in words:
                cand = (lines[-1] + " " + w_).strip()
                if d.textlength(cand, font=font_i) <= room or not lines[-1]:
                    lines[-1] = cand
                else:
                    lines.append(w_)
            for k, ln in enumerate(lines[:2]):
                d.text((470, 6 + 20 * k), ln + ("…" if k == 1 and len(lines) > 2 else ""), font=font_i, fill=(33, 36, 40))
        else:
            d.text((470, 13), text, font=F_HEAD, fill=(33, 36, 40))
        mode, col = ep.mode(T)
        n_calls = bisect.bisect_right(ep.sent_T, T)
        right = f"t = {T - ep.T0:6.1f} s    calls {n_calls}    ${ep.cost_until(T):.3f}"
        if T >= ep.T1:
            s = ep.summary
            mode = "episode over: " + ("model said arrived" if s.get("arrived_signal") else "no arrived signal")
            col = GOAL if s.get("arrived_signal") else MUTED
        tw = d.textlength(right, font=F_HEAD)
        d.text((W - 14 - tw, 4), right, font=F_HEAD, fill=(33, 36, 40))
        mw = d.textlength(mode, font=F_BOLD)
        d.text((W - 14 - mw, 26), mode, font=F_BOLD, fill=(col[2], col[1], col[0]))
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def _bottom(self, T: float) -> np.ndarray:
        ep = self.ep
        call = ep.current_call(T)
        ans = ep.answer_of(call["call"], T) if call else None
        key = (call["call"] if call else 0, ans is not None)
        if key not in self._bottom_cache:
            self._bottom_cache = {key: self._bottom_static(call, ans)}
        img = self._bottom_cache[key].copy()
        if call is not None and ans is None:
            waited = T - call["T"]
            cv2.putText(img, f"waiting for the answer: {waited:4.1f} s (predicted {call.get('pred_latency', 0):.1f} s)", (self._wait_xy[0], self._wait_xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 210), 2, cv2.LINE_AA)
        return img

    def _bottom_static(self, call: dict | None, ans: dict | None) -> np.ndarray:
        ep = self.ep
        im = np.full((BOT_H, W, 3), PANEL, np.uint8)
        cv2.line(im, (0, 0), (W, 0), (200, 195, 190), 2)
        x = 12
        thumbs = []
        if call is not None:
            names = list(call.get("images") or [])
            if ans is not None and names and (ep.dir / "images" / f"c{call['call']:03d}_answer.jpg").is_file():
                names[0] = f"c{call['call']:03d}_answer.jpg"
            labels = ["camera image sent" + (" + model's points (red)" if names and names[0].endswith("answer.jpg") else ""), "floor plan sent"]
            for n, name in enumerate(names[:2]):
                a = self._img(name)
                if a is None:
                    continue
                side = BOT_H - 40
                s = min(side / a.shape[0], 330 / a.shape[1])
                a = cv2.resize(a, (int(a.shape[1] * s), int(a.shape[0] * s)), interpolation=cv2.INTER_AREA)
                thumbs.append((labels[min(n, 1)], a))
        for lab, a in thumbs:
            im[30 : 30 + a.shape[0], x : x + a.shape[1]] = a
            cv2.putText(im, lab, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
            x += a.shape[1] + 14
        tx = max(x + 6, 700)
        col_w = (W - tx - 24) // 2
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        ink, muted, blue, red, green = (33, 36, 40), (106, 112, 120), (20, 110, 210), (210, 40, 40), (40, 150, 60)
        if call is None:
            d.text((tx, 14), "No model call yet.", font=F_TITLE, fill=ink)
            self._wait_xy = (tx, 60)
            return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        title = f"Call {call['call']}   trigger: {call.get('reason', '?')}   sent at t = {call['T'] - ep.T0:.1f} s"
        if call.get("plan_left_m", 0) > 0.05:
            title += f" with {call['plan_left_m']:.1f} m of the old path left"
        d.text((tx, 8), title, font=F_TITLE, fill=ink)
        if ans is not None:
            lat = f"answer after {ans.get('latency_s', 0):.1f} s, the robot kept moving meanwhile"
            if ans.get("cost_usd") is not None:
                u = ans.get("usage") or {}
                lat += f"   ·   {u.get('input', 0) + u.get('cache_read', 0) + u.get('cache_write', 0)} in / {u.get('output', 0)} out tokens, ${ans['cost_usd']:.4f}"
            d.text((tx, 30), lat, font=F_BODY, fill=muted)
        self._wait_xy = (tx, 46)
        y0 = 56
        line_h = 17
        max_y = BOT_H - 8

        def block(x0: int, y: int, head: str, lines: list[tuple[str, ImageFont.ImageFont, tuple]]) -> int:
            d.text((x0, y), head, font=F_BOLD, fill=muted)
            y += line_h + 3
            for text, f, colr in lines:
                for ln in wrap_text(text, f, col_w):
                    if y + line_h > max_y:
                        d.text((x0, y - line_h), "…", font=f, fill=muted)
                        return y
                    d.text((x0, y), ln, font=f, fill=colr)
                    y += line_h
            return y

        # input column
        sys_words = len(str(ep.meta.get("system_prompt", "")).split())
        inp = [(f"System prompt: {sys_words} words, fixed for the episode ({self.cond_text}).", F_SMALL, muted)]
        n_img = len(call.get("images") or [])
        inp.append((f"Images: robot camera{' + floor plan' if n_img > 1 else ''}. Text:", F_SMALL, muted))
        inp.append((call.get("text", ""), F_BODY, ink))
        block(tx, y0, "MODEL INPUT", inp)
        # output column
        ox = tx + col_w + 24
        out: list[tuple[str, ImageFont.ImageFont, tuple]] = []
        if ans is None:
            prev = ep.last_answer(call["T"])
            drv = f"Meanwhile the robot follows the path from call {prev['call']}." if prev is not None and prev.get("plan") else "No path yet: the robot stands still."
            out.append(("", F_BODY, ink))
            out.append((drv, F_BODY, muted))
        elif not ans.get("ok"):
            out.append((f"Call failed: {ans.get('error', '')}", F_BOLD, red))
        else:
            r = ans.get("reply") or {}
            outcome = ans.get("outcome", "")
            act = str(r.get("action", "?")).upper()
            harness = {"stale": "answer too old, not used", "unusable": "no sub-goal reachable, not used"}.get(outcome, "")
            if ans.get("plan_len_m"):
                harness = f"path {ans['plan_len_m']:.1f} m planned through the sub-goals"
            out.append((f"{act}   progress: {r.get('progress', '?')}" + (f"   heading to: {r['heading_to']}" if "heading_to" in r else ""), F_TITLE, blue))
            if harness:
                out.append((f"Harness: {harness}", F_SMALL, red if outcome in ("stale", "unusable") else green))
            if r.get("scene"):
                out.append((f"Scene: {r['scene']}", F_BODY, ink))
            proj = ans.get("projected") or []
            for n, s in enumerate(r.get("subgoals") or [], 1):
                pr = proj[n - 1] if n - 1 < len(proj) else {}
                where = f"{pr.get('range_m', 0):.1f} m via {pr.get('source', '?')}" if pr else ""
                if pr.get("dropped"):
                    where += f", dropped ({pr['dropped']})"
                out.append((f"Sub-goal {n} at (u={s.get('u')}, v={s.get('v')}) → {where}: {s.get('why', '')}", F_BODY, ink))
            if r.get("path_reasoning"):
                out.append((f"Why: {r['path_reasoning']}", F_BOLD, ink))
            for pp in r.get("people") or []:
                out.append((f"{pp.get('label')}: {pp.get('relevance')}, {pp.get('note', '')}", F_BODY, ink))
            if r.get("rules_applied"):
                out.append(("Rules: " + "; ".join(str(x) for x in r["rules_applied"]), F_BODY, ink))
        block(ox, y0, "MODEL OUTPUT", out)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def intro(self) -> np.ndarray:
        ep = self.ep
        pil = Image.new("RGB", (W, H), (PAPER[2], PAPER[1], PAPER[0]))
        d = ImageDraw.Draw(pil)
        d.text((80, 60), "frontier_vlm3 episode", font=font(34, bold=True), fill=(33, 36, 40))
        d.text((80, 112), self.label, font=F_BODY, fill=(106, 112, 120))
        d.text((80, 150), f"Instruction: “{ep.instruction}”", font=font(24), fill=(33, 36, 40))
        d.text((80, 190), f"Conditions {self.cond_tag}: {self.cond_text}.   Model: {ep.cfg.get('provider', '?')} {ep.cfg.get('model', '') if ep.cfg.get('provider') == 'claude' else ''}   Time axis: {ep.time_axis}.", font=F_BODY, fill=(33, 36, 40))
        d.text((80, 214), "The model never gets the goal position. It picks points in the camera image; the harness projects them with depth and drives there.", font=F_BODY, fill=(106, 112, 120))
        d.text((80, 256), "System prompt (sent with every call):", font=F_BOLD, fill=(106, 112, 120))
        y = 280
        for ln in wrap_text(str(ep.meta.get("system_prompt", "")), F_MONO, W - 160):
            if y > H - 40:
                d.text((80, y), "…", font=F_MONO, fill=(106, 112, 120))
                break
            d.text((80, y), ln, font=F_MONO, fill=(33, 36, 40))
            y += 17
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def frame(self, T: float) -> np.ndarray:
        out = np.full((H, W, 3), PAPER, np.uint8)
        out[:HEADER_H] = self._header(T)
        s = self.strip.copy()
        x = self.t2x(min(max(T, self.ep.T0), self.ep.T1))
        cv2.line(s, (x, 0), (x, STRIP_H), (255, 255, 255), 3)
        cv2.line(s, (x, 0), (x, STRIP_H), INK, 1)
        out[HEADER_H:TOP_Y] = s
        out[TOP_Y:BOT_Y, :MAP_W] = self._full_panel(T)
        out[TOP_Y:BOT_Y, MAP_W:CAM_X] = self._zoom_panel(T)
        out[TOP_Y:BOT_Y, CAM_X:] = self._camera_panel(T)
        out[BOT_Y:] = self._bottom(T)
        return out


# ------------------------------------------------------------------ output

class Writer:
    def __init__(self, path: pathlib.Path, fps: float):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ff = shutil.which("ffmpeg")
        if self.ff:
            self.proc = subprocess.Popen(
                [self.ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE,
            )
        else:
            print("ffmpeg not found: writing MPEG-4 part 2 with OpenCV (plays in VLC, maybe not in browsers)", file=sys.stderr)
            self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def write(self, img: np.ndarray) -> None:
        if self.ff:
            self.proc.stdin.write(img.tobytes())
        else:
            self.vw.write(img)

    def close(self) -> None:
        if self.ff:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.vw.release()


def default_frames_dir(ep_dir: pathlib.Path) -> pathlib.Path:
    """results/<stage>/<run_id>/calls/epNNN -> results/<stage>/<run_id> (frames/ + index.csv);
    older layout <data>/vlm_calls/<label>/epNNN -> <data>/vlm_video/<label>."""
    if ep_dir.parent.name == "calls":
        return ep_dir.parent.parent
    return ep_dir.parent.parent.parent / "vlm_video" / ep_dir.parent.name


def render(ep_dir: pathlib.Path, a: argparse.Namespace) -> pathlib.Path | None:
    frames_dir = pathlib.Path(a.frames) if a.frames else default_frames_dir(ep_dir)
    ep = Episode(ep_dir, frames_dir if frames_dir.is_dir() else None, a.world)
    r = Renderer(ep)
    print(f"{ep_dir}: {len(ep.ticks)} ticks, {len(ep.sent)} calls, {len(ep.frames)} recorded frames, {ep.T1 - ep.T0:.1f} s, time axis = {ep.time_axis}")
    if a.still is not None:
        out = pathlib.Path(a.out) if a.out else ep_dir / f"still_{a.still:.1f}.png"
        cv2.imwrite(str(out), r.frame(ep.T0 + a.still))
        print(f"wrote {out}")
        return out
    out = pathlib.Path(a.out) if a.out and not a.all else ep_dir / "episode.mp4"
    wr = Writer(out, a.fps)
    intro = r.intro()
    for _ in range(int(a.intro * a.fps)):
        wr.write(intro)
    t_start = ep.T0 + a.start
    t_end = min(ep.T1, ep.T0 + a.end) if a.end > 0 else ep.T1
    n = int((t_end - t_start) / (a.speed / a.fps)) + 1
    for i in range(n):
        wr.write(r.frame(t_start + i * a.speed / a.fps))
        if i % int(a.fps * 10) == 0:
            print(f"  {i}/{n} frames", end="\r", flush=True)
    last = r.frame(t_end + 1e-3)
    for _ in range(int(a.fps * 2)):
        wr.write(last)
    wr.close()
    print(f"wrote {out} ({n} frames, {n / a.fps:.0f} s at {a.speed}x)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="episode dir (or run dir with --all)")
    ap.add_argument("--frames", default="", help="cam_recorder output dir (default: <log_dir>/../vlm_video/<run_label>)")
    ap.add_argument("--out", default="", help="output .mp4 (default <episode_dir>/episode.mp4) or .png with --still")
    ap.add_argument("--world", default="", help="world name if the log has none (e.g. hospital_1)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed; 2 = twice real time")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the episode")
    ap.add_argument("--end", type=float, default=0.0, help="seconds into the episode (0 = to the end)")
    ap.add_argument("--intro", type=float, default=4.0, help="seconds of title card with the system prompt")
    ap.add_argument("--still", type=float, default=None, help="write one PNG at this episode time instead of a video")
    ap.add_argument("--all", action="store_true", help="path is a run dir: render every episode in it")
    a = ap.parse_args()
    p = pathlib.Path(a.path)
    eps = sorted(d for d in p.iterdir() if (d / "ticks.jsonl").is_file()) if a.all else [p]
    if not eps:
        raise SystemExit(f"no episodes under {p}")
    for ep in eps:
        render(ep, a)


if __name__ == "__main__":
    main()
