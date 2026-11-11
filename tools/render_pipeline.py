#!/usr/bin/env python3
"""Render a frontier_vlm3 VL-Nav episode (pipeline:=mission or vlnav) as key frames + video.
CONTAINER (numpy, opencv, pillow, scipy).

  python3 tools/render_pipeline.py <calls_episode_dir> [--run <run_dir>] [--out <dir>] [--fps 10] [--speed 2]
  python3 tools/render_pipeline.py <calls_episode_dir> --keyframes-only

<calls_episode_dir> = <log_dir>/<run_label>/ep###_<id>/ (calls.jsonl, ticks.jsonl, meta.jsonl, images/).
<run_dir> (optional) holds the recordings: rec/ (robot view) and chase/ (third-person), or frames/ + index.csv
(hospital_sim). Everything is aligned on the host wall clock (ticks "wall", recordings "wall_s").

Output (default <run_dir>/pipeline/ or <calls_episode_dir>/pipeline/):
  keyframes/c###.png  one 1920x1080 frame per answered call, at the moment the answer arrived
  pipeline.mp4        the episode at --fps frames per second of wall time / --speed

Layout
  top    full floor plan (route light blue, driven red, people green, target room green) | chase camera |
         robot camera over the planner view (candidates, chosen one ringed red, plan blue, laser obstacles grey)
  bottom what the model got at its last answered call (camera, map image) and what it answered:
         MISSION target / route / sub-tasks, or WAYPOINT trigger / people / choice vs. route follower / why
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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fvlm3 import floorplan as fp  # noqa: E402
from fvlm3 import vlnav as vn  # noqa: E402
from tools.render_episode import F_BODY, F_BOLD, F_HEAD, Writer, load_jsonl, wrap_text  # noqa: E402

W, H = 1920, 1080
TOP_H = 700
PLAN_W = 540
RIGHT_X = 1280
CAM_H = 340
BOT_Y = TOP_H + 8
ORANGE = (0, 165, 255)
GREEN = (60, 170, 60)


def host_path(p: str) -> str:
    """Container paths in the logs -> this machine (works inside the container and on the host)."""
    if p and not pathlib.Path(p).exists() and p.startswith("/opt/arena_ws"):
        alt = pathlib.Path.home() / "arena_ws" / p[len("/opt/arena_ws/") :]
        if alt.exists():
            return str(alt)
    return p


class Recording:
    def __init__(self, d: pathlib.Path | None) -> None:
        self.w: list[float] = []
        self.p: list[pathlib.Path] = []
        self._c: tuple[int, np.ndarray | None] | None = None
        if d is None or not (d / "index.csv").is_file():
            return
        rows = []
        for r in csv.DictReader(open(d / "index.csv")):
            try:
                rows.append((float(r["wall_s"]), d / "frames" / f"{int(r['frame']):06d}.jpg"))
            except (KeyError, ValueError):
                pass
        rows.sort()
        self.w, self.p = [r[0] for r in rows], [r[1] for r in rows]

    def at(self, wall: float) -> np.ndarray | None:
        if not self.w:
            return None
        i = max(0, bisect.bisect_right(self.w, wall) - 1)
        if self._c and self._c[0] == i:
            return self._c[1]
        img = cv2.imread(str(self.p[i]))
        self._c = (i, img)
        return img


def fit(img: np.ndarray | None, w: int, h: int, label: str = "") -> np.ndarray:
    out = np.full((h, w, 3), 245, np.uint8)
    if img is not None:
        s = min(w / img.shape[1], h / img.shape[0])
        r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
        out[y0 : y0 + r.shape[0], x0 : x0 + r.shape[1]] = r
    if label:
        cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return out


class Episode:
    def __init__(self, d: pathlib.Path, run: pathlib.Path | None) -> None:
        self.d = d
        self.meta = (load_jsonl(d / "meta.jsonl") or [{}])[0]
        cfg = self.meta.get("config", {})
        self.instruction = self.meta.get("instruction", "")
        self.provider, self.model = cfg.get("provider", ""), cfg.get("model", "")
        self.pipeline = cfg.get("pipeline", "")
        my, wy = fp.find_world_files(str(cfg.get("world", "")), host_path(str(cfg.get("map_yaml", ""))), host_path(str(cfg.get("world_yaml", ""))))
        self.sm = fp.load_static_map(host_path(my), host_path(wy))
        self.pm = vn.PlanMap(self.sm)
        self.ticks = load_jsonl(d / "ticks.jsonl")
        self.tw = [float(t["wall"]) for t in self.ticks]
        recs = load_jsonl(d / "calls.jsonl")
        self.sent = {r["call"]: r for r in recs if r.get("phase") == "sent"}
        self.answers = sorted([r for r in recs if r.get("phase") == "answer"], key=lambda r: r["wall"])
        self.aw = [float(r["wall"]) for r in self.answers]
        self.cost = np.cumsum([float(r.get("cost_usd", 0.0)) for r in self.answers]) if self.answers else []
        self.mission = next((r for r in self.answers if r.get("outcome") == "mission"), None)
        self.rec = Recording(run / "rec" if run and (run / "rec").is_dir() else run if run and (run / "index.csv").is_file() else None)
        self.chase = Recording(run / "chase" if run and (run / "chase").is_dir() else None)
        self.run_meta = json.loads((run / "run.json").read_text()) if run and (run / "run.json").is_file() else {}

    def tick(self, wall: float) -> dict:
        return self.ticks[max(0, bisect.bisect_right(self.tw, wall) - 1)]

    def trail(self, wall: float) -> list[tuple[float, float]]:
        i = bisect.bisect_right(self.tw, wall)
        return [(t["pose"][0], t["pose"][1]) for t in self.ticks[: max(i, 1) : 2]]

    def last_answer(self, wall: float) -> tuple[dict | None, int]:
        i = bisect.bisect_right(self.aw, wall) - 1
        return (self.answers[i], i) if i >= 0 else (None, -1)

    def last_plan(self, wall: float) -> list | None:
        i = bisect.bisect_right(self.aw, wall) - 1
        while i >= 0:
            a = self.answers[i]
            if a.get("plan"):
                return a["plan"]
            if a.get("outcome") in ("wait", "turn_left", "turn_right", "arrived"):
                return None
            i -= 1
        return None


def draw_people(img: np.ndarray, xf: vn.Xf, peds: list, r_px: int) -> None:
    for p in peds or []:
        u, v = xf(p[1], p[2]) if len(p) >= 3 else xf(p[0], p[1])
        cv2.circle(img, (u, v), r_px, GREEN, 2, cv2.LINE_AA)


def frame(ep: Episode, wall: float, card_only_call: dict | None = None) -> np.ndarray:
    img = np.full((H, W, 3), 246, np.uint8)
    tk = ep.tick(wall)
    pose = np.array(tk["pose"], float)
    trail = ep.trail(wall)
    ans, ai = ep.last_answer(wall)
    sent = ep.sent.get(ans["call"]) if ans else None
    target = (ep.mission or {}).get("reply", {}).get("target", "") if ep.mission and ep.mission["wall"] <= wall else ""
    route_xy = None
    for a in reversed(ep.answers[: ai + 1]):
        s = ep.sent.get(a["call"], {})
        if s.get("route_xy"):
            route_xy = s["route_xy"]
            break
    if route_xy is None and ep.mission and ep.mission["wall"] <= wall:
        route_xy = [tuple(pose[:2])] + [tuple(w["xy"]) for w in ep.mission.get("waypoints", [])]
    plan = ep.last_plan(wall)
    plan = np.asarray(plan) if plan else None

    # -- full floor plan
    vis = sorted({ep.sm.zone_of(*p) for p in trail} - {None})
    full = ep.pm.full(pose, trail, plan, route_xy, 90.0, vis, target, height_px=TOP_H - 40)
    img[0:TOP_H, 0:PLAN_W] = fit(full, PLAN_W, TOP_H)
    # people (ground truth, for the viewer only) on the full plan: recompute the transform used by fit()
    band = full.shape[0] - (TOP_H - 40)
    s = min(PLAN_W / full.shape[1], TOP_H / full.shape[0])
    ox, oy = (PLAN_W - int(full.shape[1] * s)) // 2, (TOP_H - int(full.shape[0] * s)) // 2
    k = (TOP_H - 40) / (ep.pm.y1 - ep.pm.y0)
    for p in tk.get("peds") or []:
        u = ox + int(((p[1] - ep.pm.x0) * k) * s)
        v = oy + int((band + (ep.pm.y1 - p[2]) * k) * s)
        cv2.circle(img, (u, v), 5, GREEN, 2, cv2.LINE_AA)

    # -- chase camera
    img[0:TOP_H, PLAN_W:RIGHT_X] = fit(ep.chase.at(wall), RIGHT_X - PLAN_W, TOP_H, "third-person camera" if ep.chase.w else "no chase recording")
    # -- robot camera
    img[0:CAM_H, RIGHT_X:W] = fit(ep.rec.at(wall), W - RIGHT_X, CAM_H, "robot camera")
    # -- planner view
    half_w, half_h = 8.0, (TOP_H - CAM_H) / (W - RIGHT_X) * 8.0
    scale = (W - RIGHT_X) / (2 * half_w)
    pv, xf = ep.pm.view(pose[0] - half_w, pose[1] - half_h, pose[0] + half_w, pose[1] + half_h, scale)
    vn.draw_paths(pv, xf, trail, plan, route_xy)
    ep.pm.labels(pv, xf, 0.5, 0.4, vis, target)
    tracks = [{"id": t[0], "xy": [t[1], t[2]], "v": [t[3], t[4]], "speed": math.hypot(t[3], t[4])} for t in tk.get("tracks") or []]
    vn.draw_tracks(pv, xf, tracks, scale)
    for p in tk.get("peds") or []:
        cv2.circle(pv, xf(p[1], p[2]), int(0.3 * scale), GREEN, 2, cv2.LINE_AA)
    vn.draw_robot(pv, xf, pose, 90.0, reach=3.0)
    choice = ans.get("choice", 0) if ans else 0
    if sent and sent.get("cands"):
        for c in sent["cands"]:
            vn.draw_number(pv, xf(*c["xy"]), c["id"], c["id"] == choice)
    mode = "stall (waiting for the model)" if tk.get("in_flight") and tk.get("motion") == "hold" else tk.get("motion", "")
    head = f"planner view  mode={mode}  next={tk.get('wp') or target or '-'}" + (f"  choice {choice}" if choice else "")
    pv = fit(pv, W - RIGHT_X, TOP_H - CAM_H)
    cv2.rectangle(pv, (0, 0), (W - RIGHT_X, 22), (255, 255, 255), -1)
    cv2.putText(pv, head, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    leg = "light blue = route | blue = plan sent to Nav2 | numbers = candidates, red ring = chosen | green = real people (viewer only) | grey = laser obstacles"
    cv2.rectangle(pv, (0, pv.shape[0] - 18), (W - RIGHT_X, pv.shape[0]), (255, 255, 255), -1)
    cv2.putText(pv, leg, (4, pv.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1, cv2.LINE_AA)
    img[CAM_H:TOP_H, RIGHT_X:W] = pv

    # -- bottom card
    cv2.rectangle(img, (2, BOT_Y), (W - 3, H - 3), ORANGE, 4)
    t_ep = float(tk.get("t", 0.0))
    if ans is None:
        txt_head = f"t={t_ep:.1f} s  waiting for the first answer (mission call in flight)"
        _pil_text(img, [(txt_head, F_HEAD, (20, 20, 20))], 640, BOT_Y + 14, W - 660)
        return img
    for k_im, (x0, lab) in enumerate([(12, "camera (sent)"), (322, "floor plan (sent)" if ans.get("outcome") == "mission" or ep.pipeline != "vlnav" else "local view (sent)")]):
        p = ep.d / "images" / f"c{ans['call']:03d}_in{k_im}.jpg"
        im = cv2.imread(str(p)) if p.is_file() else None
        img[BOT_Y + 24 : BOT_Y + 24 + 340, x0 : x0 + 300] = fit(im, 300, 340)
        cv2.putText(img, lab, (x0, BOT_Y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    s_ = sent or {}
    kind = "MISSION" if ans.get("outcome") == "mission" else ("WAYPOINT" if ep.pipeline == "vlnav" else "SUB-GOALS")
    lat = float(ans.get("latency_s", 0.0))
    paused = "sim paused" if ep.meta.get("config", {}).get("latency_mode") == "zero" else "sim NOT paused"
    cost = float(ep.cost[ai]) if len(ep.cost) else 0.0
    lines = [(f"call #{ans['call']}  {kind}  asked at t={float(s_.get('t', 0)):.1f} s   API {lat:.1f} s, {paused}   cost so far ${cost:.3f}   now t={t_ep:.1f} s   {ep.provider} {ep.model}", F_HEAD, (20, 20, 20)), (f"Instruction: {ep.instruction}", F_BODY, (30, 30, 30))]
    d = ans.get("reply") or {}
    if not ans.get("ok", True):
        lines.append((f"FAILED: {str(ans.get('error'))[:200]}", F_BOLD, (30, 30, 200)))
    elif kind == "MISSION":
        lines += [
            (f"Interpretation: {d.get('interpretation', '')}", F_BODY, (30, 30, 30)),
            (f"Target: {d.get('target', '')}", F_BOLD, (30, 120, 30)),
            ("Sub-tasks: " + " | ".join(d.get("subtasks") or []), F_BODY, (30, 30, 30)),
            ("Route: " + " > ".join(d.get("route") or []), F_BOLD, (30, 30, 30)),
            (f"Why: {d.get('why', '')}", F_BODY, (30, 30, 30)),
        ]
    elif kind == "WAYPOINT":
        ppl = ", ".join(f"{p.get('id')} {p.get('relevance')}" for p in d.get("people") or []) or "none"
        hint = ans.get("hint", 0)
        act = d.get("action", "")
        ch = f"CHOICE {ans.get('choice')}" if act == "goto" else act.upper()
        ch += f"  (route follower would pick {hint})" + ("" if act != "goto" else ("  = same" if ans.get("choice") == hint else "  = DIFFERENT"))
        lines += [
            (f"Trigger: {s_.get('reason', '')}", F_BODY, (30, 30, 30)),
            ("Laser obstacles: " + ("; ".join(f"{t['id']} {vn.side_of(t['bearing_deg'], t['dist'])}, {t['motion']}" for t in (s_.get("tracks") or [])[:4]) or "none near"), F_BODY, (30, 30, 30)),
            (f"People the model named: {ppl}", F_BODY, (30, 30, 30)),
            (f"Scene: {d.get('scene', '')}", F_BODY, (30, 30, 30)),
            (ch, F_BOLD, (30, 30, 200) if "DIFFERENT" in ch else (30, 30, 30)),
            (f"Why: {d.get('why', '')}", F_BODY, (30, 30, 30)),
            (f"Expects: {d.get('expects', '')}" + (f"   rules: {', '.join(d.get('rules_applied') or [])}" if d.get("rules_applied") else ""), F_BODY, (30, 30, 30)),
        ]
    else:
        lines += [(f"Action: {d.get('action', '')}  {d.get('heading_to', '')}", F_BOLD, (30, 30, 30)), (f"Why: {d.get('path_reasoning', '')}", F_BODY, (30, 30, 30))]
    _pil_text(img, lines, 640, BOT_Y + 12, W - 660)
    return img


def _pil_text(img: np.ndarray, lines: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]], x: int, y: int, width: int) -> None:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    for text, font, col in lines:
        for ln in wrap_text(text, font, width)[:4]:
            if y > H - 22:
                break
            dr.text((x, y), ln, font=font, fill=(col[2], col[1], col[0]))
            y += font.size + 5
        y += 3
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calls", help="calls episode dir (calls.jsonl, ticks.jsonl, images/)")
    ap.add_argument("--run", default="", help="run dir with rec/ + chase/ (Gazebo) or frames/ + index.csv (hospital_sim)")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--keyframes-only", action="store_true")
    a = ap.parse_args()
    d = pathlib.Path(a.calls)
    run = pathlib.Path(a.run) if a.run else None
    ep = Episode(d, run)
    if not ep.ticks:
        raise SystemExit(f"{d}: no ticks")
    out = pathlib.Path(a.out) if a.out else ((run or d) / "pipeline")
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    for r in ep.answers:
        cv2.imwrite(str(out / "keyframes" / f"c{r['call']:03d}.png"), frame(ep, float(r["wall"]) + 1e-3))
    print(f"{len(ep.answers)} key frames -> {out / 'keyframes'}")
    if a.keyframes_only:
        return
    w0, w1 = ep.tw[0], ep.tw[-1]
    step = a.speed / a.fps
    wr = Writer(out / "pipeline.mp4", a.fps)
    t = w0
    n = 0
    while t <= w1:
        wr.write(frame(ep, t))
        t += step
        n += 1
    wr.close()
    print(f"{n} frames ({(w1 - w0):.0f} s wall at {a.speed}x) -> {out / 'pipeline.mp4'}")


if __name__ == "__main__":
    main()
