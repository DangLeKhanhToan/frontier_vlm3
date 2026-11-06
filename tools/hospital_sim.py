#!/usr/bin/env python3
"""Offline hospital_1 run: no ROS, no Gazebo, no GPU (no API cost with the default mock model).

A ray-cast 2.5D view of the hospital_1 floor plan (walls tinted by room, floor,
walking people), a laser from the same map, a kinematic robot that follows the
waypoint chunks, and the mock model with a chosen latency. It writes the same
planner logs as a real run plus a 30 Hz "recording camera" in cam_recorder.py's
format, so tools/render_episode.py can be tried before any GPU time.

  python3 tools/hospital_sim.py --run-dir /opt/arena_ws/data/bench3/offline/bench3-20261105-offline-mock --seconds 60
  python3 tools/render_episode.py /opt/arena_ws/data/bench3/offline/bench3-20261105-offline-mock/calls/ep001_offline

--run-dir writes the results layout: <run>/run.json, <run>/calls/ep001_offline/ (planner logs),
<run>/frames/ + <run>/index.csv (30 Hz camera). --out <root> keeps the older
<root>/vlm_calls/<label> + <root>/vlm_video/<label> layout.

A real model on the same offline episode (costs API money; needs the key proxy env, README 5.1):
  python3 tools/hospital_sim.py --run-dir ... --provider gemini --seconds 60 --max-calls 15

Runs in real time (the harness and the model use the host clock), so the robot keeps
driving while the model thinks, as in Gazebo.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fvlm3 import floorplan as fp  # noqa: E402
from fvlm3.geometry import Camera  # noqa: E402
from fvlm3.harness import Harness  # noqa: E402
from fvlm3.providers import MockProvider  # noqa: E402
from tools.offline_sim import follow  # noqa: E402

WALL_H = 2.6
PERSON_R = 0.25
PERSON_H = 1.75
ROBOT_R = 0.3  # jackal footprint, for the offline contact count


class World:
    def __init__(self, world: str = "hospital_1"):
        map_yaml, world_yaml = fp.find_world_files(world)
        if not map_yaml:
            raise SystemExit(f"world {world!r} not found")
        self.sm = fp.load_static_map(map_yaml, world_yaml)
        h, w = self.sm.occ.shape
        self.zone_id = np.full((h, w), -1, np.int32)
        self.colors = [(150, 150, 150)]
        rng = np.random.default_rng(3)
        for k, z in enumerate(self.sm.zones):
            poly = ((np.asarray(z["polygon"]) - np.array(self.sm.origin)) / self.sm.res).astype(np.int32)
            cv2.fillPoly(self.zone_id, [poly], k)
            self.colors.append(tuple(int(c) for c in rng.integers(110, 230, 3)))
        self.colors = np.array(self.colors[1:] + [self.colors[0]], np.float32)  # index -1 -> last
        self.zone_id = cv2.dilate((self.zone_id + 1).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.int32) - 1

    def cast(self, ox: float, oy: float, ang: np.ndarray, rmax: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the first wall along each angle, and the zone index just before it."""
        sm = self.sm
        step = sm.res * 0.5
        r = np.arange(step, rmax, step)
        xs = ox + np.cos(ang)[:, None] * r
        ys = oy + np.sin(ang)[:, None] * r
        i = ((xs - sm.origin[0]) / sm.res).astype(np.int64)
        j = ((ys - sm.origin[1]) / sm.res).astype(np.int64)
        h, w = sm.occ.shape
        inb = (i >= 0) & (i < w) & (j >= 0) & (j < h)
        hit = ~inb | sm.occ[np.clip(j, 0, h - 1), np.clip(i, 0, w - 1)]
        first = np.where(hit.any(1), hit.argmax(1), len(r) - 1)
        d = r[first]
        k = np.maximum(first - 3, 0)
        rows = np.arange(len(ang))
        zid = self.zone_id[np.clip(j[rows, k], 0, h - 1), np.clip(i[rows, k], 0, w - 1)]
        return d, zid


def people_hits(ox, oy, ang, peds) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal distance to the nearest person cylinder per angle (inf if none) and its index."""
    best = np.full(len(ang), np.inf)
    who = np.full(len(ang), -1)
    dx, dy = np.cos(ang), np.sin(ang)
    for n, p in enumerate(peds):
        cx, cy = p[1] - ox, p[2] - oy
        along = cx * dx + cy * dy
        perp = np.abs(-cx * dy + cy * dx)
        ok = (along > 0) & (perp < PERSON_R)
        d = np.where(ok, along - np.sqrt(np.maximum(PERSON_R**2 - perp**2, 0)), np.inf)
        better = d < best
        best[better] = d[better]
        who[better] = n
    return best, who


def render_view(world: World, cam: Camera, w: int, h: int, pose: np.ndarray, peds: np.ndarray, want_rgb: bool = True):
    """(rgb uint8 RGB order or None, depth float32) for a w x h image with the camera's FOV."""
    rays = cam.ray_grid(w, h)
    o = cam.origin()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    ox, oy = pose[0] + c * o[0] - s * o[1], pose[1] + s * o[0] + c * o[1]
    col_ang = pose[2] + np.arctan2(rays[h // 2, :, 1], rays[h // 2, :, 0])
    d_wall, zid = world.cast(ox, oy, col_ang)
    d_p, who = people_hits(ox, oy, col_ang, peds)
    hr = np.hypot(rays[..., 0], rays[..., 1])
    rz = rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_wall = d_wall[None, :] / hr
        z_wall = o[2] + rz * t_wall
        t_wall = np.where(z_wall <= WALL_H, t_wall, np.inf)
        t_p = d_p[None, :] / hr
        z_p = o[2] + rz * t_p
        t_p = np.where((z_p >= 0) & (z_p <= PERSON_H), t_p, np.inf)
        t_floor = np.where(rz < -1e-6, -o[2] / rz, np.inf)
    t = np.minimum(np.minimum(t_wall, t_p), t_floor)
    depth = cam.depth_from_t(rays, t)
    if not want_rgb:
        return None, depth
    rgb = np.empty((h, w, 3), np.float32)
    rgb[:] = (205, 208, 212)  # ceiling
    fl = (t == t_floor) & np.isfinite(t_floor)
    with np.errstate(invalid="ignore"):
        fx = ox + (np.cos(col_ang)[None, :] * hr) * t
        fy = oy + (np.sin(col_ang)[None, :] * hr) * t
        tile = (np.floor(np.nan_to_num(fx)) + np.floor(np.nan_to_num(fy))) % 2 == 0
    rgb[fl] = np.where(tile[fl][:, None], (178, 176, 170), (164, 162, 158))
    wl = (t == t_wall) & np.isfinite(t_wall)
    wc = world.colors[zid]  # per column, RGB-ish
    rgb[wl] = np.broadcast_to(wc[None, :, :], (h, w, 3))[wl]
    pl = (t == t_p) & np.isfinite(t_p)
    zp = np.where(pl, z_p, 0)
    body = np.where((zp > 1.5)[..., None], np.array([225, 190, 160], np.float32), np.where((zp > 0.85)[..., None], np.array([60, 110, 190], np.float32), np.array([50, 55, 70], np.float32)))
    rgb[pl] = body[pl]
    shade = np.clip(1.15 - np.nan_to_num(t * hr, posinf=12) / 14.0, 0.35, 1.0)
    rgb *= shade[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8), depth


def laser(world: World, pose: np.ndarray, peds: np.ndarray, beams: int = 720, rmax: float = 10.0) -> np.ndarray:
    ang = pose[2] + np.arange(beams) * (2 * math.pi / beams)
    d, _ = world.cast(pose[0], pose[1], ang, rmax)
    dp, _ = people_hits(pose[0], pose[1], ang, peds)
    return np.minimum(np.minimum(d, dp), rmax).astype(np.float32)


class Walkers:
    """People pacing between two points (id, x, y, vx, vy)."""

    def __init__(self, lines: list[tuple[tuple[float, float], tuple[float, float], float]]):
        self.lines = lines
        self.phase = np.zeros(len(lines))

    def at(self, t: float) -> np.ndarray:
        out = []
        for k, (a, b, v) in enumerate(self.lines):
            a, b = np.array(a), np.array(b)
            L = float(np.linalg.norm(b - a))
            if L < 1e-6 or v <= 0:
                out.append([k + 1, a[0], a[1], 0.0, 0.0])
                continue
            s = (t * v) % (2 * L)
            fwd = s < L
            pos = a + (b - a) * ((s if fwd else 2 * L - s) / L)
            vel = (b - a) / L * v * (1 if fwd else -1)
            out.append([k + 1, pos[0], pos[1], vel[0], vel[1]])
        return np.array(out, np.float32).reshape(-1, 5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="", help="results run dir: <run>/run.json, <run>/calls/, <run>/frames/")
    ap.add_argument("--out", default="", help="older layout: <out>/vlm_calls/<label> and <out>/vlm_video/<label>")
    ap.add_argument("--label", default="offline-hospital")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--latency", type=float, default=2.5)
    ap.add_argument("--conds", default="P1F1R1", help="e.g. P0F1R1; Pr = people raw (VLN contract, no ground truth); PrF0R1 = profile vln")
    ap.add_argument("--rec-size", default="640x480")
    ap.add_argument("--rec-fps", type=float, default=30.0)
    ap.add_argument("--provider", default="mock", help="mock | claude | openai | gemini")
    ap.add_argument("--model", default="", help="default: the provider default (see planner.yaml)")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-calls", type=int, default=20, help="hard cap on model calls")
    ap.add_argument("--scenario", default="", help="replay a scenario's start, instruction, people and timeline (name under scenarios/ or a folder); default: the fixed lab walk")
    a = ap.parse_args()
    if bool(a.run_dir) == bool(a.out):
        ap.error("give exactly one of --run-dir or --out")
    rw, rh = (int(v) for v in a.rec_size.split("x"))
    onoff = lambda ch: {"1": "on", "r": "raw"}.get(a.conds[a.conds.index(ch) + 1], "off")  # noqa: E731  (P r = people raw)
    cfg = {"people": onoff("P"), "floorplan": onoff("F"), "rules": onoff("R"), "provider": "mock", "world": "hospital_1", "log_dir": a.run_dir or str(pathlib.Path(a.out) / "vlm_calls"), "run_label": "calls" if a.run_dir else a.label, "latency_init_s": a.latency, "mock_latency_s": a.latency, "max_calls": a.max_calls}
    world = World("hospital_1")
    if a.provider == "mock":
        h = Harness(cfg, provider=MockProvider(latency_s=a.latency, seed=2))
    else:
        cfg.update(provider=a.provider, model=a.model, effort=a.effort)
        h = Harness(cfg)  # make_provider: real API calls
        print(f"provider {a.provider}, model {cfg['model']}, effort {a.effort}, at most {a.max_calls} calls", flush=True)
    cam = Camera(**{k: h.cam.__dict__[k] for k in ("hfov_deg", "x", "y", "z", "pitch_deg", "depth_max_m")})
    import json  # noqa: PLC0415

    scen = None
    instruction = "Go to the laboratory."
    pose = np.array([10.0, 6.0, math.pi / 2])
    if a.scenario:
        from tools.scenario_peds import ScenarioPeds  # noqa: PLC0415

        sdir = pathlib.Path(a.scenario) if pathlib.Path(a.scenario).is_dir() else pathlib.Path(__file__).resolve().parents[1] / "scenarios" / a.scenario
        scen = ScenarioPeds(sdir)
        instruction = scen.instruction
        pose = np.array(scen.start, float)
        peds_at = lambda ts: (scen.update(ts, pose), scen.array())[1]  # noqa: E731
    else:
        walkers = Walkers([((10.6, 22.0), (10.6, 9.0), 0.8), ((9.4, 8.0), (9.4, 26.0), 0.6), ((11.0, 15.0), (11.0, 15.0), 0.0), ((14.0, 22.0), (15.5, 25.0), 0.5)])
        peds_at = walkers.at
    rec_dir = pathlib.Path(a.run_dir) if a.run_dir else pathlib.Path(a.out) / "vlm_video" / a.label
    if a.run_dir:
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / "run.json").write_text(json.dumps({"run_id": rec_dir.name, "stage": rec_dir.parent.name, "sim": "hospital_sim (offline, no GPU)", "world": "hospital_1", "scenario": a.scenario or "fixed lab walk", "instruction": instruction, "conditions": a.conds, "provider": a.provider, "model": cfg.get("model", "mock"), "effort": a.effort, "seconds": a.seconds, "max_calls": a.max_calls, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmd": " ".join(sys.argv)}, indent=1) + "\n")
    rec_dir.mkdir(parents=True, exist_ok=True)
    peds_log = open(rec_dir / "peds.csv", "w", buffering=1)
    peds_log.write("t,robot_x,robot_y,robot_yaw,id,name,x,y,dist\n")
    names = scen.names() if scen else {}
    min_d, min_who, contact_s = math.inf, "", 0.0
    (rec_dir / "frames").mkdir(parents=True, exist_ok=True)
    index = open(rec_dir / "index.csv", "w", buffering=1)
    index.write("frame,stamp_s,wall_s\n")
    h.reset("offline", {"instruction": instruction})
    tick = 1.0 / 5.0  # planner rate
    sub = max(1, int(round(a.rec_fps * tick)))
    wall0 = time.time()
    t_next = time.monotonic()
    t_end = t_next + a.seconds
    n_frame = 0
    chunk: list = []
    slow = 0
    while time.monotonic() < t_end and not h.arrived:
        t_sim = time.time() - wall0
        peds = peds_at(t_sim)
        for row in peds:
            d = float(math.hypot(row[1] - pose[0], row[2] - pose[1]))
            nm = names.get(int(row[0]), str(int(row[0])))
            peds_log.write(f"{t_sim:.2f},{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f},{int(row[0])},{nm},{row[1]:.3f},{row[2]:.3f},{d:.3f}\n")
            if d < min_d:
                min_d, min_who = d, nm
            if d < PERSON_R + ROBOT_R:
                contact_s += tick
        rgb, _ = render_view(world, Camera(width=224, height=224, **{k: cam.__dict__[k] for k in ("hfov_deg", "x", "y", "z", "pitch_deg", "depth_max_m")}), 224, 224, pose, peds)
        _, depth = render_view(world, cam, 256, 256, pose, peds, want_rgb=False)
        feats = {"robot_pose": pose.astype(np.float32), "rgb": rgb, "depth": depth, "scan": laser(world, pose, peds), "peds": peds}
        m = h.step(feats)
        chunk = m.chunk if m.kind in ("chunk", "turn") else []
        for k in range(sub):
            pose = follow(pose, chunk, tick / sub)
            ts = time.time() - wall0
            frame, _ = render_view(world, Camera(width=rw, height=rh, **{kk: cam.__dict__[kk] for kk in ("hfov_deg", "x", "y", "z", "pitch_deg", "depth_max_m")}), rw, rh, pose, peds_at(ts))
            n_frame += 1
            cv2.imwrite(str(rec_dir / "frames" / f"{n_frame:06d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            index.write(f"{n_frame},{ts:.4f},{time.time():.4f}\n")
        t_next += tick
        sl = t_next - time.monotonic()
        if sl > 0:
            time.sleep(sl)
        else:
            slow += 1
    s = h.finish()
    index.close()
    peds_log.close()
    if scen:
        met, why = scen.judge(pose, world.sm.zones)
        for t_ev, ev in scen.events:
            print(f"event  {t_ev:5.1f} s  {ev}")
        print(f"people: closest {min_d:.2f} m ({min_who}), contact {contact_s:.1f} s (centres < {PERSON_R + ROBOT_R:.2f} m)")
        print(f"goal:   {'met' if met else 'not met'} at the end: {why}; robot said arrived: {h.arrived}")
        if a.run_dir:
            (rec_dir / "offline_judge.json").write_text(json.dumps({"scenario": a.scenario, "instruction": instruction, "goal_met": bool(met), "goal": why, "robot_arrived_signal": bool(h.arrived), "closest_m": round(min_d, 3), "closest_who": min_who, "contact_s": round(contact_s, 2), "events": [{"t": round(t, 2), "event": e} for t, e in scen.events], "note": "offline preview: people walk blind straight lines, see tools/scenario_peds.py"}, indent=1) + "\n")
    print(f"{a.seconds:.0f} s, {n_frame} frames ({n_frame / a.seconds:.1f} fps), ticks late: {slow}")
    print(f"calls={s['calls']} ok={s['ok']} failed={s['failed']} stale={s['stale']} unusable={s['unusable']} stall={s['stall_s']:.1f}s ({100 * s['stall_frac']:.1f}%) dist={s['dist_m']:.1f} m reasons={s['reasons']}")
    if "latency_p50" in s:
        print(f"latency p50={s['latency_p50']:.1f}s p95={s['latency_p95']:.1f}s  est. cost ${s['cost_usd']:.4f}  arrived={s['arrived_signal']}")
    print(f"logs:   {h.log_root}")
    print(f"frames: {rec_dir}")


if __name__ == "__main__":
    main()
