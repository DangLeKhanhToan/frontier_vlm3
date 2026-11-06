#!/usr/bin/env python3
"""Offline check of the harness without ROS or API cost.

A straight corridor (walls at y = +-1.5 m), a flat-floor depth image, a kinematic
robot that follows the waypoint chunks like Arena's pure pursuit, and the mock
model with a chosen latency. Prints how long the robot stood waiting for the model
with prefetch on vs off.

  python tools/offline_sim.py --latency 2.0 --seconds 30
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fvlm3.geometry import Camera  # noqa: E402
from fvlm3.harness import Harness  # noqa: E402
from fvlm3.providers import MockProvider  # noqa: E402


def corridor_scan(pose: np.ndarray, half_width: float = 1.5, beams: int = 720, rmax: float = 10.0) -> np.ndarray:
    a = pose[2] + np.arange(beams) * (2 * math.pi / beams)
    s = np.sin(a)
    r = np.full(beams, rmax)
    up = s > 1e-6
    dn = s < -1e-6
    r[up] = np.minimum(rmax, (half_width - pose[1]) / s[up])
    r[dn] = np.minimum(rmax, (-half_width - pose[1]) / s[dn])
    return r.astype(np.float32)


def corridor_depth(cam: Camera, rays: np.ndarray, pose: np.ndarray, half_width: float = 1.5) -> np.ndarray:
    """Depth image in the corridor: floor plane + the two side walls (map y = +-half_width)."""
    o = cam.origin()
    c, s = math.cos(pose[2]), math.sin(pose[2])
    cam_y = pose[1] + s * o[0] + c * o[1]  # camera map-y
    ry = s * rays[..., 0] + c * rays[..., 1]  # ray map-y component
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(rays[..., 2] < -1e-6, -o[2] / rays[..., 2], np.inf)
        t = np.minimum(t, np.where(ry > 1e-6, (half_width - cam_y) / ry, np.inf))
        t = np.minimum(t, np.where(ry < -1e-6, (-half_width - cam_y) / ry, np.inf))
    return cam.depth_from_t(rays, t)


def follow(pose: np.ndarray, chunk: list, dt: float, v: float = 0.5, w_max: float = 1.0) -> np.ndarray:
    if not chunk:
        return pose
    pts = np.asarray(chunk, dtype=np.float64)
    d = np.hypot(pts[:, 0], pts[:, 1])
    idx = int(np.argmax(d >= 0.35)) if np.any(d >= 0.35) else len(pts) - 1
    tx, ty = pts[idx]
    ang = math.atan2(ty, tx)
    x, y, th = pose
    if abs(ang) > math.pi / 2:
        th += math.copysign(min(w_max * dt, abs(ang)), ang)
        return np.array([x, y, th])
    th += max(-w_max * dt, min(w_max * dt, 2.0 * ang))
    step = min(v * dt, d[-1])
    return np.array([x + step * math.cos(th), y + step * math.sin(th), th])


def run(latency: float, seconds: float, prefetch: bool, tick: float = 0.1, log_dir: str = "") -> dict:
    cfg = {"people": "off", "floorplan": "off", "rules": "off", "provider": "mock", "prefetch": prefetch, "latency_init_s": latency, "log_dir": log_dir, "run_label": f"offline-prefetch{int(prefetch)}", "save_images": bool(log_dir)}
    h = Harness(cfg, provider=MockProvider(latency_s=latency, seed=1))
    cam = Camera()
    rays = cam.ray_grid(256, 256)
    rgb = np.full((224, 224, 3), 128, np.uint8)
    rgb[112:, :, :] = 90  # floor darker than walls
    pose = np.array([0.0, 0.0, 0.0])
    h.reset("offline", {"instruction": "Drive down the corridor."})
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        depth = corridor_depth(cam, rays, pose)
        feats = {"robot_pose": pose.astype(np.float32), "rgb": rgb, "depth": depth, "scan": corridor_scan(pose), "peds": np.zeros((0, 5), np.float32)}
        m = h.step(feats)
        pose = follow(pose, m.chunk, tick)
        time.sleep(tick)
    s = h.finish()
    s["x_final"] = float(pose[0])
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--log-dir", default="")
    a = ap.parse_args()
    for pf in (False, True):
        s = run(a.latency, a.seconds, pf, log_dir=a.log_dir)
        print(f"prefetch={'on ' if pf else 'off'}  calls={s['calls']:3d}  stall={s['stall_s']:5.1f} s ({100 * s['stall_frac']:4.1f} %)  distance={s['dist_m']:5.1f} m  reasons={s['reasons']}")


if __name__ == "__main__":
    main()
