"""Camera model and frame transforms.

Frames
  robot: x forward, y left, z up, origin on the floor under base_link.
  map:   Arena map frame (pose from RobotPoseTFGenerator: x, y, theta).
  image: u right, v down, in pixels of the RGB frame. The model answers in
         normalised coordinates 0..1000 on both axes; `norm_to_px` converts.

The Gazebo depth camera reports depth along the optical axis (the camera x
axis), so a pixel's 3D point is depth * (1, -(u-cx)/fx, -(v-cy)/fy) in the
camera frame (x forward, y left, z up) before pitch and mount offset.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def robot_to_map(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts (N,2) robot frame -> (N,2) map frame."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(th), math.sin(th)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], axis=1)


def map_to_robot(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(th), math.sin(th)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


@dataclasses.dataclass
class Projection:
    """One projected sub-goal."""

    robot_xy: tuple[float, float]
    source: str  # depth | floor | object_backoff | far_clamp | invalid
    hit_z: float = 0.0
    range_m: float = 0.0


@dataclasses.dataclass
class Camera:
    width: int = 224
    height: int = 224
    hfov_deg: float = 90.0
    x: float = 0.03
    y: float = 0.0
    z: float = 1.19
    pitch_deg: float = 0.0
    depth_max_m: float = 10.0

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:  # square pixels
        return self.fx

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    def _rot(self, v: np.ndarray) -> np.ndarray:
        """Camera frame -> robot frame orientation (pitch down positive)."""
        p = math.radians(self.pitch_deg)
        c, s = math.cos(p), math.sin(p)
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        return np.stack([c * x + s * z, y, -s * x + c * z], axis=-1)

    def _rot_inv(self, v: np.ndarray) -> np.ndarray:
        p = math.radians(self.pitch_deg)
        c, s = math.cos(p), math.sin(p)
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        return np.stack([c * x - s * z, y, s * x + c * z], axis=-1)

    def ray(self, u: float, v: float) -> np.ndarray:
        """Direction (not normalised, camera-x = 1 before pitch) in the robot frame."""
        d = np.array([1.0, -(u - self.cx) / self.fx, -(v - self.cy) / self.fy])
        return self._rot(d)

    def origin(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def norm_to_px(self, un: float, vn: float) -> tuple[float, float]:
        return (min(max(un, 0.0), 1000.0) / 1000.0 * (self.width - 1), min(max(vn, 0.0), 1000.0) / 1000.0 * (self.height - 1))

    def px_to_norm(self, u: float, v: float) -> tuple[int, int]:
        return int(round(u / (self.width - 1) * 1000)), int(round(v / (self.height - 1) * 1000))

    def sample_depth(self, depth: np.ndarray | None, u: float, v: float, win: int = 2) -> float | None:
        """Median valid depth around RGB pixel (u, v); depth may have another resolution, same FOV."""
        if depth is None or depth.ndim != 2 or depth.size == 0:
            return None
        dh, dw = depth.shape
        du = int(round(u / max(self.width - 1, 1) * (dw - 1)))
        dv = int(round(v / max(self.height - 1, 1) * (dh - 1)))
        patch = depth[max(dv - win, 0) : dv + win + 1, max(du - win, 0) : du + win + 1].astype(np.float64)
        ok = np.isfinite(patch) & (patch > 0.05) & (patch < self.depth_max_m * 0.98)
        if not ok.any():
            return None
        return float(np.median(patch[ok]))

    def pixel_to_robot(
        self,
        u: float,
        v: float,
        depth: np.ndarray | None,
        max_range: float = 6.0,
        backoff_m: float = 0.5,
        object_z: float = 0.25,
    ) -> Projection:
        """Project RGB pixel (u, v) to a floor point in the robot frame.

        1. depth available: 3D point; on the floor (z < object_z) keep it, else the
           pixel is on a wall/object/person: step back `backoff_m` along the ray and
           drop to the floor (the robot drives up to the thing, not into it).
        2. no depth: intersect the ray with the floor plane (pixel below horizon).
        3. pixel above the horizon and no depth: walk `max_range` along the bearing.
        Result is clamped to `max_range` from the robot.
        """
        o = self.origin()
        d = self.ray(u, v)
        dep = self.sample_depth(depth, u, v)
        if dep is not None:
            p = o + self._rot(np.array([dep, -(u - self.cx) / self.fx * dep, -(v - self.cy) / self.fy * dep]))
            rng = float(np.hypot(p[0], p[1]))
            if p[2] < object_z:
                xy, src = p[:2], "depth"
            else:
                scale = max(rng - backoff_m, 0.0) / max(rng, 1e-6)
                xy, src = p[:2] * scale, "object_backoff"
            hit_z = float(p[2])
        elif d[2] < -1e-6:
            t = -o[2] / d[2]
            p = o + t * d
            xy, src, hit_z = p[:2], "floor", 0.0
        else:
            bearing = math.atan2(d[1], d[0])
            xy, src, hit_z = np.array([max_range * math.cos(bearing), max_range * math.sin(bearing)]), "far_clamp", 0.0
        rng = float(np.hypot(xy[0], xy[1]))
        if rng > max_range:
            xy = xy * (max_range / rng)
            rng = max_range
            if src == "depth":
                src = "depth_clamped"
        return Projection((float(xy[0]), float(xy[1])), src, hit_z, rng)

    def robot_to_pixel(self, x: float, y: float, z: float = 0.0) -> tuple[float, float] | None:
        """Robot-frame 3D point -> RGB pixel, None if behind the camera or out of view."""
        c = self._rot_inv(np.array([x, y, z]) - self.origin())
        if c[0] <= 0.05:
            return None
        u = self.cx - self.fx * c[1] / c[0]
        v = self.cy - self.fy * c[2] / c[0]
        if -0.5 <= u <= self.width - 0.5 and -0.5 <= v <= self.height - 0.5:
            return float(u), float(v)
        return None

    def robot_to_pixel_unclipped(self, x: float, y: float, z: float = 0.0) -> tuple[float, float] | None:
        c = self._rot_inv(np.array([x, y, z]) - self.origin())
        if c[0] <= 0.05:
            return None
        return float(self.cx - self.fx * c[1] / c[0]), float(self.cy - self.fy * c[2] / c[0])

    def ray_grid(self, w: int, h: int) -> np.ndarray:
        """(h, w, 3) robot-frame rays (camera-x component 1 before pitch) for a w x h image of the same FOV."""
        u = np.linspace(0, self.width - 1, w)
        v = np.linspace(0, self.height - 1, h)
        uu, vv = np.meshgrid(u, v)
        d = np.stack([np.ones_like(uu), -(uu - self.cx) / self.fx, -(vv - self.cy) / self.fy], axis=-1)
        return self._rot(d)

    def depth_from_t(self, rays: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Ray parameter t -> optical-axis depth (= camera-frame x of the hit point)."""
        ok = np.isfinite(t)
        depth = np.full(t.shape, np.inf, dtype=np.float32)
        depth[ok] = self._rot_inv(rays[ok] * t[ok][:, None])[:, 0]
        depth[depth > self.depth_max_m] = np.inf
        return depth

    def floor_depth_image(self, w: int, h: int, wall_x: float | None = None) -> np.ndarray:
        """Synthetic depth image of a flat floor (and optionally a wall at x=wall_x). For tests/mock."""
        rays = self.ray_grid(w, h)
        o = self.origin()
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(rays[..., 2] < -1e-6, -o[2] / rays[..., 2], np.inf)
            if wall_x is not None:
                t = np.minimum(t, np.where(rays[..., 0] > 1e-6, (wall_x - o[0]) / rays[..., 0], np.inf))
        return self.depth_from_t(rays, t)
