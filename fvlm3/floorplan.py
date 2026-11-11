"""Floor plan (static map + zone names), local occupancy grid, A*.

The floor plan is the world's ROS map (map.yaml + png). Zone names come from
world.yaml (arena-world/2: levels[0].zones[].polygon; older: zones[].corners).
The model only ever sees the floor plan as an image; it never gets coordinates.
The harness uses the static map for planning only when floorplan:=on, so the
floorplan:=off arm has no prior map at all (laser-only local grid).
"""

from __future__ import annotations

import dataclasses
import heapq
import math
import pathlib

import cv2
import numpy as np
import yaml

from .geometry import robot_to_map


@dataclasses.dataclass
class StaticMap:
    occ: np.ndarray  # bool (H, W), row 0 = lowest y (already flipped)
    res: float
    origin: tuple[float, float]
    zones: list[dict] = dataclasses.field(default_factory=list)  # {name, polygon: [[x,y],...]}
    doors: list[dict] = dataclasses.field(default_factory=list)  # {name, span: [[x,y],[x,y]]}
    objects: list[dict] = dataclasses.field(default_factory=list)  # {name, xy: [x, y]} furniture (grey dots on the plan)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.origin[0]) / self.res), int((y - self.origin[1]) / self.res)

    def occupied(self, x: float, y: float) -> bool:
        i, j = self.world_to_cell(x, y)
        h, w = self.occ.shape
        if not (0 <= i < w and 0 <= j < h):
            return True
        return bool(self.occ[j, i])

    def thick_occ(self, sample_res: float) -> np.ndarray:
        """occ dilated so that sampling every `sample_res` m cannot step over a thin wall (cached)."""
        cache = self.__dict__.setdefault("_thick", {})
        k = max(0, int(math.ceil(sample_res / self.res / 2)))
        if k not in cache:
            ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
            cache[k] = cv2.dilate(self.occ.astype(np.uint8), ker) > 0 if k else self.occ
        return cache[k]

    def zone_of(self, x: float, y: float) -> str | None:
        for z in self.zones:
            poly = np.asarray(z["polygon"], dtype=np.float32)
            if len(poly) >= 3 and cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return z["name"]
        return None


def _resolve_world_dir(world: str) -> pathlib.Path | None:
    roots = [
        pathlib.Path("/opt/arena_ws/src/Arena/arena_simulation_setup/worlds"),
        pathlib.Path.home() / "arena_ws/src/Arena/arena_simulation_setup/worlds",
    ]
    for r in roots:
        if (r / world).is_dir():
            return r / world
    return None


def find_world_files(world: str, map_yaml: str = "", world_yaml: str = "") -> tuple[str, str]:
    if world and (not map_yaml or not world_yaml):
        d = _resolve_world_dir(world)
        if d is not None:
            if not map_yaml:
                for cand in [d / "map" / "0.yaml", d / "0" / "map.yaml", d / "map.yaml", d / "map" / "map.yaml"]:
                    if cand.is_file():
                        map_yaml = str(cand)
                        break
            if not world_yaml:
                for cand in [d / "world.yaml", d / "0" / "world.yaml"]:
                    if cand.is_file():
                        world_yaml = str(cand)
                        break
    return map_yaml, world_yaml


def load_static_map(map_yaml: str, world_yaml: str = "") -> StaticMap:
    meta = yaml.safe_load(open(map_yaml))
    img_path = pathlib.Path(map_yaml).parent / meta["image"]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    p = img.astype(np.float32) / 255.0
    occ_p = p if int(meta.get("negate", 0)) else 1.0 - p
    occ = occ_p > float(meta.get("free_thresh", 0.196))  # unknown counts as blocked
    occ = np.flipud(occ).copy()
    origin = meta.get("origin", [0, 0, 0])
    sm = StaticMap(occ=occ, res=float(meta["resolution"]), origin=(float(origin[0]), float(origin[1])))
    if world_yaml and pathlib.Path(world_yaml).is_file():
        sm.zones, sm.doors = load_zones(world_yaml)
        sm.objects = load_objects(world_yaml)
    return sm


def load_zones(world_yaml: str) -> tuple[list[dict], list[dict]]:
    w = yaml.safe_load(open(world_yaml)) or {}
    level = (w.get("levels") or [w])[0] if isinstance(w, dict) else {}
    zones = []
    for z in level.get("zones") or []:
        poly = z.get("polygon") or [[c["x"], c["y"]] for c in z.get("corners") or []]
        if len(poly) >= 3:
            zones.append({"name": str(z.get("name", "")), "polygon": [[float(a), float(b)] for a, b, *_ in poly]})
    doors = []
    for d in level.get("doors") or []:
        if d.get("span"):
            doors.append({"name": str(d.get("name", "")), "span": [[float(a), float(b)] for a, b, *_ in d["span"]]})
    return zones, doors


def load_objects(world_yaml: str) -> list[dict]:
    w = yaml.safe_load(open(world_yaml)) or {}
    level = (w.get("levels") or [w])[0] if isinstance(w, dict) else {}
    out = []
    for o in level.get("objects") or []:
        p = o.get("pose") or o.get("xy")
        if isinstance(p, dict):
            p = [p.get("x"), p.get("y")]
        if p and len(p) >= 2 and p[0] is not None and p[1] is not None:
            out.append({"name": str(o.get("name", "")), "xy": [float(p[0]), float(p[1])]})
    return out


def scan_points_robot(scan: np.ndarray | None, max_use: float = 8.0) -> np.ndarray:
    """Canonical scan (ray 0 forward, CCW, full circle) -> (N,2) robot-frame hits."""
    if scan is None:
        return np.zeros((0, 2))
    r = np.asarray(scan, dtype=np.float64).ravel()
    if r.size == 0:
        return np.zeros((0, 2))
    a = np.arange(r.size) * (2 * math.pi / r.size)
    rmax = float(np.nanmax(r)) if np.isfinite(r).any() else 0.0
    ok = np.isfinite(r) & (r > 0.05) & (r < min(max_use, rmax - 1e-3))
    return np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1)


class LocalGrid:
    """Robot-centred window (map frame) = laser hits (+ static map when given), inflated."""

    def __init__(self, pose: np.ndarray, size_m: float = 16.0, res: float = 0.1):
        self.res = res
        self.n = int(size_m / res)
        self.x0 = float(pose[0]) - size_m / 2
        self.y0 = float(pose[1]) - size_m / 2
        self.occ = np.zeros((self.n, self.n), dtype=np.uint8)
        self.blocked: np.ndarray | None = None

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def center(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res

    def add_static(self, sm: StaticMap) -> None:
        xs = self.x0 + (np.arange(self.n) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.n) + 0.5) * self.res
        gx, gy = np.meshgrid(xs, ys)
        ii = ((gx - sm.origin[0]) / sm.res).astype(int)
        jj = ((gy - sm.origin[1]) / sm.res).astype(int)
        occ = sm.thick_occ(self.res)  # walls one map pixel wide would fall between the samples
        h, w = occ.shape
        inside = (ii >= 0) & (ii < w) & (jj >= 0) & (jj < h)
        val = np.ones_like(inside)
        val[inside] = occ[jj[inside], ii[inside]]
        self.occ |= val.astype(np.uint8)

    def add_points(self, pts_map: np.ndarray) -> None:
        for x, y in np.asarray(pts_map).reshape(-1, 2):
            i, j = self.cell(x, y)
            if 0 <= i < self.n and 0 <= j < self.n:
                self.occ[j, i] = 1

    def inflate(self, radius: float) -> None:
        k = max(1, int(math.ceil(radius / self.res)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        self.blocked = cv2.dilate(self.occ, ker) > 0

    def is_blocked(self, x: float, y: float) -> bool:
        assert self.blocked is not None
        i, j = self.cell(x, y)
        if not (0 <= i < self.n and 0 <= j < self.n):
            return False  # outside the window: unknown, optimistic
        return bool(self.blocked[j, i])

    def nearest_free(self, x: float, y: float, max_r: float = 1.2) -> tuple[float, float] | None:
        if not self.is_blocked(x, y):
            return x, y
        assert self.blocked is not None
        i0, j0 = self.cell(x, y)
        best, bd = None, 1e9
        k = int(max_r / self.res)
        for dj in range(-k, k + 1):
            for di in range(-k, k + 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < self.n and 0 <= j < self.n and not self.blocked[j, i]:
                    d = di * di + dj * dj
                    if d < bd:
                        bd, best = d, (i, j)
        return None if best is None else self.center(*best)

    def segment_free(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.res * 0.5)))
        for t in np.linspace(0, 1, n):
            if self.is_blocked(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])):
                return False
        return True

    def astar(self, a: tuple[float, float], b: tuple[float, float], partial_min_gain: float = 1.0) -> list[tuple[float, float]] | None:
        """Path a -> b. If b cannot be reached (e.g. behind a wall the laser just saw), the path
        to the reachable cell closest to b, when that gets at least `partial_min_gain` m closer."""
        path = self._astar(a, b)
        if path is not None or partial_min_gain <= 0:
            return path
        return self._astar(a, b, partial=True, min_gain=partial_min_gain)

    def _astar(self, a, b, partial: bool = False, min_gain: float = 1.0):
        assert self.blocked is not None
        blocked = self.blocked.copy()
        si, sj = self.cell(*a)
        gi, gj = self.cell(*b)
        n = self.n
        if not (0 <= si < n and 0 <= sj < n and 0 <= gi < n and 0 <= gj < n):
            return None
        # the robot's own footprint may touch the inflation: free a small disc around the start
        r = 3
        blocked[max(sj - r, 0) : sj + r + 1, max(si - r, 0) : si + r + 1] = False
        if blocked[gj, gi] and not partial:
            return None
        best_cell, best_h = (si, sj), math.hypot(gi - si, gj - sj)
        h0 = best_h
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        g = {(si, sj): 0.0}
        parent: dict = {}
        openq = [(0.0, si, sj)]
        closed = set()
        while openq:
            _, i, j = heapq.heappop(openq)
            if (i, j) in closed:
                continue
            if (i, j) == (gi, gj):
                cells = [(i, j)]
                while cells[-1] in parent:
                    cells.append(parent[cells[-1]])
                cells.reverse()
                pts = [self.center(ci, cj) for ci, cj in cells]
                pts[0], pts[-1] = a, b
                return self._shortcut(pts)
            closed.add((i, j))
            hh = math.hypot(gi - i, gj - j)
            if hh < best_h:
                best_cell, best_h = (i, j), hh
            for di, dj, c in nbrs:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not blocked[nj, ni]:
                    ng = g[(i, j)] + c
                    if ng < g.get((ni, nj), 1e18):
                        g[(ni, nj)] = ng
                        parent[(ni, nj)] = (i, j)
                        heapq.heappush(openq, (ng + math.hypot(gi - ni, gj - nj), ni, nj))
            if len(closed) > 40000:
                break
        if partial and (h0 - best_h) * self.res >= min_gain:
            cells = [best_cell]
            while cells[-1] in parent:
                cells.append(parent[cells[-1]])
            cells.reverse()
            pts = [self.center(ci, cj) for ci, cj in cells]
            pts[0] = a
            return self._shortcut(pts)
        return None

    def _shortcut(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out = [pts[0]]
        k = 0
        while k < len(pts) - 1:
            nxt = len(pts) - 1
            while nxt > k + 1 and not self.segment_free(pts[k], pts[nxt]):
                nxt -= 1
            out.append(pts[nxt])
            k = nxt
        return out


def build_grid(pose: np.ndarray, scan: np.ndarray | None, static: StaticMap | None, radius: float) -> LocalGrid:
    g = LocalGrid(pose)
    if static is not None:
        g.add_static(static)
    g.add_points(robot_to_map(pose, scan_points_robot(scan)))
    g.inflate(radius)
    return g


def resample(path: list[tuple[float, float]], step: float = 0.25) -> np.ndarray:
    if len(path) < 2:
        return np.asarray(path, dtype=np.float64).reshape(-1, 2)
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(L / step))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------- rendering

def render_floorplan(
    sm: StaticMap,
    pose: np.ndarray,
    trail: list[tuple[float, float]],
    plan: np.ndarray | None,
    hfov_deg: float,
    visited_zones: list[str],
    px: int = 640,
) -> np.ndarray:
    """North-up floor plan, whole map scaled to `px` on its long side. BGR uint8.

    Shows walls, zone names, door gaps, the robot (red arrow) with its camera view
    wedge, the trail driven so far (grey) and the current plan (blue). No numbers,
    no axes: the model reads the layout, not coordinates.
    """
    h, w = sm.occ.shape
    scale = px / max(h, w)
    W, H = int(w * scale), int(h * scale)
    base = np.where(sm.occ, 40, 245).astype(np.uint8)
    img = cv2.cvtColor(cv2.resize(np.flipud(base), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int((x - sm.origin[0]) / sm.res * scale), int(H - (y - sm.origin[1]) / sm.res * scale)

    for z in sm.zones:
        poly = np.asarray(z["polygon"])
        cx, cy = poly.mean(axis=0)
        u, v = to_px(cx, cy)
        name = z["name"].replace("_", " ")
        color = (120, 120, 120) if z["name"] in visited_zones else (30, 90, 30)
        cv2.putText(img, name, (u - 4 * len(name), v), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    if len(trail) > 1:
        cv2.polylines(img, [np.array([to_px(*p) for p in trail], np.int32)], False, (150, 150, 150), 2, cv2.LINE_AA)
    if plan is not None and len(plan) > 1:
        cv2.polylines(img, [np.array([to_px(*p) for p in plan], np.int32)], False, (220, 90, 20), 2, cv2.LINE_AA)
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    fov = math.radians(hfov_deg) / 2
    reach = 4.0
    wedge = [to_px(x, y), to_px(x + reach * math.cos(th + fov), y + reach * math.sin(th + fov)), to_px(x + reach * math.cos(th - fov), y + reach * math.sin(th - fov))]
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(wedge, np.int32)], (80, 80, 230))
    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    tip = to_px(x + 0.9 * math.cos(th), y + 0.9 * math.sin(th))
    cv2.arrowedLine(img, to_px(x, y), tip, (0, 0, 220), 3, cv2.LINE_AA, tipLength=0.5)
    return img
