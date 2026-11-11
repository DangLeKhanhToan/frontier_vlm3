"""VL-Nav pipeline (L1-L3): labelled floor plan, mission route, laser tracks, numbered candidates, prompts."""

from __future__ import annotations

import dataclasses
import math
import typing

import cv2
import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
except ImportError:  # the planner venv has no scipy: RouteGraph uses a numpy distance field
    csr_matrix = dijkstra = None

from . import floorplan as fp
from .geometry import Camera, map_to_robot, robot_to_map

XY = tuple[float, float]
Px = tuple[int, int]
Color = tuple[int, int, int]
Rec = dict[str, typing.Any]
Field = tuple[np.ndarray | None, np.ndarray | None]

PASTELS: list[Color] = [(214, 230, 250), (220, 205, 245), (210, 240, 215), (245, 215, 205), (200, 235, 240), (235, 220, 250), (215, 225, 200), (205, 215, 245), (230, 240, 205), (245, 225, 230)]
HALL: Color = (200, 236, 255)
WALL: Color = (45, 45, 45)
DOOR: Color = (0, 130, 255)
DOOR_TXT: Color = (0, 80, 190)
FURN: Color = (150, 150, 150)
ROBOT: Color = (0, 0, 220)
TRAIL: Color = (40, 40, 220)
PLAN: Color = (210, 100, 20)
ROUTE: Color = (235, 200, 140)
CAND: Color = (200, 90, 20)
CHOSEN: Color = (0, 0, 230)
TRACK: Color = (170, 170, 170)
TARGET: Color = (40, 160, 40)

LEGEND_FULL = "red arrow = robot (heading), red wedge = camera view, red line = driven so far, orange bars = doors, grey dots = furniture, light blue = mission route, blue = current plan"
LEGEND_LOCAL = "local view, north up, rings every 1 m; numbers = candidate points; grey discs = laser obstacles (o#), arrow = their motion; red line = driven"


@dataclasses.dataclass
class Waypoint:
    """A route point: a door centre or a point inside the target room."""

    name: str
    kind: str  # door | zone
    xy: XY


class Xf:
    """World metres -> view pixels, north up."""

    def __init__(self, vx0: float, vy1: float, scale: float) -> None:
        self.vx0, self.vy1, self.s = vx0, vy1, scale

    def __call__(self, x: float, y: float) -> Px:
        return int((x - self.vx0) * self.s), int((self.vy1 - y) * self.s)


class PlanMap:
    """Labelled floor plan rendered as the whole map or a robot-centred window."""

    def __init__(self, sm: fp.StaticMap, ppm: float = 40.0) -> None:
        self.sm = sm
        h, w = sm.occ.shape
        self.x0, self.y0 = sm.origin
        self.x1, self.y1 = self.x0 + w * sm.res, self.y0 + h * sm.res
        self.ppm = ppm
        W, H = int((self.x1 - self.x0) * ppm), int((self.y1 - self.y0) * ppm)
        base = np.full((H, W, 3), 238, np.uint8)
        for k, z in enumerate(sm.zones):
            col = HALL if "hall" in z["name"] else PASTELS[k % len(PASTELS)]
            cv2.fillPoly(base, [np.array([self._b(x, y) for x, y in z["polygon"]], np.int32)], col)
        walls = cv2.resize(np.flipud(sm.occ).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        base[walls] = WALL
        for o in sm.objects:
            cv2.circle(base, self._b(*o["xy"]), max(2, int(0.09 * ppm)), FURN, -1, cv2.LINE_AA)
        for d in sm.doors:
            (ax, ay), (bx, by) = d["span"]
            cv2.line(base, self._b(ax, ay), self._b(bx, by), DOOR, max(3, int(0.16 * ppm)), cv2.LINE_AA)
        self.base = base

    def _b(self, x: float, y: float) -> Px:
        return int((x - self.x0) * self.ppm), int((self.y1 - y) * self.ppm)

    def view(self, vx0: float, vy0: float, vx1: float, vy1: float, scale: float) -> tuple[np.ndarray, Xf]:
        """World window [vx0, vx1] x [vy0, vy1] at `scale` px/m, dark outside the map."""
        W, H = int((vx1 - vx0) * scale), int((vy1 - vy0) * scale)
        k = scale / self.ppm
        bu0, bv0 = (vx0 - self.x0) * self.ppm, (self.y1 - vy1) * self.ppm
        M = np.array([[k, 0.0, -bu0 * k], [0.0, k, -bv0 * k]], np.float32)
        out = cv2.warpAffine(self.base, M, (W, H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))
        return out, Xf(vx0, vy1, scale)

    def labels(self, img: np.ndarray, xf: Xf, room_scale: float, door_scale: float, visited: typing.Collection[str] = (), target: str = "") -> None:
        H, W = img.shape[:2]
        for d in self.sm.doors:
            (ax, ay), (bx, by) = d["span"]
            u, v = xf((ax + bx) / 2, (ay + by) / 2)
            if 0 <= u < W and 0 <= v < H:
                _text(img, d["name"], (u + 3, v - 3), door_scale, DOOR_TXT, center=True)
        for z in self.sm.zones:
            u, v = xf(*zone_label_point(z))
            if 0 <= u < W and 0 <= v < H:
                col = TARGET if z["name"] == target else ((110, 110, 110) if z["name"] in visited else (20, 20, 20))
                _text(img, z["name"], (u, v), room_scale, col, center=True, bold=z["name"] == target)

    def full(
        self,
        pose: np.ndarray,
        trail: typing.Sequence[XY],
        plan: np.ndarray | None = None,
        route_xy: typing.Sequence[XY] | None = None,
        hfov_deg: float = 90.0,
        visited: typing.Collection[str] = (),
        target: str = "",
        height_px: int = 900,
    ) -> np.ndarray:
        """The whole floor plan with legend (MISSION and L1 image)."""
        scale = height_px / (self.y1 - self.y0)
        img, xf = self.view(self.x0, self.y0, self.x1, self.y1, scale)
        draw_paths(img, xf, trail, plan, route_xy)
        self.labels(img, xf, 0.42, 0.30, visited, target)
        draw_robot(img, xf, pose, hfov_deg, reach=3.0)
        return _with_legend(img, LEGEND_FULL)

    def local(
        self,
        pose: np.ndarray,
        trail: typing.Sequence[XY],
        plan: np.ndarray | None = None,
        route_xy: typing.Sequence[XY] | None = None,
        cands: typing.Sequence[Rec] = (),
        chosen: int = 0,
        tracks: typing.Sequence[Rec] = (),
        hfov_deg: float = 90.0,
        half_m: float = 6.5,
        px: int = 640,
    ) -> np.ndarray:
        """Robot-centred window with range rings, candidates and laser tracks (WAYPOINT image)."""
        x, y = float(pose[0]), float(pose[1])
        scale = px / (2 * half_m)
        img, xf = self.view(x - half_m, y - half_m, x + half_m, y + half_m, scale)
        for r in range(1, int(half_m) + 1):
            cv2.circle(img, xf(x, y), int(r * scale), (175, 175, 175), 1, cv2.LINE_AA)
        draw_paths(img, xf, trail, plan, route_xy)
        self.labels(img, xf, 0.55, 0.42)
        draw_tracks(img, xf, tracks, scale)
        draw_robot(img, xf, pose, hfov_deg, reach=half_m)
        for c in cands:
            cv2.line(img, xf(x, y), xf(*c["xy"]), CAND, 1, cv2.LINE_AA)
        for c in cands:
            draw_number(img, xf(*c["xy"]), c["id"], chosen == c["id"])
        return _with_legend(img, LEGEND_LOCAL)


def zone_label_point(z: Rec) -> XY:
    poly = np.asarray(z["polygon"], dtype=np.float64)
    return float(poly[:, 0].mean()), float(poly[:, 1].mean())


def _text(img: np.ndarray, s: str, org: Px, scale: float, color: Color, center: bool = False, bold: bool = False) -> None:
    th = 2 if bold else 1
    (tw, tht), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    u, v = org
    if center:
        u, v = u - tw // 2, v + tht // 2
    cv2.putText(img, s, (u, v), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, (u, v), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def _with_legend(img: np.ndarray, legend: str) -> np.ndarray:
    W = img.shape[1]
    lines, cur = [], ""
    for w in legend.split(" "):
        if cv2.getTextSize(cur + " " + w, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0] > W - 12:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    band = np.full((10 + 15 * len(lines), W, 3), 255, np.uint8)
    for k, s in enumerate(lines):
        cv2.putText(band, s, (6, 16 + 15 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def draw_paths(img: np.ndarray, xf: Xf, trail: typing.Sequence[XY] | None, plan: np.ndarray | None = None, route_xy: typing.Sequence[XY] | None = None) -> None:
    if route_xy is not None and len(route_xy) > 1:
        cv2.polylines(img, [np.array([xf(*p) for p in route_xy], np.int32)], False, ROUTE, 5, cv2.LINE_AA)
    if trail is not None and len(trail) > 1:
        cv2.polylines(img, [np.array([xf(*p) for p in trail], np.int32)], False, TRAIL, 2, cv2.LINE_AA)
    if plan is not None and len(plan) > 1:
        cv2.polylines(img, [np.array([xf(*p) for p in plan], np.int32)], False, PLAN, 3, cv2.LINE_AA)


def draw_robot(img: np.ndarray, xf: Xf, pose: np.ndarray, hfov_deg: float, reach: float = 3.0) -> None:
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    f = math.radians(hfov_deg) / 2
    wedge = np.array([xf(x, y), xf(x + reach * math.cos(th + f), y + reach * math.sin(th + f)), xf(x + reach * math.cos(th - f), y + reach * math.sin(th - f))], np.int32)
    ov = img.copy()
    cv2.fillPoly(ov, [wedge], (90, 90, 235))
    img[:] = cv2.addWeighted(ov, 0.18, img, 0.82, 0)
    L, Wd = 0.75, 0.32
    tip = (x + L * math.cos(th), y + L * math.sin(th))
    left = (x - 0.3 * math.cos(th) - Wd * math.sin(th), y - 0.3 * math.sin(th) + Wd * math.cos(th))
    right = (x - 0.3 * math.cos(th) + Wd * math.sin(th), y - 0.3 * math.sin(th) - Wd * math.cos(th))
    tri = np.array([xf(*tip), xf(*left), xf(*right)], np.int32)
    cv2.fillPoly(img, [tri], ROBOT, cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_tracks(img: np.ndarray, xf: Xf, tracks: typing.Sequence[Rec], scale: float) -> None:
    r = max(4, int(0.3 * scale))
    for t in tracks:
        p = xf(*t["xy"])
        cv2.circle(img, p, r, TRACK, -1, cv2.LINE_AA)
        cv2.circle(img, p, r, (90, 90, 90), 1, cv2.LINE_AA)
        if t.get("speed", 0) > 0.25:
            q = xf(t["xy"][0] + 1.5 * t["v"][0], t["xy"][1] + 1.5 * t["v"][1])
            cv2.arrowedLine(img, p, q, (60, 60, 60), 2, cv2.LINE_AA, tipLength=0.3)
        _text(img, t["id"], (p[0] + max(5, r) + 2, p[1] + 4), 0.45, (30, 30, 30))


def draw_number(img: np.ndarray, p: Px, n: int, chosen: bool = False, r: int = 11) -> None:
    cv2.circle(img, p, r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, p, r, CAND, 2, cv2.LINE_AA)
    s = str(n)
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(img, s, (p[0] - tw // 2, p[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CAND, 2, cv2.LINE_AA)
    if chosen:
        cv2.circle(img, p, r + 5, CHOSEN, 3, cv2.LINE_AA)


def annotate_candidates(cam_img: np.ndarray, cam: Camera, cands: typing.Sequence[Rec], depth: np.ndarray | None, upscale: int, chosen: int = 0) -> np.ndarray:
    """Upscaled camera image with the visible floor tinted green and the in-view candidates numbered."""
    out = cam_img.copy()
    if depth is not None and depth.ndim == 2:
        rays = cam.ray_grid(depth.shape[1], depth.shape[0])
        d = np.where(np.isfinite(depth) & (depth > 0.05), depth, np.nan)
        with np.errstate(invalid="ignore"):
            z = cam.z + rays[..., 2] * d  # rays have camera-x 1 and depth is the camera-x distance
            floor = (np.abs(z) < 0.07) & np.isfinite(z)
        m = cv2.resize(floor.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out[m] = (0.55 * out[m] + 0.45 * np.array([60, 200, 60])).astype(np.uint8)
    for c in cands:
        if c.get("px") is not None:
            u, v = c["px"]
            draw_number(out, (int(u * upscale), int(v * upscale)), c["id"], chosen == c["id"], r=13)
    return out


class RouteGraph:
    """8-connected grid over the static map; edges cost more near walls, so routes keep to the middle of hallways."""

    NB: typing.ClassVar[list[tuple[int, int, float]]] = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

    def __init__(self, sm: fp.StaticMap, res: float = 0.1, inflate: float = 0.3) -> None:
        self.sm, self.res = sm, res
        h, w = sm.occ.shape
        k = sm.res / res
        occ = cv2.resize(sm.occ.astype(np.uint8), (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) > 0
        clr = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * res
        self.penalty = 1.0 + 4.0 * np.clip(1.6 - clr, 0.0, None)
        r = max(1, int(round(inflate / res)))
        self.blocked = cv2.dilate(occ.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        self.H, self.W = self.blocked.shape
        self._cache: dict[tuple[int, int], Field] = {}
        self.g = None if dijkstra is None else self._graph()

    def _graph(self) -> csr_matrix:
        idx = np.arange(self.H * self.W).reshape(self.H, self.W)
        rows, cols, vals = [], [], []
        for di, dj, c in [(0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)]:
            ja0, ja1 = max(0, -dj), self.W - max(0, dj)
            sa = (slice(0, self.H - di), slice(ja0, ja1))
            sb = (slice(di, self.H), slice(ja0 + dj, ja1 + dj))
            ok = ~self.blocked[sa] & ~self.blocked[sb]
            wgt = c * self.res * 0.5 * (self.penalty[sa][ok] + self.penalty[sb][ok])
            rows += [idx[sa][ok], idx[sb][ok]]
            cols += [idx[sb][ok], idx[sa][ok]]
            vals += [wgt, wgt]
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.H * self.W,) * 2)

    def _field_numpy(self, c: tuple[int, int]) -> np.ndarray:
        """Distance field by repeated 8-neighbour relaxation, flattened."""
        H, W = self.H, self.W
        d = np.full((H + 2, W + 2), np.inf)
        pen = np.pad(self.penalty, 1, constant_values=1e9)
        fr = np.pad(~self.blocked, 1, constant_values=False)
        d[c[0] + 1, c[1] + 1] = 0.0
        core = (slice(1, H + 1), slice(1, W + 1))
        for _ in range(4 * (H + W)):
            best = d[core].copy()
            for di, dj, w in self.NB:
                nb = (slice(1 + di, H + 1 + di), slice(1 + dj, W + 1 + dj))
                np.minimum(best, d[nb] + w * self.res * 0.5 * (pen[core] + pen[nb]), out=best)
            best[~fr[core]] = np.inf
            if np.array_equal(best, d[core]):
                break
            d[core] = best
        return d[core].ravel()

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.sm.origin[1]) / self.res), int((x - self.sm.origin[0]) / self.res)

    def xy(self, i: int, j: int) -> XY:
        return self.sm.origin[0] + (j + 0.5) * self.res, self.sm.origin[1] + (i + 0.5) * self.res

    def free_cell(self, x: float, y: float, max_r: float = 1.5) -> tuple[int, int] | None:
        i0, j0 = self.cell(x, y)
        k = int(max_r / self.res)
        best, bd = None, 1e9
        for i in range(max(0, i0 - k), min(self.H, i0 + k + 1)):
            for j in range(max(0, j0 - k), min(self.W, j0 + k + 1)):
                if not self.blocked[i, j]:
                    d = (i - i0) ** 2 + (j - j0) ** 2
                    if d < bd:
                        bd, best = d, (i, j)
        return best

    def field(self, x: float, y: float) -> Field:
        """Geodesic distance field to (x, y) and predecessors (None without scipy), cached per cell."""
        c = self.free_cell(x, y)
        if c is None:
            return None, None
        if c not in self._cache:
            if self.g is not None:
                d, pred = dijkstra(self.g, directed=False, indices=c[0] * self.W + c[1], return_predecessors=True)
                self._cache[c] = (d, pred)
            else:
                self._cache[c] = (self._field_numpy(c), None)
        return self._cache[c]

    def dist(self, field: Field, x: float, y: float) -> float:
        d, _ = field
        c = self.free_cell(x, y, 0.6)
        if d is None or c is None:
            return float("inf")
        return float(d[c[0] * self.W + c[1]])

    def path(self, field: Field, x: float, y: float) -> list[XY]:
        """Shortest path from (x, y) to the field's target."""
        d, pred = field
        c = self.free_cell(x, y, 0.8)
        if d is None or c is None or not np.isfinite(d[c[0] * self.W + c[1]]):
            return []
        out: list[XY] = []
        if pred is not None:
            n = c[0] * self.W + c[1]
            while n >= 0 and len(out) < 20000:
                out.append(self.xy(n // self.W, n % self.W))
                n = int(pred[n])
            return out
        i, j = c
        while len(out) < 20000:
            out.append(self.xy(i, j))
            if d[i * self.W + j] <= 1e-9:
                break
            best, bv = None, d[i * self.W + j]
            for di, dj, _w in self.NB:
                a, b = i + di, j + dj
                if 0 <= a < self.H and 0 <= b < self.W and d[a * self.W + b] < bv:
                    best, bv = (a, b), d[a * self.W + b]
            if best is None:
                break
            i, j = best
        return out


def door_center(d: Rec) -> XY:
    (ax, ay), (bx, by) = d["span"]
    return (ax + bx) / 2, (ay + by) / 2


def zone_goal(rg: RouteGraph, zone: Rec, from_xy: XY, inside_m: float = 1.0) -> XY:
    """The free point at least `inside_m` inside the zone that is geodesically closest to `from_xy`."""
    poly = np.asarray(zone["polygon"], dtype=np.float32)
    fld = rg.field(*from_xy)
    best, bd = None, float("inf")
    for x in np.arange(poly[:, 0].min(), poly[:, 0].max(), 0.2):
        for y in np.arange(poly[:, 1].min(), poly[:, 1].max(), 0.2):
            if cv2.pointPolygonTest(poly, (float(x), float(y)), True) < inside_m:
                continue
            i, j = rg.cell(x, y)
            if not (0 <= i < rg.H and 0 <= j < rg.W) or rg.blocked[i, j]:
                continue
            d = rg.dist(fld, x, y) if fld[0] is not None else math.hypot(x - from_xy[0], y - from_xy[1])
            if d < bd:
                bd, best = d, (float(x), float(y))
    return best if best is not None else zone_label_point(zone)


def resolve_route(sm: fp.StaticMap, rg: RouteGraph, names: typing.Sequence[str], target: str, start_xy: XY) -> list[Waypoint]:
    """Route names -> waypoints: door centres, then a point inside the target room; other rooms add none."""
    doors = {d["name"]: d for d in sm.doors}
    zones = {z["name"]: z for z in sm.zones}
    wps: list[Waypoint] = []
    for n in names:
        if n in doors and (not wps or wps[-1].name != n):
            wps.append(Waypoint(n, "door", door_center(doors[n])))
    if target in zones:
        wps.append(Waypoint(target, "zone", zone_goal(rg, zones[target], wps[-1].xy if wps else start_xy)))
    return wps


class Tracker:
    """Laser points off the static walls -> clusters -> nearest-neighbour tracks with velocity."""

    def __init__(self, sm: fp.StaticMap | None, wall_margin: float = 0.2) -> None:
        self.sm = sm
        self.tracks: list[Rec] = []
        self.next_id = 1
        self.static_dil: np.ndarray | None = None
        if sm is not None:
            r = max(1, int(round(wall_margin / sm.res)))
            self.static_dil = cv2.dilate(sm.occ.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0

    def _on_wall(self, pts: np.ndarray) -> np.ndarray:
        if self.static_dil is None or self.sm is None or len(pts) == 0:
            return np.zeros(len(pts), bool)
        i = ((pts[:, 0] - self.sm.origin[0]) / self.sm.res).astype(int)
        j = ((pts[:, 1] - self.sm.origin[1]) / self.sm.res).astype(int)
        h, w = self.static_dil.shape
        inside = (i >= 0) & (i < w) & (j >= 0) & (j < h)
        out = np.ones(len(pts), bool)
        out[inside] = self.static_dil[j[inside], i[inside]]
        return out

    def _detections(self, pose: np.ndarray, scan: np.ndarray | None, max_range: float) -> list[np.ndarray]:
        pr = fp.scan_points_robot(scan, max_range)
        pm = robot_to_map(pose, pr) if len(pr) else np.zeros((0, 2))
        pm = pm[~self._on_wall(pm)]
        clusters: list[np.ndarray] = []
        cur: list[np.ndarray] = []
        for p in pm:  # scan order, so neighbours in angle
            if cur and math.hypot(p[0] - cur[-1][0], p[1] - cur[-1][1]) > 0.3:
                clusters.append(np.array(cur))
                cur = []
            cur.append(p)
        if cur:
            clusters.append(np.array(cur))
        if len(clusters) > 1 and math.hypot(*(clusters[0][0] - clusters[-1][-1])) < 0.3:
            clusters[0] = np.vstack([clusters[-1], clusters[0]])
            clusters.pop()
        return [c.mean(axis=0) for c in clusters if len(c) >= 2 and np.ptp(c, axis=0).max() < 1.2]

    def update(self, pose: np.ndarray, scan: np.ndarray | None, t: float, max_range: float = 8.0) -> list[Rec]:
        dets = self._detections(pose, scan, max_range)
        used: set[int] = set()
        for tr in sorted(self.tracks, key=lambda q: q["last"], reverse=True):
            best, bd = None, 0.7
            pred = np.array(tr["xy"]) + np.array(tr["v"]) * (t - tr["last"])
            for k, d in enumerate(dets):
                dd = float(np.hypot(*(d - pred)))
                if k not in used and dd < bd:
                    best, bd = k, dd
            if best is None:
                continue
            used.add(best)
            v_new = (dets[best] - np.array(tr["xy"])) / max(t - tr["last"], 1e-3)
            a = 0.35
            tr["v"] = [float(a * v_new[0] + (1 - a) * tr["v"][0]), float(a * v_new[1] + (1 - a) * tr["v"][1])]
            tr["xy"] = [float(dets[best][0]), float(dets[best][1])]
            tr["last"] = t
            tr["hits"] += 1
        for k, d in enumerate(dets):
            if k not in used:
                self.tracks.append({"id": f"o{self.next_id}", "xy": [float(d[0]), float(d[1])], "v": [0.0, 0.0], "last": t, "first": t, "hits": 1})
                self.next_id += 1
        self.tracks = [tr for tr in self.tracks if t - tr["last"] < 1.0]
        for tr in self.tracks:
            tr["speed"] = float(math.hypot(*tr["v"])) if tr["hits"] >= 3 else 0.0
        return self.visible(pose)

    def visible(self, pose: np.ndarray) -> list[Rec]:
        out = []
        for tr in self.tracks:
            rel = map_to_robot(pose, np.array([tr["xy"]]))[0]
            d = float(np.hypot(*rel))
            if d <= 7.0:
                out.append({**tr, "dist": d, "bearing_deg": math.degrees(math.atan2(rel[1], rel[0])), "motion": describe_motion(pose, rel, tr)})
        return sorted(out, key=lambda q: q["dist"])


def describe_motion(pose: np.ndarray, rel: np.ndarray, tr: Rec) -> str:
    if tr.get("speed", 0.0) < 0.25:
        return "static (furniture, door or standing person)"
    c, s = math.cos(-pose[2]), math.sin(-pose[2])
    vx, vy = c * tr["v"][0] - s * tr["v"][1], s * tr["v"][0] + c * tr["v"][1]
    r = np.asarray(rel) / max(float(np.hypot(*rel)), 1e-6)
    radial = vx * r[0] + vy * r[1]
    lateral = -vx * r[1] + vy * r[0]
    sp = f"{tr['speed']:.1f} m/s"
    if abs(lateral) > abs(radial):
        return f"crossing {'right to left' if vy > 0 else 'left to right'} at {sp}"
    return f"{'coming toward the robot' if radial < 0 else 'moving away'} at {sp}"


def side_of(bearing_deg: float, dist: float) -> str:
    if abs(bearing_deg) <= 20:
        s = "ahead"
    elif abs(bearing_deg) >= 120:
        s = "behind"
    else:
        s = "left" if bearing_deg > 0 else "right"
    return f"{dist:.1f} m {s}"


def track_lines(tracks: typing.Sequence[Rec], limit: int = 6) -> list[str]:
    return [f"{t['id']}: {side_of(t['bearing_deg'], t['dist'])}, {t['motion']}" for t in tracks[:limit]]


def _route_points(x: float, y: float, route_path: typing.Sequence[XY]) -> list[tuple[float, float, str]]:
    acc, prev, pt = 0.0, (x, y), None
    for q in route_path:
        acc += math.hypot(q[0] - prev[0], q[1] - prev[1])
        prev = q
        if acc >= 3.0:
            pt = q
            break
    pt = pt or route_path[-1]
    hd = math.atan2(pt[1] - y, pt[0] - x)
    return [(pt[0], pt[1], "route")] + [(pt[0] - off * math.sin(hd), pt[1] + off * math.cos(hd), "route_shift") for off in (0.8, -0.8)]


def make_candidates(
    pose: np.ndarray,
    grid: fp.LocalGrid,
    cam: Camera,
    route_path: typing.Sequence[XY],
    tracks: typing.Sequence[Rec],
    rg: RouteGraph | None,
    field: Field | None,
    max_n: int = 7,
) -> list[Rec]:
    """Up to `max_n` free floor points in line of sight: 3 m along the route (+-0.8 m) and a fan ahead, numbered left to right."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    raw = _route_points(x, y, route_path) if route_path else []
    raw += [(x + r * math.cos(th + math.radians(b)), y + r * math.sin(th + math.radians(b)), "fan") for r in (1.5, 2.5, 3.5, 4.5) for b in (-60, -40, -20, 0, 20, 40, 60)]
    ok = [(cx, cy, src) for cx, cy, src in raw if math.hypot(cx - x, cy - y) >= 0.8 and not grid.is_blocked(cx, cy) and grid.segment_free((x, y), (cx, cy)) and all(math.hypot(cx - t["xy"][0], cy - t["xy"][1]) >= 0.6 for t in tracks)]
    if len(ok) < 3:
        for r in (1.2, 2.0):
            for b in range(-180, 180, 30):
                cx, cy = x + r * math.cos(th + math.radians(b)), y + r * math.sin(th + math.radians(b))
                if not grid.is_blocked(cx, cy) and grid.segment_free((x, y), (cx, cy)):
                    ok.append((cx, cy, "around"))
    keep = [p for p in ok if p[2] == "route"][:1]
    rest = [p for p in ok if p not in keep]
    while rest and len(keep) < max_n:
        if not keep:
            keep.append(rest.pop(0))
            continue
        dists = [min(math.hypot(p[0] - k[0], p[1] - k[1]) for k in keep) for p in rest]
        k = int(np.argmax(dists))
        if dists[k] < 0.9:
            break
        keep.append(rest.pop(k))
    cands = []
    for cx, cy, src in keep:
        rel = map_to_robot(pose, np.array([[cx, cy]]))[0]
        togo = rg.dist(field, cx, cy) if (rg is not None and field is not None and field[0] is not None) else float("nan")
        clear = min([math.hypot(cx - t["xy"][0], cy - t["xy"][1]) for t in tracks if t.get("speed", 0) > 0.25] or [99.0])
        cands.append(
            {
                "xy": [float(cx), float(cy)],
                "src": src,
                "range_m": float(np.hypot(*rel)),
                "bearing_deg": math.degrees(math.atan2(rel[1], rel[0])),
                "px": cam.robot_to_pixel(float(rel[0]), float(rel[1]), 0.0),
                "to_go_m": togo,
                "people_clear_m": clear,
            }
        )
    cands.sort(key=lambda c: -c["bearing_deg"])
    for k, c in enumerate(cands, 1):
        c["id"] = k
    return cands


def heuristic_pick(cands: typing.Sequence[Rec]) -> int:
    """A plain route follower's choice: least distance to the next waypoint, penalised near moving obstacles."""
    best, bs = 0, float("inf")
    for c in cands:
        tg = c["to_go_m"] if np.isfinite(c["to_go_m"]) else 50.0 - c["range_m"]
        s = tg + 3.0 * max(0.0, 1.5 - c["people_clear_m"])
        if s < bs:
            best, bs = c["id"], s
    return best


def candidate_lines(cands: typing.Sequence[Rec]) -> list[str]:
    out = []
    for c in cands:
        b = c["bearing_deg"]
        side = "straight ahead" if abs(b) < 10 else f"{abs(b):.0f} deg {'left' if b > 0 else 'right'}"
        seen = "in camera" if c.get("px") is not None else "not in camera"
        tg = f", {c['to_go_m']:.1f} m left to the next waypoint after it" if np.isfinite(c.get("to_go_m", float("nan"))) else ""
        pc = f", {c['people_clear_m']:.1f} m from the nearest moving obstacle" if c["people_clear_m"] < 5 else ""
        out.append(f"{c['id']}: {c['range_m']:.1f} m, {side} ({seen}){tg}{pc}")
    return out


def mission_system(zone_names: typing.Sequence[str], door_names: typing.Sequence[str]) -> str:
    return "\n".join(
        [
            "You are the mission planner of a wheeled robot (about 0.5 m wide) in a building. You get a task in plain language, the robot's camera image and the floor plan of the building.",
            "",
            "The floor plan is north up: walls dark, rooms coloured with their names, doors as orange bars with their names, furniture as grey dots, the robot as a red arrow (pointing where it faces) with its camera view as a red wedge.",
            "",
            "Work out which room the task asks for and a route to it through the named doors and rooms, starting where the robot is. "
            "List the route as names from the plan in driving order, e.g. ['central_hallway', 'pharmacy_door', 'pharmacy']; the route must "
            "end with the target room and pass through the target room's door. Then break the task into short sub-tasks a driver would "
            "follow, including where people may be met and how to behave around them.",
            "",
            "The robot then drives the route; at every later step you will pick the next point to drive to from numbered candidates.",
            "Rooms: " + ", ".join(zone_names),
            "Doors: " + ", ".join(door_names),
        ]
    )


def mission_schema(zone_names: typing.Sequence[str], door_names: typing.Sequence[str]) -> Rec:
    return {
        "type": "object",
        "properties": {
            "interpretation": {"type": "string", "description": "The task in your own words, one sentence."},
            "target": {"type": "string", "enum": list(zone_names)},
            "route": {"type": "array", "items": {"type": "string", "enum": list(zone_names) + list(door_names)}},
            "subtasks": {"type": "array", "items": {"type": "string"}},
            "why": {"type": "string", "description": "What in the floor plan and the camera image the route is based on."},
        },
        "required": ["interpretation", "target", "route", "subtasks", "why"],
        "additionalProperties": False,
    }


def mission_user(instruction: str, zone_here: str | None) -> str:
    return "\n".join([f"Task: {instruction}", f"The robot is in: {zone_here or 'unknown'}.", "Image 1: robot camera. Image 2: floor plan."])


def waypoint_system(rules: typing.Sequence[str] | None) -> str:
    parts = [
        "You are the navigation brain of a wheeled robot (about 0.5 m wide, camera at 1.2 m height) that follows a mission route you planned earlier. At each step you see the camera image and a local map view, both with the same numbered candidate points on the floor, and you choose where the robot drives next.",
        "",
        "Camera image: floor the robot can see is tinted green; candidates in view are numbered circles. "
        "Local map: north up, rings every 1 m around the robot, the robot as a red arrow with its camera wedge, the mission route "
        "in light blue, the part driven in red, laser obstacles as grey discs (o1, o2, ...) with an arrow when they move; "
        "moving ones are usually people. Room and door names are written on the map.",
        "",
        "Actions: 'goto' with 'choice' = the candidate number; 'wait' (stay still for a moment, e.g. to let a person pass); 'turn_left' / 'turn_right' (rotate about 45 degrees); 'arrived' (the robot is inside the target room: the task is done). 'choice' is 0 unless the action is goto.",
        "",
        "Choose the candidate that makes progress along the route while being polite to people: the numbers of metres left "
        "to the next waypoint and the distance to moving obstacles are given for each candidate. A route follower's choice is "
        "given as a hint; choose differently when people, the camera image or the task call for it.",
        "Keep the answer short: what you see, which people matter, your choice and why, what you expect next.",
    ]
    if rules:
        parts += ["", "Follow these rules when you move:", *rules, "In 'rules_applied' name the rules that shaped this decision, or leave it empty."]
    return "\n".join(parts)


def waypoint_schema(rules: bool) -> Rec:
    props: Rec = {
        "scene": {"type": "string", "description": "What the camera shows, one sentence."},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "o-number from the local map, or 'camera' for a person only seen in the image"},
                    "relevance": {"type": "string", "enum": ["blocks_path", "may_cross_path", "nearby", "irrelevant"]},
                },
                "required": ["id", "relevance"],
                "additionalProperties": False,
            },
        },
        "action": {"type": "string", "enum": ["goto", "wait", "turn_left", "turn_right", "arrived"]},
        "choice": {"type": "integer"},
        "expects": {"type": "string", "description": "What you expect to see after this move."},
        "why": {"type": "string"},
    }
    req = ["scene", "people", "action", "choice", "expects", "why"]
    if rules:
        props["rules_applied"] = {"type": "array", "items": {"type": "string"}}
        req.append("rules_applied")
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


def waypoint_user(
    instruction: str,
    mission: Rec,
    route_names: typing.Sequence[str],
    next_wp: str,
    zone_here: str | None,
    reason: str,
    tracks: typing.Sequence[str],
    cands: typing.Sequence[str],
    hint: int,
    history: typing.Sequence[str],
    elapsed_s: float,
    plan_left_m: float,
    note: str = "",
) -> str:
    in_target = " (the target room)" if zone_here and zone_here == mission.get("target") else ""
    lines = [
        f"Task: {instruction}",
        f"Mission: target {mission.get('target', '?')}; route {' > '.join(route_names)}.",
        f"Next waypoint: {next_wp}. The robot is in: {zone_here or 'unknown'}{in_target}.",
        f"{elapsed_s:.0f} s into the task. Why you are asked now: {reason}.",
        f"The robot is still driving the previous choice: {plan_left_m:.1f} m left." if plan_left_m > 0.05 else "The robot is standing still.",
        "Laser obstacles: " + ("; ".join(tracks) if tracks else "none near."),
        "Candidates:",
        *([f"  {c}" for c in cands] or ["  (none: turn or wait)"]),
        f"A route follower would pick {hint}." if hint else "No route-follower hint.",
    ]
    if history:
        lines += ["Your recent decisions (oldest first):", *[f"  - {h}" for h in history]]
    if note:
        lines.append(note)
    lines.append("Image 1: robot camera. Image 2: local map view.")
    return "\n".join(lines)
