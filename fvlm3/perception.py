"""What the model sees in the camera image.

people:=on   visible people get a box and a label p1, p2, ... and a text line each.
people:=off  visible people are inpainted out of the RGB image and the depth image,
             and nothing about people reaches the prompt. (The laser safety stop in
             the harness still sees them; that is the robot, not the model.)

"Visible" = the ground-truth pedestrian (arena_peds) projects into the image and
the depth image agrees that nothing closer hides it.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from .geometry import Camera, map_to_robot

PERSON_H = 1.75
PERSON_W = 0.55


@dataclasses.dataclass
class SeenPerson:
    label: str
    ped_id: int
    box: tuple[int, int, int, int]  # u0, v0, u1, v1 in RGB pixels
    dist: float
    bearing_deg: float  # + = left
    motion: str
    map_xy: tuple[float, float]


def _motion(rx: float, ry: float, rvx: float, rvy: float) -> str:
    speed = math.hypot(rvx, rvy)
    if speed < 0.15:
        return "standing"
    d = math.hypot(rx, ry) + 1e-6
    radial = (rx * rvx + ry * rvy) / d  # + = moving away from robot
    lateral = (-ry * rvx + rx * rvy) / d  # + = moving to robot's left (CCW)
    if radial < -0.5 * speed:
        return "toward_robot"
    if radial > 0.5 * speed:
        return "away_from_robot"
    return "crossing_to_left" if lateral > 0 else "crossing_to_right"


def visible_people(peds: np.ndarray | None, pose: np.ndarray, cam: Camera, depth: np.ndarray | None, max_dist: float = 10.0) -> list[SeenPerson]:
    if peds is None or len(peds) == 0:
        return []
    peds = np.asarray(peds, dtype=np.float64).reshape(-1, 5)
    rel = map_to_robot(pose, peds[:, 1:3])
    th = float(pose[2])
    c, s = math.cos(th), math.sin(th)
    out: list[SeenPerson] = []
    order = np.argsort(np.hypot(rel[:, 0], rel[:, 1]))
    for k in order:
        rx, ry = rel[k]
        dist = float(math.hypot(rx, ry))
        if rx <= 0.2 or dist > max_dist:
            continue
        # box from the person's footprint corners and head height
        corners = []
        for dy in (-PERSON_W / 2, PERSON_W / 2):
            for z in (0.0, PERSON_H):
                p = cam.robot_to_pixel_unclipped(rx, ry + dy, z)
                if p is not None:
                    corners.append(p)
        if len(corners) < 4:
            continue
        us, vs = [p[0] for p in corners], [p[1] for p in corners]
        u0, u1 = int(max(min(us), 0)), int(min(max(us), cam.width - 1))
        v0, v1 = int(max(min(vs), 0)), int(min(max(vs), cam.height - 1))
        if u1 - u0 < 2 or v1 - v0 < 2:
            continue
        # occlusion: depth at the torso must not be much closer than the person
        cu, cv = (u0 + u1) / 2, cam.robot_to_pixel_unclipped(rx, ry, 1.0)
        cv = cv[1] if cv is not None else (v0 + v1) / 2
        cv = min(max(cv, 0), cam.height - 1)
        d = cam.sample_depth(depth, cu, cv, win=1)
        if d is not None and d < rx - 0.6:
            continue
        vx, vy = peds[k, 3], peds[k, 4]
        rvx, rvy = c * vx + s * vy, -s * vx + c * vy
        out.append(
            SeenPerson(
                label=f"p{len(out) + 1}",
                ped_id=int(peds[k, 0]),
                box=(u0, v0, u1, v1),
                dist=dist,
                bearing_deg=math.degrees(math.atan2(ry, rx)),
                motion=_motion(rx, ry, rvx, rvy),
                map_xy=(float(peds[k, 1]), float(peds[k, 2])),
            )
        )
    return out


def remove_people(rgb: np.ndarray, depth: np.ndarray | None, people: list[SeenPerson], cam: Camera) -> tuple[np.ndarray, np.ndarray | None]:
    """Inpaint people out of RGB; mark their depth pixels invalid."""
    if not people:
        return rgb, depth
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for p in people:
        u0, v0, u1, v1 = p.box
        mask[max(v0 - 2, 0) : v1 + 3, max(u0 - 2, 0) : u1 + 3] = 255
    out = cv2.inpaint(np.ascontiguousarray(rgb), mask, 5, cv2.INPAINT_TELEA)
    if depth is not None and depth.ndim == 2 and depth.size:
        dm = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        depth = depth.copy()
        depth[dm] = np.nan
    return out, depth


def people_lines(people: list[SeenPerson]) -> list[str]:
    lines = []
    for p in people:
        side = "ahead" if abs(p.bearing_deg) < 10 else (f"{abs(p.bearing_deg):.0f} deg left" if p.bearing_deg > 0 else f"{abs(p.bearing_deg):.0f} deg right")
        lines.append(f"{p.label}: {p.dist:.1f} m, {side}, {p.motion.replace('_', ' ')}")
    return lines


def annotate_camera(
    rgb: np.ndarray,
    cam: Camera,
    people: list[SeenPerson] | None,
    plan_px: list[tuple[float, float]] | None,
    upscale: int = 3,
) -> np.ndarray:
    """RGB (camera pixels) -> BGR image for the model, `upscale`x larger.

    Draws a light 0..1000 coordinate frame on the border (ticks every 100) so the
    model can name pixels, the plan the robot is still driving (blue), and people
    labels (people:=on only; pass None otherwise).
    """
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[0] * upscale, bgr.shape[1] * upscale
    img = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_CUBIC)
    f = upscale
    for k in range(0, 1001, 100):
        x = int(k / 1000 * (W - 1))
        y = int(k / 1000 * (H - 1))
        cv2.line(img, (x, 0), (x, 8), (255, 255, 255), 1)
        cv2.line(img, (x, H - 9), (x, H - 1), (255, 255, 255), 1)
        cv2.line(img, (0, y), (8, y), (255, 255, 255), 1)
        cv2.line(img, (W - 9, y), (W - 1, y), (255, 255, 255), 1)
        if 0 < k < 1000:
            cv2.putText(img, str(k), (x - 10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(k), (11, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    if plan_px and len(plan_px) > 1:
        pts = np.array([[u * f, v * f] for u, v in plan_px], np.int32)
        cv2.polylines(img, [pts], False, (255, 140, 0), 3, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 6, (255, 140, 0), -1)
    for p in people or []:
        u0, v0, u1, v1 = p.box
        cv2.rectangle(img, (u0 * f, v0 * f), (u1 * f, v1 * f), (0, 220, 255), 2)
        cv2.putText(img, p.label, (u0 * f + 2, max(v0 * f - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    return img


def draw_answer(img: np.ndarray, points_norm: list[tuple[int, int]], color=(60, 60, 255)) -> np.ndarray:
    """Overlay the model's chosen sub-goals (for logs / cards)."""
    out = img.copy()
    H, W = out.shape[:2]
    prev = (W // 2, H - 1)
    for k, (un, vn) in enumerate(points_norm, 1):
        p = (int(un / 1000 * (W - 1)), int(vn / 1000 * (H - 1)))
        cv2.line(out, prev, p, color, 2, cv2.LINE_AA)
        cv2.circle(out, p, 8, color, 2)
        cv2.putText(out, str(k), (p[0] + 9, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        prev = p
    return out
