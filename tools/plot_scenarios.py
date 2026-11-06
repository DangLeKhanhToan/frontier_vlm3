#!/usr/bin/env python3
"""Top-down design figure of a scenario: map, robot start + goal, people and their scripted paths,
timeline triggers (E1, E2, ...), source/sink regions. CPU, no ROS.

  python3 tools/plot_scenarios.py --out <dir> <scenario_dir> [<scenario_dir> ...]

Behaviour-tree people: their go_to targets are drawn in sequence order (the path they take once
their stimulus fires); waypoint people: their waypoint polyline. BLOCK steps (pursue the robot)
are drawn as a dashed line to the robot route. What the figure cannot show: humansim's local
avoidance, timing, and how far the robot has got when a trigger fires.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fvlm3 import floorplan as fp  # noqa: E402
from tools.render_episode import F_BODY, F_BOLD, F_HEAD, F_SMALL, INK, MUTED, PANEL, WALL, Xf, wrap_text  # noqa: E402

PAL = [(200, 80, 30), (30, 140, 200), (150, 60, 170), (40, 150, 90), (20, 110, 230), (170, 120, 20), (90, 90, 200), (60, 160, 160)]
MAP_PX = 900


def _rgb(c):
    return (c[2], c[1], c[0])


def _dashed(img, p, q, col, th=2, dash=8):
    p, q = np.array(p, float), np.array(q, float)
    L = np.linalg.norm(q - p)
    for s in np.arange(0, L, 2 * dash):
        a = p + (q - p) * s / max(L, 1e-6)
        b = p + (q - p) * min(s + dash, L) / max(L, 1e-6)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), col, th, cv2.LINE_AA)


def _arrow(img, pts, col):
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.arrowedLine(img, a, b, col, 2, cv2.LINE_AA, tipLength=min(0.3, 12 / max(1, np.hypot(b[0] - a[0], b[1] - a[1]))))


def plot(sdir: pathlib.Path, sm: fp.StaticMap, out: pathlib.Path) -> pathlib.Path:
    raw = yaml.safe_load((sdir / "scenario.yaml").read_text())
    import re  # noqa: PLC0415

    text = " ".join(ln[1:].strip() for ln in (sdir / "scenario.yaml").read_text().splitlines() if ln.startswith("#"))
    header = [h.strip() for h in re.split(r"(?=\bE\d+(?: \(t [^)]*\))?: )|(?=Background: )|(?=If BLOCK )", text) if h.strip()]
    h, w = sm.occ.shape
    wx, wy = w * sm.res, h * sm.res
    k = MAP_PX / wy
    mw = int(wx * k) + 40
    img = np.full((MAP_PX + 40, mw + 640, 3), PANEL, np.uint8)
    xf = Xf(k, sm.origin[0], sm.origin[1] + wy, 20, 20)
    for z in sm.zones:
        cv2.fillPoly(img, [xf.many(z["polygon"])], (240, 235, 230))
    occ = cv2.resize(np.flipud(sm.occ).astype(np.uint8) * 255, (int(wx * k), int(wy * k)), interpolation=cv2.INTER_AREA)
    sub = img[20 : 20 + occ.shape[0], 20 : 20 + occ.shape[1]]
    sub[occ > 60] = WALL
    labels: list[tuple[tuple[int, int], str, tuple]] = []
    # regions
    for name, reg in (raw.get("regions") or {}).items():
        pts = xf.many([[p["x"], p["y"]] for p in reg["polygon"]])
        col = (200, 140, 40) if reg["type"] == "source" else (160, 80, 160)
        ov = img.copy()
        cv2.fillPoly(ov, [pts], col)
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        cv2.polylines(img, [pts], True, col, 2)
        rp = (reg.get("config") or {}).get("rate_profile") or []
        txt = f"{name}" + (" rate " + " ".join(f"{p['t']:.0f}s:{p['rate']}" for p in rp) if rp else "")
        labels.append((tuple(pts.min(axis=0)), txt, col))
    # robot
    for rb in raw.get("robots") or []:
        x, y, th = rb["start"]
        route = [(x, y)]
        for ph in rb.get("phases") or []:
            if isinstance(ph.get("pose"), list):
                route.append(tuple(ph["pose"][:2]))
            elif isinstance(ph.get("goto"), str):
                z = next((z for z in sm.zones if z["name"] == ph["goto"]), None)
                if z:
                    route.append(tuple(np.mean(np.asarray(z["polygon"]), axis=0)))
        for a, b in zip(route[:-1], route[1:]):
            _dashed(img, xf(*a), xf(*b), (40, 40, 210), 2, 10)
        p = xf(x, y)
        cv2.circle(img, p, 9, (40, 40, 210), -1, cv2.LINE_AA)
        cv2.line(img, p, xf(x + 1.0 * np.cos(th), y + 1.0 * np.sin(th)), (40, 40, 210), 3, cv2.LINE_AA)
        g = xf(*route[-1])
        cv2.drawMarker(img, g, (40, 150, 40), cv2.MARKER_STAR, 26, 3)
        labels.append(((p[0] + 10, p[1] + 4), "robot", (40, 40, 210)))
        labels.append(((g[0] + 12, g[1] - 8), "goal", (40, 150, 40)))
    # people
    triggered_by: dict[str, list[str]] = {}
    for n, ev in enumerate(raw.get("timeline") or [], 1):
        for st in ev.get("stimulus") or []:
            for ag in st.get("agents") or []:
                triggered_by.setdefault(ag, []).append(f"E{n}")
    for i, hm in enumerate(raw.get("humans") or []):
        col = PAL[i % len(PAL)]
        x, y = hm["pose"][:2]
        p0 = xf(x, y)
        at = (hm.get("agent") or {}).get("agent_type", "adult")
        pts = [p0]
        block = False
        if isinstance(at, str) and at.startswith("./"):
            t = yaml.safe_load((sdir / at).read_text())
            for sname, seq in (t.get("sequences") or {}).items():
                if sname == "ring":
                    continue  # fallback path, not drawn
                for st in (seq.get("steps") or {}).values():
                    if isinstance(st, dict) and st.get("target_pose"):
                        pts.append(xf(st["target_pose"]["x"], st["target_pose"]["y"]))
                    if isinstance(st, dict) and st.get("interaction") == "BLOCK":
                        block = True
        else:
            pts += [xf(*wp[:2]) for wp in hm.get("waypoints") or []]
        if block:
            rx, ry = raw["robots"][0]["start"][:2]
            _dashed(img, p0, xf(rx, ry + 8.0), col, 2, 5)
        if len(pts) > 1:
            _arrow(img, pts, col)
        cv2.circle(img, p0, 7, col, -1, cv2.LINE_AA)
        tag = hm["name"] + (" [" + ",".join(triggered_by[hm["name"]]) + "]" if hm["name"] in triggered_by else "") + (f" +{hm['spawn_delay']:.0f}s" if hm.get("spawn_delay") else "")
        labels.append(((p0[0] + 9, p0[1] - 16), tag, col))
    # triggers
    for n, ev in enumerate(raw.get("timeline") or [], 1):
        wh = str(ev.get("when", ""))
        if "within" in wh and " of " in wh:
            r = float(wh.split("within")[1].split("of")[0])
            cx, cy = (float(v) for v in wh.split(" of ")[1].strip().split(","))
            c = xf(cx, cy)
            cv2.circle(img, c, int(r * k), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(img, c, 4, (0, 0, 0), -1)
            labels.append(((c[0] - int(r * k) - 34, c[1] - 8), f"E{n}", (0, 0, 0)))
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for (x, y), t, col in labels:
        d.text((x, y), t, font=F_SMALL if t[:1] != "E" or " " in t else F_BOLD, fill=_rgb(col))
    x0, y = mw + 10, 20
    d.text((x0, y), sdir.name, font=F_HEAD, fill=_rgb(INK))
    y += 30
    for rb in raw.get("robots") or []:
        for ph in rb.get("phases") or []:
            for ln in wrap_text(f'Instruction: "{ph.get("text", "")}"', F_BOLD, 610):
                d.text((x0, y), ln, font=F_BOLD, fill=_rgb(INK))
                y += 18
    y += 8
    for para in header:
        for ln in wrap_text(para, F_BODY, 610):
            d.text((x0, y), ln, font=F_BODY, fill=_rgb(INK))
            y += 18
    y += 10
    d.text((x0, y), "Timeline triggers", font=F_BOLD, fill=_rgb(INK))
    y += 20
    for n, ev in enumerate(raw.get("timeline") or [], 1):
        who = "; ".join(f"{s['need']} -> {', '.join(s['agents'])}" for s in ev.get("stimulus") or [])
        for ln in wrap_text(f"E{n}: when {ev.get('when')}: {who}", F_SMALL, 610):
            d.text((x0, y), ln, font=F_SMALL, fill=_rgb(MUTED))
            y += 16
    y += 10
    for ln in ["Legend: red dot + bar = robot start and heading, dashed red = straight line to the goal (star), not a path,",
               "coloured dot = person start, arrows = scripted path (waypoints or go_to steps in order),",
               "short dashes = BLOCK: pursues the robot (humansim agent id -1), black circle = trigger zone,",
               "orange area = source (people spawn), purple = sink. Tags [E1] = person driven by that trigger."]:
        d.text((x0, y), ln, font=F_SMALL, fill=_rgb(MUTED))
        y += 16
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{sdir.name}.png"
    pil.save(f)
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", default="hospital_1")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    my, wy = fp.find_world_files(a.world)
    sm = fp.load_static_map(my, wy)
    for s in a.dirs:
        print("wrote", plot(pathlib.Path(s), sm, pathlib.Path(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
