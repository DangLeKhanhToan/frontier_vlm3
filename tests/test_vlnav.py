"""VL-Nav pipeline (vlnav.py) offline tests: no ROS, no API. Run: python -m pytest tests -q"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fvlm3 import floorplan as fp  # noqa: E402
from fvlm3 import vlnav as vn  # noqa: E402
from fvlm3.geometry import Camera  # noqa: E402
from fvlm3.harness import Harness  # noqa: E402
from fvlm3.providers import MockProvider  # noqa: E402

WORLD = "hospital_1"


@pytest.fixture(scope="module")
def sm() -> fp.StaticMap:
    my, wy = fp.find_world_files(WORLD)
    if not my:
        pytest.skip("hospital_1 not found")
    return fp.load_static_map(my, wy)


@pytest.fixture(scope="module")
def rg(sm: fp.StaticMap) -> vn.RouteGraph:
    return vn.RouteGraph(sm)


def test_static_map_has_objects_and_thin_walls_block(sm: fp.StaticMap) -> None:
    assert len(sm.objects) > 50
    g = fp.build_grid(np.array([10.0, 4.0, math.pi / 2]), None, sm, 0.35)
    # the hallway / nurse room wall at x = 12 is one map pixel thick: it must still block
    assert not g.segment_free((10.0, 4.0), (13.9, 6.25))


def test_render_full_and_local(sm: fp.StaticMap) -> None:
    pm = vn.PlanMap(sm)
    pose = np.array([10.0, 4.0, math.pi / 2])
    full = pm.full(pose, [(10, 2), (10, 4)], None, [(10, 4), (10, 20)], 90, [], "pharmacy")
    loc = pm.local(pose, [(10, 2), (10, 4)], None, None, [{"id": 1, "xy": [10, 6]}], 1, [{"id": "o1", "xy": [10.5, 7], "v": [0.5, 0], "speed": 0.5}])
    assert full.shape[0] > 900 and full.shape[2] == 3
    assert loc.shape[1] == 640
    # crops partly outside the map do not fail
    pm.view(-5, -5, 3, 3, 40)


def test_route_through_named_door(sm: fp.StaticMap, rg: vn.RouteGraph) -> None:
    wps = vn.resolve_route(sm, rg, ["central_hallway", "pharmacy_door", "pharmacy"], "pharmacy", (10, 4))
    assert [w.name for w in wps] == ["pharmacy_door", "pharmacy"]
    assert sm.zone_of(*wps[-1].xy) == "pharmacy"
    f = rg.field(*wps[0].xy)
    path = rg.path(f, 10, 4)
    assert len(path) > 100 and math.hypot(path[-1][0] - 8.0, path[-1][1] - 27.65) < 0.3


def test_numpy_field_matches_scipy(sm: fp.StaticMap) -> None:
    a = vn.RouteGraph(sm)
    if a.g is None:
        pytest.skip("no scipy here: nothing to compare")
    b = vn.RouteGraph(sm)
    b.g = None
    fa, fb = a.field(8.0, 27.65), b.field(8.0, 27.65)
    assert abs(a.dist(fa, 10, 4) - b.dist(fb, 10, 4)) < 0.05
    assert len(b.path(fb, 10, 4)) > 100


def test_candidates_ahead_free_and_hint(sm: fp.StaticMap, rg: vn.RouteGraph) -> None:
    pose = np.array([10.0, 4.0, math.pi / 2])
    g = fp.build_grid(pose, None, sm, 0.35)
    f = rg.field(8.0, 27.65)
    cands = vn.make_candidates(pose, g, Camera(), rg.path(f, 10, 4), [], rg, f)
    assert 3 <= len(cands) <= 7
    assert [c["id"] for c in cands] == list(range(1, len(cands) + 1))
    assert all(c["range_m"] >= 0.8 and not g.is_blocked(*c["xy"]) for c in cands)
    assert any(c["src"] == "route" for c in cands)
    assert vn.heuristic_pick(cands) in {c["id"] for c in cands}


def test_tracker_velocity() -> None:
    tr = vn.Tracker(None)
    pose = np.array([0.0, 0.0, 0.0])
    beams = 720
    for k in range(8):  # a 0.3 m blob 3 m ahead moving left at 0.5 m/s
        t = 0.2 * k
        y = -1.0 + 0.5 * t
        scan = np.full(beams, 30.0)
        for i in range(beams):
            a = i * 2 * math.pi / beams
            # ray-circle intersection with centre (3, y), r 0.15
            dx, dy = math.cos(a), math.sin(a)
            b = dx * 3 + dy * y
            disc = b * b - (9 + y * y - 0.0225)
            if disc > 0 and b > 0:
                scan[i] = b - math.sqrt(disc)
        vis = tr.update(pose, scan, t)
    assert vis and vis[0]["speed"] > 0.3
    assert "right to left" in vis[0]["motion"]


def _feats(cam: Camera, pose: np.ndarray) -> dict[str, np.ndarray]:
    rgb = np.full((224, 224, 3), 120, np.uint8)
    return {"robot_pose": pose, "rgb": rgb, "depth": cam.floor_depth_image(256, 256), "scan": np.full(720, 8.0)}


def test_harness_mission_then_waypoint(tmp_path: pathlib.Path) -> None:
    cfg = {"provider": "mock", "world": WORLD, "floorplan": "on", "people": "raw", "pipeline": "vlnav", "latency_mode": "zero", "log_dir": str(tmp_path)}
    h = Harness(cfg, provider=MockProvider(latency_s=0.0))
    h.reset("t1", {"instruction": "Go to the pharmacy."})
    pose = np.array([10.0, 4.0, math.pi / 2])
    m1 = h.step(_feats(h.cam, pose))  # mission (answered at once in latency_mode zero)
    assert h.mission and h.mission["target"] == "pharmacy"
    assert [w.name for w in h.waypoints][-1] == "pharmacy"
    m2 = h.step(_feats(h.cam, pose))  # first waypoint call: a candidate -> plan
    assert h.plan is not None and m2.kind in ("chunk", "hold")
    calls = (tmp_path / h.log_root.name).glob("ep*/calls.jsonl")
    text = next(calls).read_text()
    assert '"kind": "mission"' in text and '"kind": "waypoint"' in text and '"cands"' in text
    assert m1.kind in ("hold", "chunk")
