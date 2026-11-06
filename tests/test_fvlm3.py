"""Offline tests: no ROS, no API. Run: python -m pytest tests -q"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fvlm3 import floorplan as fp  # noqa: E402
from fvlm3 import perception as pc  # noqa: E402
from fvlm3 import prompts as pr  # noqa: E402
from fvlm3.geometry import Camera, map_to_robot, robot_to_map  # noqa: E402
from fvlm3.harness import Harness, Snapshot  # noqa: E402
from fvlm3.providers import MockProvider, Reply  # noqa: E402


@pytest.fixture(scope="module")
def cam():
    return Camera()


@pytest.fixture(scope="module")
def floor_depth(cam):
    return cam.floor_depth_image(256, 256)


def test_frames_roundtrip():
    pose = np.array([3.0, -2.0, 0.7])
    pts = np.array([[1.0, 0.5], [4.0, -1.0]])
    assert np.allclose(map_to_robot(pose, robot_to_map(pose, pts)), pts)


def test_pixel_robot_pixel_roundtrip(cam):
    for x, y in [(2.0, 0.0), (3.5, 1.0), (5.0, -1.5)]:
        uv = cam.robot_to_pixel(x, y, 0.0)
        assert uv is not None
        p = cam.pixel_to_robot(*uv, depth=None)
        assert p.source == "floor"
        assert math.hypot(p.robot_xy[0] - x, p.robot_xy[1] - y) < 1e-6


def test_depth_projection_matches_floor(cam, floor_depth):
    uv = cam.robot_to_pixel(3.0, 0.8, 0.0)
    p = cam.pixel_to_robot(*uv, depth=floor_depth)
    assert p.source == "depth"
    assert math.hypot(p.robot_xy[0] - 3.0, p.robot_xy[1] - 0.8) < 0.15


def test_point_on_wall_backs_off(cam):
    depth = cam.floor_depth_image(256, 256, wall_x=4.0)
    p = cam.pixel_to_robot(cam.cx, cam.cy - 20, depth)  # above the horizon: hits the wall
    assert p.source == "object_backoff"
    assert 3.2 < p.robot_xy[0] < 3.7


def test_far_clamp(cam):
    p = cam.pixel_to_robot(cam.cx, 10, None, max_range=6.0)
    assert p.source == "far_clamp" and abs(p.range_m - 6.0) < 1e-6


@pytest.mark.parametrize("people,floorplan,rules", [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])
def test_conditions_change_prompt_and_schema(people, floorplan, rules):
    c = pr.Conditions(bool(people), bool(floorplan), bool(rules))
    sch = pr.reply_schema(c)
    sysp = pr.system_prompt(c, ["pharmacy", "reception"])
    assert ("people" in sch["properties"]) == bool(people)
    assert ("heading_to" in sch["properties"]) == bool(floorplan)
    assert ("rules_applied" in sch["properties"]) == bool(rules)
    assert ("R1 " in sysp) == bool(rules)
    assert ("pharmacy" in sysp) == bool(floorplan)
    assert ("labelled p1" in sysp) == bool(people)
    assert set(sch["required"]) == set(sch["properties"])
    # the model must never be asked for coordinates of the goal
    assert "goal_x" not in str(sch) and "coordinates of the goal" not in sysp


def test_people_visible_and_removed(cam, floor_depth):
    pose = np.array([0.0, 0.0, 0.0])
    peds = np.array([[7, 3.0, 0.3, -0.8, 0.0], [8, -2.0, 0.0, 0.0, 0.0]], np.float32)  # one ahead, one behind
    seen = pc.visible_people(peds, pose, cam, floor_depth)
    assert [p.ped_id for p in seen] == [7]
    assert seen[0].motion == "toward_robot"
    rgb = np.full((224, 224, 3), 100, np.uint8)
    u0, v0, u1, v1 = seen[0].box
    rgb[v0:v1, u0:u1] = 255
    out, d2 = pc.remove_people(rgb, floor_depth, seen, cam)
    assert out[(v0 + v1) // 2, (u0 + u1) // 2].mean() < 200
    assert np.isnan(d2).any()


def test_people_hidden_behind_wall(cam):
    depth = cam.floor_depth_image(256, 256, wall_x=2.0)
    peds = np.array([[1, 5.0, 0.0, 0.0, 0.0]], np.float32)
    assert pc.visible_people(peds, np.zeros(3), cam, depth) == []


def test_astar_goes_around_block():
    pose = np.array([0.0, 0.0, 0.0])
    g = fp.LocalGrid(pose)
    wall = [(2.0, y) for y in np.arange(-1.0, 1.01, 0.05)]
    g.add_points(np.array(wall))
    g.inflate(0.3)
    path = g.astar((0.0, 0.0), (4.0, 0.0))
    assert path is not None
    assert max(abs(p[1]) for p in path) > 1.0  # detoured around the block
    assert all(not g.is_blocked(*p) for p in fp.resample(path, 0.1)[3:])


def test_scan_points_drop_non_returns():
    r = np.full(720, 10.0, np.float32)
    r[0] = 2.0
    pts = fp.scan_points_robot(r)
    assert len(pts) == 1 and abs(pts[0, 0] - 2.0) < 1e-6


def _harness(**over):
    cfg = {"people": "on", "floorplan": "off", "rules": "on", "provider": "mock", "log_dir": ""}
    cfg.update(over)
    return Harness(cfg, provider=MockProvider(latency_s=0.0))


def test_answer_projects_with_snapshot_pose(cam, floor_depth):
    """A late answer is projected with the pose of the frame the model saw, not the current pose."""
    h = _harness()
    h.reset("t", {"instruction": "go"})
    snap_pose = np.array([0.0, 0.0, 0.0])
    now_pose = np.array([1.0, 0.0, 0.0])  # robot drove 1 m while the model thought
    uv = cam.robot_to_pixel(4.0, 0.0, 0.0)
    un, vn = cam.px_to_norm(*uv)
    snap = Snapshot(h.clock(), snap_pose, floor_depth, [], 1, "test", 0.0, [])
    grid = fp.build_grid(now_pose, None, None, 0.35)
    data = {"action": "goto", "subgoals": [{"u": un, "v": vn, "why": "x"}], "path_reasoning": "r", "scene": "s", "progress": "searching", "people": [], "rules_applied": []}
    h._apply(Reply(True, data, 1.0), snap, now_pose, grid, h.clock())
    assert h.plan is not None
    end = h.plan[-1]
    assert abs(end[0] - 4.0) < 0.2 and abs(end[1]) < 0.2  # map point stays where the model pointed
    assert abs(h.plan[0][0] - 1.0) < 1e-6  # path starts at the current pose


def test_stale_answer_dropped(floor_depth):
    h = _harness(max_answer_age_s=2.0)
    h.reset("t", {"instruction": "go"})
    snap = Snapshot(h.clock() - 5.0, np.zeros(3), floor_depth, [], 1, "test", 0.0, [])
    grid = fp.build_grid(np.zeros(3), None, None, 0.35)
    data = {"action": "goto", "subgoals": [{"u": 500, "v": 700, "why": "x"}], "path_reasoning": "", "scene": "", "progress": "searching"}
    h._apply(Reply(True, data, 5.0), snap, np.zeros(3), grid, h.clock())
    assert h.plan is None and h.stats["stale"] == 1


def test_prefetch_trigger():
    h = _harness(latency_init_s=4.0, prefetch_margin_s=1.0, speed_mps=0.5)
    h.reset("t", {"instruction": "go"})
    grid = fp.build_grid(np.zeros(3), None, None, 0.35)
    h.plan = fp.resample([(0.0, 0.0), (5.0, 0.0)])  # 10 s of driving left
    h.plan_t = h.clock()
    assert h._trigger(np.zeros(3), grid, [], h.clock()) == ""
    h.plan = fp.resample([(0.0, 0.0), (2.0, 0.0)])  # 4 s left < 4 s latency + 1 s margin
    assert h._trigger(np.zeros(3), grid, [], h.clock()) == "prefetch"
    h.cfg["prefetch"] = False
    assert h._trigger(np.zeros(3), grid, [], h.clock()) == ""


def test_prefetch_reduces_stall():
    from offline_sim import run  # noqa: PLC0415

    off = run(latency=2.0, seconds=20.0, prefetch=False, tick=0.05)
    on = run(latency=2.0, seconds=20.0, prefetch=True, tick=0.05)
    assert on["x_final"] > off["x_final"]
    assert on["stall_s"] < off["stall_s"]


def test_render_episode_from_logs(tmp_path):
    """Planner logs + a recorder index -> frames on the simulation-time axis."""
    import json  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    from offline_sim import run  # noqa: PLC0415
    from render_episode import H, W, Episode, Renderer  # noqa: PLC0415

    run(latency=0.5, seconds=4.0, prefetch=True, tick=0.05, log_dir=str(tmp_path / "vlm_calls"))
    ep_dir = next((tmp_path / "vlm_calls").glob("*/ep*"))
    ticks = [json.loads(x) for x in (ep_dir / "ticks.jsonl").read_text().splitlines()]
    assert "wall" in ticks[0] and "peds" in ticks[0]
    calls = [json.loads(x) for x in (ep_dir / "calls.jsonl").read_text().splitlines()]
    assert any(c.get("plan") for c in calls if c["phase"] == "answer")
    # fake recording: simulation clock = host clock - 1000 s, running at half speed
    rec = tmp_path / "rec"
    (rec / "frames").mkdir(parents=True)
    w0, w1 = ticks[0]["wall"] - 1.0, ticks[-1]["wall"] + 1.0
    img = np.full((48, 64, 3), 90, np.uint8)
    with open(rec / "index.csv", "w") as fh:
        fh.write("frame,stamp_s,wall_s\n")
        for n, w in enumerate(np.arange(w0, w1, 0.1), 1):
            cv2.imwrite(str(rec / "frames" / f"{n:06d}.jpg"), img)
            fh.write(f"{n},{(w - w0) * 0.5 + 1000.0:.4f},{w:.4f}\n")
    ep = Episode(ep_dir, rec)
    assert ep.time_axis.startswith("simulation")
    assert ep.T0 == pytest.approx(1000.5, abs=0.05)
    assert ep.T1 - ep.T0 == pytest.approx(0.5 * (ticks[-1]["wall"] - ticks[0]["wall"]), abs=0.05)
    assert ep.frames
    r = Renderer(ep)
    for T in (ep.T0, (ep.T0 + ep.T1) / 2, ep.T1 + 0.5):
        assert r.frame(T).shape == (H, W, 3)
    # without a recording: host time since the first tick
    ep2 = Episode(ep_dir, None)
    assert ep2.T0 == 0.0 and not ep2.frames
    assert Renderer(ep2).frame(1.0).shape == (H, W, 3)


# ---------------------------------------------------------------- OpenAI / Gemini wire format (fake servers, no network)


class _Fake:
    """Tiny HTTP server: records requests, answers from a queue of (status, body) pairs."""

    def __init__(self, answers):
        import http.server
        import json
        import threading

        self.answers, self.seen = list(answers), []
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, body=b""):
                fake.seen.append({"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": json.loads(body) if body else None})
                code, ans = fake.answers.pop(0) if fake.answers else (200, {"data": []})
                raw = json.dumps(ans).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                self._reply(self.rfile.read(int(self.headers.get("content-length", 0))))

            def do_GET(self):
                self._reply()

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


def _req():
    from fvlm3.providers import Request

    c = pr.Conditions(True, True, True)
    return Request("SYS", "USER", [np.zeros((32, 32, 3), np.uint8), np.zeros((16, 16, 3), np.uint8)], pr.reply_schema(c))


REPLY = {"scene": "hall", "progress": "searching", "action": "goto", "subgoals": [{"u": 500, "v": 700, "why": "open"}], "path_reasoning": "ahead", "people": [], "heading_to": "laboratory", "rules_applied": []}


def test_openai_provider_format_and_retry(monkeypatch):
    import json

    from fvlm3.providers import OpenAIProvider

    ok = {"model": "gpt-5.5-2026-04-23", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(REPLY)}}], "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 200}}}
    f = _Fake([(400, {"error": {"message": "Unsupported parameter: 'reasoning_effort'"}}), (200, ok), (200, {"choices": [{"finish_reason": "length", "message": {"content": ""}}]})])
    monkeypatch.setenv("OPENAI_BASE_URL", f.url + "/openai/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    try:
        p = OpenAIProvider("gpt-5.5", effort="low")
        r = p(_req())
        assert r.ok and r.data == REPLY and r.model.startswith("gpt-5.5")
        assert r.usage == {"input": 1000, "output": 300, "cache_read": 200, "cache_write": 0}
        first, second = f.seen[0], f.seen[1]
        assert first["path"] == "/openai/v1/chat/completions" and first["headers"]["authorization"] == "Bearer dummy"
        assert first["body"]["reasoning_effort"] == "low" and "reasoning_effort" not in second["body"]
        msgs = second["body"]["messages"]
        assert msgs[0] == {"role": "system", "content": "SYS"}
        kinds = [c["type"] for c in msgs[1]["content"]]
        assert kinds == ["image_url", "image_url", "text"] and msgs[1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        rf = second["body"]["response_format"]
        assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True and rf["json_schema"]["schema"]["required"]
        r2 = p(_req())
        assert not r2.ok and r2.error == "max_tokens"
    finally:
        f.close()


def test_gemini_provider_format_and_parse(monkeypatch):
    import json

    from fvlm3.providers import GeminiProvider

    txt = json.dumps(REPLY)
    ok = {"modelVersion": "gemini-3.1-pro-preview", "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "thinking...", "thought": True}, {"text": txt[:20]}, {"text": txt[20:]}]}}], "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 250, "thoughtsTokenCount": 400}}
    f = _Fake([(200, ok), (200, {"candidates": [{"finishReason": "SAFETY"}]}), (429, {"error": {"message": "quota"}})])
    monkeypatch.setenv("GEMINI_BASE_URL", f.url + "/gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    try:
        p = GeminiProvider("gemini-3.1-pro-preview", effort="low")
        r = p(_req())
        assert r.ok and r.data == REPLY
        assert r.usage == {"input": 1500, "output": 650, "cache_read": 0, "cache_write": 0}
        s = f.seen[0]
        assert s["path"] == "/gemini/v1beta/models/gemini-3.1-pro-preview:generateContent" and s["headers"]["x-goog-api-key"] == "dummy"
        b = s["body"]
        assert b["systemInstruction"]["parts"][0]["text"] == "SYS"
        parts = b["contents"][0]["parts"]
        assert [list(q)[0] for q in parts] == ["inline_data", "inline_data", "text"] and parts[0]["inline_data"]["mime_type"] == "image/jpeg"
        g = b["generationConfig"]
        assert g["responseMimeType"] == "application/json" and g["responseJsonSchema"]["required"] and g["thinkingConfig"] == {"thinkingLevel": "low"}
        r2 = p(_req())
        assert not r2.ok and r2.error.startswith("refusal")
        r3 = p(_req())
        assert not r3.ok and r3.error.startswith("rate_limit")
        assert GeminiProvider("gemini-2.5-flash", effort="low")._thinking() == {"thinkingBudget": 1024}
    finally:
        f.close()


def test_keyproxy_routes_and_ledger(tmp_path, monkeypatch):
    import json
    import threading
    import http.server
    import urllib.request

    import keyproxy as kp

    env = tmp_path / "keys.env"
    env.write_text("OPENAI_API_KEY=sk-o\nCLAUDE_API_KEY=sk-c\nexport GEMINI_API_KEY='g-k'\n")
    env.chmod(0o600)
    keys = kp.load_keys(env)
    assert keys == {"openai": "sk-o", "claude": "sk-c", "gemini": "g-k"}
    assert kp.route("/gemini/v1beta/models") == "gemini" and kp.route("/openai/v1/models") == "openai" and kp.route("/v1/messages") == "claude"
    up = _Fake([(200, {"modelVersion": "gemini-3.1-pro-preview", "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 1000}}), (200, {"models": []})])
    monkeypatch.setitem(kp.ROUTES, "gemini", ("/gemini", up.url, "x-goog-api-key", "{}", ("GEMINI_API_KEY",)))
    st = kp.State(keys, max_usd=10.0, max_requests=1, ledger=tmp_path / "ledger.jsonl")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), kp.make_handler(st))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        rq = urllib.request.Request(base + "/gemini/v1beta/models/x:generateContent?key=leak", data=b"{}", headers={"x-goog-api-key": "dummy", "content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(rq) as r:
            assert r.status == 200
        sent = up.seen[0]
        assert sent["path"] == "/v1beta/models/x:generateContent" and sent["headers"]["x-goog-api-key"] == "g-k"
        rec = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])
        assert rec["provider"] == "gemini" and rec["usage"]["output"] == 1000 and rec["usd"] == pytest.approx((1000 * 2.0 + 1000 * 12.0) / 1e6)
        with urllib.request.urlopen(base + "/gemini/v1beta/models") as r:  # GET is free and not counted
            assert r.status == 200
        try:  # request cap reached
            urllib.request.urlopen(urllib.request.Request(base + "/gemini/v1beta/models/x:generateContent", data=b"{}", method="POST"))
            raise AssertionError("expected 402")
        except urllib.error.HTTPError as e:
            assert e.code == 402
        st2 = kp.State(keys, 10.0, 0, tmp_path / "ledger.jsonl")  # ledger is reloaded
        assert st2.by["gemini"]["requests"] == 1
    finally:
        srv.shutdown()
        up.close()


def test_make_provider_defaults():
    from fvlm3.providers import DEFAULT_MODEL, GeminiProvider, OpenAIProvider, cost_usd, make_provider

    cfg = {"provider": "gemini", "model": ""}
    p = make_provider(cfg)
    assert isinstance(p, GeminiProvider) and cfg["model"] == DEFAULT_MODEL["gemini"]
    assert isinstance(make_provider({"provider": "openai", "model": "gpt-5.4-mini"}), OpenAIProvider)
    with pytest.raises(ValueError):
        make_provider({"provider": "llama"})
    u = {"input": 1_000_000, "output": 0, "cache_read": 0, "cache_write": 0}
    assert cost_usd("gpt-5-mini-2025-08-07", u) == pytest.approx(0.25) and cost_usd("gpt-5.5", u) == pytest.approx(5.0) and cost_usd("gpt-5-2025-08-07", u) == pytest.approx(1.25)
    assert cost_usd("gemini-3.1-pro-preview", u) == pytest.approx(2.0) and cost_usd("mock", u) == 0.0


def _scen():
    from scenario_peds import ScenarioPeds

    return ScenarioPeds, ROOT / "scenarios"


def test_scenario_follow_slow_nurse_slows_on_trigger():
    ScenarioPeds, S = _scen()
    sp = ScenarioPeds(S / "fvlm3_follow_slow")
    nurse = sp.by_name["nurse"]
    assert sp.instruction.startswith("Follow the nurse") and sp.start[:2] == [10.0, 6.0]
    sp.update(7.9, [10.0, 6.0])
    assert nurse.pos[1] == pytest.approx(8.5)  # talking for 8 s
    sp.update(11.0, [10.0, 6.0])
    assert nurse._speed() == pytest.approx(0.8) and nurse.pos[1] > 10.0
    sp.update(20.0, [10.0, 6.0])  # passed y = 14: "when" fired, need set after the reaction time
    assert any("stimulus slow" in e for _, e in sp.events)
    assert nurse._speed() == pytest.approx(0.28)
    ok, why = sp.judge(np.array([10.0, 6.0, 0.0]), [])
    assert not ok and "from nurse" in why


def test_scenario_sudden_turn_reverses_toward_robot():
    ScenarioPeds, S = _scen()
    sp = ScenarioPeds(S / "fvlm3_sudden_turn")
    tr = sp.by_name["trickster"]
    sp.update(5.0, [10.0, 6.0])
    assert not sp.events  # robot 4 m away: no trick yet
    sp.update(6.0, [10.0, 7.5])  # robot within 3 m
    sp.update(7.0, [10.0, 7.5])
    assert tr.seq == "back" and tr.vel[1] < -1.0  # rushing south at 1.2 m/s
    assert "cutter" in sp.by_name and sp.by_name["cutter"].seq == "default"


def test_scenario_plain_walkers_spawn_delay_and_once():
    ScenarioPeds, S = _scen()
    sp = ScenarioPeds(S / "fvlm3_group_oncoming")
    sp.update(1.0, [10.0, 6.0])
    assert len(sp.array()) == 0  # group spawns at 2 s
    sp.update(3.0, [10.0, 6.0])
    rows = sp.array()
    assert len(rows) == 3 and np.all(rows[:, 4] < 0)  # three abreast, walking south
    sp.update(60.0, [10.0, 6.0])
    g = sp.by_name["group_c"]
    assert g.pos[1] == pytest.approx(3.2) and g.v0 == 0.0  # once: stands at the last waypoint


def test_people_raw_and_vln_profile():
    c = pr.Conditions.from_config({"people": "raw", "floorplan": "on", "rules": "off"})
    assert (c.people, c.people_mode, c.floorplan, c.rules, c.tag) == (False, "raw", True, False, "PrF1R0")
    v = pr.Conditions.from_config({"profile": "vln", "people": "on", "floorplan": "on", "rules": "on"})
    assert (v.people, v.people_mode, v.floorplan, v.rules, v.tag) == (False, "raw", False, True, "PrF0R1")
    assert pr.Conditions.from_config({"people": "off"}).tag.startswith("P0")
    assert pr.Conditions(True, True, True).people_mode == "on"
    with pytest.raises(ValueError):
        pr.Conditions.from_config({"profile": "nope"})
    with pytest.raises(ValueError):
        pr.Conditions(True, True, True, "raw")
    # raw: no people block in the prompt or schema, like people:=off
    assert "People in view" not in pr.user_text(v, "Go to the lab.", 1, 0.0, [], ["p1 2 m ahead"], 0.0, None, "")
