"""The harness around the model: one `step()` per planner tick (5 Hz).

Per tick
  1. read pose, RGB, depth, laser, pedestrians
  2. if an API answer has arrived: project its image sub-goals to the map with the
     depth image *of the frame the model saw* and the pose *at that frame*, plan a
     path through them from where the robot is now, swap it in (no stop)
  3. decide whether to start the next call (one call in flight at most):
       idle      : nothing to drive
       prefetch  : remaining plan time < predicted latency + margin   (the latency fix)
       blocked   : laser says the plan is blocked ahead
       person    : a new person within event_person_m (people:=on)
       refresh   : plan older than refresh_s
  4. drive: next ~2 m of the plan as a waypoint chunk, or turn / wait / hold,
     with a laser safety stop that the model cannot override

The model never sees or returns coordinates; its only geometric output is
pixels in the camera image.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import json
import logging
import math
import pathlib
import time

import cv2
import numpy as np

from . import floorplan as fp
from . import perception as pc
from . import prompts as pr
from . import vlnav as vn
from .geometry import Camera, map_to_robot, robot_to_map, wrap
from .providers import Reply, Request, cost_usd, make_provider

_log = logging.getLogger("fvlm3")


@dataclasses.dataclass
class Snapshot:
    t: float
    pose: np.ndarray
    depth: np.ndarray | None
    people: list
    call_no: int
    reason: str
    plan_left_m: float
    images: list
    kind: str = "pixels"  # pixels | mission | waypoint
    extra: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Motion:
    """What the harness tells the bridge this tick."""

    kind: str  # chunk | turn | hold | arrived
    chunk: list[list[float]] = dataclasses.field(default_factory=list)
    signal: str = ""
    why: str = ""


def _pose(features: dict) -> np.ndarray | None:
    p = features.get("robot_pose")
    if p is None:
        return None
    a = np.asarray(p, dtype=np.float64).ravel()
    if a.size < 3 or not np.isfinite(a[:3]).all():
        return None
    return a[:3]


def _img(features: dict, key: str) -> np.ndarray | None:
    a = features.get(key)
    if a is None:
        return None
    a = np.asarray(a)
    return a if a.ndim >= 2 and a.size else None


def path_length(path: np.ndarray) -> float:
    if path is None or len(path) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))


class EpisodeLog:
    def __init__(self, root: pathlib.Path | None, save_images: bool):
        self.root = root
        self.save_images = save_images and root is not None
        if root is not None:
            (root / "images").mkdir(parents=True, exist_ok=True)

    def write(self, name: str, rec: dict) -> None:
        if self.root is None:
            return
        with open(self.root / f"{name}.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")

    def image(self, name: str, img: np.ndarray) -> None:
        if self.save_images:
            cv2.imwrite(str(self.root / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def summary(self, rec: dict) -> None:
        if self.root is not None:
            (self.root / "summary.json").write_text(json.dumps(rec, indent=2, default=float))


class Harness:
    def __init__(self, cfg: dict, provider=None, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        mode = str(cfg.get("latency_mode", "real") or "real").strip().lower()
        if mode not in ("real", "zero"):
            raise ValueError(f"latency_mode must be 'real' or 'zero', got {mode!r}")
        self.zero_latency = mode == "zero"  # step() waits for the model, so a lockstep simulator waits with it
        self.cond = pr.Conditions.from_config(cfg)
        self.cam = Camera(
            hfov_deg=float(cfg.get("cam_hfov_deg", 90)),
            x=float(cfg.get("cam_x", 0.03)),
            y=float(cfg.get("cam_y", 0.0)),
            z=float(cfg.get("cam_z", 1.19)),
            pitch_deg=float(cfg.get("cam_pitch_deg", 0.0)),
            depth_max_m=float(cfg.get("depth_max_m", 10.0)),
        )
        self.static: fp.StaticMap | None = None
        if self.cond.floorplan:
            map_yaml, world_yaml = fp.find_world_files(str(cfg.get("world", "")), str(cfg.get("map_yaml", "")), str(cfg.get("world_yaml", "")))
            if map_yaml:
                self.static = fp.load_static_map(map_yaml, world_yaml)
            else:
                _log.warning("floorplan:=on but no map found (set robot.mobile.config.world or map_yaml); the model gets no floor plan image")
        zone_names = [z["name"] for z in self.static.zones] if self.static else []
        door_names = [d["name"] for d in self.static.doors] if self.static else []
        self.system = pr.system_prompt(self.cond, zone_names, self.zero_latency)
        self.schema = pr.reply_schema(self.cond)
        # VL-Nav pipeline (vlnav.py): "" = pixel sub-goals only, mission = L2, vlnav = L3
        self.pipeline = str(cfg.get("pipeline", "") or "").strip().lower()
        if self.pipeline not in ("", "mission", "vlnav"):
            raise ValueError(f"pipeline must be '', 'mission' or 'vlnav', got {self.pipeline!r}")
        self.planmap = vn.PlanMap(self.static) if self.static is not None else None
        if self.pipeline:
            if self.static is None:
                raise ValueError("pipeline mission/vlnav needs the floor plan (floorplan:=on and a map)")
            self.route_graph = vn.RouteGraph(self.static)
            self.mission_system = vn.mission_system(zone_names, door_names)
            self.mission_schema = vn.mission_schema(zone_names, door_names)
            if self.pipeline == "vlnav":
                self.system = vn.waypoint_system(pr.SOCIAL_RULES if self.cond.rules else None)
                self.schema = vn.waypoint_schema(self.cond.rules)
        self.provider = provider if provider is not None else make_provider(cfg)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="fvlm3-call")
        self.speed = float(cfg.get("speed_mps", 0.5))
        self.radius = float(cfg.get("robot_radius", 0.35))
        label = str(cfg.get("run_label", "")) or f"fvlm3-{self.cond.tag}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        self.log_root = pathlib.Path(str(cfg.get("log_dir", ""))) / label if cfg.get("log_dir") else None
        self.latencies: list[float] = []
        self.episode_idx = 0
        self._new_episode("", "")

    # ------------------------------------------------------------ episode

    def _new_episode(self, episode_id: str, instruction: str) -> None:
        self.instruction = instruction
        self.episode_id = episode_id
        self.paused_s = 0.0  # time spent waiting for the model in latency_mode zero
        self.t0 = self._now()
        self.plan: np.ndarray | None = None
        self.plan_t = 0.0
        self.turn_target: float | None = None
        self.wait_until = 0.0
        self.arrived = False
        self.future: concurrent.futures.Future | None = None
        self.snap: Snapshot | None = None
        self.call_no = 0
        self.history: list[str] = []
        self.trail: list[tuple[float, float]] = []
        self.visited: list[str] = []
        self.seen_ped_ids: set[int] = set()
        self.fail_streak = 0
        self.backoff_until = 0.0
        self.note = ""
        self.last_tick = None
        self.last_pose = None
        self.rgb_checked = False
        self.mission: dict | None = None
        self.route_names: list[str] = []
        self.waypoints: list = []
        self.wp_idx = 0
        self.route_event = ""
        self.tracker = vn.Tracker(self.static) if self.pipeline == "vlnav" else None
        self.tracks: list[dict] = []
        self.announced: set[str] = set()
        self.chosen_xy = None
        self.stats = dict(calls=0, ok=0, failed=0, stale=0, unusable=0, stall_s=0.0, safety_stop_s=0.0, drive_s=0.0, dist_m=0.0, cost_usd=0.0, reasons={}, latencies=[], usage={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
        if self.log_root is not None and episode_id != "":
            self.log = EpisodeLog(self.log_root / f"ep{self.episode_idx:03d}_{episode_id}"[:120], bool(self.cfg.get("save_images", True)))
            self.log.write("meta", {"episode_id": episode_id, "instruction": instruction, "conditions": dataclasses.asdict(self.cond), "config": self.cfg, "system_prompt": self.system, "schema": self.schema, "wall_t0": time.time()})
        else:
            self.log = EpisodeLog(None, False)

    def reset(self, episode_id: str, initial_state: dict | None) -> None:
        self.finish()
        self.episode_idx += 1
        state = initial_state or {}
        self._new_episode(str(episode_id), str(state.get("instruction", "") or ""))

    def finish(self) -> dict:
        if self.future is not None:
            self.future.cancel()
            self.future = None
        s = dict(self.stats)
        lat = s.pop("latencies")
        if lat:
            s["latency_p50"] = float(np.percentile(lat, 50))
            s["latency_p95"] = float(np.percentile(lat, 95))
        s["elapsed_s"] = self._now() - self.t0
        s["stall_frac"] = s["stall_s"] / max(s["elapsed_s"], 1e-6)
        s["arrived_signal"] = self.arrived
        s["conditions"] = self.cond.tag
        if self.episode_id:
            self.log.summary(s)
        return s

    # ------------------------------------------------------------ latency model

    def _now(self) -> float:
        """Harness time. In latency_mode zero the time spent waiting for the model does not count."""
        return self.clock() - self.paused_s

    def predicted_latency(self) -> float:
        if self.zero_latency:
            return 0.0
        lat = self.latencies[-12:]
        if not lat:
            return float(self.cfg.get("latency_init_s", 4.0))
        return float(np.quantile(lat, float(self.cfg.get("latency_quantile", 0.8))))

    # ------------------------------------------------------------ main tick

    def step(self, features: dict) -> Motion:
        now = self._now()
        pose = _pose(features)
        rgb = _img(features, "rgb")
        depth = _img(features, "depth")
        scan = features.get("scan")
        peds = features.get("peds")
        dt = 0.0 if self.last_tick is None else min(now - self.last_tick, 1.0)
        self.last_tick = now
        if pose is None or rgb is None:
            return Motion("hold", why="no pose or camera yet")
        if self.last_pose is not None:
            self.stats["dist_m"] += math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
        self.last_pose = pose.copy()
        if not self.trail or math.hypot(pose[0] - self.trail[-1][0], pose[1] - self.trail[-1][1]) > 0.2:
            self.trail.append((float(pose[0]), float(pose[1])))
        if self.static is not None:
            z = self.static.zone_of(pose[0], pose[1])
            if z and z not in self.visited:
                self.visited.append(z)
        if depth is not None and depth.dtype != np.float32:
            depth = depth.astype(np.float32) / (1000.0 if depth.dtype == np.uint16 else 1.0)
        grid = fp.build_grid(pose, scan, self.static, self.radius)
        self._advance_plan(pose)
        if self.tracker is not None:
            self.tracks = self.tracker.update(pose, scan, now)
        self._advance_route(pose)

        # 2. answer arrived?
        if self.future is not None and self.future.done():
            reply: Reply = self.future.result() if not self.future.cancelled() else Reply(False, {}, 0.0, error="cancelled")
            snap = self.snap
            self.future, self.snap = None, None
            self._apply(reply, snap, pose, grid, now)

        # 3. start the next call?
        people_now = pc.visible_people(peds, pose, self.cam, depth) if (peds is not None) else []
        reason = self._trigger(pose, grid, people_now, now)
        if reason:
            asked_at = self.clock()
            self._start_call(reason, pose, rgb, depth, people_now, now)
            if self.zero_latency:
                now = self._wait_for_answer(pose, grid, asked_at)

        # 4. drive
        m = self._motion(pose, scan, grid, now)
        if m.kind == "hold" and not self.arrived and self.future is not None:
            self.stats["stall_s"] += dt
        if m.kind == "hold" and m.why == "safety_stop":
            self.stats["safety_stop_s"] += dt
        if m.kind in ("chunk", "turn"):
            self.stats["drive_s"] += dt
        if not self.rgb_checked:
            self.rgb_checked = True
            if rgb.shape[1] > 400:
                _log.warning("planner camera is %dx%d; expected the 224 px vln camera (did the adapter bind the recording camera?)", rgb.shape[1], rgb.shape[0])
            self.log.write("camera", {"rgb_shape": list(rgb.shape), "depth_shape": None if depth is None else list(depth.shape)})
        peds_log = [] if peds is None else np.round(np.asarray(peds, dtype=np.float64).reshape(-1, 5), 3).tolist() if np.asarray(peds).size else []
        self.log.write("ticks", {"t": now - self.t0, "wall": time.time(), "pose": pose.tolist(), "peds": peds_log, "call": self.call_no if self.future is not None else 0, "motion": m.kind, "why": m.why, "plan_left_m": path_length(self.plan) if self.plan is not None else 0.0, "in_flight": self.future is not None, "pred_latency": self.predicted_latency(), **self._tick_extra()})
        return m

    # ------------------------------------------------------------ plan bookkeeping

    def _advance_plan(self, pose: np.ndarray) -> None:
        if self.plan is None:
            return
        d = np.hypot(self.plan[:, 0] - pose[0], self.plan[:, 1] - pose[1])
        k = int(np.argmin(d))
        self.plan = self.plan[k:]
        if len(self.plan) < 2 or (len(self.plan) <= 2 and d[-1] < 0.25) or d[-1] < 0.2:
            self.plan = None

    def _plan_blocked(self, grid: fp.LocalGrid, pose: np.ndarray, ahead_m: float = 2.5) -> bool:
        if self.plan is None:
            return False
        acc, prev = 0.0, (pose[0], pose[1])
        for x, y in self.plan[1:]:
            acc += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
            if acc > ahead_m:
                break
            if acc > self.radius + 0.2 and grid.is_blocked(x, y):
                return True
        return False

    # ------------------------------------------------------------ call trigger

    def _trigger(self, pose, grid, people_now, now) -> str:
        if self.future is not None or self.arrived or now < self.backoff_until:
            return ""
        if self.call_no >= int(self.cfg.get("max_calls", 120)):
            return ""
        if self.pipeline and self.mission is None:
            return "mission"
        if self.turn_target is not None or now < self.wait_until:
            return ""
        if self.route_event:
            ev, self.route_event = self.route_event, ""
            return ev
        if self.plan is None:
            return "idle"
        left_s = path_length(self.plan) / max(self.speed, 1e-3)
        if bool(self.cfg.get("prefetch", True)) and left_s < self.predicted_latency() + float(self.cfg.get("prefetch_margin_s", 1.0)):
            return "prefetch"
        if self._plan_blocked(grid, pose):
            return "blocked"
        if self.cond.people:
            near_new = [p for p in people_now if p.dist < float(self.cfg.get("event_person_m", 3.0)) and p.ped_id not in self.seen_ped_ids]
            if near_new:
                return "person"
        if self.tracker is not None:
            near = [t for t in self.tracks if t.get("speed", 0) > 0.25 and t["dist"] < float(self.cfg.get("event_person_m", 3.0))
                    and abs(t["bearing_deg"]) < 100 and t["id"] not in self.announced]
            if near:
                self.announced.update(t["id"] for t in near)
                return "moving obstacle " + ", ".join(f"{t['id']} {vn.side_of(t['bearing_deg'], t['dist'])}" for t in near)
        if now - self.plan_t > float(self.cfg.get("refresh_s", 10.0)):
            return "refresh"
        return ""

    def _start_call(self, reason: str, pose, rgb, depth, people_now, now) -> None:
        self.call_no += 1
        self.stats["calls"] += 1
        key = "moving_obstacle" if reason.startswith("moving") else ("waypoint" if reason.startswith("waypoint") else reason)
        self.stats["reasons"][key] = self.stats["reasons"].get(key, 0) + 1
        if reason == "mission" or self.pipeline == "vlnav":
            return self._start_vlnav_call(reason, pose, rgb, depth, now)
        rgb = np.asarray(rgb)[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        self.cam.width, self.cam.height = rgb.shape[1], rgb.shape[0]
        gt_people = people_now
        if self.cond.people_mode == "off":
            rgb, depth = pc.remove_people(rgb, depth, gt_people, self.cam)
        # raw: the camera image as it is, nothing about people derived from ground truth
        people = [] if self.cond.people_mode == "raw" else gt_people
        for p in people:
            self.seen_ped_ids.add(p.ped_id)
        plan_left = path_length(self.plan) if self.plan is not None else 0.0
        plan_px = None
        if self.plan is not None:
            rel = map_to_robot(pose, self.plan)
            plan_px = [q for q in (self.cam.robot_to_pixel(x, y, 0.0) for x, y in rel) if q is not None]
        cam_img = pc.annotate_camera(rgb, self.cam, people if self.cond.people else None, plan_px, int(self.cfg.get("image_upscale", 3)))
        images = [cam_img]
        if self.cond.floorplan and self.static is not None:
            images.append(self.planmap.full(pose, self.trail, self.plan, self._route_xy(pose), self.cam.hfov_deg, self.visited, (self.mission or {}).get("target", "")))
        zone_here = self.static.zone_of(pose[0], pose[1]) if self.static is not None else None
        text = pr.user_text(self.cond, self.instruction, self.call_no, now - self.t0, self.history[-5:], pc.people_lines(people), plan_left, zone_here, self.note)
        if self.mission is not None:  # L2: the mission route as text; the floor plan shows it in light blue
            nxt = self.waypoints[self.wp_idx].name if self.wp_idx < len(self.waypoints) else self.mission.get("target", "")
            text += f"\nYour mission plan: target {self.mission.get('target')}; route {' > '.join(self.route_names)}; next waypoint {nxt} (light blue line on the floor plan)."
        self.note = ""
        req = Request(self.system, text, images, self.schema, depth=depth, cam=self.cam)
        self.snap = Snapshot(now, pose.copy(), None if depth is None else depth.copy(), people, self.call_no, reason, plan_left, images)
        self.future = self.pool.submit(self.provider, req)
        self.log.write("calls", {"call": self.call_no, "phase": "sent", "t": now - self.t0, "wall": time.time(), "reason": reason, "images": [f"c{self.call_no:03d}_in{k}.jpg" for k in range(len(images))], "pose": pose.tolist(), "plan_left_m": plan_left, "pred_latency": self.predicted_latency(), "text": text, "people": [dataclasses.asdict(p) for p in people], "gt_people": [dataclasses.asdict(p) for p in gt_people]})
        for k, im in enumerate(images):
            self.log.image(f"c{self.call_no:03d}_in{k}.jpg", im)

    # ------------------------------------------------------------ apply an answer

    def _wait_for_answer(self, pose, grid, asked_at: float) -> float:
        """latency_mode zero: block this tick until the model answers, then use the answer at once."""
        try:
            reply = self.future.result()
        except Exception as exc:  # noqa: BLE001
            reply = Reply(False, {}, self.clock() - asked_at, error=repr(exc))
        snap, self.future, self.snap = self.snap, None, None
        self.paused_s += self.clock() - asked_at
        now = self._now()
        self._apply(reply, snap, pose, grid, now)
        return now

    def _apply(self, reply: Reply, snap: Snapshot | None, pose, grid, now) -> None:
        if snap is None:
            return
        age = now - snap.t
        rec = {"call": snap.call_no, "phase": "answer", "t": now - self.t0, "wall": time.time(), "latency_s": reply.latency_s, "age_s": age, "ok": reply.ok, "error": reply.error, "usage": reply.usage, "model": reply.model, "reply": reply.data}
        self.stats["latencies"].append(reply.latency_s)
        if reply.usage:
            for k in self.stats["usage"]:
                self.stats["usage"][k] += int(reply.usage.get(k, 0))
            c = cost_usd(reply.model or str(self.cfg.get("model", "")), reply.usage)
            self.stats["cost_usd"] += c
            rec["cost_usd"] = c
        if reply.ok:
            self.latencies.append(reply.latency_s)
        if not reply.ok:
            self.stats["failed"] += 1
            self.fail_streak += 1
            if self.fail_streak >= 3:
                self.backoff_until = now + min(2.0 * self.fail_streak, 15.0)
            rec["outcome"] = "failed"
            self.log.write("calls", rec)
            return
        self.fail_streak = 0
        self.stats["ok"] += 1
        if age > float(self.cfg.get("max_answer_age_s", 12.0)) and snap.kind != "mission":
            self.stats["stale"] += 1
            rec["outcome"] = "stale"
            self.note = "Your previous answer arrived too late and was not used."
            self.log.write("calls", rec)
            return
        d = reply.data
        if snap.kind == "mission":
            self._apply_mission(d, rec, pose)
            return
        if snap.kind == "waypoint":
            self._apply_waypoint(d, rec, snap, pose, grid, now)
            return
        action = str(d.get("action", "wait"))
        summary = f"t={snap.t - self.t0:.0f}s {action}: {str(d.get('path_reasoning', ''))[:110]}"
        outcome = action
        if action == "arrived":
            self.arrived = True
            self.plan = None
        elif action in ("turn_left", "turn_right"):
            step = math.radians(float(self.cfg.get("turn_step_deg", 45)))
            self.turn_target = wrap(pose[2] + (step if action == "turn_left" else -step))
            self.plan = None
        elif action == "wait":
            self.wait_until = now + 1.5
            self.plan = None
        else:
            path, proj = self._project_and_plan(d.get("subgoals") or [], snap, pose, grid)
            rec["projected"] = proj
            if path is None:
                self.stats["unusable"] += 1
                outcome = "unusable"
                self.note = "None of your previous points was reachable (on an obstacle, behind the robot, or too close). Choose points on open floor farther ahead, or turn."
                summary += " [points unreachable]"
            else:
                self.plan = path
                self.plan_t = now
                rec["plan_len_m"] = path_length(path)
                rec["plan"] = np.round(path, 3).tolist()
        rec["outcome"] = outcome
        self.history.append(summary)
        self.log.write("calls", rec)
        if snap.images and self.log.save_images:
            pts = [(int(s.get("u", 0)), int(s.get("v", 0))) for s in (d.get("subgoals") or [])][: int(self.cfg.get("max_subgoals", 3))]
            self.log.image(f"c{snap.call_no:03d}_answer.jpg", pc.draw_answer(snap.images[0], pts))

    # ------------------------------------------------------------ VL-Nav pipeline (L2 / L3)

    def _route_xy(self, pose) -> list | None:
        """Route still to drive, robot -> next waypoint -> ... -> target (for drawing)."""
        if not self.waypoints or self.wp_idx >= len(self.waypoints):
            return None
        pts, cur = [(float(pose[0]), float(pose[1]))], (float(pose[0]), float(pose[1]))
        for w in self.waypoints[self.wp_idx:]:
            seg = self.route_graph.path(self.route_graph.field(*w.xy), *cur)
            pts += seg[::3] + ([seg[-1]] if seg else [])
            cur = w.xy
        return pts

    def _advance_route(self, pose) -> None:
        if not self.waypoints or self.wp_idx >= len(self.waypoints):
            return
        w = self.waypoints[self.wp_idx]
        here = self.static.zone_of(pose[0], pose[1]) if self.static is not None else None
        reached = (w.kind == "door" and math.hypot(pose[0] - w.xy[0], pose[1] - w.xy[1]) < 1.0) or (w.kind == "zone" and here == w.name)
        if reached:
            self.wp_idx += 1
            nxt = self.waypoints[self.wp_idx].name if self.wp_idx < len(self.waypoints) else ""
            self.route_event = f"waypoint {w.name} reached" + (f", next {nxt}" if nxt else " (inside the target room)")

    def _tick_extra(self) -> dict:
        if not self.pipeline:
            return {}
        out = {"wp": self.waypoints[self.wp_idx].name if self.wp_idx < len(self.waypoints) else ""}
        if self.tracker is not None:
            out["tracks"] = [[t["id"], round(t["xy"][0], 2), round(t["xy"][1], 2), round(t["v"][0], 2), round(t["v"][1], 2)] for t in self.tracks]
        return out

    def _prep_rgb(self, rgb):
        rgb = np.asarray(rgb)[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        self.cam.width, self.cam.height = rgb.shape[1], rgb.shape[0]
        return rgb

    def _start_vlnav_call(self, reason: str, pose, rgb, depth, now) -> None:
        rgb = self._prep_rgb(rgb)
        up = int(self.cfg.get("image_upscale", 3))
        zone_here = self.static.zone_of(pose[0], pose[1])
        plan_left = path_length(self.plan) if self.plan is not None else 0.0
        extra: dict = {}
        if reason == "mission":
            kind = "mission"
            cam_img = pc.annotate_camera(rgb, self.cam, None, None, up)
            images = [cam_img, self.planmap.full(pose, self.trail, None, None, self.cam.hfov_deg, self.visited)]
            text = vn.mission_user(self.instruction, zone_here)
            system, schema = self.mission_system, self.mission_schema
        else:
            kind = "waypoint"
            grid = fp.build_grid(pose, None, self.static, self.radius)
            for t in self.tracks:  # laser obstacles into the grid used for candidates (static map + tracks)
                grid.add_points(np.array([t["xy"]]))
            grid.inflate(self.radius)
            w = self.waypoints[self.wp_idx] if self.wp_idx < len(self.waypoints) else None
            field = self.route_graph.field(*w.xy) if w is not None else None
            route_path = self.route_graph.path(field, pose[0], pose[1]) if field is not None else []
            cands = vn.make_candidates(pose, grid, self.cam, route_path, self.tracks, self.route_graph, field, int(self.cfg.get("max_candidates", 7)))
            hint = vn.heuristic_pick(cands)
            route_xy = self._route_xy(pose)
            cam_img = vn.annotate_candidates(pc.annotate_camera(rgb, self.cam, None, None, up), self.cam, cands, depth, up)
            local = self.planmap.local(pose, self.trail, self.plan, route_xy, cands, 0, self.tracks, self.cam.hfov_deg)
            images = [cam_img, local]
            text = vn.waypoint_user(self.instruction, self.mission or {}, self.route_names, w.name if w else (self.mission or {}).get("target", "?"),
                                    zone_here, reason, vn.track_lines(self.tracks), vn.candidate_lines(cands), hint, self.history[-4:], now - self.t0, plan_left, self.note)
            system, schema = self.system, self.schema
            extra = {"cands": cands, "hint": hint, "route_xy": route_xy, "tracks": [dict(t) for t in self.tracks]}
        self.note = ""
        req = Request(system, text, images, schema, depth=depth, cam=self.cam)
        self.snap = Snapshot(now, pose.copy(), None if depth is None else depth.copy(), [], self.call_no, reason, plan_left, images, kind, extra)
        self.future = self.pool.submit(self.provider, req)
        rec = {"call": self.call_no, "phase": "sent", "kind": kind, "t": now - self.t0, "wall": time.time(), "reason": reason,
               "images": [f"c{self.call_no:03d}_in{k}.jpg" for k in range(len(images))], "pose": pose.tolist(), "plan_left_m": plan_left,
               "pred_latency": self.predicted_latency(), "text": text, "zone": zone_here,
               "wp": self.waypoints[self.wp_idx].name if self.wp_idx < len(self.waypoints) else ""}
        if extra:
            rec.update(cands=[{k: c[k] for k in ("id", "xy", "px", "src", "range_m", "bearing_deg", "to_go_m", "people_clear_m")} for c in extra["cands"]],
                       hint=extra["hint"], route_xy=extra["route_xy"], tracks=[{k: t[k] for k in ("id", "xy", "v", "speed", "dist", "bearing_deg", "motion")} for t in extra["tracks"]])
        self.log.write("calls", rec)
        for k, im in enumerate(images):
            self.log.image(f"c{self.call_no:03d}_in{k}.jpg", im)

    def _apply_mission(self, d: dict, rec: dict, pose) -> None:
        self.mission = d
        self.route_names = [str(n) for n in d.get("route") or []]
        target = str(d.get("target", ""))
        self.waypoints = vn.resolve_route(self.static, self.route_graph, self.route_names, target, (float(pose[0]), float(pose[1])))
        self.wp_idx = 0
        rec["outcome"] = "mission"
        rec["waypoints"] = [dataclasses.asdict(w) for w in self.waypoints]
        self.history.append(f"mission: target {target}, route {' > '.join(self.route_names)}")
        self.log.write("calls", rec)

    def _apply_waypoint(self, d: dict, rec: dict, snap: Snapshot, pose, grid, now) -> None:
        action = str(d.get("action", "wait"))
        choice = int(d.get("choice", 0) or 0)
        cands = {c["id"]: c for c in snap.extra.get("cands", [])}
        hint = snap.extra.get("hint", 0)
        outcome = action
        if action == "arrived":
            self.arrived = True
            self.plan = None
        elif action in ("turn_left", "turn_right"):
            step = math.radians(float(self.cfg.get("turn_step_deg", 45)))
            self.turn_target = wrap(pose[2] + (step if action == "turn_left" else -step))
            self.plan = None
        elif action == "wait":
            self.wait_until = now + 1.5
            self.plan = None
        elif choice in cands:
            c = cands[choice]
            goal = grid.nearest_free(*c["xy"])
            path = None
            if goal is not None:
                seg = grid.astar((float(pose[0]), float(pose[1])), goal)
                if seg is None and grid.segment_free((float(pose[0]), float(pose[1])), goal):
                    seg = [(float(pose[0]), float(pose[1])), goal]
                path = fp.resample(seg, 0.25) if seg and len(seg) >= 2 else None
            if path is None:
                self.stats["unusable"] += 1
                outcome = "unusable"
                self.note = f"Candidate {choice} could not be reached any more; choose again."
            else:
                self.plan, self.plan_t, self.chosen_xy = path, now, c["xy"]
                rec["plan_len_m"] = path_length(path)
                rec["plan"] = np.round(path, 3).tolist()
        else:
            outcome = "unusable"
            self.stats["unusable"] += 1
            self.note = f"Choice {choice} is not one of the candidates."
        rec.update(outcome=outcome, choice=choice, hint=hint, agrees_with_hint=(choice == hint))
        self.stats.setdefault("hint_agree", 0)
        self.stats.setdefault("choices", 0)
        if action == "goto":
            self.stats["choices"] += 1
            self.stats["hint_agree"] += int(choice == hint)
        self.history.append(f"t={snap.t - self.t0:.0f}s {action}{' ' + str(choice) if action == 'goto' else ''}: {str(d.get('why', ''))[:100]}")
        self.log.write("calls", rec)
        if self.log.save_images and snap.images:
            up = int(self.cfg.get("image_upscale", 3))
            self.log.image(f"c{snap.call_no:03d}_answer.jpg", vn.annotate_candidates(snap.images[0], self.cam, [c for c in snap.extra.get("cands", []) if c["id"] == choice], None, up, choice))

    def _project_and_plan(self, subgoals: list, snap: Snapshot, pose, grid: fp.LocalGrid):
        """Image points -> map points (with the snapshot's depth + pose) -> path from the current pose."""
        proj = []
        goals: list[tuple[float, float]] = []
        for s in subgoals[: int(self.cfg.get("max_subgoals", 3))]:
            u, v = self.cam.norm_to_px(float(s.get("u", 500)), float(s.get("v", 800)))
            p = self.cam.pixel_to_robot(u, v, snap.depth, max_range=float(self.cfg.get("max_subgoal_m", 6.0)))
            mx, my = robot_to_map(snap.pose, np.array([p.robot_xy]))[0]
            rec = {"u": s.get("u"), "v": s.get("v"), "source": p.source, "range_m": p.range_m, "map_xy": [float(mx), float(my)]}
            free = grid.nearest_free(float(mx), float(my))
            if free is None:
                rec["dropped"] = "no free cell nearby"
            elif math.hypot(free[0] - pose[0], free[1] - pose[1]) < 0.4:
                rec["dropped"] = "too close"
            else:
                rec["snapped_xy"] = [free[0], free[1]]
                goals.append(free)
            proj.append(rec)
        if not goals:
            return None, proj
        path: list[tuple[float, float]] = [(float(pose[0]), float(pose[1]))]
        for g in goals:
            seg = grid.astar(path[-1], g)
            if seg is None:
                if grid.segment_free(path[-1], g):
                    seg = [path[-1], g]
                else:
                    break
            path += seg[1:]
            if math.hypot(path[-1][0] - g[0], path[-1][1] - g[1]) > 0.3:
                break  # partial path: g not reachable now, later goals are beyond it
        if len(path) < 2:
            return None, proj
        return fp.resample(path, 0.25), proj

    # ------------------------------------------------------------ motion

    def _motion(self, pose, scan, grid, now) -> Motion:
        if self.arrived:
            return Motion("arrived", signal="arrived", why="model says arrived")
        if self.turn_target is not None:
            err = wrap(self.turn_target - pose[2])
            if abs(err) < math.radians(8):
                self.turn_target = None
                return Motion("hold", why="turn done")
            side = 1.0 if err > 0 else -1.0
            return Motion("turn", chunk=[[-0.05, 0.3 * side]], why="turn")
        if now < self.wait_until:
            return Motion("hold", why="model wait")
        if self.plan is None:
            return Motion("hold", why="no plan" + (" (waiting for model)" if self.future is not None else ""))
        rel = map_to_robot(pose, self.plan)
        acc, out, prev = 0.0, [], np.zeros(2)
        for q in rel[1:]:
            acc += float(np.hypot(*(q - prev)))
            prev = q
            out.append([float(q[0]), float(q[1])])
            if acc > 2.0:
                break
        if not out:
            return Motion("hold", why="plan end")
        if self._safety_stop(scan, out):
            return Motion("hold", why="safety_stop")
        return Motion("chunk", chunk=out, why="follow plan")

    def _safety_stop(self, scan, chunk) -> bool:
        pts = fp.scan_points_robot(scan, 2.0)
        if len(pts) == 0:
            return False
        hx, hy = chunk[0] if math.hypot(*chunk[0]) > 0.3 or len(chunk) == 1 else chunk[1]
        h = math.atan2(hy, hx)
        c, s = math.cos(h), math.sin(h)
        fx = c * pts[:, 0] + s * pts[:, 1]
        fy = -s * pts[:, 0] + c * pts[:, 1]
        stop = float(self.cfg.get("safety_stop_m", 0.45))
        return bool(np.any((fx > 0) & (fx < self.radius + stop) & (np.abs(fy) < self.radius)))
