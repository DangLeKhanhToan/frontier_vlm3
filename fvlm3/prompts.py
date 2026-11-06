"""System prompt, per-call user content and the reply schema, per condition.

The three switches change only what the model is told and shown:
  people    : people labels + people list + `people` field in the reply
  floorplan : second image (floor plan) + zone names + `heading_to` field
  rules     : social rules in the system prompt + `rules_applied` field
Everything else (task text, image format, reply format) is identical across arms.
"""

from __future__ import annotations

import dataclasses

SOCIAL_RULES = [
    "R1 Keep at least 1.2 m from every person; 2 m in front of a person who is walking.",
    "R2 In corridors keep to the right side of the free space, and pass oncoming people on your right (they are on your left).",
    "R3 Never drive between people who are standing or walking together.",
    "R4 Do not cross directly in front of a walking person; pass behind them, or wait.",
    "R5 Yield at doors, corners and narrow passages: let people come through first.",
    "R6 Slow down or wait rather than squeeze through a gap that is only just wide enough.",
]


PEOPLE_MODES = ("on", "off", "raw")


@dataclasses.dataclass(frozen=True)
class Conditions:
    """What the model is given. ``people`` is True only for people_mode "on" (labels + list).

    people_mode "off" inpaints people out (uses ground truth to find them), "raw" passes the camera
    image untouched and uses no ground truth at all: the input the VLN planners get.
    """

    people: bool = True
    floorplan: bool = True
    rules: bool = True
    people_mode: str = ""

    def __post_init__(self) -> None:
        if not self.people_mode:
            object.__setattr__(self, "people_mode", "on" if self.people else "off")
        if self.people_mode not in PEOPLE_MODES:
            raise ValueError(f"people must be one of {PEOPLE_MODES}, got {self.people_mode!r}")
        if self.people != (self.people_mode == "on"):
            raise ValueError(f"people={self.people} contradicts people_mode={self.people_mode!r}")

    @property
    def tag(self) -> str:
        p = {"on": "1", "off": "0", "raw": "r"}[self.people_mode]
        return f"P{p}F{int(self.floorplan)}R{int(self.rules)}"

    @staticmethod
    def from_config(cfg: dict) -> "Conditions":
        def on(v) -> bool:
            return str(v).strip().lower() in ("1", "on", "true", "yes")

        def mode(v) -> str:
            v = str(v).strip().lower()
            return "raw" if v == "raw" else ("on" if on(v) else "off")

        profile = str(cfg.get("profile", "") or "").strip().lower()
        if profile not in ("", "frontier", "vln"):
            raise ValueError(f"profile must be '', 'frontier' or 'vln', got {profile!r}")
        if profile == "vln":  # the VLN planners' contract: egocentric camera + instruction only
            return Conditions(False, False, on(cfg.get("rules", "on")), "raw")
        pm = mode(cfg.get("people", "on"))
        return Conditions(pm == "on", on(cfg.get("floorplan", "on")), on(cfg.get("rules", "on")), pm)


def system_prompt(c: Conditions, zone_names: list[str]) -> str:
    parts = [
        "You are the navigation brain of a wheeled robot (about 0.5 m wide, camera at 1.2 m height). "
        "You receive a task in plain language. You are NOT told where the goal is: find it by looking.",
        "",
        "How you drive: pick 1 to 3 sub-goal points in the camera image, ordered near to far, that the robot should drive through. "
        "Give each point as [u, v] on a 0..1000 scale (u from the left edge, v from the top edge; the image border shows ticks every 100). "
        "Put points ON THE FLOOR where the robot should be, not on walls, furniture or people. "
        "The robot measures the depth of each point, turns it into a position on its map, plans around obstacles and drives there. "
        "A point on the floor 1 to 5 m away works best; the farthest point should be 2 to 6 m away when the floor is open.",
        "",
        "Latency: the robot keeps driving while you think. The blue line in the image is the part of the previous plan "
        "it is still driving; a blue dot marks where that plan ends. When a blue line is shown, start your points near or beyond its end so the robot can continue without stopping, "
        "unless the plan is no longer safe or no longer leads toward the goal.",
        "",
        "Actions: 'goto' (with sub-goals), 'turn_left' / 'turn_right' (rotate about 45 degrees to look around, when the goal or a way forward is not visible), "
        "'wait' (stay still for a moment), 'arrived' (the task is completed: the robot is at or right in front of the goal).",
        "",
        "Before choosing, describe briefly what you see, then explain why the chosen path is a good one: what it leads toward, what it avoids, and what you expect to see next.",
    ]
    if c.floorplan:
        parts += [
            "",
            "You also get a floor plan (north up): walls dark, the robot as a red arrow with its camera view as a red wedge, "
            "the path driven so far in grey, the current plan in blue, zone names in green (grey once visited). "
            "Use it to decide which way to go; still choose your points in the camera image. "
            "Name the zone you are heading to in 'heading_to'.",
            "Zones: " + ", ".join(zone_names) if zone_names else "Zones: (no names in this map)",
        ]
    if c.people:
        parts += [
            "",
            "People in view are boxed and labelled p1, p2, ... in the camera image, and listed with distance, side and motion. "
            "List every labelled person in 'people' and say whether they matter for your path.",
        ]
    if c.rules:
        parts += ["", "Follow these rules when you move:"] + SOCIAL_RULES + [
            "In 'rules_applied' name the rules that shaped this decision (e.g. 'R2'), or leave it empty."
        ]
    return "\n".join(parts)


def reply_schema(c: Conditions) -> dict:
    props: dict = {
        "scene": {"type": "string", "description": "What the camera shows, one or two sentences."},
        "progress": {"type": "string", "enum": ["searching", "goal_visible", "approaching", "arrived"]},
        "action": {"type": "string", "enum": ["goto", "turn_left", "turn_right", "wait", "arrived"]},
        "subgoals": {
            "type": "array",
            "description": "1-3 floor points, near to far, [u, v] on 0..1000. Empty unless action is goto.",
            "items": {
                "type": "object",
                "properties": {
                    "u": {"type": "integer"},
                    "v": {"type": "integer"},
                    "why": {"type": "string"},
                },
                "required": ["u", "v", "why"],
                "additionalProperties": False,
            },
        },
        "path_reasoning": {"type": "string", "description": "Why this path: where it leads, what it avoids, what you expect next."},
    }
    req = ["scene", "progress", "action", "subgoals", "path_reasoning"]
    if c.people:
        props["people"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "relevance": {"type": "string", "enum": ["blocks_path", "may_cross_path", "nearby", "irrelevant"]},
                    "note": {"type": "string"},
                },
                "required": ["label", "relevance", "note"],
                "additionalProperties": False,
            },
        }
        req.append("people")
    if c.floorplan:
        props["heading_to"] = {"type": "string", "description": "Zone name you are heading to, or 'unknown'."}
        req.append("heading_to")
    if c.rules:
        props["rules_applied"] = {"type": "array", "items": {"type": "string"}}
        req.append("rules_applied")
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


def user_text(
    c: Conditions,
    instruction: str,
    call_no: int,
    elapsed_s: float,
    history: list[str],
    people_lines: list[str],
    plan_left_m: float,
    zone_here: str | None,
    note: str = "",
) -> str:
    lines = [f"Task: {instruction or '(no instruction given: explore and report what you find)'}", f"Decision {call_no}, {elapsed_s:.0f} s into the task."]
    if plan_left_m > 0.05:
        lines.append(f"The robot is still driving the previous plan: {plan_left_m:.1f} m left (blue line).")
    else:
        lines.append("The robot is standing still and waits for your decision.")
    if c.floorplan and zone_here:
        lines.append(f"The robot is in: {zone_here}.")
    if c.people:
        lines.append("People in view: " + ("; ".join(people_lines) if people_lines else "none."))
    if history:
        lines.append("Your recent decisions (oldest first):")
        lines += [f"  - {h}" for h in history]
    if note:
        lines.append(note)
    lines.append("Image 1: robot camera." + (" Image 2: floor plan." if c.floorplan else ""))
    return "\n".join(lines)
