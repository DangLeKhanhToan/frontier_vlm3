"""frontier_vlm3 entry point: Arena bridge <-> harness.

Runs as the planner subprocess (`ros2 run arena_planner_frontier_vlm3 python planner.py`).
step() never blocks on the model: API calls run in a background thread inside the harness.
"""

from __future__ import annotations

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from arena_planners.sdk import Step, load_manifest, main_loop  # noqa: E402

from fvlm3.harness import Harness  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[fvlm3] %(levelname)s %(message)s")
_log = logging.getLogger("fvlm3")

_MANIFEST = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
_harness: Harness | None = None


def on_init(planner_config: dict) -> None:
    global _harness
    cfg = dict(_MANIFEST.get("config") or {})
    cfg.update(planner_config or {})
    _harness = Harness(cfg)
    _log.info("conditions %s, provider %s, model %s, prefetch %s, log %s", _harness.cond.tag, cfg.get("provider"), cfg.get("model"), cfg.get("prefetch"), _harness.log_root)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _harness is not None
    _harness.reset(episode_id, initial_state)
    _log.info("episode %s instruction=%r", episode_id, _harness.instruction)


def on_cancel() -> None:
    if _harness is not None:
        s = _harness.finish()
        _log.info("episode summary %s", {k: s[k] for k in ("calls", "ok", "stale", "stall_s", "cost_usd") if k in s})


def step(features: dict) -> Step:
    assert _harness is not None
    m = _harness.step(features)
    if m.kind == "arrived":
        return Step(signal="arrived")
    if m.kind in ("chunk", "turn"):
        return Step(chunk=m.chunk)
    return Step()


if __name__ == "__main__":
    main_loop(step, manifest=_MANIFEST, on_reset=on_reset, on_cancel=on_cancel, on_init=on_init)
