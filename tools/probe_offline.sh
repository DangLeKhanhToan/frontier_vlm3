#!/usr/bin/env bash
# One offline episode with a real model: the hospital_1 ray-cast sim, no Gazebo, no GPU. HOST only.
#
#   tools/probe_offline.sh <provider> <model> <tag> [seconds=45] [max_calls=6] [conditions=P1F1R1]
#   tools/probe_offline.sh gemini gemini-3.5-flash gemini35flash
#   tools/probe_offline.sh openai gpt-5.5 gpt55
#   tools/probe_offline.sh claude claude-opus-5-5 opus55
#   tools/probe_offline.sh mock mock mock 20          # free
#
# Same start, instruction ("Go to the laboratory.") and walker routes for every model.
# SCENARIO=<name> replays a scenario instead (start, instruction, people, timeline triggers;
# people walk blind straight lines, tools/scenario_peds.py):
#   SCENARIO=fvlm3_follow_slow tools/probe_offline.sh openai gpt-5-mini follow-mini 90 20
#   tools/probe_offline.sh --all-scenarios mock mock allmock 60   # every scenario, free
# Needs the key proxy on the host (README 5.1) for claude / openai / gemini.
# Writes ~/work/bench3/results/offline/bench3-<date>-offline-<tag>/ (= ~/arena_ws/data/bench3/offline/...):
#   run.json, calls/ep001_offline/ (planner logs), frames/ + index.csv (30 Hz camera).
# Render the video afterwards (CPU):  tools/probe_offline.sh --render <run_dir>
set -euo pipefail
IMAGE="${IMAGE:-localhost/arena:arena-arena_ws}"
KEYPROXY="${KEYPROXY:-http://127.0.0.1:18765}"
P=/opt/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3
PYLIB=/opt/arena_ws/.fvlm3-pylib          # HOST ~/arena_ws/.fvlm3-pylib: anthropic SDK for the probe
run() { podman run --rm --network host --user 0:0 --entrypoint bash -v "$HOME/arena_ws:/opt/arena_ws" -w "$P" \
          -e PYTHONPATH="$PYLIB" \
          -e ANTHROPIC_BASE_URL="$KEYPROXY" -e ANTHROPIC_API_KEY=via-keyproxy \
          -e OPENAI_BASE_URL="$KEYPROXY/openai/v1" -e OPENAI_API_KEY=via-keyproxy \
          -e GEMINI_BASE_URL="$KEYPROXY/gemini" -e GEMINI_API_KEY=via-keyproxy \
          "$IMAGE" -c "$1"; }

if [ "${1:-}" = "--render" ]; then
  d="$(cd "${2:?run dir}" && pwd -P)"; c="/opt/arena_ws/${d#"$HOME"/arena_ws/}"
  run "python3 tools/render_episode.py $c/calls --all"
  exit 0
fi

if [ "${1:-}" = "--all-scenarios" ]; then
  shift; base="${3:?tag}"
  for d in "$(dirname "$0")"/../scenarios/fvlm3_*/; do
    n="$(basename "$d")"
    SCENARIO="$n" "$0" "$1" "$2" "$base-${n#fvlm3_}" "${4:-45}" "${5:-6}" "${6:-P1F1R1}" || true
  done
  exit 0
fi

provider="${1:?provider: mock | claude | openai | gemini}"; model="${2:?model}"; tag="${3:?tag}"
seconds="${4:-45}"; max_calls="${5:-6}"; conds="${6:-P1F1R1}"
run_id="bench3-$(date +%Y%m%d)-offline-$tag"
host_dir="$HOME/arena_ws/data/bench3/offline/$run_id"
if [ -e "$host_dir" ]; then echo "$host_dir exists: use a new tag"; exit 2; fi
if [ "$provider" != mock ] && ! curl -sf "$KEYPROXY/_status" > /dev/null; then
  echo "key proxy not reachable at $KEYPROXY: start tools/keyproxy.py (README 5.1)"; exit 2
fi
if [ "$provider" = claude ] && [ ! -d "$HOME/arena_ws/.fvlm3-pylib/anthropic" ]; then
  echo "installing the anthropic SDK into ~/arena_ws/.fvlm3-pylib (once)"
  run "uv pip install --quiet --python /usr/bin/python3 --target $PYLIB anthropic"
fi
mm=$([ "$provider" = mock ] && echo "" || echo "--model $model")
run "python3 tools/hospital_sim.py --run-dir /opt/arena_ws/data/bench3/offline/$run_id --provider $provider $mm \
     --seconds $seconds --max-calls $max_calls --conds $conds ${SCENARIO:+--scenario $SCENARIO} 2>&1 | grep -v -e RuntimeWarning -e 'tile ='"
[ "$provider" != mock ] && { curl -s "$KEYPROXY/_status"; echo; }
echo "results: $host_dir"
