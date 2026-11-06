#!/usr/bin/env bash
# Run frontier_vlm3 over study conditions, one `arena launch` per condition. CONTAINER only.
#
#   tools/run_conditions.sh <scenario> <tag> [provider] [conditions...]
#   tools/run_conditions.sh fvlm3_lab_crowd smoke mock P1F1R1
#   tools/run_conditions.sh fvlm3_lab_crowd pilot claude              # all 8 conditions
#   MODEL=gemini-3.5-flash tools/run_conditions.sh fvlm3_lab_crowd probe gemini P1F1R1
#
# provider: mock | claude | openai | gemini. The API calls go to the HOST key proxy
# (tools/keyproxy.py, README 5.1); this script sets the proxy env for the chosen provider.
# Env: EP_SECONDS (wall-clock cap per launch, default 240), WORLD (default hospital_1),
#      SIM (default gazebo), EXTRA (more launch args, e.g. "robot.mobile.config.prefetch:=false"),
#      REC (1 = add the 640x480 30 Hz recording camera and record it, default 1; 0 = vln_rgbd only),
#      RENDER (1 = render the episode videos right after each condition, default 0: render
#              later without the GPU, see README 5.7),
#      MODEL (default: provider default, see planner.yaml), EFFORT (default low),
#      EPISODES (episodes per launch, default 1), KEYPROXY (default http://127.0.0.1:18765),
#      STAGE (results folder: smoke | probe | pilot | study; default smoke for mock, else pilot),
#      RESULTS (default /opt/arena_ws/data/bench3 = HOST ~/work/bench3/results),
#      HUMAN (pedestrian backend, default arena: the only one that acts on scenario stimuli).
# One folder per run: $RESULTS/<stage>/bench3-<date>-<tag>-<cond>/
#   run.json (what ran), calls/epNNN_*/ (planner logs, episode.mp4 once rendered),
#   frames/ + index.csv (30 Hz recording, REC=1), launch.log, recorder.log
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
scenario="${1:?scenario}"; tag="${2:?tag}"; provider="${3:-mock}"; shift $(( $# >= 3 ? 3 : $# ))
conds=("$@")
if [ ${#conds[@]} -eq 0 ]; then
  conds=(P1F1R1 P1F1R0 P1F0R1 P1F0R0 P0F1R1 P0F1R0 P0F0R1 P0F0R0)
fi
WORLD="${WORLD:-hospital_1}"; SIM="${SIM:-gazebo}"; EP_SECONDS="${EP_SECONDS:-240}"
REC="${REC:-1}"; RENDER="${RENDER:-0}"; EPISODES="${EPISODES:-1}"; EFFORT="${EFFORT:-low}"
KEYPROXY="${KEYPROXY:-http://127.0.0.1:18765}"; HUMAN="${HUMAN:-arena}"
case "$provider" in
  claude) export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-$KEYPROXY}" ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-via-keyproxy}" ;;
  openai) export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$KEYPROXY/openai/v1}" OPENAI_API_KEY="${OPENAI_API_KEY:-via-keyproxy}" ;;
  gemini) export GEMINI_BASE_URL="${GEMINI_BASE_URL:-$KEYPROXY/gemini}" GEMINI_API_KEY="${GEMINI_API_KEY:-via-keyproxy}" ;;
  mock) ;;
  *) echo "unknown provider $provider (mock | claude | openai | gemini)"; exit 2 ;;
esac
if [ "$provider" != mock ] && ! curl -sf "$KEYPROXY/_status" > /dev/null; then
  echo "key proxy not reachable at $KEYPROXY: start tools/keyproxy.py on the HOST (README 5.1)"; exit 2
fi
camera=vln_rgbd; [ "$REC" = 1 ] && camera=vln_rgbd_rec
RESULTS="${RESULTS:-/opt/arena_ws/data/bench3}"
STAGE="${STAGE:-$([ "$provider" = mock ] && echo smoke || echo pilot)}"
stage_dir="$RESULTS/$STAGE"
mkdir -p "$stage_dir"
arena_sha="$(git -C /opt/arena_ws/src/Arena rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
date_tag="$(date +%Y%m%d)"
onoff() { [ "$1" = 1 ] && echo on || echo off; }
for c in "${conds[@]}"; do
  p=$(onoff "${c:1:1}"); f=$(onoff "${c:3:1}"); r=$(onoff "${c:5:1}")
  label="bench3-${date_tag}-${tag}-${c}"
  run="$stage_dir/$label"
  if [ -e "$run/calls" ]; then echo "    $run exists: use a new tag"; continue; fi
  mkdir -p "$run"
  cat > "$run/run.json" <<JSON
{"run_id": "$label", "stage": "$STAGE", "sim": "$SIM", "world": "$WORLD", "scenario": "$scenario", "conditions": "$c",
 "provider": "$provider", "model": "${MODEL:-default}", "effort": "$EFFORT", "episodes": $EPISODES, "ep_seconds": $EP_SECONDS,
 "camera": "$camera", "human": "$HUMAN", "extra": "${EXTRA:-}", "arena_sha": "$arena_sha", "started": "$(date +%Y-%m-%dT%H:%M:%S)"}
JSON
  echo "=== $label  people=$p floorplan=$f rules=$r provider=$provider model=${MODEL:-default} scenario=$scenario camera=$camera episodes=$EPISODES"
  rec_pid=""
  if [ "$REC" = 1 ]; then
    python3 "$here/cam_recorder.py" --out "$run" > "$run/recorder.log" 2>&1 &
    rec_pid=$!
  fi
  timeout --signal=INT "$EP_SECONDS" arena launch sim:="$SIM" world:="$WORLD" human:="$HUMAN" headless:=true \
    robot:="jackal[camera=$camera]" robot.planner:=frontier_vlm3 robot.mobile.goal:=instruction \
    task.robots:=scenario task.obstacles:=scenario task.scenario.file:="$scenario" task.episode.count:="$EPISODES" \
    robot.mobile.config.world:="$WORLD" robot.mobile.config.provider:="$provider" \
    ${MODEL:+robot.mobile.config.model:="$MODEL"} robot.mobile.config.effort:="$EFFORT" \
    robot.mobile.config.people:="$p" robot.mobile.config.floorplan:="$f" robot.mobile.config.rules:="$r" \
    robot.mobile.config.log_dir:="$run" robot.mobile.config.run_label:=calls ${EXTRA:-} 2>&1 | tee "$run/launch.log"
  if [ -n "$rec_pid" ]; then
    kill -INT "$rec_pid" 2>/dev/null; wait "$rec_pid" 2>/dev/null
    n=$(( $(wc -l < "$run/index.csv" 2>/dev/null || echo 1) - 1 ))
    echo "    recorded $n frames -> $run/frames"
  fi
  if [ "$RENDER" = 1 ] && [ -d "$run/calls" ]; then
    python3 "$here/render_episode.py" "$run/calls" --all
  fi
  sleep 5
done
python3 "$here/summarize.py" "$stage_dir"
[ "$provider" != mock ] && curl -s "$KEYPROXY/_status" && echo
