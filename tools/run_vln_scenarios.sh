#!/usr/bin/env bash
# One Gazebo episode per (planner x scenario) for the VLN comparison, with the decision trace,
# the robot camera, the third-person chase camera and the people recorded. CONTAINER only, inside
# an arena-gpu claim (the 7-8B VLNs need 16-18 GB GPU memory each, Gazebo needs its share too).
#
#   tools/run_vln_scenarios.sh <tag> <planner> [scenario ...]
#   tools/run_vln_scenarios.sh vln1 navila                        # every bench3_* scenario
#   tools/run_vln_scenarios.sh vln1 cma bench3_cross_repeat bench3_feint_cross
#   PROVIDER=openai MODEL=gpt-5.5 tools/run_vln_scenarios.sh vln1 frontier_vlm3   # frontier, profile vln
#
# planner: cma | navid | uninavid | navila | streamvln | frontier_vlm3. Each gets its own
# training camera plus the recording cameras (camera components *_rec, arena_robots branch
# bench3/vln-rgbd-rec): cma + frontier_vlm3 vln_rgbd_rec, navid vln_navid_rec, uninavid
# vln_uninavid_rec, navila vln_navila_rec, streamvln vln_streamvln_rec.
# frontier_vlm3 runs with profile:=vln (camera + instruction only, no people ground truth, no floor
# plan), RULES=on|off picks the rules switch, PROVIDER/MODEL/EFFORT the frontier model (key proxy),
# EXECUTOR=dwb sends its paths to Nav2 DWB, LATENCY_MODE=zero makes the simulator wait for the model (lockstep).
# Env: EP_SECONDS (default 300), SIM (gazebo), WORLD (hospital_1), HUMAN (arena: needed for the
#      scenario stimuli and for BLOCK), STAGE (probe), RESULTS (/opt/arena_ws/data/bench3),
#      RENDER (1 = render right away; default 0: render later on the CPU), EXTRA (more launch args),
#      SEED (run_seed, default 1; part of the run folder name), EPISODES (episodes per launch, default 1).
# One folder per run: $RESULTS/<stage>/bench3-<date>-<tag>-<planner>-<scenario short>/
#   run.json, trace/<planner>/<episode>/trace.jsonl + frames/ (ARENA_PLANNER_TRACE_DIR),
#   rec/ (robot view 30 Hz), chase/ (third-person 30 Hz), peds.csv, launch.log
# Render: python3 tools/render_vln.py --all <run_dir>     -> trace/<planner>/<episode>/episode.mp4
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tag="${1:?tag}"; planner="${2:?planner: cma | navid | uninavid | navila | streamvln | frontier_vlm3}"; shift 2
scen=("$@")
if [ ${#scen[@]} -eq 0 ]; then
  mapfile -t scen < <(cd /opt/arena_ws/src/Arena/arena_simulation_setup/worlds/hospital_1/scenarios && ls -d bench3_*/ | tr -d /)
fi
WORLD="${WORLD:-hospital_1}"; SIM="${SIM:-gazebo}"; EP_SECONDS="${EP_SECONDS:-300}"; HUMAN="${HUMAN:-arena}"
STAGE="${STAGE:-probe}"; RESULTS="${RESULTS:-/opt/arena_ws/data/bench3}"; RENDER="${RENDER:-0}"
SEED="${SEED:-1}"; EPISODES="${EPISODES:-1}"
KEYPROXY="${KEYPROXY:-http://127.0.0.1:18765}"
args=(); arena_planner="$planner"
case "$planner" in
  cma)        camera=vln_rgbd_rec ;;
  navid)      camera=vln_navid_rec ;;
  uninavid)   camera=vln_uninavid_rec; arena_planner=navid; args+=(robot.mobile.config.variant:=uninavid) ;;
  navila)     camera=vln_navila_rec ;;
  streamvln)  camera=vln_streamvln_rec ;;
  frontier_vlm3)
    camera=vln_rgbd_rec; PROVIDER="${PROVIDER:-mock}"
    args+=(robot.mobile.config.profile:=vln robot.mobile.config.rules:="${RULES:-on}" robot.mobile.config.world:="$WORLD"
           robot.mobile.config.provider:="$PROVIDER" robot.mobile.config.effort:="${EFFORT:-low}" ${MODEL:+robot.mobile.config.model:="$MODEL"})
    if [ "${LATENCY_MODE:-real}" = zero ]; then  # the simulator waits for the model: no delay in sim time
      args+=(robot.mobile.config.latency_mode:=zero lockstep:=true lockstep.paused:=false lockstep.rtf:=1)
    fi
    [ -n "${EXECUTOR:-}" ] && args+=(robot.mobile.executor:="$EXECUTOR")
    case "$PROVIDER" in
      claude) export ANTHROPIC_BASE_URL="$KEYPROXY" ANTHROPIC_API_KEY=via-keyproxy ;;
      openai) export OPENAI_BASE_URL="$KEYPROXY/openai/v1" OPENAI_API_KEY=via-keyproxy ;;
      gemini) export GEMINI_BASE_URL="$KEYPROXY/gemini" GEMINI_API_KEY=via-keyproxy ;;
    esac
    if [ "$PROVIDER" != mock ] && ! curl -sf "$KEYPROXY/_status" > /dev/null; then
      echo "key proxy not reachable at $KEYPROXY: start tools/keyproxy.py on the HOST (README 5.1)"; exit 2
    fi ;;
  *) echo "unknown planner $planner"; exit 2 ;;
esac
python3 "$here/check_scenarios.py" "${scen[@]/#//opt/arena_ws/src/Arena/arena_simulation_setup/worlds/hospital_1/scenarios/}" || { echo "fix the scenarios first"; exit 2; }
arena_sha="$(git -C /opt/arena_ws/src/Arena rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
planners_sha="$(git -C /opt/arena_ws/src/Arena/arena_planners rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
date_tag="$(date +%Y%m%d)"; t0=$(date +%s)
for s in "${scen[@]}"; do
  label="bench3-${date_tag}-${tag}-${planner}-${s#bench3_}-s${SEED}"
  run="$RESULTS/$STAGE/$label"
  if [ -e "$run/launch.log" ]; then echo "    $run exists: use a new tag"; continue; fi
  mkdir -p "$run/trace"
  cat > "$run/run.json" <<JSON
{"run_id": "$label", "stage": "$STAGE", "sim": "$SIM", "world": "$WORLD", "scenario": "$s", "planner": "$planner",
 "model": "${MODEL:-$planner}", "provider": "${PROVIDER:-local}", "rules": "${RULES:-}", "camera": "$camera", "human": "$HUMAN", "executor": "${EXECUTOR:-bridge}", "latency_mode": "${LATENCY_MODE:-real}",
 "ep_seconds": $EP_SECONDS, "seed": "$SEED", "episodes": $EPISODES, "ros_domain_id": "${ROS_DOMAIN_ID:-}", "extra": "${EXTRA:-}", "arena_sha": "$arena_sha", "arena_planners_sha": "$planners_sha", "started": "$(date +%Y-%m-%dT%H:%M:%S)"}
JSON
  echo "################ $label  camera=$camera  ($(( ($(date +%s) - t0) / 60 )) min so far)"
  python3 "$here/cam_recorder.py" --out "$run/rec" --suffix /rec_image > "$run/rec.log" 2>&1 & p1=$!
  python3 "$here/cam_recorder.py" --out "$run/chase" --suffix /chase_image > "$run/chase.log" 2>&1 & p2=$!
  python3 "$here/peds_recorder.py" --out "$run" > "$run/peds.log" 2>&1 & p3=$!
  # `arena` is a shell function (cd $ARENA_WS_DIR + python3 -m arena_cli); timeout needs a program, so call the CLI directly
  ( cd "$ARENA_WS_DIR" && ARENA_PLANNER_TRACE_DIR="$run/trace" ARENA_PLANNER_TRACE_NAME="$planner" \
    PYTHONPATH="$ARENA_DIR/_meta${PYTHONPATH:+:$PYTHONPATH}" \
    timeout --signal=INT --kill-after=30 "$EP_SECONDS" python3 -m arena_cli launch sim:="$SIM" world:="$WORLD" human:="$HUMAN" headless:=true \
    robot:="jackal[camera=$camera]" robot.planner:="$arena_planner" robot.mobile.goal:=instruction \
    task.robots:=scenario task.obstacles:=scenario task.scenario.file:="$s" task.episode.count:="$EPISODES" run_seed:="bench3-$SEED" \
    "${args[@]}" ${EXTRA:-} ) 2>&1 | tee "$run/launch.log"
  kill -INT $p1 $p2 $p3 2>/dev/null; sleep 3; kill -TERM $p1 $p2 $p3 2>/dev/null; sleep 2; kill -KILL $p1 $p2 $p3 2>/dev/null; wait $p1 $p2 $p3 2>/dev/null
  n_tr=$(find "$run/trace" -name trace.jsonl | wc -l)
  echo "    trace episodes: $n_tr, rec frames: $(( $(wc -l < "$run/rec/index.csv" 2>/dev/null || echo 1) - 1 )), chase frames: $(( $(wc -l < "$run/chase/index.csv" 2>/dev/null || echo 1) - 1 ))"
  [ "$n_tr" = 0 ] && echo "    WARNING: no trace: did ARENA_PLANNER_TRACE_DIR reach the planner process? (see launch.log)"
  [ "$RENDER" = 1 ] && python3 "$here/render_vln.py" --all "$run" --world "$WORLD"
done
echo "done: ${#scen[@]} scenarios in $(( ($(date +%s) - t0) / 60 )) min, results in $RESULTS/$STAGE/"
