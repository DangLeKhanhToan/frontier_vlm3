#!/usr/bin/env bash
# One episode of every frontier_vlm3 scenario with one model, one condition. CONTAINER only,
# inside an arena-gpu claim for Gazebo (SIM=flatsim runs on the CPU).
#
#   tools/run_scenarios.sh <tag> <provider> [scenario ...]
#   MODEL=gpt-5.5 EP_SECONDS=300 tools/run_scenarios.sh gpt55 openai            # all 8
#   MODEL=gpt-5.5 tools/run_scenarios.sh gpt55 openai fvlm3_follow_slow fvlm3_sudden_turn
#
# Each scenario goes through tools/run_conditions.sh (same env: MODEL, EFFORT, SIM, REC, ...)
# with STAGE=probe and condition COND (default P1F1R1), so the runs land in
#   $RESULTS/probe/bench3-<date>-<tag>-<scenario short name>-<COND>/
# Checks every scenario first (tools/check_scenarios.py) and stops on a broken one.
# Afterwards, on the CPU: RENDER the videos (README 5.7) and build the scenario report.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tag="${1:?tag}"; provider="${2:?provider: mock | claude | openai | gemini}"; shift 2
scen=("$@")
if [ ${#scen[@]} -eq 0 ]; then
  mapfile -t scen < <(cd "$here/../scenarios" && ls -d fvlm3_*/ | tr -d /)
fi
COND="${COND:-P1F1R1}"
export STAGE="${STAGE:-probe}" EP_SECONDS="${EP_SECONDS:-300}"
python3 "$here/check_scenarios.py" "${scen[@]}" || { echo "fix the scenarios above first"; exit 2; }
"$here/../scenarios/link.sh"
t0=$(date +%s)
for s in "${scen[@]}"; do
  echo "################ $s  ($(( ($(date +%s) - t0) / 60 )) min so far)"
  "$here/run_conditions.sh" "$s" "$tag-${s#fvlm3_}" "$provider" "$COND"
done
echo "done: ${#scen[@]} scenarios in $(( ($(date +%s) - t0) / 60 )) min, results in ${RESULTS:-/opt/arena_ws/data/bench3}/$STAGE/"
