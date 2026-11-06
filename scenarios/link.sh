#!/usr/bin/env bash
# Link these scenarios into the hospital_1 world (run once, on the host or in the container).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
worlds="$(cd "$here/../../../../arena_simulation_setup/worlds" && pwd)"
for d in "$here"/fvlm3_*/; do
  n="$(basename "$d")"
  ln -sfn "../../../../arena_planners/planners/frontier_vlm3/scenarios/$n" "$worlds/hospital_1/scenarios/$n"
  echo "linked hospital_1/scenarios/$n"
done
