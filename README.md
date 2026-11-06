# frontier_vlm3: a frontier VLM as an instruction-only VLN planner

A frontier VLM (Claude, GPT or Gemini; same prompt, images and JSON schema for all three) drives
an Arena robot from a sentence such as "Go to the laboratory".
**The model never receives the goal pose or any coordinates.** At every decision it looks at
the RGB frame, picks 1–3 sub-goal points *in the image*, and explains why that path. The
harness turns each pixel into a map position with the depth image, plans around obstacles
and drives there, while the next decision is already being computed.

Code: `arena_planners/planners/frontier_vlm3` on branch `bench3/frontier-vlm3` of the
`arena_planners` submodule. Bench3 project: `~/work/bench3` (session notes in `docs/log/`).

## 1. One decision, step by step

```
 RGB 224x224 + depth 256x256 + pose (TF) + laser + arena_peds
        │
        ▼ [harness, 5 Hz]
 people:=on   → people boxed and labelled p1, p2 … + listed (distance, side, motion)
 people:=off  → people inpainted out of RGB and depth, never mentioned
 floorplan:=on→ 2nd image: floor plan with zone names, robot, view wedge, trail, current plan
 rules:=on    → 6 social rules (R1–R6) in the system prompt
        │
        ▼ [Claude / GPT / Gemini, background thread, strict JSON schema]
 {scene, progress, action: goto|turn_left|turn_right|wait|arrived,
  subgoals: [{u, v, why}] (0..1000 image coords, near→far), path_reasoning,
  people[] (P1), heading_to (F1), rules_applied[] (R1)}
        │
        ▼ [harness]
 pixel → depth (median 5x5) → 3D point in the camera frame of THAT frame → robot → map
   point on wall/person/furniture (z > 0.25 m) → back off 0.5 m along the ray, drop to floor
   no depth → ray ∩ floor plane;  above horizon → 6 m along the bearing
 → snap to free cell → A* on local grid (laser hits [+ static map if F1], inflated 0.35 m)
   unreachable → drive to the reachable cell closest to it
 → path from the CURRENT pose through the sub-goals → waypoint chunk (next 2 m) → Arena bridge
 → laser safety stop (the model cannot override it)
```

`arrived` sends Arena the `arrived` signal; the scenario's goto phase is judged at that moment
(`signal: arrived`, `tolerance_radius`).

## 2. The three conditions (2 × 2 × 2 = 8 arms)

| switch | on | off |
|---|---|---|
| `people` (P) | people labelled in the image + text list + `people[]` in the reply | people removed from the image (inpainted) and from depth; no mention anywhere |
| `floorplan` (F) | floor plan image + zone names + `heading_to`; harness A* also uses the static map | camera only; harness map = laser only (no prior map at all) |
| `rules` (R) | rules R1–R6 in the system prompt + `rules_applied[]` | no rules |

Condition tag in logs: `P1F1R1` … `P0F0R0`. Everything else (task text, image format,
reply format, harness) is identical across arms. The laser safety stop always sees people;
that is the robot's reflex, not the model's perception.

## 3. Latency: deciding before the robot reaches its sub-goal

An API call takes 3–5 s, so stop-and-ask leaves the robot standing 25–40 % of the time (v3 data).
v3.1 overlaps thinking and driving:

1. **Prefetch.** The next call starts when `remaining plan length / speed < predicted latency + margin`
   (predicted latency = 80th percentile of the last 12 calls, start 4 s, margin 1 s). The answer
   arrives about when the current plan runs out.
2. **Answers are anchored to their frame.** The harness keeps the depth image and pose of the
   frame the model saw. Pixels are projected with those, so a 4 s old answer still points at
   the right spot on the map. The new path starts at where the robot is *now*: it continues
   without stopping.
3. **The model extends the plan.** The remaining plan is drawn in blue on the camera image
   with a dot at its end, and the prompt says how many metres are left; the model is asked to
   put its points near or beyond that end.
4. **Several sub-goals per answer** (up to 3, farthest 2–6 m): one answer covers 4–12 s of driving.
5. **Early calls on events:** plan blocked by the laser, a new person within 3 m (P1), plan older than 10 s.
6. **Stale guard:** an answer whose frame is older than 12 s is dropped and asked again.
7. **Low effort** (`effort: low`) and a short fixed system prompt (cached) keep each call short.

Offline check (no ROS, mock model, straight corridor, `tools/offline_sim.py`):

| mock latency | prefetch | stall (robot waiting) | distance in window |
|---|---|---|---|
| 2.5 s, 30 s | off | 7.7 s (25.5 %) | 10.6 m |
| 2.5 s, 30 s | on | 2.5 s (8.4 %) | 13.1 m |
| 4.0 s, 40 s | off | 12.2 s (30.4 %) | 13.1 m |
| 4.0 s, 40 s | on | 4.1 s (10.3 %) | 17.0 m |

The remaining stall is the first call of the episode (nothing to drive yet). Turn
`prefetch:=false` to get the stop-and-ask baseline for the paper.

## 4. Files

| file | what |
|---|---|
| `planner.yaml` | manifest: `waypoints` action, `goal_inputs: [instruction]`, `signals: [arrived]`, datasources, all config keys |
| `planner.py` | Arena SDK glue: on_init / on_reset / step / on_cancel |
| `fvlm3/harness.py` | the loop: call trigger, async call, projection, A*, chunk, safety stop, logs |
| `fvlm3/geometry.py` | camera model, pixel ↔ robot ↔ map, depth projection |
| `fvlm3/perception.py` | visible people (ground truth + depth occlusion), label / inpaint, image overlay |
| `fvlm3/floorplan.py` | map.yaml + world.yaml zones, local grid, A*, floor plan image |
| `fvlm3/prompts.py` | system prompt, user text, JSON schema per condition |
| `fvlm3/providers.py` | Claude (Anthropic SDK, structured output, refusal fallback), OpenAI and Gemini (plain HTTP, JSON schema), mock |
| `tools/keyproxy.py` | HOST key proxy for all three APIs: real keys stay on the host, spend and request caps, ledger |
| `tools/run_conditions.sh` | one launch per condition, then a summary table |
| `tools/summarize.py` | table of episode summaries |
| `tools/offline_sim.py` | ROS-free corridor sim for the latency check |
| `tools/cam_recorder.py` | CONTAINER: records the 30 Hz recording camera as JPEG frames + time index |
| `tools/render_episode.py` | episode video (MP4, 1920x1080, 30 fps) from the logs + recorded frames; no GPU |
| `tools/probe_offline.sh` | HOST: one offline hospital episode with any model into `results/offline/`; `--render` makes its video |
| `tools/hospital_sim.py` | ROS-free hospital_1 sim (ray-cast view, walking people) with the mock or a real model: try the video and each model without Gazebo |
| `tools/run_scenarios.sh` | CONTAINER: one episode of every scenario (or the ones named) with one model, via `run_conditions.sh`, `STAGE=probe` |
| `tools/check_scenarios.py` | checks scenarios with Arena's parser, loads their agent types, checks free floor under every point; no launch |
| `tools/scenario_peds.py` | replays a scenario's people and timeline triggers for `hospital_sim.py --scenario` (people walk blind straight lines) |
| `scenarios/` | hospital_1: `fvlm3_lab_empty`, `fvlm3_lab_crowd`, `fvlm3_pharmacy_crowd`, and the interaction set `fvlm3_pass_crowd`, `fvlm3_follow_slow`, `fvlm3_sudden_turn`, `fvlm3_cut_and_stop`, `fvlm3_group_oncoming` (behaviour files in `<scenario>/agent_types/`) + `link.sh` |
| `tests/test_fvlm3.py` | 30 offline tests (incl. OpenAI / Gemini wire format and the proxy against fake servers, the scenario replay, `profile: vln`) |
| `tools/run_vln_scenarios.sh` | CONTAINER: one episode per (VLN planner x bench3 scenario) with trace, robot view, chase view and people recorded (section 8) |
| `tools/render_vln.py` | episode video for ANY bridge planner from the SDK decision trace (floor plan, third-person, robot view, model input / output / execution); no GPU |
| `tools/peds_recorder.py` | CONTAINER: records `arena_peds` to `peds.csv` (VLN planners never see people, the video takes them from here) |
| `tools/plot_scenarios.py` | top-down design figure of a scenario (paths, triggers, regions); no ROS |

Outside this folder: the recording camera `camera/vln_rgbd_rec` in `arena_robots`
(branch `bench3/vln-rgbd-rec`): `vln_rgbd` unchanged plus a 640x480 / 30 Hz RGB sensor on the
same frame, topic `<ns>/vln_camera/rec_image`. The model still gets the 224 px frames.

## 4b. Episode video

One MP4 per episode, on the simulation-time axis. The simulator is not paused while the model
thinks, so the video shows what really happened: the robot keeps driving its old path during a call.

```
+-----------------------------------------------------------------------------------------+
| run / episode   P1F1R1 (people on · floor plan on · rules on)  "Go to the laboratory."  |
|                                              t = 59.0 s  calls 16  $0.150   driving     |
| [timeline: blue = call in flight | green = driving | red = waiting for model | orange = safety stop] |
+--------------+-------------------------+------------------------------------------------+
| full floor   | 12 m close-up around    | robot camera, 640x480 at 30 Hz                 |
| plan, goal   | the robot: path driven  | blue = path being followed,                    |
| room framed  | (red), path followed    | numbered dots = sub-goals of the last answer,  |
| in green     | (blue), sub-goals 1-3,  | people labelled as the model saw them          |
|              | dropped points (x),     |                                                |
|              | people, camera wedge    |                                                |
+--------------+--------+----------------+------------------------------------------------+
| camera image | floor  | Call 16  trigger: prefetch  sent at t = 56.9 s with 1.5 m left   |
| sent (+ the  | plan   | answer after 2.9 s / waiting for the answer: 1.2 s               |
| model's red  | sent   | MODEL INPUT: system prompt size, images, the full user text      |
| points)      |        | MODEL OUTPUT: action, progress, scene, each sub-goal (u, v) ->   |
|              |        | metres + depth source + why, path reasoning, people, rules       |
+--------------+--------+------------------------------------------------------------------+
```

The video opens with a 4 s card that shows the full system prompt of the condition. People
the model was told about carry the model's label (`p1`, `p2`, by distance in that call) with a
yellow ring; others show their simulator id (`#3`). Logs without a recording still render: the
camera panel then shows the last image sent to the model and the time axis is host time.

## 5. Run it

### Where results go (every stage)

Everything a run produces goes to the bench3 project, never to `/tmp`:
`~/work/bench3/results/` is a link to `~/arena_ws/data/bench3/`, which the Arena container sees as
`/opt/arena_ws/data/bench3/`. One folder per stage, one folder per run:

```
results/offline/  hospital_sim, no GPU        results/pilot/   Gazebo, all 8 conditions
results/smoke/    Gazebo + mock model         results/study/   production runs
results/probe/    Gazebo, one episode/model   results/ledgers/ key-proxy ledgers
results/reports/  HTML reports (analysis/model_report.sh)

results/<stage>/bench3-<date>-<tag>[-<cond>]/
  run.json             what ran: provider, model, scenario, conditions, Arena SHA, command
  calls/ep001_*/       planner logs, images sent, answers, summary.json, episode.mp4 once rendered
  frames/, index.csv   30 Hz recording camera
  launch.log, recorder.log
```

`results/` is git-ignored. Study runs that go into the paper are also copied to
`~/work/bench3/data/raw/` with a line in `data_manifest.md`.

### 5.0 What needs the GPU, and the 4 h / 6 GB plan

Only the Gazebo simulation needs the GPU, for rendering the cameras (224 px RGB, 256 px depth,
640x480 recording camera). Without a GPU the cameras fall back to CPU rendering and the
simulation runs far below real time. Building, the API calls, the recorder and the video
rendering run on the CPU, so do 5.1–5.2 **before** the claim and 5.7 **after** it.

The Arena container gets the NVIDIA devices automatically (`_meta/docker/lib` enables them when
`nvidia-smi` works on the host). It uses no GPU memory until Gazebo starts. 6 GB should be
plenty: one headless Gazebo with three small cameras is expected to use 1–3 GB. Check the real
number in 5.3 and shrink the claim if it is much lower (`arena-gpu modify <ID> --gb 4`).

| step | claim time |
|---|---|
| 5.3 mock smoke test + checks | ~20 min |
| 5.4 one episode per model (Gemini, GPT, Claude) | ~25 min |
| 5.5 8 conditions × `fvlm3_lab_crowd`, 240 s each | ~40 min |
| 5.5 again for `fvlm3_lab_empty` and `fvlm3_pharmacy_crowd` | ~80 min |
| spare (restarts, a `prefetch:=false` baseline) | ~90 min |

```bash
# HOST
arena-gpu                                   # your allowance and the card's state
arena-gpu claim 6 --for 4h --detach         # prints a claim ID
arena-gpu wait <ID>                         # returns when the 6 GB are granted
arena-gpu bookings                          # check it is running
# when finished early:  arena-gpu cancel <ID>
```

The claim ending (or the maintenance window, `arena-gpu maint`) kills the processes, so run
everything in tmux and keep an eye on the end time.

### 5.1 HOST: API keys and key proxy (no GPU)

The keys stay on the host in `~/work/bench3/.env` (chmod 600, git-ignored):
`CLAUDE_API_KEY=...`, `OPENAI_API_KEY=...`, `GEMINI_API_KEY=...`. The container never sees
them: the planner sends every request to the proxy on 127.0.0.1:18765 with a dummy key, and the
proxy adds the real key. Every request lands in the ledger.

```bash
# HOST, in tmux (the container reaches it on 127.0.0.1: network_mode host)
tmux new -s keyproxy
python3 ~/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3/tools/keyproxy.py \
  --env ~/work/bench3/.env --max-usd 10 --ledger ~/work/bench3/results/ledgers/fvlm3.jsonl
# prints: keyproxy on 127.0.0.1:18765  keys: claude, gemini, openai ...      detach: Ctrl-b d

# HOST, any shell: spend, requests per provider, and which models each key may use (free)
curl -s 127.0.0.1:18765/_status; echo
curl -s '127.0.0.1:18765/gemini/v1beta/models?pageSize=200' | python3 -c 'import json,sys; print([m["name"][7:] for m in json.load(sys.stdin)["models"] if "generateContent" in m.get("supportedGenerationMethods", [])])'
curl -s 127.0.0.1:18765/openai/v1/models | python3 -c 'import json,sys; print(sorted(m["id"] for m in json.load(sys.stdin)["data"] if m["id"].startswith("gpt-5")))'
```

The caps count everything in that ledger file; use a new `--ledger` file per study to reset
them. `--max-requests N` adds a hard cap on the number of calls. Cost in the ledger is an
estimate from a price table (unknown models at a conservative rate); the providers' billing
pages are authoritative.

**Gemini key status (checked 2026-11-05):** the key is on the Gemini **free tier**. The free
tier allows 0 requests for `gemini-3.1-pro-preview` (HTTP 429 `limit: 0`). `gemini-3.5-flash`
works, `gemini-3.8-flash` answered 503 "high demand", and `gemini-2.5-*` is closed to new
users. To use 3.1 Pro, turn on billing for the key's project in Google AI Studio. Free-tier
Flash also has per-minute limits; a 429 shows up as `failed` calls and the robot waits.

### 5.2 CONTAINER: build and link scenarios (no GPU; do it before the claim)

```bash
# HOST
tmux new -s fvlm3
cd ~/arena_ws && source arena          # enters the container
# CONTAINER
git -C /opt/arena_ws/src/Arena/arena_robots branch --show-current     # bench3/vln-rgbd-rec
git -C /opt/arena_ws/src/Arena/arena_planners branch --show-current   # bench3/frontier-vlm3
arena build arena_robots arena_planner_frontier_vlm3
source /opt/arena_ws/install/setup.bash
ls /opt/arena_ws/install/arena_robots/share/arena_robots/components/camera/ | grep vln_rgbd_rec
/opt/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3/scenarios/link.sh   # already done once on 2026-11-05
# the recording camera expands (no simulator needed):
xacro <(printf '%s\n' '<robot name="t" xmlns:xacro="http://wiki.ros.org/xacro"><xacro:arg name="sim_path" default="w"/>' \
  '<xacro:include filename="$(find arena_robots)/components/camera/vln_rgbd_rec/vln_rgbd_rec.urdf.xacro"/><link name="top"/>' \
  '<xacro:vln_rgbd_rec name="top" parent_link="top" sensor="top" topic="vln_camera" height="0.9815"><origin xyz="0 0 0"/></xacro:vln_rgbd_rec></robot>') \
  | grep -E 'sensor name|<width>|update_rate'      # expect top_rgb_camera, top_depth_camera, top_rec_camera 640 30
```

### 5.3 CONTAINER: free smoke test with the mock model (GPU claimed, no API cost)

Window 1 (tmux `fvlm3`, inside the container): the recorder, then the launch.

```bash
cd /opt/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3
L=bench3-$(date +%Y%m%d)-smoke-mock
R=/opt/arena_ws/data/bench3/smoke/$L && mkdir -p $R      # HOST: ~/work/bench3/results/smoke/$L
python3 tools/cam_recorder.py --out $R &                # waits for the camera topic
arena launch sim:=gazebo world:=hospital_1 headless:=true \
  robot:='jackal[camera=vln_rgbd_rec]' robot.planner:=frontier_vlm3 robot.mobile.goal:=instruction \
  task.robots:=scenario task.obstacles:=scenario task.scenario.file:=fvlm3_lab_crowd \
  robot.mobile.config.world:=hospital_1 robot.mobile.config.provider:=mock \
  robot.mobile.config.log_dir:=$R robot.mobile.config.run_label:=calls
# stop with Ctrl-c after ~3 min, then stop the recorder:
kill -INT %1
```

Window 2 (`Ctrl-b c`, then `cd ~/arena_ws && source arena`): checks while it runs.

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # HOST or CONTAINER: gz should be listed; note the MiB
ros2 topic list | grep -E 'vln_camera|arena_peds'          # image, depth_image, rec_image, arena_peds
ros2 topic hz $(ros2 topic list | grep rec_image)          # ~30 Hz; much lower = CPU rendering or overload
ros2 topic hz $(ros2 topic list | grep 'vln_camera/image$')  # ~15 Hz
ros2 run tf2_ros tf2_echo base_link top_rgb_camera_frame   # z = camera height (frame may carry the robot prefix); set cam_z if not 1.19
S=$(ls -d /opt/arena_ws/data/bench3/smoke/bench3-*-smoke-mock | tail -1)
cat $S/calls/*/camera.jsonl        # rgb_shape must be [224, 224, 3]
ls $S/calls/*/                     # calls.jsonl ticks.jsonl images/
tail -2 $S/index.csv               # frames arrive
```

If `rgb_shape` is `[480, 640, 3]`, the adapter bound the recording camera to the planner (it
takes the first `image` sensor); the planner also logs a warning. Then run with `REC=0` /
`camera=vln_rgbd` and report it. If gz is missing from `nvidia-smi` and `rec_image` is far below
30 Hz, the cameras render on the CPU: check `nvidia-smi` inside the container.

Render one still from the smoke test right away to check the overlay (CPU, a few seconds):

```bash
python3 tools/render_episode.py $R/calls/ep001_* --still 30 --out $R/still30.png
```

### 5.4 One episode, one seed per model: Gemini first, then GPT, then Claude

One episode = one `arena launch` of one condition with `EPISODES=1`. The seed is fixed: the
scenario fixes the start, the goal and every walker's route, and the pedestrian simulator
uses its default seed 0, so all three models face the same episode. `run_conditions.sh` sets
the proxy env for the chosen provider and stops if the proxy is not running.

**a) No GPU, a few cents: the same models on the offline hospital (do this first).** This
checks the key, the model name, the latency and the answers without Gazebo. It runs in the
container image with no GPU device.

```bash
# HOST (the key proxy from 5.1 must be running)
cd ~/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3
tools/probe_offline.sh gemini gemini-3.5-flash gemini35flash      # <provider> <model> <tag> [seconds=45] [max calls=6]
tools/probe_offline.sh openai gpt-5.5 gpt55
tools/probe_offline.sh claude claude-opus-5-5 opus55             # installs the anthropic SDK once into ~/arena_ws/.fvlm3-pylib
tools/probe_offline.sh --render ~/work/bench3/results/offline/bench3-$(date +%Y%m%d)-offline-gpt55   # episode.mp4 (CPU)
curl -s 127.0.0.1:18765/_status; echo
```

Each prints calls, ok/failed, latency p50/p95 and the estimated cost, and writes
`~/work/bench3/results/offline/bench3-<date>-offline-<tag>/`. Compare the models on one page
(summary, paths on the floor plan, call timeline, call 1 side by side, every answer):

```bash
cd ~/work/bench3
d=results/offline/bench3-$(date +%Y%m%d)-offline
analysis/model_report.sh $d-gemini35flash $d-gpt55 $d-opus55 --title "Three Models, One Corridor" \
  --ledger results/ledgers/fvlm3.jsonl --out results/reports/$(date +%F)-models.html
```

First comparison (2026-11-05, 45 s each, `results/offline/bench3-20261105-offline-*-cmp`):
Claude 9.3 m driven, 5.0 s median latency, robot waiting 15 %; GPT-5.5 6.8 m, 8.2 s, 36 %;
Gemini 3.5 Flash 7.2 m, 5.1 s, 43 % (one wrong turn into a wall, one 429). Report:
`results/reports/2026-11-05-models.html`.

**b) GPU claimed: one Gazebo episode per model (in tmux).**

```bash
# CONTAINER (after 5.2 build and the 5.3 smoke test)
cd /opt/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3
STAGE=probe MODEL=gemini-3.5-flash EP_SECONDS=300 tools/run_conditions.sh fvlm3_lab_crowd gemini35flash gemini P1F1R1
# check it, then the other two:
STAGE=probe MODEL=gpt-5.5          EP_SECONDS=300 tools/run_conditions.sh fvlm3_lab_crowd gpt55  openai P1F1R1
STAGE=probe MODEL=claude-opus-5-5  EP_SECONDS=300 tools/run_conditions.sh fvlm3_lab_crowd opus55 claude P1F1R1
```

Each writes `results/probe/bench3-<date>-<tag>-P1F1R1/` (logs, recorded frames, launch.log), then
prints the summary table (one row per run: calls, ok, failed, latency, stall, distance, cost)
and the proxy status. If a run shows only `failed` calls, read the error in its `calls.jsonl`
(`"error": "http_429 ..."` = quota, `http_404` = model name, `http_400` = request format) and
the proxy ledger. Without `MODEL` each provider uses its default: `claude-opus-5-5`, `gpt-5.5`,
`gemini-3.1-pro-preview`.

### 5.4c Every scenario, one model (offline preview, then Gazebo)

The scenarios and what each tests are in `~/work/bench3/README.md` section 4.1.

```bash
# HOST: check them (CPU, no network), then preview offline
podman run --rm --network none --user 0:0 --entrypoint bash -v ~/arena_ws:/opt/arena_ws -v ~/arena_ws/src/Arena/.venv:/opt/venv \
  -w /opt/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3 localhost/arena:arena-arena_ws -c \
  'source /opt/ros/jazzy/setup.bash; source /opt/arena_ws/install/setup.bash; export PYTHONPATH=$PYTHONPATH:$(echo /opt/venv/lib/python3*/site-packages); python3 tools/check_scenarios.py'
tools/probe_offline.sh --all-scenarios mock mock mockall 45                  # free
SCENARIO=fvlm3_sudden_turn tools/probe_offline.sh openai gpt-5-mini trick-mini 90 20
# CONTAINER, inside the claim, in tmux
MODEL=gpt-5.5 EP_SECONDS=300 tools/run_scenarios.sh gpt55 openai
```

Offline runs add `peds.csv` and `offline_judge.json` (trigger times, closest approach, contact
seconds, goal met) to the run folder. In Gazebo the people are humansim agents (`human:=arena`,
the default for gazebo and flatsim, set explicitly by `run_conditions.sh`), which also make
room for the robot; offline they do not.

### 5.5 CONTAINER: all 8 conditions (in tmux)

```bash
cd /opt/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3
~/work/shared/pin_arena.sh ~/work/bench3                  # HOST, before production runs: refresh arena.lock
EP_SECONDS=240 tools/run_conditions.sh fvlm3_lab_crowd pilot claude              # 8 arms, recorded
EP_SECONDS=240 tools/run_conditions.sh fvlm3_lab_empty pilot-empty claude
EP_SECONDS=240 tools/run_conditions.sh fvlm3_pharmacy_crowd pilot-pharm claude
EP_SECONDS=240 tools/run_conditions.sh fvlm3_lab_crowd smoke mock P1F1R1 P0F0R0   # chosen arms, free
EXTRA="robot.mobile.config.prefetch:=false" tools/run_conditions.sh fvlm3_lab_crowd noprefetch claude P1F1R1   # latency baseline
REC=0 tools/run_conditions.sh fvlm3_lab_crowd norec mock P1F1R1                   # without the recording camera
```

`run_conditions.sh` starts `cam_recorder.py` before each launch and stops it after (`REC=1`,
default), so every run folder gets `frames/` + `index.csv`. `STAGE` picks the results folder
(default `smoke` for mock, `pilot` otherwise). `RENDER=1` renders right after
each condition, but that spends claim time; render later instead (5.7).

Expected cost (estimate, not measured yet): about one call per 3–5 s of driving; each call is
~2k input tokens (two images + prompt) and a few hundred output tokens at low effort, about
$0.01–0.02, so roughly $0.15–0.40 per episode. Read the real numbers from the proxy ledger.
Disk: recorded frames are ~1 MB/s (≈250 MB per 240 s episode), the MP4 ~5 MB per minute.

### 5.6 Results

```bash
python3 tools/summarize.py /opt/arena_ws/data/bench3/pilot --csv /opt/arena_ws/data/bench3/pilot/summary.csv
python3 tools/summarize.py /opt/arena_ws/data/bench3          # every stage; HOST: ~/work/bench3/results
```

Per episode (`results/<stage>/<run_id>/calls/epNNN_<id>/`): `meta.jsonl` (prompt, schema, config, host
clock at start), `calls.jsonl` (sent: reason, text, people, images; answer: latency, age, usage,
reply, projected points, planned path, outcome), `ticks.jsonl` (pose, people, motion, plan left,
call in flight, host clock), `camera.jsonl` (image sizes the planner got), `images/cNNN_in0.jpg`
(camera as sent), `cNNN_in1.jpg` (floor plan), `cNNN_answer.jpg` (chosen points),
`summary.json`, and `episode.mp4` once rendered. Success, collisions and social metrics come from
Arena's evaluation and recording. Move finished runs into `~/data/bench3/raw` and add a line to
`~/work/bench3/data_manifest.md` with the Arena SHA.

### 5.7 Episode videos (no GPU: after the claim)

```bash
cd ~/arena_ws && source arena          # CONTAINER (has ffmpeg, pillow, fonts)
cd /opt/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3
R=/opt/arena_ws/data/bench3/pilot
python3 tools/render_episode.py $R/bench3-20261106-pilot-P1F1R1/calls --all      # -> calls/<episode>/episode.mp4
for d in $R/bench3-20261106-pilot-*; do python3 tools/render_episode.py $d/calls --all; done
python3 tools/render_episode.py $R/bench3-20261106-pilot-P1F1R1/calls/ep001_* --speed 2 --out $R/bench3-20261106-pilot-P1F1R1/p1f1r1_2x.mp4
python3 tools/render_episode.py $R/bench3-20261106-pilot-P1F1R1/calls/ep001_* --start 40 --end 70   # a 30 s clip
python3 tools/render_episode.py $R/bench3-20261106-pilot-P1F1R1/calls/ep001_* --still 55               # one PNG
```

Replace `20261106` with the run date. The frames are looked up in the run folder (`frames/` +
`index.csv` next to `calls/`); pass `--frames <dir>` if they live elsewhere. Rendering takes about
half the episode length on the CPU. On the HOST the files are under `~/work/bench3/results/`.

### 5.8 Offline tests and a demo video (no ROS, no GPU, no API)

```bash
# HOST (uses the Arena image's Python; --network none keeps it offline)
cd ~/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3
podman run --rm --network none --entrypoint bash -v "$PWD":/w -w /w localhost/arena:arena-arena_ws \
  -c 'python3 -m pytest tests -q -p no:cacheprovider && python3 tools/offline_sim.py --latency 4 --seconds 40'

# demo video from the offline hospital_1 sim (mock model, synthetic camera)
tools/probe_offline.sh mock mock mockdemo 60
tools/probe_offline.sh --render ~/work/bench3/results/offline/bench3-$(date +%Y%m%d)-offline-mockdemo
ls -la ~/work/bench3/results/offline/bench3-*-offline-mockdemo/calls/ep001_offline/episode.mp4
```

## 6. Config keys (override with `robot.mobile.config.<key>:=<value>`)

Conditions `people`, `floorplan`, `rules` (on/off) · model `provider` (claude/openai/gemini/mock), `model`
("" = provider default), `effort` (claude effort / openai reasoning_effort / gemini thinkingLevel; off), `fallbacks` (claude: default/off), `max_tokens`, `timeout_s`, `mock_latency_s` · world `world`,
`map_yaml`, `world_yaml` · camera `cam_hfov_deg`, `cam_x`, `cam_y`, `cam_z`, `cam_pitch_deg`,
`depth_max_m`, `image_upscale` · motion `speed_mps`, `robot_radius`, `max_subgoal_m`,
`max_subgoals`, `turn_step_deg`, `safety_stop_m` · latency `prefetch`, `prefetch_margin_s`,
`latency_init_s`, `latency_quantile`, `max_answer_age_s`, `refresh_s`, `event_person_m`,
`max_calls` · logs `log_dir`, `run_label`, `save_images`. Defaults and comments: `planner.yaml`.

## 7. Not yet verified in the live simulator (check in the smoke test)

- 2026-11-06: `vln_rgbd_rec` and `vln_*_rec` now expand with xacro (pip xacro in a throwaway
  container, sensors and optics as expected), the chase camera has not been rendered by Gazebo yet.

- Models: on 2026-11-05 only `gemini-3.5-flash` was run for real (offline hospital, 4 of 4 calls
  OK, 5.5 s median). `gpt-5.5` and `claude-opus-5-5` have only been tested against fake servers
  (request format) and a free model list; 5.4a is their first real call.

- The recording camera `vln_rgbd_rec` has not been through xacro or Gazebo yet (the xacro check
  in 5.2 is the first test). Its 640x480 sensor is listed after the 224 px one so the adapter,
  which takes the first `image` sensor, keeps feeding the model the 224 px frames; `camera.jsonl`
  shows what the planner got.
- GPU memory for Gazebo + three cameras (expected 1–3 GB) and whether gz renders on the GPU
  inside the container.

- `robot:='jackal[camera=vln_rgbd]'` attaches the RGB-D mast to the jackal's top mount (syntax from
  `arena_robots/robots/README.md`; the default jackal has lidar only).
- Camera height `cam_z: 1.19` is computed from the mount (0.205 m) + mast (0.9815 m) above
  `chassis_link`; confirm with `tf2_echo`.
- `arena_peds` topic name, pedestrian poses in the map frame, and the depth encoding (`32FC1`, metres).
- The scenario `pose:` on the laboratory goto is only for Arena's success check;
  `robot.mobile.goal:=instruction` withholds it from the planner.
- `tools/run_conditions.sh` stops each launch with a wall-clock `timeout`; moving it to
  `arena evaluation benchmark` suites is the next step.

## 8. As a VLN module next to CMA / NaVid / NaVILA / StreamVLN (2026-11-06)

frontier_vlm3 is a submodule `vln/frontier_vlm3` of arena_planners like the other VLNs
(`arena feature planners ls` lists it; its own repo, branch `master`, not pushed yet). Same contract:
`goal_inputs: [instruction]`, `signals: [arrived]`, `robot.mobile:=vla`, the goal pose withheld.

- `robot.mobile.config.profile:=vln` gives it exactly the VLN inputs: the camera image untouched
  (`people: raw`, no people ground truth, no labels, no inpainting) and no floor plan. `rules` stays a
  switch, so "frontier VLN +/- rules" is the clean ablation against the trained VLNs. Differences that
  remain: waypoint chunks + pure pursuit instead of discrete primitives, and the lidar local grid /
  safety stop. See `~/work/bench3/docs/vln_planners.md`.
- Decision trace (arena_planners SDK, all planners): `ARENA_PLANNER_TRACE_DIR=<dir>` writes
  `<dir>/<planner>/<episode>/trace.jsonl` (reset, every step: command / chunk / signal, pose, latency,
  the model answer the planner logged) and the frame the model saw at each call.
- Recording cameras with a third-person view: `vln_rgbd_rec` (CMA, frontier_vlm3) and `vln_rgb_rec`
  variants `vln_navid_rec`, `vln_uninavid_rec`, `vln_navila_rec`, `vln_streamvln_rec`: the planner's own
  camera unchanged + `rec_image` (640x480, 30 Hz) + `chase_image` (2.6 m behind, 1.6 m up, 28 deg down,
  30 Hz, branch `bench3/vln-rgbd-rec` of arena_robots).

```bash
# CONTAINER, once (no GPU needed): build the VLN planners (each gets its pinned environment) and the cameras
arena build arena_robots arena_planners arena_planners_cma arena_planners_navid arena_planners_navila arena_planners_streamvln arena_planner_frontier_vlm3
source /opt/arena_ws/install/setup.bash
# HOST: copy the bench3 scenarios into the world after any edit
~/work/bench3/scripts/sync_scenarios.sh
# CONTAINER, inside the GPU claim, in tmux: one planner over all bench3_* scenarios
cd /opt/arena_ws/src/Arena/arena_planners/planners/frontier_vlm3
tools/run_vln_scenarios.sh vln1 cma                                  # smallest first
tools/run_vln_scenarios.sh vln1 navila bench3_cross_repeat           # one scenario
PROVIDER=openai MODEL=gpt-5.5 RULES=on tools/run_vln_scenarios.sh vln1 frontier_vlm3
# afterwards, CPU: videos
python3 tools/render_vln.py --all /opt/arena_ws/data/bench3/probe/bench3-<date>-vln1-cma-cross_repeat
```

