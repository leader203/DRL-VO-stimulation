#!/usr/bin/env bash
# v13: the third_3 scene as it actually runs in evaluation, 512-way parallel.
#
# What this run is, in one sentence: the same room, the same robot and the same
# four walkers that `run_model_eval_peds.sh` drives, trained instead of frozen.
#
# Room     `simulation/rl/assets/third_room.usd` is the cloneable extraction of
#          `simulation/assets/THIRD_EXP_1.usd`, the asset `configs/third_3/
#          simulation.yaml` points at.  `configs/model_eval/simulation.toml`
#          only overrides pedestrians, so evaluation and training stand in the
#          same geometry.
#
# Robot    `simulation/rl/assets/evt_robot.usd`, the de-ROS'd copy of
#          `configs/third_3/evt_ros_robot.usda`: same 38 joints, same masses,
#          same wheel layout, minus the OmniGraph nodes that cannot be cloned
#          512 times.  No added body, no added collider.
#
# Walkers  `runs/ped_trace.npz` is the bake of `configs/third_3/simulation.yaml`
#          itself: walker_01..04, `speed: 0.9`, the same waypoints and waits.
#          v12 halved their speed with `--ped-speed-scale 0.5`, which is not
#          what the evaluation does; this run uses 1.0.
#
# Clock    `global` keeps one corridor clock for the whole run, so a reset drops
#          the robot into the middle of the crowd's walk instead of rewinding
#          it.  That is what evaluation sees; `reset_robot_random` now keeps the
#          spawn clear of the walkers' *current* disks to match.
#
# Reward   `rl/rewards.py::drl_vo_reward`, unchanged from v12 -- the four paper
#          terms plus the deadlock charge, speed gate at 0.9 m/s.
#
# Per-iteration review
#     `--video-every 1` makes the trainer score the policy after every update and
#     `--save-every 1` writes `drl_vo_iterNNNN.pt` to match, so each iteration has
#     a number and a weight.  `--video-steps 0` keeps RTX out of this process: a
#     camera across 512 clones never finished building, which is why the clip is
#     recorded by `watch_v13_video.sh` in a one-environment process instead.
set -eu

PY=/workspace/miniconda3/envs/isaaclab62/bin/python
REPO=/workspace/wujunlin/DRL-VO-stimulation
RUNS=/workspace/wujunlin/runs
OUT="$RUNS/v13_eval_scene"

exec 9> "$RUNS/v13.lock"
flock -n 9 || { echo "another trainer already holds $RUNS/v13.lock, refusing to double-start"; exit 1; }

cd "$REPO"
exec env OMNI_KIT_ACCEPT_EULA=YES PXR_USDC_EMIT_DEPRECATION_WARNINGS=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u \
  simulation/train_drl_vo_vec.py \
  --mode train \
  --sim-device cuda:0 \
  --num-envs 512 \
  --n-steps 64 \
  --batch-size 256 \
  --epochs 4 \
  --total-steps 9830400 \
  --lr 1e-4 \
  --target-kl 0.02 \
  --gamma 0.995 \
  --gae-lambda 0.97 \
  --episode-steps 1000 \
  --save-every 1 \
  --lookahead 0.25 \
  --goal-radius 0.20 \
  --goal-clearance 0.60 \
  --plan-margin 0.02 \
  --plan-downsample 2 \
  --replan-steps 20 \
  --pedestrians \
  --ped-trace "$RUNS/ped_trace.npz" \
  --ped-speed-scale 1.0 \
  --ped-clock global \
  --video-every 1 \
  --eval-episodes 8 \
  --video-steps 0 \
  --seed 13001 \
  --map-cache "$RUNS/map512_v13.npz" \
  --output "$OUT"
