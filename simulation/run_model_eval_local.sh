#!/usr/bin/env bash
# Same pipeline as simulation/run_model_eval.sh, but pointed at the Isaac Sim
# install that exists on this box (isaaclab62 instead of isaacsim61_env) and at
# the in-repo copy of the frozen model package. The model file itself is the
# same artefact: sha256 bef18b93... matches drl_eval.yaml's expected_sha256.
#
# Usage:
#   ./run_model_eval_local.sh                 # overhead camera, records MP4
#   ./run_model_eval_local.sh --dry-run       # resolve settings only
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .package_root/pedstrain_simulation ]; then
  mkdir -p .package_root/pedstrain_simulation
  : > .package_root/pedstrain_simulation/__init__.py
  ln -sfn "$PWD" .package_root/pedstrain_simulation/simulation
fi

ISAAC_ENV=/workspace/miniconda3/envs/isaaclab62
J=$ISAAC_ENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy
export PYTHONPATH="$PWD/.package_root:$J/rclpy"
export LD_LIBRARY_PATH="$J/lib"
export AMENT_PREFIX_PATH="$J"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY

# Recording validates ffmpeg and ffprobe on PATH; there is no system copy, so
# borrow the conda-forge pair from the lerobot env (both start cleanly).
mkdir -p /workspace/wujunlin/bin
ln -sfn /workspace/miniconda3/envs/lerobot/bin/ffmpeg /workspace/wujunlin/bin/ffmpeg
ln -sfn /workspace/miniconda3/envs/lerobot/bin/ffprobe /workspace/wujunlin/bin/ffprobe
export PATH="/workspace/wujunlin/bin:$PATH"

export OMNI_KIT_ACCEPT_EULA=YES
export PXR_USDC_EMIT_DEPRECATION_WARNINGS=0

exec "$ISAAC_ENV/bin/python" run.py run configs/model_eval/recording.toml \
  --drl-policy-config configs/model_eval/drl_eval_local.yaml "$@"
