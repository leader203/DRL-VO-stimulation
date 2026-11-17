#!/usr/bin/env bash
# Model-only DRL-VO evaluation in the third_3 scene with its four pedestrians.
# The guard is logged but not applied.
# Usage:
#   ./run_model_eval_peds.sh                         # GUI, configured 60 s
#   ./run_model_eval_peds.sh --headless --duration 40
set -euo pipefail
cd "$(dirname "$0")"

# Local import root: pedstrain_simulation.simulation -> this repository.
if [ ! -d .package_root/pedstrain_simulation ]; then
  mkdir -p .package_root/pedstrain_simulation
  : > .package_root/pedstrain_simulation/__init__.py
  ln -sfn "$PWD" .package_root/pedstrain_simulation/simulation
fi

# Isaac Sim's bundled ROS 2 Jazzy, no /tmp scripts required.
J=/home/stardust/isaacsim61_env/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy
export PYTHONPATH="$PWD/.package_root:$J/rclpy"
export LD_LIBRARY_PATH="$J/lib"
export AMENT_PREFIX_PATH="$J"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY

export OMNI_KIT_ACCEPT_EULA=YES

exec /home/stardust/isaacsim61_env/bin/python run.py run configs/third_3 \
  --navigation-settings configs/runtime/ros2_navigation_local.yaml \
  --drl-policy-config configs/model_eval/drl_eval_peds.yaml "$@"
