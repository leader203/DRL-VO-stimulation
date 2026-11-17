#!/bin/bash
# 启动终端 WASD 机器人底盘遥控。参数原样传给 robot_keyboard_teleop.py。
set -e

E=/home/stardust/isaacsim61_env
J="$E/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy"

exec env -u PYTHONPATH -u LD_LIBRARY_PATH \
  PYTHONPATH="$J/rclpy" \
  LD_LIBRARY_PATH="$J/lib" \
  AMENT_PREFIX_PATH="$J" \
  ROS_DISTRO=jazzy \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  ROS_DOMAIN_ID=37 \
  "$E/bin/python" "$(dirname "$0")/robot_keyboard_teleop.py" "$@"
