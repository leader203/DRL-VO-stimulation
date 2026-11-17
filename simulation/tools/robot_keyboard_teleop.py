#!/usr/bin/env python3
"""Terminal WASD teleop for the robot chassis over ROS 2 ``/cmd_vel``."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WASD teleop for the simulation robot chassis.")
    parser.add_argument("--linear", type=float, default=0.3, help="前进/后退速度 [m/s]，默认 0.3")
    parser.add_argument("--angular", type=float, default=0.7, help="左右转向角速度 [rad/s]，默认 0.7")
    parser.add_argument("--timeout", type=float, default=0.35, help="松开按键后保持时间 [s]，默认 0.35")
    parser.add_argument("--rate", type=float, default=20.0, help="发布频率 [Hz]，默认 20")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("需要在交互式终端中运行（stdin 不是 TTY）。")

    rclpy.init()
    node = rclpy.create_node("robot_keyboard_teleop")
    publisher = node.create_publisher(Twist, "/cmd_vel", 10)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.count_subscribers("/cmd_vel") == 0:
        print("警告：/cmd_vel 当前没有订阅者；请确认仿真已启动并出现 crowd controller ready。", flush=True)
    command = Twist()
    last_key = 0.0
    last_printed = None
    old_settings = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("机器人底盘遥控")
        print(f"  w 前进  s 后退  a 左转  d 右转")
        print(f"  空格 立即停车  q 退出")
        print(f"  线速度 {args.linear:g} m/s，角速度 {args.angular:g} rad/s，松开按键 {args.timeout:g}s 后停车")
        period = 1.0 / args.rate
        next_tick = time.monotonic()
        while True:
            now = time.monotonic()
            key = None
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
            if key == "":
                break
            if key in ("q", "Q"):
                break
            if key in ("w", "W"):
                command.linear.x = args.linear
                last_key = now
            elif key in ("s", "S"):
                command.linear.x = -args.linear
                last_key = now
            elif key in ("a", "A"):
                command.angular.z = args.angular
                last_key = now
            elif key in ("d", "D"):
                command.angular.z = -args.angular
                last_key = now
            elif key == " ":
                command = Twist()
                last_key = 0.0
            if last_key and now - last_key > args.timeout:
                command = Twist()
                last_key = 0.0
            current = (round(command.linear.x, 3), round(command.angular.z, 3))
            if current != last_printed:
                print(f"  linear={current[0]:+.2f} m/s  angular={current[1]:+.2f} rad/s", flush=True)
                last_printed = current
            publisher.publish(command)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        for _ in range(5):
            publisher.publish(Twist())
            time.sleep(0.02)
        node.destroy_node()
        rclpy.try_shutdown()
        print("已发送停车指令。")


if __name__ == "__main__":
    main()
