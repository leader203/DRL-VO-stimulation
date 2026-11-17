"""Evaluate actual post-physics robot telemetry, independently of wheel targets."""
import math
import numpy as np


def evaluate_robot_smoke(rows):
    segments = {}
    for name in ("settle", "forward", "stop_after_forward", "backward", "left", "stop_after_left", "right", "final_stop"):
        selected = [r for r in rows if r["segment"] == name]
        if not selected:
            segments[name] = {"status": "NO_SAMPLES"}
            continue
        start, end = selected[0], selected[-1]
        delta = np.asarray(end["root_position_m"]) - np.asarray(start["root_position_m"])
        heading = start["yaw_rad"]
        forward = float(delta[0]*math.cos(heading) + delta[1]*math.sin(heading))
        yaw = math.atan2(math.sin(end["yaw_rad"]-start["yaw_rad"]), math.cos(end["yaw_rad"]-start["yaw_rad"]))
        tail = selected[-min(15, len(selected)):]
        linear = float(np.mean([abs(r["physical_forward_velocity_m_s"]) for r in tail]))
        angular = float(np.mean([abs(r["physical_yaw_velocity_rad_s"]) for r in tail]))
        wheel_error = float(np.mean([np.abs(np.asarray(r["wheel_actual_velocity_rad_s"]) - np.asarray(r["wheel_target_velocity_rad_s"])).mean() for r in tail]))
        passed = (forward > .10 if name == "forward" else forward < -.10 if name == "backward"
            else yaw > .20 if name == "left" else yaw < -.20 if name == "right"
            else linear < .05 and angular < .15)
        segments[name] = {"status":"PASS" if passed else "FAIL", "samples":len(selected),
            "actual_forward_displacement_m":forward, "actual_yaw_change_rad":yaw,
            "tail_mean_abs_forward_speed_m_s":linear, "tail_mean_abs_yaw_speed_rad_s":angular,
            "tail_mean_wheel_velocity_error_rad_s":wheel_error}
    return {"status":"ROBOT_MOVEMENT_SMOKE_PASS" if all(s["status"] == "PASS" for s in segments.values()) else "ROBOT_MOVEMENT_SMOKE_FAIL",
        "segments":segments, "root_pose_integration":False, "root_velocity_injection":False,
        "scope":"Bounded signed forward/reverse/yaw and stop telemetry only; not contact safety or navigation success",
        "full_action_range_validated":False, "collision_safety_validated":False,
        "RL_ready":False, "training":False}
