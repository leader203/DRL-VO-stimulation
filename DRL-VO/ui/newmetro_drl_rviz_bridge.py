#!/usr/bin/env python3
"""Publish the current owned Isaac demo's actual state to ROS 2 for RViz."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT


def owned_report(run):
    owner = json.loads((run / 'OWNER.json').read_text())
    pid = int(owner['pid'])
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != int(owner['uid']):
        raise RuntimeError('Isaac owner UID changed')
    ticks = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    argv = [v.decode(errors='replace') for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
    if ticks != str(owner['start_ticks']) or str(run) not in argv or not any(v.endswith('/show_newmetro_robot_preview.py') for v in argv):
        raise RuntimeError('Isaac owner process changed')
    report_path = run / 'REPORT.json'
    if time.time() - report_path.stat().st_mtime > 120:
        raise RuntimeError('Isaac report stale')
    report = json.loads(report_path.read_text())
    if int(report['pid']) != pid:
        raise RuntimeError('Isaac report owner changed')
    return report


class Bridge(Node):
    def __init__(self, run):
        super().__init__('newmetro_drl_demo_' + str(os.getpid()))
        self.run = run
        self.owner_pid = int(json.loads((run / 'OWNER.json').read_text())['pid'])
        reliable = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                              reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/newmetro/reference_map', latched)
        self.robot_pub = self.create_publisher(Marker, '/newmetro/robot', reliable)
        self.people_pub = self.create_publisher(MarkerArray, '/newmetro/pedestrians', reliable)
        self.goal_pub = self.create_publisher(Marker, '/newmetro/goal', reliable)
        self.subgoal_pub = self.create_publisher(Marker, '/newmetro/subgoal', reliable)
        self.route_pub = self.create_publisher(NavPath, '/newmetro/planned_route', latched)
        self.trace_pub = self.create_publisher(NavPath, '/newmetro/actual_trace', reliable)
        self.status_pub = self.create_publisher(String, '/newmetro/demo_status', reliable)
        self.scan_pubs = {name: self.create_publisher(PointCloud2,
                          f'/newmetro/{name}/points_world', reliable)
                          for name in ('scan_01', 'scan_02')}
        self.map_sent = False
        self.last_scan_mtime = -1
        self.last_route_mtime = -1
        self.trace = []
        self.stop = False
        self.create_timer(.25, self.tick)

    def header(self):
        h = Header()
        h.frame_id = 'map'
        h.stamp = self.get_clock().now().to_msg()
        return h

    def publish_map(self):
        with np.load(self.run / 'robot_navigation_grid.npz', allow_pickle=False) as data:
            labels = data['labels']
        if labels.shape != (76, 262) or not set(np.unique(labels)).issubset({0, 205, 254}):
            raise RuntimeError('Unexpected native map labels')
        msg = OccupancyGrid()
        msg.header = self.header()
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = .1
        msg.info.width = 262
        msg.info.height = 76
        msg.info.origin.position.x = -15.
        msg.info.origin.position.y = -6.2
        msg.info.origin.orientation.w = 1.
        cells = np.full(labels.shape, -1, dtype=np.int8)
        cells[labels == 254] = 0
        cells[labels == 0] = 100
        msg.data = np.flipud(cells).ravel().tolist()
        self.map_pub.publish(msg)
        self.map_sent = True

    def marker(self, namespace, ident, marker_type, x, y, z, color, scale):
        m = Marker()
        m.header = self.header()
        m.ns = namespace
        m.id = ident
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        m.pose.orientation.w = 1.
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    def publish_state(self, report):
        obs = report.get('robot_observation') or {}
        position = obs.get('root_position_m')
        yaw = obs.get('yaw_rad')
        if position and yaw is not None and all(math.isfinite(float(v)) for v in [*position[:2], yaw]):
            robot = self.marker('physical_robot', 1, Marker.ARROW, float(position[0]), float(position[1]), .2,
                                (.05, 1., .1, 1.), (.72, .16, .16))
            robot.pose.orientation.z = math.sin(float(yaw)/2)
            robot.pose.orientation.w = math.cos(float(yaw)/2)
            self.robot_pub.publish(robot)
            label = self.marker('physical_robot', 2, Marker.TEXT_VIEW_FACING,
                                float(position[0]), float(position[1]) + .65, .5,
                                (1., 1., 1., 1.), (0., 0., .23))
            distance = report.get('demo_last_goal_distance_m')
            distance_text = f'{float(distance):.2f}m' if distance is not None else '--'
            label.text = (f"{report.get('status')}  goal={distance_text}  "
                          f"reached={report.get('demo_completed_goals', 0)}")
            self.robot_pub.publish(label)
            point = (float(position[0]), float(position[1]))
            if not self.trace or math.dist(point, self.trace[-1]) > .03:
                self.trace.append(point)
            self.publish_path(self.trace_pub, self.trace)
        people = MarkerArray()
        for i, xyz in enumerate(report.get('positions', [])):
            if len(xyz) >= 2 and all(math.isfinite(float(v)) for v in xyz[:2]):
                people.markers.append(self.marker('physical_pedestrians', i, Marker.CYLINDER,
                    float(xyz[0]), float(xyz[1]), .45, (1., .42, .07, .9), (.42, .42, .9)))
        self.people_pub.publish(people)
        goal = report.get('demo_goal_xy')
        gm = self.marker('selected_goal', 1, Marker.SPHERE,
                         float(goal[0]) if goal else 0., float(goal[1]) if goal else 0., .2,
                         (1., .85, .0, 1.), (.38, .38, .38))
        if not goal or report.get('status', '').startswith('DEMO_WAITING'):
            gm.action = Marker.DELETE
        self.goal_pub.publish(gm)
        subgoal = report.get('demo_subgoal_world_xy_m')
        sm = self.marker('policy_subgoal', 1, Marker.SPHERE,
                         float(subgoal[0]) if subgoal else 0., float(subgoal[1]) if subgoal else 0., .28,
                         (.9, .05, 1., 1.), (.18, .18, .18))
        if not subgoal or report.get('status') != 'DEMO_RUNNING':
            sm.action = Marker.DELETE
        self.subgoal_pub.publish(sm)
        status = String()
        status.data = json.dumps({'status': report.get('status'),
            'goal_xy_m': goal, 'goal_distance_m': report.get('demo_last_goal_distance_m'),
            'completed_goals': report.get('demo_completed_goals', 0),
            'frozen_policy_actions': report.get('demo_actions'),
            'subgoal_world_xy_m': subgoal, 'subgoal_lookahead_m': report.get('demo_local_goal_lookahead_m'),
            'actual_controls': report.get('actual_controls'),
            'contact_safety': report.get('contact_safety', 'UNVERIFIED')})
        self.status_pub.publish(status)

    def publish_path(self, publisher, xy):
        path = NavPath()
        path.header = self.header()
        for x, y in xy:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.
            path.poses.append(pose)
        publisher.publish(path)

    def publish_route(self):
        path = self.run / 'ROBOT_GLOBAL_ROUTE.json'
        if not path.exists():
            return
        stat = path.stat()
        if stat.st_mtime_ns == self.last_route_mtime:
            return
        route = json.loads(path.read_text())
        self.publish_path(self.route_pub, [pose[:2] for pose in route['poses']])
        self.last_route_mtime = stat.st_mtime_ns

    def publish_scans(self):
        path = self.run / 'latest_lidar.npz'
        if not path.exists():
            return
        stat = path.stat()
        if stat.st_mtime_ns == self.last_scan_mtime:
            return
        with np.load(path, allow_pickle=False) as data:
            payload = [(name, data[name].copy(), data[name + '_points_world'].copy())
                       for name in self.scan_pubs]
        for name, ranges, points in payload:
            if ranges.shape != (2000,) or points.shape != (2000, 3):
                raise RuntimeError('Actual LiDAR packet shape changed')
            valid = np.isfinite(ranges) & np.isfinite(points).all(axis=1)
            self.scan_pubs[name].publish(point_cloud2.create_cloud_xyz32(self.header(),
                                          points[valid].astype(np.float32).tolist()))
        self.last_scan_mtime = stat.st_mtime_ns

    def tick(self):
        try:
            report = owned_report(self.run)
            if int(report['pid']) != self.owner_pid:
                raise RuntimeError('Isaac PID changed')
            if not self.map_sent:
                self.publish_map()
            self.publish_state(report)
            self.publish_route()
            self.publish_scans()
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            self.get_logger().error(f'Owned Isaac demo unavailable: {exc}')
            self.stop = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    if run.parent != (DEMO / 'runs').resolve():
        parser.error('Expected isolated demo run directory')
    rclpy.init()
    node = Bridge(run)
    try:
        while rclpy.ok() and not node.stop:
            rclpy.spin_once(node, timeout_sec=.5)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
