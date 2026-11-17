"""Configurable pure-pursuit circle target using original executable methods."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'reference/pure_pursuit.py'
tree = ast.parse(SOURCE.read_text())
methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
           and node.name in ('find_closest_point', 'find_goal')]
namespace = dict(np=np, rospy=SimpleNamespace(logwarn=lambda *a: None))
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(SOURCE), 'exec'), namespace)


def local_goal_base(path_xy, robot_pose, lookahead=1.5):
    path = np.asarray(path_xy, dtype=float)
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1)>1e-10]]
    if len(path) < 2:
        target = path[-1]
    else:
        class Pursuit:
            find_closest_point = namespace['find_closest_point']
            find_goal = namespace['find_goal']
        controller = Pursuit()
        controller.lookahead = float(lookahead)
        controller.path = SimpleNamespace(poses=[SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=p[0], y=p[1]),
            orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))) for p in path])
        closest, distance, segment = controller.find_closest_point(np.asarray(robot_pose[:2]))
        target, _, _ = controller.find_goal(np.asarray(robot_pose[:2]), closest, distance, segment)
    delta = np.asarray(target)-robot_pose[:2]
    yaw = robot_pose[2]
    result = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])@delta
    if not np.isfinite(result).all(): raise ValueError('Pure-pursuit produced a non-finite local goal')
    return result


class ActionClock:
    """20 Hz deadlines quantized causally onto the unchanged 30 Hz physics owner."""
    def __init__(self, period=.05):
        self.period = float(period)
        self.origin = None
        self.index = 0

    def due(self, time):
        return self.origin is None or time+1e-9 >= self.origin+self.index*self.period

    def commit(self, time):
        if self.origin is None: self.origin = float(time)
        self.index = max(self.index+1, int(math.floor((time-self.origin)/self.period+1e-8))+1)
