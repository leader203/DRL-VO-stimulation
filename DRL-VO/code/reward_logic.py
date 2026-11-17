"""Stateful original reward/done semantics; author angular threshold is 1.0 rad/s."""
import math
import random
import numpy as np


class OriginalReward:
    def __init__(self, seed=1337, *, angular_threshold=1.0, angular_scale=.1):
        self.angular_threshold = float(angular_threshold)
        self.angular_scale = float(angular_scale)
        self.rng = random.Random(int(seed))
        self.bump_num = 0
        self.reset_task()

    def reset_task(self):
        self.num_iterations = 0
        self.distances = np.zeros(10, dtype=float)
        # Source _init_env_variables comments out bump_num reset. Preserve the
        # counter across goal/timeout/reset, and clear only on collision terminal.

    def theta_reward(self, goal_base_xy, pedestrians_base_xy_vxy, actual_v):
        goal = np.asarray(goal_base_xy, dtype=float)
        peds = np.asarray(pedestrians_base_xy_vxy, dtype=float).reshape(-1, 4)
        preferred = float(np.arctan2(goal[1], goal[0]))
        direction = preferred
        guards = 0
        if len(peds):
            direction = math.pi/2
            best = 1000.
            for _ in range(60):
                theta = self.rng.uniform(-math.pi, math.pi)
                free = True
                for px, py, vx, vy in peds:
                    distance = float(np.hypot(px, py))
                    if distance <= 7.:
                        bearing = float(np.arctan2(py, px))
                        radicand = distance**2-.9**2
                        if radicand < 0: guards += 1
                        # Explicit numerical repair to the source's negative sqrt.
                        half_cone = float(np.arctan2(.9, math.sqrt(max(0., radicand))))
                        relative = float(np.arctan2(actual_v*math.sin(theta)-vy,
                                                    actual_v*math.cos(theta)-vx))
                        if bearing-half_cone <= relative <= bearing+half_cone:
                            free = False
                            break
                if free:
                    error = (theta-preferred)**2
                    if error < best:
                        best, direction = error, theta
        return .6*(math.pi/6-abs(direction)), guards

    def step(self, *, distance, minimum_scan, actual_v, actual_w, goal_base_xy, pedestrians_base_xy_vxy):
        distance, minimum_scan, actual_v, actual_w = map(float, (distance, minimum_scan, actual_v, actual_w))
        if not np.isfinite([distance, minimum_scan, actual_v, actual_w]).all() or distance < 0 or minimum_scan < 0:
            raise ValueError('Invalid measured reward inputs')
        index = self.num_iterations % 10
        if self.num_iterations == 0: self.distances[:] = distance
        if distance <= .3:
            goal = 20.
        elif self.num_iterations >= 512:
            goal = -20.
        else:
            goal = 3.2*(self.distances[index]-distance)
        self.distances[index] = distance
        obstacle = (-20. if .02 <= minimum_scan <= .3 else
                    -.2*(.9-minimum_scan) if minimum_scan < .9 else 0.)
        angular = -self.angular_scale*abs(actual_w) if abs(actual_w) > self.angular_threshold else 0.
        vo, guards = self.theta_reward(goal_base_xy, pedestrians_base_xy_vxy, actual_v)
        reward = goal+obstacle+angular+vo
        self.num_iterations += 1
        if distance <= .3:
            done, reason = True, 'goal'
        else:
            if .02 <= minimum_scan <= .3: self.bump_num += 1
            if self.bump_num >= 3:
                self.bump_num = 0
                done, reason = True, 'collision'
            elif self.num_iterations > 512:
                done, reason = True, 'timeout'
            else:
                done, reason = False, 'running'
        return float(reward), done, reason, dict(goal=goal, obstacle=obstacle, angular=angular,
            vo=vo, vo_close_radicand_guards=guards, bump_num=self.bump_num,
            iteration_after_done_check=self.num_iterations, reward_terms_added=True)

    def state(self):
        return dict(num_iterations=self.num_iterations, distances=self.distances.copy(),
                    bump_num=self.bump_num, rng=self.rng.getstate())

    def restore(self, state):
        self.num_iterations = int(state['num_iterations'])
        self.distances = np.asarray(state['distances'], dtype=float).copy()
        self.bump_num = int(state['bump_num'])
        self.rng.setstate(state['rng'])
