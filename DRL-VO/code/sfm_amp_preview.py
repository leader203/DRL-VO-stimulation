"""Gazebo-style SFM trajectory targets for physical AMP, without A* queries."""
import math
import numpy as np
import torch


def social_force(position, velocity, other_position, other_velocity, neighbor_range=10.0):
    """Same center/relative-velocity interaction as gazebo_social_kernel.py."""
    diff = other_position - position
    distance = float(np.linalg.norm(diff))
    if distance < 1e-6 or distance > neighbor_range:
        return np.zeros(2)
    direction = diff / distance
    interaction = 2.0 * (velocity - other_velocity) + direction
    length = float(np.linalg.norm(interaction))
    if length < 1e-6:
        return np.zeros(2)
    interaction /= length
    theta = math.atan2(interaction[0]*direction[1]-interaction[1]*direction[0],
                       float(np.dot(interaction, direction)))
    b = max(1e-6, .35 * length)
    velocity_amount = -math.exp(-distance/b - (3.0*b*theta)**2)
    angle_amount = -math.copysign(1.0, theta)*math.exp(-distance/b - (2.0*b*theta)**2) if theta else 0.0
    return velocity_amount*interaction + angle_amount*np.array([-interaction[1], interaction[0]])


class SfmAmpPreview:
    """Receding-horizon SFM predictions sampled at PACER's 0..4.5 s times."""
    def __init__(self, controller, routes_xy, arrival_radius, native_map, robot, report, *, speed=.8):
        self.controller, self.robot, self.report = controller, robot, report
        self.routes = np.asarray(routes_xy, dtype=float)
        if self.routes.ndim != 3 or self.routes.shape[-1] != 2 or self.routes.shape[1] < 2 or not np.isfinite(self.routes).all():
            raise ValueError("Finite cyclic SFM routes [actors,vertices,2] required")
        self.arrival_radius = float(arrival_radius)
        if not math.isfinite(self.arrival_radius) or self.arrival_radius <= 0:
            raise ValueError("Positive SFM waypoint arrival radius required")
        # Occupancy distance supplies a soft obstacle force only. The map's
        # A* walkability and start-recovery clearance masks are never queried.
        self.obstacle_map = native_map
        self.speed = speed
        self.last_control = -1
        self.safety_braking_plans = 0

    def advance_route(self, measured_xy, route_indices):
        position = np.asarray(torch.as_tensor(measured_xy).detach().cpu(), dtype=float)
        indices = np.asarray(torch.as_tensor(route_indices).detach().cpu(), dtype=np.int64)
        if position.shape != (len(self.routes), 2) or indices.shape != (len(self.routes),):
            raise ValueError("SFM measured positions and route indices must match actors")
        reached = np.linalg.norm(position-self.routes[np.arange(len(position)), indices], axis=1) <= self.arrival_radius
        updated = (indices+reached.astype(np.int64)) % self.routes.shape[1]
        device = torch.as_tensor(route_indices).device
        return torch.as_tensor(updated, device=device), torch.as_tensor(reached, device=device)

    def _obstacle_force(self, point):
        grid = self.obstacle_map._signed_grid
        pixel = self.obstacle_map.world_to_pixel(np.asarray(point))
        row, col = int(pixel[0]), int(pixel[1])
        if not (1 <= row < grid.shape[0]-1 and 1 <= col < grid.shape[1]-1):
            return np.zeros(2), math.nan
        distance = float(grid[row, col])
        gradient = np.array([grid[row, col+1]-grid[row, col-1],
                             grid[row-1, col]-grid[row+1, col]], dtype=float)
        length = float(np.linalg.norm(gradient))
        if length < 1e-6:
            return np.zeros(2), distance
        amount = math.exp(-max(-2., distance-.75)/.2)
        return amount*gradient/length, distance

    def action(self, world, control, route_indices, **_):
        if control != self.last_control + 1 or abs(world.dt-1/30) > 1e-8:
            raise ValueError("SFM preview requires consecutive measured 30 Hz controls")
        measured = world.planar().detach().cpu().numpy()
        indices = np.asarray(torch.as_tensor(route_indices).detach().cpu(), dtype=np.int64).copy()
        if measured.shape != (len(self.routes), 5) or indices.shape != (len(self.routes),):
            raise ValueError("SFM input shape must match route actor count")
        robot_state = self.robot.observe()
        robot_xy = np.asarray(robot_state['root_position_m'][:2], dtype=float)
        robot_v = np.asarray(robot_state['linear_velocity_world'][:2], dtype=float)
        position = measured[:, :2].astype(float).copy()
        velocity = measured[:, 2:4].astype(float).copy()
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise FloatingPointError("Nonfinite measured pedestrian state")
        future = np.empty((len(position), 10, 2), dtype=np.float32)
        future[:, 0] = position
        min_peer = math.inf
        min_robot = math.inf
        min_wall = math.inf
        # Predict all actors simultaneously, feeding the next real measured
        # state back on the following control; no predicted pose is written.
        for tick in range(1, 46):
            dt = .1
            force = np.zeros_like(position)
            goal = self.routes[np.arange(len(position)), indices]
            for i in range(len(position)):
                offset = goal[i] - position[i]
                length = float(np.linalg.norm(offset))
                if length <= self.arrival_radius:
                    indices[i] = (indices[i]+1) % self.routes.shape[1]
                    offset = self.routes[i, indices[i]] - position[i]
                    length = float(np.linalg.norm(offset))
                desired_v = self.speed*offset/length if length > 1e-6 else np.zeros(2)
                force[i] = (desired_v-velocity[i])/.5
                for j in range(len(position)):
                    if i != j:
                        force[i] += 5.1*social_force(position[i], velocity[i], position[j], velocity[j])
                        min_peer = min(min_peer, float(np.linalg.norm(position[i]-position[j])))
                predicted_robot_xy = robot_xy + robot_v*(tick*dt)
                force[i] += 5.1*social_force(position[i], velocity[i], predicted_robot_xy, robot_v)
                delta = position[i]-predicted_robot_xy
                distance = float(np.linalg.norm(delta))
                min_robot = min(min_robot, distance)
                direction = delta/distance if distance > 1e-6 else np.array([1., 0.])
                force[i] += 6.0*math.exp(-max(-2., distance-1.0)/.2)*direction
                obstacle_force, wall_distance = self._obstacle_force(position[i])
                force[i] += 10.0*obstacle_force
                if math.isfinite(wall_distance):
                    min_wall = min(min_wall, wall_distance)
            proposed_v = velocity + dt*force
            norms = np.linalg.norm(proposed_v, axis=1)
            proposed_v *= np.minimum(1., self.speed/np.maximum(norms, 1e-9))[:, None]
            position, velocity = position + dt*proposed_v, proposed_v
            if tick % 5 == 0:
                future[:, tick//5] = position
        amp_targets = torch.as_tensor(future, device=world.device, dtype=world.planar().dtype)
        self.last_targets = amp_targets
        action = self.controller.act(world.state(), amp_targets, world, return_raw=True)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite SFM to AMP action")
        self.last_control = control
        self.report.update(sfm_predicted_min_peer_center_m=min_peer if math.isfinite(min_peer) else None,
                           sfm_predicted_min_robot_center_m=min_robot if math.isfinite(min_robot) else None,
                           sfm_predicted_min_wall_distance_m=min_wall if math.isfinite(min_wall) else None,
                           sfm_target_xy_1s=future[:, 2].tolist(),
                           sfm_target_xy_4s=future[:, 8].tolist())
        return action, {"replanned": True, "safety_braking": False}
