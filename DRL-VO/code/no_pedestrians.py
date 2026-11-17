"""Remove the initialized crowd before the first training physics transition.

The common startup retains policy initialization/RNG ordering. After removal,
only the EVT robot and the unchanged static scene participate in physics.
Empty tensors represent zero actors, never fabricated live actor states.
"""
import json
import random
import numpy as np
from types import SimpleNamespace
import torch


def install(m):
    from pxr import Usd, UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    policy = m.training_bridge.driver.policy
    assert policy.steps == 0 and policy.pending is None and m.control == 0
    assert m.args.isolated_demo
    sim = m.world.simulator
    dt = sim._sim.get_physics_dt()
    assert abs(dt * sim.decimation - m.world.dt) < 1e-9
    roots = [f'/World/envs/env_{i}/Robot' for i in range(m.count)]

    def retained_colliders():
        return sorted(str(p.GetPath()) for p in stage.Traverse()
                      if p.HasAPI(UsdPhysics.CollisionAPI)
                      and not any(str(p.GetPath()).startswith(r + '/') or str(p.GetPath()) == r for r in roots))

    before = retained_colliders()
    state = m.world.state()
    old_positions = state.root_pos.detach().cpu().tolist()
    initial_robot = m.robot.observe()
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(), np.random.get_state(), random.getstate())
    empty = SimpleNamespace(**{k: getattr(state, k)[:0].clone() for k in
                              ('root_pos', 'root_rot', 'root_vel', 'root_ang_vel',
                               'dof_pos', 'dof_vel', 'dof_forces')})
    assert sim._scene._articulations['robot'] is sim._robot
    sim._sim.stop()
    sim._robot._clear_callbacks()
    # Do not write/update a removed articulation through InteractiveScene.
    sim._scene._articulations.pop('robot')
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for root in roots:
            prim = stage.GetPrimAtPath(root)
            assert prim.IsValid() and prim.IsActive(), root
            prim.SetActive(False)
        stage.GetPrimAtPath(m.mesh.visualizer.cfg.prim_root).SetActive(False)
    assert retained_colliders() == before, 'Non-pedestrian geometry changed'
    assert all(not stage.GetPrimAtPath(r).IsActive() for r in roots)
    # PhysX tensor views are globally invalidated by articulation removal.
    # Rebuild the remaining scene before reading or controlling the EVT view.
    sim._sim.reset()
    m.robot.initialize(spawn_pose=[*initial_robot['root_position_m'][:2], initial_robot['yaw_rad']])
    sim._scene.write_data_to_sim()
    sim._sim.forward()
    sim._scene.update(dt=0.)
    restored_robot = m.robot.observe()
    assert np.allclose(restored_robot['root_position_m'], initial_robot['root_position_m'], atol=1e-5, rtol=0)
    assert abs(restored_robot['yaw_rad'] - initial_robot['yaw_rad']) < 1e-5
    import warp
    dampings = warp.to_torch(m.robot.robot.root_view.get_dof_dampings())[:, m.robot.wheel_joint_ids].tolist()
    forces = warp.to_torch(m.robot.robot.root_view.get_dof_max_forces())[:, m.robot.wheel_joint_ids].tolist()
    assert dampings == m.report['robot_effective_wheel_damping']
    assert forces == m.report['robot_effective_wheel_max_force']
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1]); np.random.set_state(rng[2]); random.setstate(rng[3])

    m.mesh.update = lambda: None
    m.world.planar = lambda: torch.empty((0,4),device=m.world.device)
    m.count = 0
    m.source = dict(m.source, actor_count=0)
    m.world.state = lambda: empty
    m.world.num_envs = 0
    m.start = empty.root_pos.clone()
    m.first_joints = empty.dof_pos.clone()
    m.max_displacements = torch.zeros(0, device=m.world.device)
    m.route_indices = torch.zeros(0, device=m.world.device, dtype=torch.long)
    m.runtime.advance_route = lambda xy, indices: (indices, torch.zeros(0, dtype=torch.bool, device=xy.device))
    m.runtime.action = lambda world, control, indices: (torch.empty((0, empty.dof_pos.shape[1]), device=world.device), {'replanned': False})

    query_checked = [False]
    def step(action):
        assert action.shape[0] == 0
        # Same substep order and dt as the original simulator._physics_step.
        for _ in range(sim.decimation):
            m.robot.pre_physics_step()
            sim._scene.write_data_to_sim()
            sim._sim.step(render=True)
            sim._scene.update(dt=dt)
        if not query_checked[0]:
            hits = []
            def collect(hit):
                for key in ('collision', 'rigidBody'):
                    path = str(hit.get(key, '') if isinstance(hit, dict) else getattr(hit, key, ''))
                    if any(path == r or path.startswith(r + '/') for r in roots):
                        hits.append(path)
                return True
            query = m.training_bridge.driver.lidar.query
            for x, y, _z in old_positions:
                for height in (.3, .8, 1.3):
                    query.raycast_all((x-1., y, height), (1., 0., 0.), 2., collect, True)
                    query.raycast_all((x, y-1., height), (0., 1., 0.), 2., collect, True)
            assert not hits, 'Removed humanoid colliders still present in PhysX: ' + str(hits)
            record['post_physics_absence_rays'] = 24
            record['post_physics_humanoid_hits'] = hits
            (m.OUT/'NO_PEDESTRIANS_VERIFIED.json').write_text(json.dumps(record, indent=2))
            query_checked[0] = True
        return empty, torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.bool)

    m.world.step = step
    # Keep original path labels in the LiDAR classifier: any stale physical
    # humanoid hit must be detectable rather than relabeled as a static wall.
    record = dict(mode='zero_physical_pedestrians', removed_roots=roots,
                  active_pedestrian_count=0, observation_pedestrian_count=0,
                  no_reward_formula_change=True, static_and_robot_colliders_unchanged=True,
                  retained_collider_count=len(before), physics_dt=dt,
                  control_dt=m.world.dt, decimation=sim.decimation,
                  installed_before_first_transition=True,
                  physics_views_rebuilt_before_training=True,
                  initial_robot_pose_and_wheel_parameters_verified=True,
                  common_initialization_retained=True,
                  original_task_improvement_claim=False)
    (m.OUT/'NO_PEDESTRIANS_VERIFIED.json').write_text(json.dumps(record, indent=2))
    m.report.update(pedestrian_ablation=record, route_vertices_reached_per_actor=[],
                    training_source='diagnostic zero-pedestrian scene; real EVT wheels',
                    pedestrian_observation='empty (0,4); no oracle pedestrians')
