# ROS 2 行人训练接入

`Ros2TrainingScene` 和 `Ros2PedestrianActionCfg` 将现有 ROS 2 行人控制接入任务工程中的 `ManagerBasedRLEnv`。训练任务持有 Kit、PhysX 和机器人动作。行人动作项占零维，每个物理子步前调用一次桥接器。

训练场景使用米制 Z-up USD，行人需要显式出生位置，并通过物理占用地图的出生和原路线检查。环境配置使用 `num_envs=1`、Isaac Sim PhysX、`sim.create_stage_in_memory=False` 和场景查询。物理频率必须是 ROS 控制频率的整数倍。120 Hz 物理配合 60 Hz ROS 及 `decimation=2` 时，每个策略步有两个物理子步和一次 ROS 更新。

先启动 Kit 并加载 ROS 2 环境，再准备舞台。任务自己的动作、机器人、观测组和训练算法照常配置。

```python
from pathlib import Path

import omni.usd
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg
from isaaclab.sim import use_stage

from pedstrain_simulation.simulation import load_simulation_config
from pedstrain_simulation.simulation.rl import (
    Ros2PedestrianActionCfg,
    Ros2TrainingScene,
    pedestrian_observations,
)

# app 是已启动的 Kit，env_cfg 由机器人任务提供。
config = load_simulation_config(Path("path/to/simulation.yaml"))
env = None
stage = None
with Ros2TrainingScene(config, max_pedestrians=8) as prepared:
    try:
        app.run_coroutine(prepared.setup())
        stage = prepared.stage
        env_cfg.sim.physics_prim_path = prepared.physics_prim_path
        env_cfg.sim.create_stage_in_memory = False
        env_cfg.sim.enable_scene_query_support = True
        env_cfg.actions.pedestrian_clock = Ros2PedestrianActionCfg(asset_name="robot")
        env_cfg.observations.policy.pedestrians = ObservationTermCfg(func=pedestrian_observations)
        with use_stage(stage), prepared.bind():
            env = ManagerBasedRLEnv(env_cfg)
        obs, info = env.reset()
        # 用任务的机器人动作调用 env.step(action)。
    finally:
        try:
            if env is not None:
                try:
                    env.pedestrian_bridge.close()
                finally:
                    env.close()
        finally:
            context = omni.usd.get_context()
            if stage is not None and context.get_stage() == stage:
                app.run_coroutine(context.close_stage_async())
```

- `pedestrian_observations` 返回 `[1, 7 × max_pedestrians]`。行人按名称排列，每槽是机器人朝向坐标下的位置 [m]、相对速度 [m/s]、身体半径 [m]、平面净距 [m] 和有效标记。空槽清零。
- `pedestrian_clearance`、`pedestrian_proximity_cost` 和 `pedestrian_collision` 可用作任务项。它们以平面圆形轮廓计算，PhysX 接触传感和终止策略由任务选择。
- 手动与自动回合重置保留行人路线、状态和 ROS 序号，只清除动态障碍速度历史。任务负责在安全位置重置机器人。ROS 请求失败后，桥接器拒绝继续步进，需释放并重建该训练场景。
- 行人观测取最近一次 ROS 控制批次后的状态。训练桥使用独立的会话命名空间，不发布全局 `/clock`。CUDA PhysX 通过刚体张量视图同步 ROS 校验后的碰撞体位姿，奖励计算仍按任务指定的传感与几何语义执行。

从仓库根目录运行真实管理器冒烟：

```bash
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/tests/run_ros2_rl.py \
  --output /tmp/ros2_rl_smoke
```

冒烟使用受力驱动的球形机器人和一个行人。输出包括策略步、物理子步、DDS 批次、重置、实测物理位姿和资源清理记录。完整训练任务可以使用不同的机器人、观测和奖励。
