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

## DRL-VO PPO 训练

`simulation/train_drl_vo.py` 在现有 ROS 2 舞台上直接训练：奖励、终止和观测全部来自本包（`rl/rewards.py` 的四个分量、`rl/drl_observations.py` 的 19,202 维观测），PPO 超参与 DRL-VO 的 `TRAINING_CONFIG.json` 一致（`n_steps=512`、`batch=128`、`epochs=10`、`lr=1e-3`、`gamma=0.99`、`gae=0.95`、`clip=0.2`、`ent=0`、`vf=0.5`、`max_grad_norm=0.5`、Adam `eps=1e-5`）。

```bash
OMNI_KIT_ACCEPT_EULA=YES python simulation/train_drl_vo.py \
  --task-config simulation/configs/third_3/simulation.yaml \
  --device cuda:0 --total-steps 480000 --output runs/drl_vo_ppo
```

`--warm-start path/to/frozen_policy.pt` 用已发布的 DRL-VO 权重初始化策略。没有 ROS 2 时用 `simulation/configs/third_3/no_crowd.toml` 冒烟：它跳过行人桥接，观测、奖励和终止照常计算。

### 设备分工

| 参数 | 含义 | GPU 运行时的取值 |
| --- | --- | --- |
| `--device` | 跑 PhysX 仿真和策略网络的 GPU | `cuda:0` |
| `--sim-device` | 环境 torch 缓冲区所在设备 | 默认 `cpu`，`auto` 表示跟随 `--device` |

Isaac Sim 4.5 的 PhysX 张量接口返回主机张量，`set_root_transforms` 只接受 CPU 索引，因此 GPU 运行时缓冲区留在主机侧：物理和网络在 GPU 上计算，位姿读写走主机侧。把 `--sim-device` 强行设成 `cuda:0` 会在第一次复位时报设备不一致；`--device cpu` 则把物理也压回主机，吞吐下降约一个数量级。

### 实测吞吐

单张 RTX A6000、单环境：策略步约 66 ms（物理子步 26 ms、观测 4 ms、终止 12 ms、策略前向 19 ms），一次 512 步迭代约 47 s（`steps_per_s ≈ 11`）。舞台上首次构建 0.02 m 占用栅格（593×313）约 2 分钟，与训练步数无关，长时间训练可忽略。
### Isaac Sim 6.2 / Isaac Lab 3.0 环境

项目自带的 `simulation/configs/runtime/ros2_app.kit` 面向 Isaac Sim 6.2，机器上原有的 Isaac Sim 4.5 + Isaac Lab 0.54 因为缺 `omni.physics.stageupdate` 起不来它。为此另装了一个隔离环境 `isaaclab62`（Python 3.12.16、Isaac Sim 6.2.0.0、Isaac Lab 3.0.0、warp-lang 1.17.0），原有的 `isaaclab` 环境保持不动：

```bash
/workspace/miniconda3/envs/isaaclab62/bin/python simulation/train_drl_vo.py \
  --task-config simulation/configs/third_3/no_crowd.toml \
  --launcher isaaclab --device cuda:0 --sim-device cpu \
  --total-steps 1024 --n-steps 512 --epochs 10 --batch-size 128
```

三点须知：

- **启动器**：`--launcher isaaclab` 走 `AppLauncher → isaaclab.app.launch_simulation → isaaclab_physx.app.KitLauncher`，使用 Isaac Lab 3.0 自带的 experience。`--launcher project` 需要 `isaacsim-extscache-kit-sdk`，而 NVIDIA 只为 aarch64 和 Windows 发布了 wheel，x86_64 Linux 上只有空 sdist，`pip` 会卡在建元数据，因此该路径在 x86_64 上不可用。
- **`--sim-device cpu` 是必须的**：Isaac Sim 6.2 的 PhysX 张量接口在这台机器上仍返回主机张量——`/physics/suppressReadback`、`/physics/fabricUseGPUInterop`、`/physics/cudaDevice=0` 全部打开，甚至关掉 Fabric 直读张量接口（`--no-fabric`），`get_root_transforms()` 依旧在 CPU。Isaac Lab 3.0 把位姿读写做成 warp kernel，kernel 的设备取自 articulation 的设备，因此 `--sim-device cuda` 会在第一次复位时报 `expects an array of type array(ndim=1, dtype=transformf), but passed value has type ProxyArray`。物理与策略仍在 GPU（`--device cuda:0`），只有缓冲区在主机侧。
- **`--kit-arg SETTING`** 透传任意 Kit 设置，例如 `--kit-arg=--/physics/suppressReadback=true`。

#### isaaclab_physx 3.0.0 的补丁

`ArticulationData.update` 在每个物理步刷新 `joint_acc`，该访问器启动的 `get_joint_acc_from_joint_vel` warp kernel 在 6.2 上触发 CUDA error 700（非法访存），任何环境都建不起来。DRL-VO 的奖励、观测和差速动作都不读关节加速度，`simulation/tools/patch_isaaclab_physx.py` 去掉这次刷新（属性仍按需计算），`--revert` 可还原。

#### 复位位姿

任务不 spawn 机器人：Isaac Lab 的 `default_root_pose` 来自 `cfg.scene.robot.init_state`（单位阵），而机器人实际位置由 `simulation/configs/*/simulation.yaml` 的 `robots[].position` 在合成舞台时写入。按默认值复位会把机器人搬到世界原点（那一格在墙里），碰撞预算三步就耗尽，于是每个回合只活 3~4 步。`capture_spawn_state` 作为 startup 事件记录舞台上的真实位姿，`reset_robot_default` 每次复位写回它，回合才能跑满 `--episode-steps`。

#### 实测（Isaac Sim 6.2 / Isaac Lab 3.0，RTX A6000）

| 场景 | 每步耗时 | 吞吐 |
| --- | --- | --- |
| 512 步迭代（`--episode-steps 64`） | 0.088 s | 11.3 steps/s |
| 5120 步（10 次迭代）均值 | 0.079 s | 12.9 steps/s |

10 次迭代的 `value_loss` 从 245.7 收敛到 0.20，回合长度稳定在 57~61 步（上限 64）。`--warm-start DRL-VO/frozen_policy.pt` 的 163 个张量全部载入（0 missing / 0 unexpected），报告里的 `warm_start` 字段会记下来源与张量数。首次构建 0.02 m 占用栅格仍是约 2 分钟，与训练步数无关。
