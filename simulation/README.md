# 行人场景仿真

行人仿真采用 **ROS 2 导航进程 + Isaac Sim 物理地图与人物视觉**。
ROS 节点执行栅格 A*、SFM 期望速度、内置 RVO2 避碰及 YAML 路线调度/积分；
Isaac 基本应用采样真实碰撞体并显示角色，不启用 IRA、行为代理或 NavMesh。
场景、角色和动作库继续引用原资源，通用代码不按场景名称或固定路点分支。
控制责任、协议和限制见 [ROS 2 导航](docs/ros2_navigation.md)。
操作指南和按运行区分的验收记录见[文档索引](docs/index.md)。

## 启动现有场景

从仓库根目录启动当前四人 `third_3` 的 ROS 2 窗口：

```bash
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py run \
  src/pedstrain_simulation/simulation/configs/third_3
```

无窗口、录像和场景验收记录见 [THIRD_3 启动指南](configs/third_3/README.md)。
迁移机器时，先核对导航配置中的 `asset_root` 和 `ros_setup_path`。内置场景通过共享角色清单引用本机 Isaac 资源，设置方法见[配置索引](configs/README.md#角色资源清单)。

## 创建并运行自己的项目

从仓库根目录创建项目；`init` 和 `check` 不启动 SDK，也不要求预先加载 ROS：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py init \
  src/pedstrain_simulation/simulation/configs/my_room \
  --scene /absolute/path/to/room.usd --characters /absolute/path/to/characters --count 3
```

CLI 默认生成 `simulation.yaml` 与 `navigation_settings.yaml`，选择 `navigation.backend: ros2`
和 `navigation.waypoint_frame: stage_world`。出生点和路点填写已组合的 Isaac 场景世界坐标，
采用舞台原生单位和上轴；带父变换对象的 local translate 不能直接当 world 坐标。
已有目录不覆盖，资产不复制。默认人口随机出生并漫游；`--behavior standing` 生成待机计划。
编辑 YAML 中的出生点、行路点、开始时间、逐点 `wait` / `yaw` 后运行：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py workflow \
  src/pedstrain_simulation/simulation/configs/my_room --record
```

`workflow` 默认隐藏 Isaac 窗口，`--no-headless` 显示窗口，`--no-record` 只运行并评估。
工作流为子进程加载所选 ROS 环境，串行执行预检、仿真、坐标核对和轨迹评估，在新目录写入 `workflow_report.json`。
实时 RViz 须在该会话仍活动时连接它的 `monitor.json`。选项与输出见[自动工作流](docs/workflow.md#自动运行核对与评估)和[监控说明](docs/rviz_monitoring.md)。

`check PROJECT` 和 `run PROJECT --dry-run` 可离线核对原始行人计划、位置框架、参数来源和 ROS 净空。
在 `run` 或 `workflow` 中用 `--navigation-settings FILE` 临时试用另一套 ROS 参数。
地图、实际通行和动画由实际仿真验证。目录入口和路径规则见[配置索引](configs/README.md)。

## 按需配置

| 需求 | 入口 |
|---|---|
| ROS 设置、物理地图、启动与排错 | [ROS 2 导航](docs/ros2_navigation.md) |
| RViz2 地图、路径、实测行人与门状态监控 | [只读监控](docs/rviz_monitoring.md) |
| 工程入口、默认来源与共享参数 | [配置索引](configs/README.md) |
| 路点、停留、朝向、往返及角色模型字段 | [行人 YAML](docs/pedestrian_yaml.md) |
| 米/厘米、舞台轴、ROS 原点和四元数 | [ROS 坐标合同](docs/ros2_navigation.md#坐标地图与动画) |
| 真实门状态、有限退让与原路线恢复 | [门与导航协作](docs/ros2_navigation.md#门与导航协作) |
| CPU 回归和真实场景验收 | [测试说明](tests/README.md) |
| 仿真录像 | [录制](recording/README.md) |
| Isaac Lab 强化学习任务接入 | [ROS 2 训练桥](rl/README.md) |
| 旧行人场景与训练配置迁移 | [迁移指南](docs/custom_scenes.md#从旧格式迁移) |

ROS 配置是独立的平铺 YAML，不使用旧 `pedestrian_settings.yaml` 的原生避障开关。
此方案通过标准 ROS 消息交换地图、时钟和路径，通过有序 JSON 批次交换人物状态与命令；
它没有提供 Nav2 的 `NavigateToPose` 动作服务器或 `cmd_vel` 机器人执行器。

## Python 入口

```python
from pathlib import Path
from pedstrain_simulation.simulation import load_simulation_config, run_simulation

config = load_simulation_config(Path("src/pedstrain_simulation/simulation/configs/my_room"))
run_simulation(config)
```

Python 创建新工程时可用 `create_simulation_project(..., waypoint_frame="stage_world")` 明确选择舞台坐标。
配置省略 `navigation.waypoint_frame` 时，位置按 `world_m` 解释。
完整读取文件使用 `load_simulation_config`，覆盖时基于已加载对象及其 `settings` 使用 `dataclasses.replace`。
