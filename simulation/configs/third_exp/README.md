# THIRD_EXP 场景

[simulation.yaml](simulation.yaml) 引用 `pedstrain_simulation/assets/THIRD_EXP.usd`，当前配置 **4 人**，
按 YAML 中各自的出生点、路线和停留设置往返行走，运行 **60 秒**，随机种子为 **42**。
本配置使用项目自定义 YAML；场景文件保持原样，门控、碰撞和导航排除区使用资产自身的定义。
当前已显式选择 `navigation.backend: ros2`，并引用[共享 ROS 参数](../runtime/ros2_navigation.yaml)。
ROS 执行路线、避碰与积分，Isaac 负责物理地图与视觉；原计划、时长、种子、物理和场景设置保留。
ROS 不使用资产中的 NavMesh 排除区；门按[元数据协作合同](../../docs/ros2_navigation.md#门与导航协作)处理。

## 启动与检查

从仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python src/pedstrain_simulation/simulation/run.py run src/pedstrain_simulation/simulation/configs/third_exp
```

将命令中的 `run` 换成 `check`，可在启动前检查配置与本地资源；离线检查不证明运行时路线可达性。
正常退出或运行失败时，状态写入本目录的 `output/status.json`。
当前后端选择不代表本场景已完成通行验收；物理地图、动画与整条路线仍需独立验证。

## 修改配置

| 文件与字段 | 作用 |
|---|---|
| `simulation.yaml` → `scene.asset_path` | 场景 USD 路径，相对当前 YAML |
| `pedestrians.actors` | 人数由条目数推导，当前为 4；维护逐人出生点、路线、停留、朝向和往返 |
| `pedestrians.defaults.gender` | 可选 `any`、`male`、`female`；未指定时不限制性别 |
| `simulation.duration` / `seed` | 运行时长 [s] 与随机种子 |
| `navigation.settings_path` | 当前共享的平铺 ROS 地图、身体、控制与通信参数 |

需要指定出生点、路线、逐点停留及朝向时，参阅[配置工作流](../../docs/workflow.md)
和[完整字段参考](../../docs/pedestrian_yaml.md)。移除 `defaults.behavior: wander` 或在路线角色上设置
`behavior: walk`；若保留 `count`，其值必须与 `actors` 条目数一致。路径和坐标必须按当前场景重新定义。
