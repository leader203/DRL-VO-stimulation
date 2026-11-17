# THIRD 场景

从仓库根目录启动：

```bash
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py run \
  src/pedstrain_simulation/simulation/configs/third
```

`simulation.yaml` 当前引用 `/project/Third_Floor/THIRD_1.usd`，通过 `scene.layers` 加载
`scene_overrides.usda`。覆盖层的来源元数据仍记录原 `navigation_data_synthesis/assets/THIRD.usd`；
它与当前场景的几何及门控兼容性需要另行运行验证。原始 USD 不被改写；只打开原始 USD 不会应用这份修正。
行人数量、路线、停留和运行时长以自己的 YAML 为准。字段见[行人配置参考](../../docs/pedestrian_yaml.md)。
人数由 `actors` 条目推导；退出或运行失败时，状态写入本工程的 `output/status.json`。
本工程已显式选择 `navigation.backend: ros2`，引用[共享 ROS 参数](../runtime/ros2_navigation.yaml)。
ROS 执行路线、避碰与积分，Isaac 负责物理地图与视觉；原计划、时长、种子、物理和场景设置保留。
## 碰撞和门覆盖层

覆盖层中的碰撞设置仍按当前合成结果保留；ROS 门适配器不主动恢复已关闭的碰撞。
ROS 使用[独立门协作](../../docs/ros2_navigation.md#门与导航协作)，
启用 ROS 行人时，临时组合层按 `automaticDoor:joint` 元数据识别控制器根，
将根内同时包含 `automaticDoor:` 与 `DriveAPI` 的旧内联门脚本替换为空操作，避免重复驱动。
这不依赖脚本导入 IRA/NavMesh；源 USD 不改写，无行人模式不应用这项门所有权规则。
独立回归已覆盖此识别边界，本场景的真实门开关和通行仍需原生复验。

### 隔间门与门框

| 文件 | 内容 |
|---|---|
| `simulation.yaml` | 场景、行人路线与修正层引用 |
| `scene_overrides.usda` | 4 扇门、固定门框与隔板的 USD 覆盖，以及自包含的 Script Node 门控脚本 |

门接近检测和关节驱动逻辑直接保存在覆盖层的 `/World/AutomaticDoors/Graph/Controller.inputs:script`。
脚本只依赖 Kit/Isaac Sim SDK，不导入项目 Python 包；通用 `simulation/scene` 仅负责按配置装配场景。
修改场景专用行为时编辑这份覆盖层，不向通用仿真代码添加场景名称、prim 路径或特殊判断。

按用户选择，**4 扇隔间门的门扇及门把手碰撞永久关闭，保留可见模型和开关动画**。
播放、停止和离开触发区不会重新启用门扇碰撞。门扇不再物理阻挡或推挤行人、机器人。
门框、固定填充板和隔板仍然是实体障碍；不修改行人的路线来规避门扇扫掠范围。

修正层中的 `/World/AutomaticDoors` 属性控制触发与动画：

| 属性 | 当前值 | 含义 |
|---|---:|---|
| `automaticDoor:approachDistance` | 0.75 | 行人中心接近门平面的触发距离 [m]，还须朝向门口运动 |
| `automaticDoor:doorwayDepth` | 0.05 | 门口保持开启区的半深度 [m]，检测时加上角色占用半径 |
| `automaticDoor:closeDelay` | 0.35 | 最后一次接近/门口占用检测后的关门延时 [s] |
| `automaticDoor:speed` | 120 | 关节目标角度的变化上限 [deg/s]；实际门角度由 PhysX 驱动 |

实际开门条件和通行结果需用 ROS 场景运行核对。固定门框与隔板的实体碰撞仍由覆盖层提供。
