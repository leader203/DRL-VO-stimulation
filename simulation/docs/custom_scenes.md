# 配置迁移与复用

现行行人模拟由 ROS 2 控制。场景入口使用本项目的 `schema_version: 1` YAML，也可使用通过 `scenario` 引用它的 TOML 包装。场景不依赖特定文件名、门名或固定坐标。

## 适配不同场景

每个场景使用自己的 USD、人口与几何覆盖层。

| 差异 | 配置位置 |
|---|---|
| 场景 USD 与碰撞、交互覆盖 | `scene.asset_path`、`scene.layers` |
| 人数、模型、出生点与路线 | `pedestrians.count`、`defaults`、`actors` |
| 路点位置输入框架 | `navigation.waypoint_frame` 选 `stage_world` 或 `world_m` |
| 身体、地图、ROS 设置和动态对象 | `navigation.settings_path` 所选的导航 YAML |
| 窗口、时长、录像与机器人 | `simulation`、`recording`、`robots` |

`stage_world` 取已组合父变换的 Isaac WORLD 坐标，使用舞台的单位和上轴。`world_m` 取规范米制 Z-up 世界位置，字段省略时使用该框架。速度、到达容差和地图尺寸均用 SI。出生 `yaw` 相对于模型作者朝向，路点 `yaw` 表示到达后的世界朝向。见[单位与坐标](units.md)。

ROS 地图要求单层可通行碰撞与地面支撑。自动门需要有效的 [`automaticDoor:*` 元数据](ros2_navigation.md#门与导航协作)，其几何、开闭状态和运动空间都须在本场景核对。活动 PhysicsScene 需唯一。指定步频或求解器时，该场景还须可编辑。机器人资产及明确声明指标的覆盖层须匹配环境长度、质量和上轴。

运行时在可编辑的场景根下分配独立角色容器。它按当前几何生成物理地图。门元数据、动态对象根和物理覆盖层须指向当前舞台的真实对象。复制另一个场景的对象路径、路线或 NavMesh 排除区无法建立当前地图。

迁移机器时，核对导航设置中的 `asset_root` 与 `ros_setup_path`，并核对角色资源和 USD 内部引用。[角色清单](pedestrian_yaml.md#角色资源清单)的角色路径与默认动作库都使用本地资产根。也可在场景中直接指定文件相对的角色 USD 或 `pedestrians.motion_library`。

`check PROJECT --dependencies` 可核对声明的引用。运行 `workflow PROJECT --require-route-goals` 后，再核对实际地图、原目标覆盖、通行和可见动画。见[工作流](workflow.md#自动运行核对与评估)。

## 从旧格式迁移

选择或生成 ROS 2 项目，沿用经确认的场景、资产和逐人计划。内置场景使用各自的 `simulation.yaml`。

| 旧入口或字段 | 现行用法 |
|---|---|
| `ira.toml` 或 `navigation.backend: ira` | 选择场景目录或 `simulation.yaml`；配置中的后端为 `ros2` |
| TOML 的 `ira_config` | 改用 `scenario`，指向本项目场景 YAML |
| `native_ira_settings`、`pedestrians.settings_path` | 移除旧选择，通过 `navigation.settings_path` 选择 ROS 2 导航参数 |
| `pedestrians.root_prim_path` | 使用会话分配的人物根，实际路径见运行状态 |
| IRA `environment.base_stage_asset_path`、角色组、`spawn_positions` | 分别填写 `scene.asset_path`、`pedestrians.actors` 和角色 `position` |
| IRA 的 `stop`、`wander`、`patrol.path_points` | 分别用 `behavior: stand`、`behavior: wander`、有序 `waypoints` |
| `run_ira_simulation`、`setup_ira_simulation` | 独立仿真使用 `run_simulation(load_simulation_config(path))` |
| 依赖上述 IRA 导入的旧 `spawn_simulation_sence.py` 脚本 | 从仓库根目录使用 `src/pedstrain_simulation/simulation/run.py run PROJECT` |
| IRA 训练桥、`TrainingScene`、`PedestrianClockActionCfg` | 使用 [ROS 2 训练桥](../rl/README.md) 的 `Ros2TrainingScene` 与 `Ros2PedestrianActionCfg` |
| `--legacy-example`、`--native-ira-settings`、`--pedestrian-settings` | 传入 ROS 2 项目目录或配置文件，导航参数可通过 `--navigation-settings FILE` 选择 |

旧行为树、原生触发器和 SDK 的加权 routines 需要按现行字段重新描述。行人启动时刻、点位、停留、朝向和往返用 `start_time`、`waypoints` 与 `repeat` 表达。IRA 的避让和身体参数须依据 ROS 物理地图与角色尺寸重新配置。完整字段见[行人 YAML](pedestrian_yaml.md)。

## TOML 包装与人口复用

同一份场景可通过不同包装选择无窗口或录像等运行方式。包装先继承 YAML，再覆盖明确字段。`scene.layers` 追加，`robots=[]` 清空继承机器人。

```toml
scenario = "simulation.yaml"

[simulation]
headless = true
duration = 120.0

[pedestrians]
config_path = "other_population.yaml"
```

`other_population.yaml` 使用完整场景格式，也可通过另一份 TOML 指向它。替换只取其人口，不读取下一层替换。角色点位按主工程的最终输入框架解释。身体和避碰参数来自主工程选择的导航设置。

旧参数式 CLI 在已有场景来源时，`--pedestrians-config FILE` 也只替换人口。单独使用时，完整项目 YAML 或包装 TOML 提供会话基底。
