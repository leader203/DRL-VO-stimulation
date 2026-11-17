# 配置工作流

日常完整流程是 **创建一次 → 编辑 YAML → workflow**：自动检查、运行、核对坐标并评估轨迹。
只需交互仿真时仍可使用 `run`，它自动检查最终配置后启动。
`check` 与 `--dry-run` 用于需要单独查看配置时，无需每次重复执行三条命令。
命令从仓库根目录执行，使用现有 `env_isaaclab`，不重新同步依赖；不会复制场景、模型或纹理。
文件之间如何引用见[配置依赖图](../configs/README.md#配置依赖关系)；参数字段集中在[行人 YAML 参考](pedestrian_yaml.md)。
仿真使用 ROS 2。`workflow` 为原生子进程自动加载配置的 ROS setup。
直接启动启用了行人的 ROS 2 项目时，先在 shell 加载所选设置中的 `ros_setup_path`，当前包级路径为 `/opt/ros/jazzy/setup.bash`。
地图、通信与运行边界见 [ROS 2 导航](ros2_navigation.md)。

## 创建项目

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py init \
  src/pedstrain_simulation/simulation/configs/my_room \
  --scene /absolute/path/to/room.usd --characters /absolute/path/to/characters --count 3
```

目标目录必须不存在。CLI 默认生成 `simulation.yaml` 与 `navigation_settings.yaml`，并明确 ROS 后端；
角色随机出生并漫游。`--behavior standing` 改为待机。`--characters` 可指定单个角色 USD、角色目录或[角色清单](pedestrian_yaml.md#角色资源清单)。
场景需具有真实实体碰撞和地面支撑。
CLI 新 ROS 项目显式生成 `navigation.waypoint_frame: stage_world`。出生点和路点填写
Isaac 已组合的 WORLD 位置 `[x,y,z]`，使用舞台单位/上轴；不要抄带父变换对象的 local translate。
已有规范米制 Z-up 数据可选 `world_m`；省略字段的旧配置保留这个兼容默认。
速度、到达容差、地图参数和相机参数仍用 SI，详见[单位与坐标](units.md)。
新场景的出生点、路线、对象排除路径和几何覆盖层都在自己的配置中定义，不继承历史三楼数据。
几何根节点、运行时人物容器和环境配置边界见[跨场景适配](custom_scenes.md#适配不同场景)。

| 创建选项 | 结果 |
|---|---|
| 默认 | ROS 场景和导航参数两文件，使用专用基本应用 |
| `--navigation-settings FILE` | ROS 直接引用已有导航 YAML，不生成参数副本 |
多个工程可共享同一导航设置。生成器在创建目录前检查显式输入。
共享文件修改后，各工程在下一次加载时采用新值，配置不热加载。
生成的参数文件是独立副本，修改包级参数不会同步到已有工程；[复制与引用说明](../configs/README.md#生成时复制与运行时引用)列出了各选项的影响。
迁移机器时核对导航设置中的 `asset_root` 与 `ros_setup_path`，并核对场景中的角色资源、动作库及 USD 内部引用。

成功时 stdout 只有一个 JSON 对象：

```json
{"simulation_config": "/absolute/project/simulation.yaml"}
```

stderr 给出实际编辑文件、参数来源和当前 Python 解释器对应的 Check/Run 命令。
脚本只解析 stdout，不合并 stderr。

## 自动运行核对与评估

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py workflow \
  src/pedstrain_simulation/simulation/configs/my_room \
  --output /absolute/path/to/new_run --record --require-route-goals
```

此入口要求正的有限仿真时长。`--output` 必须是全新目录；
省略时在项目的 `output/workflow_<随机后缀>` 新建。`--record` / `--no-record` 只覆盖本次录像选择，
省略时沿用 YAML；已配置的 FPS、预热与相机保留，录像目的地改为新目录的 `video.mp4`。
`--no-headless` 显示窗口并开启视口，即使入口是关闭视口的无窗口包装。
仅使用 `--no-record` 会保留入口的视口设置。无窗口且无视口的验收可使用 [headless.toml 示例](../configs/third_3/headless.toml)的包装方式。
`--navigation-settings FILE` 为启用行人的 ROS 2 工程选择本次使用的导航 YAML。工作流用最终文件生成 `preflight.json` 和子进程包装，并将该文件纳入声明输入哈希。
`--require-route-goals` 要求在原配置时长内到达每个原始行走路点；不指定时仍报告覆盖情况。
`--timeout-s` 限制本次拥有的原生进程组墙钟运行时间，不改变仿真时长；超时只终止该进程组。
原生主进程退出、超时或中断由外层程序处理时，工作流会清理该次创建的进程组残余子进程，
包括主进程先退出后仍存活的 ROS 节点；其他独立进程不受影响。
清理失败会使工作流失败；已有运行异常时，清理错误附在原异常说明中。

父进程不导入 SDK，只在独立 shell 中加载 `ros_setup_path`，将得到的环境交给原生子进程；
它不安装 ROS、编译器、ffmpeg 或其他依赖。原生子进程复用当前解释器，整个工作流按顺序执行：

1. 保存最终配置检查 `preflight.json` 和仅覆盖本次运行的 `workflow.toml`，原 YAML 不改。
2. 原生运行与可选视频编码，保存 `native.log`、`status.json` 及实际运行目录。
3. 校验原输入、规范世界、ROS map 与原生 stage world 坐标，以及出发时间、路点停留和朝向的转换，输出
   `alignment/coordinate_alignment.json` 和独立 SVG。图上的虚线只连接原路点，不是规划或实际轨迹。
4. 输出 `evaluation.json`，核对实际位移、净距和路点覆盖。`waypoint_visits` 记录每次实测到达、目标位置和停留区间，
   报告截至最后一帧的已观测停留。最后写入 `workflow_report.json`。

报告保留源码/声明输入哈希和各阶段结果；声明的目录不会递归当作已哈希文件。
禁用行人时不建立导航地图，坐标/轨迹阶段跳过；运行或后处理失败保留已有证据并返回失败，
不能把 `preflight` 通过当作整个工作流通过。是否通过以本次最终报告为准。
外层工作流被强制终止时，`workflow_report.json` 可能缺失。核对已保存的 `status.json`、录像元数据和运行轨迹，分别判断原生执行和录像是否完成。

## 从随机漫游改为指定路线

日常修改 `simulation.yaml`。保留已生成的 `scene`、`simulation` 和资源路径，
将 `pedestrians` 中的 `count` 和旧 `defaults` 替换为完整的具名角色配置。
**若由 standing 预设生成，必须移除 `defaults.behavior: stand`，或在路线角色上显式设置 `behavior: walk`。**

以下行人块使用工程自己的 `assets` 和顶层 `navigation`。将示例坐标换成目标场景的实际可通行坐标。

```yaml
pedestrians:
  assets: ../assets/characters
  defaults:
    speed: 0.9
    yaw: 0.0
  actors:
    walker_01:
      position: [1.0, 2.0, 0.0]
      start_time: 3.0
      repeat: true
      waypoints:
        - position: [1.0, 3.0, 0.0]
          wait: 2.0
          yaw: 90.0
        - position: [2.0, 3.0, 0.0]
          wait: 4.0
          yaw: -90.0
    walker_02:
      behavior: wander
    walker_03:
      behavior: stand
```

人数按三个 `actors` 条目自动确定；若保留 `count`，必须填写 `3`。
路线角色有初始 `position` 与非空 `waypoints`，且未被默认行为覆盖，因此自动选择 `walk`。
其他两人随机出生后执行各自行为。每人可配置自己的模型，见[完整混合示例](pedestrian_yaml.md#完整混合示例)。
需要指定性别时，在 `defaults` 或单个角色下设置 `gender: male` / `female`；`any` 表示不限，规则见[性别与选模](pedestrian_yaml.md#行人性别与模型选择)。

`start_time` 是行为开始时间，所有角色在初始化时已经出生并占位。
ROS 默认保留明确出生坐标。仅在 `spawn_tolerance_m` 为正且原出生点无效时，才在指定半径内寻找可用点。
随机点在身体可通过的空闲格采样，内部路径保留用户路点。
往返、朝向、容差和重置的精确语义见[行人字段参考](pedestrian_yaml.md)。

## 运行与按需预览

直接启动前，先加载该工程所选的 ROS setup。

```bash
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py run \
  src/pedstrain_simulation/simulation/configs/my_room
```

| 目的 | 入口参数 |
|---|---|
| 自动运行、坐标核对与评估 | `workflow PROJECT` |
| 交互启动，自动检查后运行 | `run PROJECT` |
| 无窗口、无视口运行 | `run PROJECT --headless --no-render-viewport`，适用于录像关闭的配置 |
| 只检查配置并查看逐人计划 | `check PROJECT` |
| 预览本次启动覆盖 | `run PROJECT --dry-run --headless --duration 30` |
| 预览另一套 ROS 导航参数 | `run PROJECT --dry-run --navigation-settings FILE` |
| 只看配置与资源引用 | `check PROJECT --dependencies` |

`check` 与 `run --dry-run` 检查配置、人数、数值和本地资源。ROS 固定出生点诊断会在本地单位和上轴可用时核对身体间距。
启用录像时还检查新输出名字和编码工具。占位图、可达性和动画需要实际运行验证。

`run` 在应用显式 CLI 覆盖后检查同一份最终配置，再启动仿真。
覆盖只作用于该次命令，不写回 YAML。长期使用的窗口、时长与录像选项写入 YAML，日常只需 `run PROJECT`。
预览中的临时覆盖不会自动应用到下一次运行。
直接从关闭视口的包装恢复 GUI 时使用 `--no-headless --render-viewport`。工作流使用 `--no-headless` 即可。

### 试用导航参数

从仓库根目录用候选导航文件预览同一份场景和路线。

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py run \
  src/pedstrain_simulation/simulation/configs/my_room \
  --dry-run --navigation-settings /absolute/path/to/navigation_test.yaml
```

`--navigation-settings` 的相对路径以当前终端目录为准。此选项仅适用于已启用行人的 ROS 2 配置，优先于场景和 TOML 包装选中的导航文件。
它替换整份文件选择，遗漏字段使用类型默认值。选中的文件会在启动前校验，覆盖仅作用于本次命令。
确认预览后，加载候选文件指定的 `ros_setup_path`，再去掉 `--dry-run` 启动。
自动验收可用 `workflow PROJECT --navigation-settings FILE`，由工作流加载相应 ROS 环境。

### 选择工程入口

PROJECT 可以是目录或显式 YAML/TOML。目录仅选择本层唯一的 `simulation.yaml` 或 `simulation.toml`；
两者同时存在、都不存在或入口不是文件均拒绝，不递归搜索其他场景或录像配置。
`run PROJECT` 不与另一个 `--simulation-config`、`--config` 或 agent 配置组合。

如果终端已经位于自己的场景工程目录，可以省略 PROJECT：`run`、`check`、`run --dry-run` 和
`check --dependencies` 都只选择当前目录。使用本仓库的 uv 环境和绝对入口，例如：

```bash
ISAACLAB_REPO=/absolute/path/to/IsaacLab
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT="$ISAACLAB_REPO/env_isaaclab" \
  uv run --project "$ISAACLAB_REPO" --no-sync python \
  "$ISAACLAB_REPO/src/pedstrain_simulation/simulation/run.py" run
```

`--project` 只指定 uv 环境所属仓库，不改变当前场景目录；换仓库位置时相应修改绝对路径。
缺少入口、存在多个入口或入口不是文件时明确报错，不搜索父/子目录，也不记忆上次工程。
显式 `run` / `check` 可使用这一规则。

## 检查逐人计划

`check PROJECT` 和 `run PROJECT --dry-run` 的 `pedestrian_plans` 列出合并默认值与角色覆盖后的完整计划，
可逐人核对模型、出生请求、速度、开始时刻及每个导航点。`schedules` 保留原有紧凑摘要。
例如以下是检查输出的一个片段，不是新的配置格式：

```json
{
  "pedestrian_plans": [{
    "name": "walker_01",
    "position": [1.0, 2.0, 0.0],
    "yaw": 0.0,
    "behavior": "walk",
    "speed": 0.9,
    "start_time": 3.0,
    "repeat": true,
    "waypoints": [{"position": [1.0, 3.0, 0.0], "wait": 2.0, "yaw": 90.0}],
    "arrival_tolerance": 0.15,
    "asset_path": null,
    "gender": "any"
  }]
}
```

`pedestrian_plans_frame` 与 `navigation.waypoint_frame` 标明原始位置框架：
`stage_world` 为舞台 WORLD 单位/上轴，`world_m` 为规范米制 Z-up。
速度为米/秒，到达容差为米，时间为仿真秒，角度为度。点的 `wait` 已包含角色/公共默认值，
点 `yaw: null` 保留到达朝向；角色 `yaw` 仍是相对模型作者朝向的初始旋转。
两种yaw与Y-up的对应关系见[单位与坐标](units.md#出生旋转与路点朝向)。

- `position: null` 表示在初始化时从物理地图采样。
- `asset_path: null` 表示继承公共模型池；非空目录仍是模型池，具体模型在启动时选定。
- `gender` 表示合并后的模型标签筛选条件，不是观测或推断得到的身份信息。
- 明确出生坐标和导航点尚未经过后端的实际几何验证；`repeat` 不展开未知实际出生点的返程。
- 启用行人时，自定义配置为每人生成计划；禁用行人时列表为空。

这是配置预览，不是仿真测量结果。它使用最终选中的人口来源；`--dependencies` 仍只返回依赖摘要。
Python 可用 `inspect_simulation_config(path)["pedestrian_plans"]` 读取同一结果，
或用 `PedestrianPlan.to_dict()` 导出独立、可 JSON 编码的数据副本。
运行 bootstrap 的 `input_plans` 保留原始框架，`world_plans` 是规范米制计划，`plans` 是 ROS map 计划；
录像的 `pedestrian_plans` 使用规范米制，另保留 `input_plans` 与 `coordinate_contract`，不能混读。

## 检查实际参数来源

普通检查 JSON 包含 `simulation_config`、最终 `simulation`、`navigation`、人数与 `schedules`。
`pedestrian_parameters` 用于核对补齐默认值后的导航参数。
`pedestrian_controller` 为 `ros2_sfm_orca` 或 `disabled`。
启用行人时，`navigation.settings_path` 给出显式导航文件，使用包级默认文件时为 `null`。实际选择的文件与来源见 `dependencies.pedestrian_settings`。
`random_spawn_minimum_center_distance_m` 在 ROS 下为身体直径加 `pair_clearance_m`。
禁用行人时，行人参数、避障摘要和派生间距均为 `null`。

启用 ROS 行人时，`check`、`run --dry-run` 和工作流预检的 `ros2_clearance` 汇总以下配置派生量。禁用行人时该对象为 `null`。

| `ros2_clearance` 字段 | 计算与用途 |
|---|---|
| `planning_radius_m` | `body_radius_m + map_clearance_m`，规划与站位的硬净空半径 [m] |
| `preferred_path_radius_m` | `planning_radius_m + preferred_path_clearance_m`，条件允许时的路径净空半径 [m] |
| `fixed_spawn_body_center_distance_m` | `2 * body_radius_m`，固定出生点的身体重叠阈值 [m] |
| `pedestrian_target_center_distance_m` | `2 * body_radius_m + pair_clearance_m`，行人间目标中心距及随机出生最小中心距 [m] |

`fixed_spawn_check` 另用实际填写的固定出生坐标计算间距诊断。地面、场景障碍和实测身体净距由原生验收核对。

只需查看依赖时：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py check \
  src/pedstrain_simulation/simulation/configs/my_room --dependencies
```

此模式仍检查配置和声明的本地资源，省去录像输出新名字与工具发现，`outputs_checked: false` 不代表可启动。
输出中的路径均已解析；主要字段如下：

| 字段 | 用来核对什么 |
|---|---|
| `entrypoint` | 本次实际选择的场景入口 |
| `configuration_files` | 当前解析和选择的配置文件；不含创建模板 |
| `population.path` | 行人来源入口，可能为 YAML 或 TOML 包装；`selection` 为 `scenario`、`population_override` 或 `disabled` |
| `pedestrian_settings.path` | 实际 ROS 导航文件及其选择来源 |
| `runtime_settings.path` | 本次使用的 Kit 应用配置 |
| `runtime_settings.asset_root` | 导航设置中声明的资产根，未声明为 `null` |
| `scene.path` / `assets` | 最终场景路径及声明的资源路径，包含角色清单和解析后的 USD 文件 |

这里的 `explicit` 表示最终主配置已选定文件，可能来自 YAML、TOML 包装或 CLI 覆盖，
不表示文件是复制生成还是多人共用。禁用行人时，人口与行人参数路径均为 `null`。
当人口入口为 TOML 时，其引用的 YAML 会列在 `configuration_files` 中，不能只看 `population.path` 判断是否加载了 YAML。
ROS 的默认动作库和角色清单都通过导航设置的本地 `asset_root` 解析，接受本地目录或 `file://` URI。也可在场景中直接指定本地 `pedestrians.motion_library`。预检、运行和报告路径检查使用同一份解析规则。
迁移机器时核对实际动作库是否存在。
`scene.selection` 是内部来源标签；自定义 YAML 正常也可能显示 `scene_override`，不表示一定使用了 CLI 覆盖。
检查要求声明的场景和角色资源可在本地访问。它不展开 USD 内部引用。

ROS 参数按本次 `--navigation-settings`、最终 `navigation.settings_path`、包级 ROS 文件的顺序选择。
自己的部分设置文件省略字段时继承类型默认值，不与包级文件合并，详见[参数来源与优先级](../configs/README.md#参数来源与优先级)。

## 运行控制与报告

在 `simulation.yaml` 添加：

```yaml
simulation:
  duration: 0.0
  start_paused: true
  console: true
  report_path: output/status.json
```

这是完整 `simulation` 块的示例，合并到已有块中，避免重复 YAML 键。
控制台支持 `play`、`pause`、`status`、`quit`。暂停保留时间和路线进度。重置路线或时间轴时重新启动会话。
`duration: 0` 等待退出，正数按时间轴阈值退出。
CLI 的 `--console`、`--start-paused`、`--report FILE` 可临时覆盖；`--no-console`、`--no-report` 可禁用。
报告不能覆盖输入配置、已声明资产或录像数据，父路径不能是普通文件。
普通状态报告可更新；录像要求全新输出名字。交互仿真可用 Ctrl+C 退出并清理应用。

## 录像

`recording` 配置观察当前 ROS 会话并生成视频。

在 YAML 中设置正时长、偶数渲染宽高，并增加独立的 `recording` 块，例如：

```yaml
simulation:
  duration: 120.0
  headless: true
  width: 1600
  height: 900
recording:
  enabled: true
  output_path: output/native_demo.mp4
  fps: 12
  warmup_s: 2.0
```

录制要求自动播放，关闭初始暂停、控制台与外部控制器；它不发出导航指令。
ROS 的 `fps` 还须不超过 `control_hz` 并整除它；例如 60 Hz 控制可按 12 fps 录制。
相邻 JSON 记录时间戳和采样信息，`*_frames/` 保存 PNG。ROS 的逐帧位姿保存在 `samples.jsonl`，逻辑身体净距见工作流的 `evaluation.json`。
实际 PhysX 净距由独立场景验收测量。
已有 MP4、相邻 JSON、帧目录或悬空链接均拒绝覆盖，重录需换新名字。
`fps` 是采样频率，不代表仿真实际性能；预热不计入视频时长。
相机可自动选取，或同时填写世界坐标 `[m]` 的 `camera_eye_m` 与 `camera_target_m`。

持续行走验收可加 `max_stationary_s`；含站立或长停留时应省略。
录制程序能处理的失败或中断会记录结果并清理本次资源，包括恢复相机与视口状态。
胶囊间距验收只针对行人之间的物理近似边界，不代表环境碰撞或外观已全部验证。
详细输出和清理规则见[录像说明](../recording/README.md)。录像依赖系统已有的 ffmpeg 和 ffprobe。

## Python 复用

```python
from pathlib import Path
from pedstrain_simulation.simulation import create_simulation_project, inspect_simulation_config

profile = create_simulation_project(
    Path("configs/my_room"),
    scene_asset=Path("/absolute/path/to/room.usd"),
    character_assets=Path("/absolute/path/to/characters"),
    pedestrian_count=3,
    configuration_format="yaml",
    waypoint_frame="stage_world",
)
print(inspect_simulation_config(profile))
```

上例明确选择原生舞台 WORLD 位置框架。`inspect_config_dependencies(path)` 提供与 CLI 相同的依赖摘要。

运行已有工程用 `run_simulation(load_simulation_config(path))`。
旧工程的字段和入口调整见[配置迁移与复用](custom_scenes.md#从旧格式迁移)。
