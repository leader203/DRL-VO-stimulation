# THIRD_3 场景

本工程在 `THIRD_EXP_1.usd` 中运行 **4 名行人**，默认时长为 **60 仿真秒**。
出生点、路点、开始时间、朝向和停留以 [simulation.yaml](simulation.yaml) 为准。主配置使用 ROS 2 和[共享导航参数](../runtime/ros2_navigation.yaml)。
角色来自[共享清单](../runtime/characters.yaml)，默认动作库和清单内的 USD 路径都由导航设置的 `asset_root` 解析。

## 准备与预检

在仓库根目录运行以下命令，使用现有 uv 环境。预检无需加载 ROS，也不启动 Isaac Sim。

```bash
cd /project/EXP/IsaacLab
export UV_PROJECT_ENVIRONMENT="$PWD/env_isaaclab"
PEDSTRAIN_RUN=src/pedstrain_simulation/simulation/run.py
PEDSTRAIN_SCENE=src/pedstrain_simulation/simulation/configs/third_3
uv run --no-sync python "$PEDSTRAIN_RUN" check "$PEDSTRAIN_SCENE"
```

普通检查应显示 `pedestrians: 4`、`pedestrian_controller: ros2_sfm_orca`、
`fixed_spawn_check.status: no_body_overlap` 和 `simulation.duration: 60.0`。
只核对声明资源时追加 `--dependencies`。试用另一份导航设置时，通过 `run PROJECT --dry-run --navigation-settings FILE` 预览本次选择。相对路径以终端当前目录为准。[临时导航设置](../../docs/workflow.md#试用导航参数)
固定出生点诊断核对身体间距，实际地面、障碍净空和动态通行由原生运行验证。

## 直接启动

在完成上方变量设置的同一 shell 中，加载当前导航设置指定的 ROS 环境，再启动 GUI。

```bash
source /opt/ros/jazzy/setup.bash
uv run --no-sync python "$PEDSTRAIN_RUN" run "$PEDSTRAIN_SCENE"
```

其他模式使用同一主 YAML，可把上方命令的项目目录换成以下入口。路径均相对 `configs/third_3`。

| 模式 | 入口 | 默认输出 |
|---|---|---|
| ROS 2 窗口 | [目录入口](simulation.yaml) | `output/status.json` |
| ROS 2 无窗口、无视口 | [headless.toml](headless.toml) | `output/headless_status.json`，该包装关闭录像 |
| ROS 2 俯视录像 | [recording.toml](recording.toml) | `output/recording.mp4`、同名 JSON、帧目录和 `output/recording_status.json` |

无窗口示例为 `uv run --no-sync python "$PEDSTRAIN_RUN" run "$PEDSTRAIN_SCENE/headless.toml"`。
直接从该入口恢复 GUI 时同时追加 `--no-headless --render-viewport`。
只需观察启动和初段移动时，可临时追加 `--duration 10 --report "$PEDSTRAIN_SCENE/output/smoke_status.json"`。
这会单独记录 10 仿真秒的短测，60 秒路线覆盖需完整运行核对。

ROS 运行中连接 RViz 时，从 `ROS2_SIMULATION_READY` 输出读取当前运行目录或 `monitor.json`，参见[实时监控](../../docs/rviz_monitoring.md)。

## 自动验收与录像

`workflow` 为原生子进程加载 ROS 环境，并创建新的运行目录。按用途选择下列一条命令。

```bash
uv run --no-sync python "$PEDSTRAIN_RUN" workflow "$PEDSTRAIN_SCENE/headless.toml" \
  --no-record --require-route-goals

uv run --no-sync python "$PEDSTRAIN_RUN" workflow "$PEDSTRAIN_SCENE/recording.toml" \
  --record --require-route-goals
```

`--require-route-goals` 检查原配置时长内的全部原始路点。运行失败时先查新目录的 `workflow_report.json`、`native.log` 和已有状态记录。
工作流录像成功后，`video.mp4` 位于新运行目录。原配置完整运行时目标为 1280×720、12 FPS、720 帧。
录像包装启用 `camera_view = "overhead"`。相机从场景正上方自动取景，范围涵盖占用地图、行人路线及全身高度。
直接运行录像包装时，MP4、同名 JSON 和帧目录须使用全新路径。参数见[录像说明](../../recording/README.md)。

## 当前运动参数

| 项目 | 保留值与语义 |
|---|---|
| 人数 | `count: 4`，与 `actors` 一致 |
| `defaults.speed` | `0.9 m/s`，实际速度受可行空间和局部避碰限制 |
| `defaults.arrival_tolerance` | 原始路点的三维到达容差 `0.2 m`，不放宽来取得通过 |
| `defaults.repeat` | `true`，按既有闭合/往返语义执行 |
| `simulation.duration` | `60.0` 仿真秒，初始化另计，墙钟耗时由实时率决定 |
| 控制与物理 | ROS 控制 `60 Hz`，场景物理 `120 Hz` |
| 规划和站位有效半径 | `0.2775 m`，由 `0.25 m` 胶囊半径和 `0.0275 m` 场景边距组成 |
| 行人间目标净距 | `0.225 m`，对应目标中心距 `0.725 m` |
| 提前探测范围 | `4.8 m`，用于识别有预测冲突的邻近行人 |
| `walker_04.start_time` | 仿真第 `5 s` 开始行走 |
| 出生修正、规划投影、让位距离 | 共享导航设置默认 `0.0 m`、`null`、`0.4 m` |

ROS 节点拥有 A* 路径、SFM 期望速度、RVO2 避碰、路线调度及积分；Isaac 仅应用完整命令和显示距离驱动动画。
不启用 IRA 或 NavMesh。源资产中的冲突脚本在临时组合层抑制，源 USD 不改写。
自动门通过现有 `automaticDoor:*` 元数据驱动实际关节，保持实体碰撞，不使用旧 CCT 关闭门扇碰撞的绕过方式。
ROS 门协作使用门口距离、实际朝向和扫掠净空，必要时按 `max_yield_distance_m` 选择临时让位点，待真实门开度稳定后继续原目标。
场景通过 [door_activation.usda](door_activation.usda) 启用门前站位检查。走廊侧在门洞前 `0.55 m` 内，隔间内侧在实际扫掠安全站位的 `0.72 m` 范围内，朝向门洞中点的偏差须在 `30°` 内。
元数据要求和超时规则见[门与导航协作](../../docs/ros2_navigation.md#门与导航协作)。

出生点与路点使用已组合父变换的 Isaac WORLD 坐标，不是 Prim 的 local Translate。
本场景为米制 Z-up，当前 ROS map 原点与轴偏航均为零，因此原位置数字保持不变。
出生 `yaw` 相对模型作者朝向；路点 `yaw` 是到达后的绝对世界身体朝向，二者含义不同，见[单位与坐标](../../docs/units.md)。

共享 ROS 设置的 `auto_return_heading: follow_route` 只影响自动追加的返程朝向。显式路点仍按 YAML 中的 `yaw` 执行。

## 路线维护与验证

转向完成后才开始指定停留；显式闭合路线按顺序重复，其他重复路线按原往返规则执行。
更改人数时同步维护 `count`。路点不因内部栅格路径而被写回；内部终点选择不能扩大原到达容差。

运行目录保存地图、计划、完整轨迹、ROS 日志及结果。整场景验收需核对原目标到达、等待解除、相遇净空和可见步态，见[测试说明](../../tests/README.md)。

入口绕行修正后的四人原生复验中，`walker_01` 在第 4.42 秒越过入口内沿，早于 `walker_04` 的第 5 秒启动时间。`walker_04` 在第 6.98 秒随后通过，入口阶段未触发两人互相让位。[入口顺序对照](../../tests/output/startup_entry_bypass_20261109/evidence/entrance_order.json) · [120 秒原生回执](../../tests/output/startup_entry_bypass_20261109/native120/result.json)

门外候位、缓存拐角重规划和 `0.225 m` 行人间净距的四人无视口复验运行了 120 仿真秒。前 60 秒到达全部 8 个原始路点，实际最小扫掠身体净距约 `0.225 m`，旧轨迹中外侧门洞附近约 6.45 秒的排队未复现。另一处返程目标附近仍有约 3.23 秒的普通行走近零速区间。[原生回执](../../tests/output/portal_queue_clearance_20261109/native120_corner_refresh/result.json) · [候位对照](../../tests/output/portal_queue_clearance_20261109/evidence_corner_refresh/portal_queue_comparison.json) · [门控核对](../../tests/output/portal_queue_clearance_20261109/evidence_corner_refresh/door_audit.json)

其他原生验收、参数覆盖短测和不同时期的录像见[历史验收索引](../../docs/index.md#历史验收索引)。
