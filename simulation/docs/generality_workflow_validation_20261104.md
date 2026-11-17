# 场景独立性、坐标、门控与自动工作流验收 — 2026-11-04

> 历史验收快照：本报告绑定当次回执中的source/input指纹、运行模式与时限。正文的“当前”指该冻结版本；后续整理后的源码须重新验收。现行操作与版本索引见[文档索引](index.md)，整理后结果见[维护记录](project_maintenance_20261105.md)。

已保留用户的原始行路点数值、速度、停留、朝向和到达容差，完成本轮修正。
最终源码通过 657 项 CPU 回归、完整 `third_3` 自动工作流及独立厘米制 Y-up 原生场景验证。

## 审查发现与修正

| 问题 | 根因与当前行为 |
|---|---|
| 距门较远时提前开门 | 原逻辑使用离扫掠区 1.5m 或室内归属判断；旧视频 Door2 在距门口 1.570m 时请求。现按 USD `automaticDoor:approachDistance` 和实际过门路径首次准入，本场景阈值为 0.55m。 |
| 安全退让后反复开关门 | 全开后过早清除通行请求，退让者暂时超出接近区便被当成无请求。现首次准入建立固定期限授权，实际穿门且连续两帧清出扫掠区后释放；不按 tick 续租。 |
| 原生场景坐标与规范米制输入混淆 | 显式提供 `stage_world` 与兼容的 `world_m`，在读取源场景单位/上轴后只转换一次；保留原始、规范世界与 ROS 三份计划记录。 |
| 不同 USD 帧率导致控制步错误 | 目标帧率不是精确控制步。现以 loop-isaac 双精度秒步驱动 60Hz 控制，保留源 USD FPS/TCPS 和动画语义；关闭恢复原 loop/player 设置。 |
| 无交会风险也额外侧移或让行 | 互避偏置改为按实际半径、净距与预测最近距离连续衰减；已有足够间距、平行错开或背离路线不误触发有限让行。 |
| 朝向约束停车后丢失名义净距 | 原执行保护只检查身体半径和，没有重新保证配置净距。现对最终速度弦再次认证，优先单侧缩速、保留分离运动；真实身体重叠明确报错。 |
| 缺少完整自动化验收 | 新增独立父进程工作流，自动准备子进程 ROS 环境，串联配置、原生执行、坐标图表、轨迹与门准入检查。独立检查会保留各自结果，不因某一项失败而伪报整体通过。 |

门的首次接近距离是**行人中心到有限门口线段**的距离。安全退让后人体可能暂时远于阈值，此时延续已取得的授权，避免门扇碰人；这不是重新远距离触发。

## 坐标使用约定

当前 `third_3` 显式配置：

```yaml
navigation:
  backend: ros2
  waypoint_frame: stage_world
```

`stage_world` 应填写 Isaac Sim **已组合父变换的世界位置**，单位和上轴沿用源 USD；带父变换的 Prim 的 local Translate 不能直接作为世界位置。
OpenUSD 规定长度单位应施加在完全变换后的世界数据，直接解释原始局部平移并不可靠。[OpenUSD 长度单位说明](https://openusd.org/release/api/group___usd_geom_linear_units__group.html)

内部规范世界采用米制 Z-up，ROS map 再应用 `origin_m` 与 `yaw_rad`；速度、身体尺寸、到达容差、地图和录像相机参数保持 SI，不随原生位置单位缩放。
ROS 使用 SI 和右手坐标；地图 cell 数据沿 X 优先、按行展开，原点位于 cell(0,0) 的左下角。[REP-103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)、[OccupancyGrid](https://github.com/ros2/common_interfaces/blob/jazzy/nav_msgs/msg/OccupancyGrid.msg)、[MapMetaData](https://github.com/ros2/common_interfaces/blob/jazzy/nav_msgs/msg/MapMetaData.msg)

`third_3` 源舞台为 1m/unit、Z-up，因此原数字无需改写；原始点、规范米制点、ROS 点及往返结果最大误差为 0m。
新 ROS CLI 工程默认显式 `stage_world`；旧文件省略字段仍保留 `world_m`，Python 创建 API 的兼容默认也保持。

## 最终原生结果

| 验证 | 结果 |
|---|---|
| 当前源码全量 CPU | 657 项通过；simulation 范围 Ruff 通过 |
| `third_3` 自动工作流 | 60 仿真秒、三人原始路点 7/7，通过全部阶段 |
| Door2 首次准入 | t=12.7833s，距实际门口 0.54436m，低于 0.55m |
| 全部首次准入记录 | 5 条，零超距准入；Door3/1 首次分别 0.54206/0.42813m |
| 最小执行弦身体净距 | 约 0.25m，与配置一致，未减小半径或净距 |
| 最大单步位移 | 0.015000002m 以下，保持名义 0.9m/s、60Hz |
| 普通 walking 连续低于 0.03m/s | 三人最长分别 0.55/0.65/1.317s；指定停留、门等待和有限让行分开统计 |
| 完整成片 | 1280×720、12FPS、720 帧、60s；编码及全片解码通过 |
| 独立场景 | 厘米制 Y-up、24FPS/24TCPS、父节点平移旋转、非零 ROS origin/yaw，双人原目标全部到达 |
| 独立场景物理姿态 | 1440 次实际 PhysX 人物根观测通过；最小实际胶囊净距 0.25704m |
| 控制时钟精度 | 原 24 码场景 3600 控制帧末误差约 2.12e-12s；每控制帧两个 120Hz 物理步，暂停零推进 |
| 源码、配置与清理 | 该次验收结束时文件指纹与最终原生记录一致；stage、临时目录、自有 ROS 进程清理通过 |

独立时钟探针明确分列了 play/commit 初始化的额外两个物理回调，不将其混算为 3600 控制更新内的 7200 步。
本次完整录制总墙钟约 303s；60s 是仿真及视频时间，不代表已经达到 1× 实时。

## 一键复验与产物

从仓库根目录运行，无需手工 source ROS：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py workflow \
  src/pedstrain_simulation/simulation/configs/third_3 \
  --record --require-route-goals
```

默认创建全新的 `output/workflow_*`；也可用 `--output NEW_DIR` 明确选择。原工程不被重写，已有输出拒绝覆盖。
工作流自动检查原目标覆盖、声明坐标、时钟、位移、角速、静态扫掠与行人净距，产出 JSON/SVG 和最终报告。
`--require-route-goals` 适用于要求在所配时长内走完全部原路点的验收；普通短测试会报告覆盖情况而不假定路线已走完。

- [本轮工作流报告](../configs/third_3/output/workflow_generality_20261103_v4/workflow_report.json)
- [修正后视频](../configs/third_3/output/workflow_generality_20261103_v4/video.mp4)
- [原始点与 ROS 站位图](../configs/third_3/output/workflow_generality_20261103_v4/alignment/coordinate_alignment.svg)
- [逐点坐标核对表](../configs/third_3/output/workflow_generality_20261103_v4/alignment/coordinate_alignment.json)
- [轨迹验收](../configs/third_3/output/workflow_generality_20261103_v4/evaluation.json)
- [当前指纹复核](../configs/third_3/output/workflow_generality_20261103_v4/final_verification.json)
- [厘米 Y-up 原生结果](../tests/output/workflow_yup_cm_native_20261103_v2/result.json)及[该场景站位图](../tests/output/workflow_yup_cm_native_20261103_v2/alignment/coordinate_alignment.svg)
- [CPU JUnit](../tests/output/ros2_generality_20261103_all_final.xml)

## 通用性与支持范围

生产导航、门控、坐标及工作流不按场景目录名、行人名或固定路点分支；场景几何、门参数、资源与行路点属于输入数据。
更换项目时指定新的 PROJECT 与其资源/导航配置即可复用流程；不会复制第三场景的几何或改写用户点位。

当前全局地图仍是单层平面，未知地面按障碍；多层/楼梯及自定义门控制器需要对应适配，不能猜测。
现有 `automaticDoor:*` schema 仍明确限其声明的米制 Z-up、直立父框架和 Z 轴铰链；普通厘米/Y-up 无该门 schema 的场景已原生验证。
坐标组件支持某种单位/上轴，不等于所有第三方门格式或资产骨架都自动兼容。详细边界见[场景适配](custom_scenes.md)和[ROS 说明](ros2_navigation.md)。
图中虚线表示原路点顺序，不是实际规划路径；轨迹评价是逻辑位置和占据图扫掠验证，不冒称 PhysX 接触深度认证。
