# 文档索引

行人运动由 ROS 2 独立导航控制，Isaac Sim 负责物理和人物显示。

## 创建、运行与观察

| 需求 | 文档 |
| --- | --- |
| `third_3` 窗口、无窗口和录像 | [THIRD_3 启动指南](../configs/third_3/README.md) |
| 创建新工程、自动运行、录制与失败检查 | [工作流](workflow.md) |
| 人数、角色资源、出生点、路线、等待与往返 | [行人YAML](pedestrian_yaml.md) |
| 场景迁移、覆盖层与配置边界 | [自定义场景](custom_scenes.md) · [配置索引](../configs/README.md) |
| 坐标、单位、出生yaw与目标heading | [单位与坐标](units.md) |
| 物理地图、SFM/RVO2、目标占用与门协议 | [ROS 2导航](ros2_navigation.md) |
| 活动会话的RViz地图/路径/实测状态 | [只读监控](rviz_monitoring.md) |
| 录像帧、相机、编码与清理 | [录制](../recording/README.md) |
| 仿真时间、RTF与不同渲染模式 | [性能测量](performance.md) |
| CPU回归、真实DDS/PhysX与场景验收 | [测试说明](../tests/README.md) |

## 开发

| 需求 | 文档 |
| --- | --- |
| 模块责任与扩展位置 | [架构](architecture.md) |
| CLI与项目API | [CLI](../cli/README.md)；[项目工具](../projects/README.md) |
| 会话所有权和失败清理 | [生命周期](../runtime/README.md) |
| 共享 USD 组合、物理设置及可选机器人显示 | [场景](../scene/README.md) · [机器人显示](../behaviors/README.md) |

## 历史验收索引

每项回执绑定当次源码、输入、仿真时长和运行模式。复用结果前先核对 `source_sha256` 以及 `input_sha256` 或 `declared_input_sha256`。

| 冻结记录 | 证据范围 |
| --- | --- |
| [2026-11-09 三处停滞、拥塞与墙角复验](../tests/output/third_3_three_encounters_20261109_2QZU5a/native125_stall_gated/result.json) · [时间窗对照](../tests/output/third_3_three_encounters_20261109_2QZU5a/evidence_stall_gated/encounter_comparison.json) | 四人、125 仿真秒、60 Hz。7 个原路点全部到达，实测最小行人身体净距约 0.225 m。新旧交会时序不同，原墙角场面另用实测状态复放 |
| [2026-11-09 排队安全前进段](../tests/output/wait71_queue_prefix_20261109_qK3jiw/native120/result.json) · [等待对照](../tests/output/wait71_queue_prefix_20261109_qK3jiw/evidence/queue_recovery_comparison.json) | 四人、120 仿真秒原生无视口。队首在指定停留结束后首次移动的延迟由 3.33 秒降至 0.35 秒，整段近零速由 3.33 秒降至 2.53 秒 |
| [2026-11-09 走廊会车与等待恢复](../tests/output/corridor_recovery_20261109_kZTAOx/native120_certified/result.json) · [前后对照](../tests/output/corridor_recovery_20261109_kZTAOx/evidence_certified/corridor_comparison.json) | 四人、120 仿真秒原生无视口。通过者在让位期间提前沿安全段行进，目标队首在原停留结束后约 3.33 秒恢复，原路线和门控复核通过 |
| [2026-11-09 入口绕行](../tests/output/startup_entry_bypass_20261109/native120/result.json) · [实际入门顺序](../tests/output/startup_entry_bypass_20261109/evidence/entrance_order.json) | 四人、120 仿真秒原生无视口。先出发行人绕过尚未启动者，在后者启动前进入，前 60 秒到达 8 个原始路点 |
| [2026-11-09 外侧门洞候位与拐角](../tests/output/portal_queue_clearance_20261109/native120_corner_refresh/result.json) · [实测帧复放](../tests/output/portal_queue_clearance_20261109/frozen_outer_portal_replay.json) | 四人、120 仿真秒原生无视口。验证门外候位、缓存拐角重规划与 `0.225 m` 行人目标净距，前 60 秒到达 8 个原始路点 |
| [2026-11-09 导航参数覆盖短测](../tests/output/configuration_override_20261109/native_smoke/workflow_report.json) | 四人、4 仿真秒。用 `--navigation-settings` 选择候选文件，核对预检、原生子进程、坐标和输入指纹。时长仅覆盖启动及初段运动 |
| [2026-11-09 规划和站位净空](../tests/output/planning_footprint_20261109/native120_quarter_cm_verified/result.json) · [门控核对](../tests/output/planning_footprint_20261109/evidence_quarter_cm/door_audit.json) | 四人、120 仿真秒原生无视口。共享设置的有效半径为 0.2775 m，前 60 秒已到达 8 个原始路点，实际最小行人身体净距约 0.25 m，三个隔间门均记录到入室后关稳 |
| [2026-11-09 运动连续性](../tests/output/motion_flow_20261109/native120_verified/result.json) · [轨迹对照](../tests/output/motion_flow_20261109/evidence/native120_verified/motion_comparison.json) · [录像核验](../tests/output/motion_flow_overhead120_20261109_1358/verification.json) · [成片](../tests/output/motion_flow_overhead120_20261109_1358/recording/video.mp4) | 四人、120 仿真秒，覆盖拐角和同向跟随。录像通过 8/8 原始路点评估，使用净空边距更新前的导航设置 |
| [2026-11-09 门控与路线](../tests/output/door_route_alignment_20261109/native100_verified/result.json) · [等待与坐标](../tests/output/door_route_alignment_20261109/evidence/native100_verified/wait_coordinate_summary.json) · [门控核对](../tests/output/door_route_alignment_20261109/evidence/native100_verified/door_audit.json) | 四人、100 仿真秒原生运行，覆盖重复路线和 8 个原始路点。核对首次开门、两次 8 秒关门驻留和零停留入室后的实际关门 |
| [2026-11-09 早期俯视录像](../tests/output/door_facing_20261109/recording_overhead75/video.mp4) · [两分钟对照](../tests/output/overhead120_20261109_1238/recording_verified/video.mp4) · [较早四人时序失败](../tests/output/configuration_update_20261109/native_current_four/result.json) | 75 秒视频记录后续门控和路径调整前的行为。两分钟对照早于运动连续性与净空更新。另一次原生运行中，`walker_04` 第 1 秒出发，运行于第 18.43 秒因相向交会无安全让行点而停止 |
| [全片转向复审](turning_followup_20261106.md) · [两分钟录像](../tests/output/turning_followup_20261106/recording120_final/video.mp4) | `third_3` 三人、120 仿真秒，覆盖共享拐角前瞻、低速回摆和通行 |
| [上一轮转向验证](turning_validation_20261105.md) | 近点、多人准入、返程策略和冻结录像 |
| [全链路记录](end_to_end_validation_20261105.md) | 碰撞地图、根变换和流程清理 |
| [避让专项验证](avoidance_heading_validation_20261105.md) | SFM/ORCA、避让朝向及坐标 |
| [路径导航审查](ros2_navigation_review_20261105.md) | 路径可达性、漫游、出生点、进度与ROS 2时序 |
| [2026-11-05维护记录](project_maintenance_20261105.md) | 当次配置整理、归档与三人原生工作流 |
| [2026-11-05导航/门控/监控](goal_door_navigation_validation_20261105.md) | 整理前770项CPU、120秒物理审计、60秒录像/普通模式及live RViz；V3–V7故障材料单列 |
| [2026-11-04工作流与录像复审](review_recording_20261104.md) | 整理前670项CPU、TOML工作流、截图写盘/相机与退出所有权 |
| [2026-11-04通用场景/单位工作流](generality_workflow_validation_20261104.md) | 当次657项CPU、地图/坐标、门准入及厘米制Y-up组件 |
| [2026-11-03ROS原生验证](ros2_validation_20261103.md) | 当次RVO2、受限通道、原生单位/胶囊、完整场景输入 |
| [2026-11-03早期录制](ros2_recording_validation_20261103.md) | 早期录像与采样合同，后续修复另见新日期报告 |

历史视频、原生回执和回归夹具保留原路径。历史维护回执中的临时归档路径仅记录当次操作。
