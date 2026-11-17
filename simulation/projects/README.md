# 项目工具

本包提供工程创建、离线检查和自动工作流。用户步骤见[工作流](../docs/workflow.md)，参数来源见[配置索引](../configs/README.md)。

| API | 职责 |
|---|---|
| `create_simulation_project` | 校验资源、生成新项目并返回绝对入口路径 |
| `inspect_simulation_config` | 检查有效配置、声明资源与输出条件，预览逐人计划 |
| `inspect_config_dependencies` | 查看场景配置和资源引用 |
| `workflow.run_simulation_workflow` | 给子进程加载 ROS 环境，运行仿真并核对坐标、轨迹 |
| `alignment.export_coordinate_alignment` | 核对计划坐标与舞台位置，导出 JSON 与 SVG |
| `evaluation.evaluate_ros2_run` | 离线评估运行 trace，可要求覆盖所有原始路点 |

项目创建只生成配置。角色和场景从本地路径引用，已有目录不会覆盖。默认生成 `simulation.yaml` 和导航参数副本；提供已有 `navigation_settings` 时直接引用。CLI 的默认位置框架是舞台 WORLD，可显式选择规范米制 Z-up。

检查使用最终选中的人口和导航设置。`pedestrian_plans` 包含有效角色计划，位置框架由 `pedestrian_plans_frame` 标注。所选导航参数和来源见 `dependencies.pedestrian_settings`，`runtime_settings.asset_root` 为配置中声明的资产根。`ros2_clearance` 汇总规划、出生和行人目标净距的派生量。

`run --dry-run` 和 `workflow` 都可用 `--navigation-settings FILE` 临时选择一份导航参数。工作流将同一选择写入本次子进程包装并纳入输入指纹。父进程负责编排，不加载 Kit；仿真、录像和评估结果写入独立运行目录。最终结果见 `workflow_report.json`。
