# 命令行接口

| 文件 | 职责 |
|---|---|
| `arguments.py` | `create_parser`：分组声明组合入口参数、帮助和默认值 |
| `overrides.py` | 将配置选择和显式 CLI 覆盖合并为新的不可变 SimulationConfig |
| `launcher.py` | 启动编排、run 项目预检和项目命令分发 |
| `projects.py` | 解析 init/check 并调用 projects API，不启动 SDK |
| `workflow.py` | 自动准备子进程ROS环境，运行/录制并核对坐标和轨迹 |

日常ROS流程为 `init PROJECT → 编辑 simulation.yaml → workflow PROJECT`，
完整命令见 [配置工作流](../docs/workflow.md)。`init` 标准输出保留原 JSON，
标准错误输出提供当前解释器对应的 Edit/Check/Run 提示；`check` 输出检查 JSON。
`workflow` 默认无窗口，`--no-headless` 显示 Isaac 窗口并开启视口，`--record` 或 `--no-record` 覆盖录制选择。
直接启动启用了行人的 ROS 2 项目前，先在 shell 加载所选配置的 ROS setup。活动会话可用 `monitor RUN_DIR_OR_STATUS` 启动 RViz，见[监控](../docs/rviz_monitoring.md)。
`run PROJECT --dry-run` 在应用 CLI 覆盖后输出同份最终检查 JSON，不启动 SDK。
`run` 和 `workflow` 均可用 `--navigation-settings FILE` 临时选择一份 ROS 2 导航参数。命令行相对路径以当前目录为准。
该选项要求 ROS 2 行人已启用。有效文件和规划、站位净空见预检的 `dependencies.pedestrian_settings` 与 `ros2_clearance`。
正式 `run PROJECT` 也先预检，再进入唯一运行循环。
`check` 与 `--dry-run` 是按需预览，不是运行前必须重复执行的步骤。

PROJECT 可为目录或显式 YAML/TOML。目录必须恰好提供一个 `simulation.yaml` 或 `simulation.toml`。
显式 `run` / `check` 省略 PROJECT 时使用当前工作目录。
其他名称的包装需显式传入。缺失、歧义或非文件入口均拒绝，不搜索子目录。
`run PROJECT` 不与 `--simulation-config` 或 `--config` 并用。旧参数式入口可选择有效的 ROS 2 项目 YAML/TOML。
离线检查验证本地资源、数值和输出位置，启用录像时还检查编码工具；不保证物理地图、后端导航、路径或碰撞。
额外 Kit 参数仍交给正式启动处理，离线模式不验证其效果。

`simulation/run.py` 与 `python -m pedstrain_simulation.simulation` 使用同一 CLI 和 ROS 2 会话。
使用旧脚本或旧配置的工程见[迁移指南](../docs/custom_scenes.md#从旧格式迁移)。
新增命令先在这里解析，配置读取放在 configuration，项目生成与检查放在 projects，仿真控制放在 runtime。
增加组合入口参数时，分别修改 `arguments.py` 的声明和 `overrides.py` 的合并规则；
`launcher.py` 不再同时维护这两种细节。公共解析入口由 CLI 包统一导出。

[自定义场景指南](../docs/custom_scenes.md)
