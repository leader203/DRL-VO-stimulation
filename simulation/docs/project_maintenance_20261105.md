# 项目整理与验收记录（2026-11-05）

> 这是配置整理完成时的冻结记录。后续避让算法与朝向修复、完整新原生验收见[避让专项验证](avoidance_heading_validation_20261105.md)，本页回执仍绑定其当时源码。

本轮仅整理 `src/pedstrain_simulation/simulation`。已完成入口、配置依赖、通用运行链、文档/历史证据和中间产物审查，并用整理后的源码与配置完成第三场景的新 60 秒原生工作流。用户行路点、出生点、朝向、停留、速度、人数、往返、到达容差和控制频率保持不变。

## 修正与清理

| 已确认问题 | 完成的更新 |
|---|---|
| 四份 ROS 主配置保留不生效的原生设置引用，容易把两个后端的参数混为一谈 | 从 ROS 主 YAML 移除 6 处 IRA 引用；新增各场景 `ira.toml` 明确选择原始原生设置，ROS 仍使用共享平铺导航设置 |
| 直接录像的默认目的地被钉死为已存在的旧日期文件，预检因此拒绝运行 | `third_3/recording.toml` 改为 `output/recording.mp4` 与 `output/recording_status.json`；12 FPS、60 秒、相机/预热不变；重复录制推荐 `workflow` 的全新目录 |
| IRA 检查摘要显示继承而来但未加载的 ROS 参数路径 | 仅在 ROS 后端显示生效的导航设置路径；保留字段和原配置对象，不改变加载/执行语义 |
| 运行器内部仍依赖历史录制模块别名 | 内部导入改用现行 `recording` 包路径；外部兼容模块/API 保留 |
| CLI 仍把公共人数/seed/控制功能称为 IRA 专用，部分说明混淆后端 | 更正帮助文字和错误提示，不改变参数名或行为 |
| 原生验收命令在加载 ROS 后覆盖 `PYTHONPATH`，导致 `rclpy` 不可见 | 移除该命令中多余的覆盖；保留 ROS setup 提供的路径，验收脚本自行加入本项目源码目录 |
| 操作文档、坐标/yaw 说明及历史验收入口易被误读 | 更新配置/模块文档、补充文档索引，明确历史报告绑定其当时源码和输入指纹 |
| 存在已被后续运行替代的中间产物与字节码缓存 | 归档 3 个无精确外部路径引用的中间工作流目录，逐文件复核 2220 个文件、1,255,095,011 字节；另删除 13 个可再生缓存目录、85 个缓存文件 |

原生参数文件、行为脚本包装、旧 Python 模块别名、IRA/RL 接口仍有有效调用或 USD/测试引用，因此保留。没有为减少文件数删除有效功能或用户资源。最新视频、最终工作流、原生审计、回归夹具及其来源证据均保留。归档位置和每个文件的 SHA256 见[清理回执](../tests/output/maintenance_cleanup_20261105.json)。

2026-11-09 复核时，回执中记录的 `/tmp` 归档已不存在。回执保留为当次归档的历史记录。

## 当前操作入口

- ROS：`run.py run configs/<project>`，主文件仍为 `simulation.yaml`。
- IRA 兼容：`run.py run configs/<project>/ira.toml`；它复用同一用户计划和原始身体/Kit 设置。不要只切换 ROS YAML 的 backend 后假定原生设置也随之恢复。
- 自动检查、执行、坐标核对和轨迹评估：`run.py workflow PROJECT`；重复录制可使用 `workflow PROJECT/recording.toml --record`，每次保存到新目录。
- 具体仓库根目录命令与 ROS 准备见[项目入口](../README.md)、[配置索引](../configs/README.md)、[文档索引](index.md)。

## 检查与结果

下面命令从仓库根目录使用现有 uv 环境执行，不安装或升级依赖。输出目录需全新；本轮已使用的目录不能再次覆盖。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python -m pytest \
  src/pedstrain_simulation/simulation/tests -q

UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py workflow \
  src/pedstrain_simulation/simulation/configs/third_3 \
  --output /absolute/path/to/NEW_DIRECTORY --no-record --require-route-goals
```

| 检查 | 实际结果 |
|---|---|
| 全套 CPU 回归 | **770 passed**，0 fail/error/skip；[JUnit](../tests/output/project_maintenance_20261105/unit_regression.xml) |
| 配置检查 | `main(['check', PROJECT])` 检查 4 个 ROS、4 个 IRA 及 headless/recording 两个包装，共 10 个入口均通过；[逐项结果](../tests/output/project_maintenance_20261105/config_checks.json) |
| 原计划与行为比较 | 四场景与修改前备份的场景/人口/运行设置字典完全相等；ROS/IRA 包装共享原计划；[合同结果](../tests/output/project_maintenance_20261105/scenario_contracts_after.json)及[有效参数等值](../tests/output/project_maintenance_20261105/simulation_config_maintenance_20261105_equivalence.json) |
| 摘要显示回归 | 最近既有 IRA 用例加入未使用 ROS 路径：修复前 3 个参数用例失败，修复后与 ROS 正例共 4 个通过；最终全套已覆盖 |
| ROS 环境说明 | 加载 ROS setup 后 `find_spec('rclpy')` 成功；将 `PYTHONPATH` 覆盖为 `src` 后失败；[复现结果](../tests/output/project_maintenance_20261105/ros_environment_check.json) |
| Ruff | `ruff check` 与 `ruff format --check`（排除 vendor）通过，106 个 Python 文件格式已一致 |
| 本地文档链接 | 本轮脚本遍历排除 output/vendor 的 35 份项目 Markdown，检查 289 个本地链接和 45 个标题锚点，均有效；结果保存在 `tests/output/project_maintenance_20261105/documentation_links.json` |
| 第三场景原生工作流 | **passed**；原配置 60 仿真秒、60Hz 控制；7/7 原始目标到达，原容差仍为 0.2m |
| 坐标 | 原生生成地图与 input/world/map 计划核对通过，最大位置残差 **0m** |
| 逻辑轨迹与占据图 | 扫掠静态净空和行人间距评估通过；最小逻辑身体净距 **0.25m** |
| 控制隔离 | 原生报告 `ira_enabled=false`、`navmesh_enabled=false` |
| 运行结束 | 场景关闭、临时组合删除、ROS 子进程停止均为 true |
| 本轮指纹 | 工作流 `source_unchanged=true`、`declared_inputs_unchanged=true`，运行后重新核验源码/声明输入 |

[本轮汇总](../tests/output/project_maintenance_20261105/summary.json) · [原生工作流](../tests/output/project_maintenance_20261105/native_ros2/workflow_report.json) · [轨迹评估](../tests/output/project_maintenance_20261105/native_ros2/evaluation.json) · [坐标表与站位图](../tests/output/project_maintenance_20261105/native_ros2/alignment/coordinate_alignment.json)

本轮原生无录制运行的播放墙钟耗时 **135.809s**，初始化及退出计入后的总墙钟耗时 **197.772s**，实时率约 **0.442**。本轮没有修改导航/动画内核，也不将清理描述为实时率提升。

## 验证边界

本轮原生运行覆盖 `third_3` 的 60 秒与首轮全部原路点，不等于无限循环、任意新场景或所有动态障碍已证明无问题。其他三个工程本轮完成配置加载和参数/行为等值检查，没有另跑完整原生通行；IRA 做配置与 CPU 兼容回归，没有新增 GPU 验收。本轮没有重新录制视频或开展新的视觉步态评估。

0.25m 净距是独立 ROS 轨迹/占据图评估结果，不应冒充本轮新测的 PhysX 接触深度或可见模型边界。之前 120 秒真实物理审计、视频和坐标复核仍是维护前冻结版本证据，不能因为本轮文档更新就改称当前完整源码的原生验收。工作流仅哈希声明的文件，不递归哈希资产目录。

当前靠墙目标的精确中心净空与坐标匹配是不同问题：到达依原 0.2m 三维容差判定；本轮未移动任何点或放宽容差来获取通过。
