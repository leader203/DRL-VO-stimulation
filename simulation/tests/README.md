# 仿真回归与场景验收

ROS 2 仿真分别验证 CPU 契约、真实 PhysX、DDS 和用户场景；
通过配置检查或跑满时长均不能替代路线和可见动画验收。

## CPU 回归

从仓库根目录使用既有 uv 环境：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_PROJECT_ENVIRONMENT=env_isaaclab \
  uv run --no-sync python -m pytest src/pedstrain_simulation/simulation/tests -q
```

| 测试 | 覆盖范围 |
|---|---|
| `test_ros2_mapping.py` | 坐标/单位、地面与地洞、静态固定约束、普通实例与 PointInstancer 区分、导出翻转、动态凸包和拓扑 Observer |
| `test_ros2_actors.py` | 根位姿唯一写入、父变换/单位、命令批次、距离步态相位与清理 |
| `test_ros2_doors.py` | 元数据门控、连续转动外包、真实开度读回、导航 hold 优先、碰撞保留、父变换与自有层恢复 |
| `test_ros2_navigation.py` | 栅格路径、原容差、SFM/RVO2 与相遇/静态边界的独立内核合同 |
| `test_ros2_door_interactions.py` | 有限安全点、两批清空后授权、门超时与恢复原路线 |
| `test_ros2_goal_occupancy.py` / `test_ros2_passing.py` | 目标区域、局部队列/完整退出、门内外交换、内外圈证书及Passing所有权 |
| `test_ros2_wall_force.py` | 连续墙面软力，不替代硬几何guard |
| `test_ros2_monitor.py` | 实测/请求分层、仿真限频、canonical参考TF及只读CLI |
| `test_ros2_recorder.py` / `test_ros2_recording_viewport.py` | 捕获时间冻结、实际帧、相机/层所有权及失败清理 |
| `test_workflow.py` / `test_workflow_alignment.py` / `test_workflow_evaluation.py` | 子进程、配置保持、坐标、原路线/驻留和错误报告 |
| `test_ros2_runtime.py` | 独立后端生命周期、门观测/命令协议与失败清理 |
| `test_configuration_review.py` | 严格配置、后端选择、生成/检查、来源、计划与兼容 API |
| `test_scene_composition_units.py` / `test_scene_physics.py` | USD 组合、物理单位和场景设置 |

含真实 USD 的 CPU 用例需要现有 `pxr`，但不启动 Kit/GPU。
上述命令使用项目环境；`uv --no-project` 不会按 `UV_PROJECT_ENVIRONMENT` 选择解释器。
不要为执行测试安装另一版 USD 或修改现有 SDK。测试替身的通过不证明真实碰撞或 DDS 正确。

## ROS 场景验收

先在启动 shell 中加载 ROS，再运行独立观察脚本；输出目录必须全新：

```bash
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/tests/run_ros2_navigation.py \
  src/pedstrain_simulation/simulation/configs/third_3 \
  --output /tmp/ros2_acceptance --headless
```

保留ROS setup生成的`PYTHONPATH`；不要随后用`PYTHONPATH=src`覆盖它。
该验收脚本自行加入项目src，以上命令无需额外设置该变量。

默认使用工程时长；`--duration` 可明确覆盖仿真秒数。脚本使用工程声明的人口、路点、速度、等待、朝向、往返和到达容差。
`--require-route-goals` 要求本次运行实际到达全部原始行路点；首点通过不足以代替完整路线验证。
报告记录每个阶段的时长及行走速度低于 `0.03 m/s` 的最长连续时间，区分指定停留与局部通行问题。
需要检查可见动画时启用 `--capture` 并保留 `render_viewport: true`；无视口包装不支持捕获。

`result.json` 记录源码/输入指纹、有效计划、控制频率、竞争扩展是否关闭、到达/身体距离观测和清理结果；
`audit.jsonl` 保存逐帧观测，`runtime/` 保留实际地图、DDS 日志与控制 trace。
暂停采图不应推进时间或改写行人状态。真实 ROS 通信应核对序列重发不重积分、超时失败和退出后进程结束。
入口、topic 与输出字段见 [ROS 2 导航](../docs/ros2_navigation.md)。

应验证首轮目标、返程后的进展、原定停留解除和相遇净空；不能以“人物出现移动”代替整条路线通过。
对向、交叉、跟随、门、目标占用和受限通道分别保留正例与负例。
门交互应核对 `door_yield` / `door_wait`、不超过 `0.4 m` 的临时退让、真实稳定开度后恢复原目标，
以及无法找到安全点或超时的明确失败；协议见[门与导航协作](../docs/ros2_navigation.md#门与导航协作)。
只有起点/终点安全也不能代替中间整段扫掠净空。逻辑圆盘、物理胶囊与可见模型的证据不相互替代。
独立物理地图、角色显示和 DDS 验证不代表所有用户场景已通过；当前整场景结果以其源码/输入绑定的验收报告为准。

## 运行产物与清理

`tests/output/` 和场景自己的 `output/` 保存验收运行。回执、控制轨迹、实际地图、录像原帧、编码日志和运行内 `build/librvo2_*.so` 可用于追查结果或复放同次导航输入，按该次运行保留。
缓存清理限于确认未被活跃进程使用的 Python 字节码和测试工具缓存。清理前先区分运行证据与缓存，录像和验收使用全新输出路径。
