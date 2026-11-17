# Isaac Sim／ROS2 全链路审查与两分钟录像（2026-11-05）

后续转向复查发现：这段录像仍有近点追逐造成的低位移大角回摆，例如 walker_01 在约 46.73–49.72 秒移动约 10.5cm、累计转向约 333°。
本记录的角速、轨迹一致与到达检查通过，不等于排除了此类转向问题；后续转向修复的回执需另行核对。

本轮修改限于 `simulation`，沿用真实 PhysX occupancy map、独立 ROS2 路线调度、A*、SFM/RVO2 与距离驱动动画。
没有修改用户场景 USD、既有配置、出生点、路点、停留、朝向、速度、人数、到达容差或 60Hz 控制频率。
两分钟验收使用独立的 [120 秒包装](../tests/output/end_to_end_120_20261105/recording_120.toml)，原 `third_3/simulation.yaml` 仍为 60 秒。

[机器汇总](../tests/output/end_to_end_120_20261105/summary.json) · [本轮完整差异](../tests/output/end_to_end_120_20261105/evidence/final_changes.patch)

## 已复现并修复的问题

| 严重性 | 原问题及影响 | 修正 |
| --- | --- | --- |
| P1 | 配置的固定机器人被静态地图无条件排除，随后又因固定关节证明而不进入实时观测，碰撞体从两条链路同时消失 | `runtime.py` 让机器人使用与其他对象相同的静态／实时分类，不再按机器人身份排除 |
| P1 | `dynamic_roots` 指向非刚体父节点时，子刚体仍读取父节点的旧 USD 变换，未使用当前 PhysX 位姿 | `mapping.py` 优先选择碰撞体最近的启用刚体；没有刚体时才使用显式根／时变变换。显式动态子树保留实时归属 |
| P1 | 运行中增加／删除静态碰撞体或动态转静态后，继续使用原烘焙地图，物体可能漏检 | `DynamicObstacleObserver` 检测静态成员集合变化并明确失败，要求重建地图与会话 |
| P1 | 位姿命令的竖直速度为零，却能直接改变位置高度，绕过平面运动合同 | `Ros2Actors.apply` 在整批提交前拒绝高度变化；无任何人物根或动画相位被部分更新 |
| P2 | 外部写入缩放、剪切或倾斜时，位置与偏航仍可能正常，实际胶囊与导航半径不符 | `Ros2Actors.states` 在既有容差内核对完整单位旋转基，发现冲突即失败，不自动传送复位；Visual 子节点动画／单位缩放不受影响 |
| P2 | 关门合成扫掠障碍丢失高度，另一高度平面的行人被无关门扇阻挡 | `interactions.py` 保留已知高度带，`solver._cooperate` 仅排除明确不相交的障碍；未知／无效高度仍保守阻挡 |
| P2 | 工作流子进程正常或非零退出后，其 ROS 等后代可能存活 | `_run_child` 在正常、失败、超时、中断各退出路径均清理本次拥有的进程组；保留原异常，不操作独立进程 |

每项修复均有旧实现失败、修复后通过的回归证据；真实 USD 几何测试与替身 PhysX 边界测试分别保留。
未加入场景名、walker ID 或特定坐标分支，没有放宽身体净空、缩小胶囊或替换用户路线。

## 本轮验证

- 完整 CPU 回归：**876 passed**，0 failure、error、skip；基线为 843 项。
- Ruff 检查通过，106 个 Python 文件格式检查通过。
- 真实本机 DDS 复验通过：同一时刻的新请求被拒绝，下一合法请求正常；晚加入的 Map／Path／Marker／TF 订阅收到数据；明确错误保持原错误，没有退化为超时。
- 原生 Isaac Sim／PhysX 120 秒验收通过，原始路点全覆盖；共 **21,600 个物理根位姿样本**，速度、单步位移、角速度、实际根姿态、到达和原停留检查通过。
- 最小逻辑身体净距 `0.24999999999999967 m`，物理胶囊净距 `0.24999995627937577 m`；约 `4.4e-8 m` 的差异属于物理读回浮点量级。
- 原生验收的场景关闭、临时组合删除和 ROS 进程退出均成功，源码／输入哈希未变化。
- 与上一轮已验收导航基线的全部 7,199 个控制批次逐项比较，实测与指令的位置、朝向、速度、时间和路线阶段均无差异；本轮边界修复没有改变正常场景轨迹。
- 普通已前向行走的反向采样、引导回指、角速超限及非零移动方向／身体朝向错配均为 0。已朝向且远离引导终点时，近零移动最长为 0.300 秒；这不等于所有阶段都没有停顿。
- 合法目标占用等待的单段最长分别约为 6.083、6.800、8.717 秒。3 次关门让位／归位均完成，无越界准入或闭门返程扫掠失败；最后一次归位后尚未走满原 8 秒驻留，运行在 120 秒正常截止。

[完整 JUnit](../tests/output/end_to_end_120_20261105/unit_regression.xml) · [原生结果](../tests/output/end_to_end_120_20261105/native120/result.json) · [独立 DDS 及边界证据](../tests/output/end_to_end_120_20261105/evidence)

[独立原生汇总](../tests/output/end_to_end_120_20261105/evidence/independent/native120_independent_summary.json)保留完整轨迹、朝向、等待与门流程统计。
原生审计器保存外层 `result.json`，普通工作流保存 `runtime/result.json`；独立离线评估的临时适配仅映射已保存字段，原地图、轨迹和审计证据保持只读。

## 坐标与录制合同

原路点来自已组合的 Isaac 世界坐标，`navigation.waypoint_frame: stage_world` 保留该含义。
当前源场景为米制 Z-up，ROS map 原点与偏航均为零；输入、规范世界和 ROS map 的位置数值一致。
通用转换仍覆盖厘米制、Y-up、非零 map 原点／偏航和父变换；不能将带父变换对象的 local translate 当作世界坐标。

录像沿用现有 1280×720、12FPS、零预热及自动相机，保留源灯光；行人控制仍为 60Hz。
最终 H.264 视频经 ffprobe 确认为 **120.000000 秒、1,440 帧、12FPS**，全片 ffmpeg 解码成功且无错误输出。
与独立原生验收的 7,199 个控制批次逐项比较，位置、朝向、速度、时间及路线阶段差异均为 0。
全部 1,440 个视频样本与原生轨迹匹配；最大时间采样误差约 `5.53e-12 s`。
捕获等待期间冻结仿真，完成后继续原控制步。
没有通过降低控制频率或改变播放速度压缩录制时间。

工作流的地图／路点坐标往返最大误差 **0 m**，7/7 原始目标覆盖、门准入和退出清理检查均通过。
原生与录制使用同一冻结源码，录制前后所有记录的源码／输入哈希一致。
视频 SHA256：`8a5c1dc89855dd927ae141138164aedfb9513eb7fabdcdffd81953847ad67d8e`。

[两分钟视频](../tests/output/end_to_end_120_20261105/recording120/video.mp4) · [工作流报告](../tests/output/end_to_end_120_20261105/recording120/workflow_report.json) · [逐批次与逐帧比较](../tests/output/end_to_end_120_20261105/evidence/independent/recording120_comparison.json) · [全片解码回执](../tests/output/end_to_end_120_20261105/evidence/independent/recording120_full_decode.json) · [坐标对齐图](../tests/output/end_to_end_120_20261105/recording120/alignment/coordinate_alignment.svg)

抽查 5、15、30、50、75、95、119 秒的编码画面，并核对原 PNG 与人物投影；源墙体造成的遮挡保留。
录制播放阶段墙钟约 `548.565 s`，包含冻结截图与读回等待，不能作为普通运行的实时率。

## 边界与复验

原生验收播放墙钟约 `334.593 s`，120 秒仿真的实测 RTF 约 **0.359**，当前仍未达到实时。
CPU 性能模式、PCIe 和 IOMMU 提示仍见原始日志；本轮未修改范围外的驱动、系统参数或环境依赖。
这次验收覆盖当前配置及列出的回归边界，不证明任意场景、无限运行时间或所有几何输入均无问题。
地图仍是单个可行走平面，门关节仍遵守现有直立米制 Z-up 合同；静态拓扑变化需新会话，不自动重烘焙。

从仓库根目录复验；输出必须使用新目录：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python -m pytest \
  src/pedstrain_simulation/simulation/tests -q

UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py workflow \
  src/pedstrain_simulation/simulation/tests/output/end_to_end_120_20261105/recording_120.toml \
  --record --require-route-goals --timeout-s 1200
```

旧 [导航审查](ros2_navigation_review_20261105.md) 和 [避让专项](avoidance_heading_validation_20261105.md) 保留其各自的冻结源码与时长，不能作为本轮新录像回执。
