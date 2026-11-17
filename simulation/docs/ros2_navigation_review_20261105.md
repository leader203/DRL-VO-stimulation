# ROS2 行人路径规划与导航审查（2026-11-05）

本轮修复了可达性误判、漫游目标采样、低速进度判断、出生点校验和ROS请求/只读监控隔离。项目改动限于simulation，没有修改用户YAML、场景USD、路点、停留、人数、速度、原到达容差或60Hz控制。

最终 **843项回归、真实本机DDS互操作、120秒原生物理验收、原配置60秒自动工作流均通过**。当前正常场景的7199个控制批次与上轮避让验收逐项核对：位置、朝向、时间和phase差异均为0；故障边界通过独立回归修复。

[机器汇总](../tests/output/navigation_review_20261105/summary.json) · [全量JUnit](../tests/output/navigation_review_20261105/unit_regression.xml) · [原生结果](../tests/output/navigation_review_20261105/native120/result.json) · [原60秒工作流](../tests/output/navigation_review_20261105/workflow60/workflow_report.json)

## 已确认并修复的问题

| 严重性 | 可复现问题 | 修正及源码 |
|---|---|---|
| P2 | actual起点安全，但只能连接到超过原2格搜索窗的中心；原Overlay/Reservation错误无路 | 仅原种子集为空时，作至身体净空半径的有界第二窗，每条actual→anchor仍用原完整圆盘证书。r=.25、margin=.03、gap=.25、res=.02的双peer狭口例有完整合法witness，新旧负例仍拒绝；[planner.py:306](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/planner.py:306) |
| P2 | actual已在原goal球内，目标中心不安全、无可用目标anchors时仍判无路 | actual硬净空通过且已满足原到达球时返回零运动路径；Passing排除新事务中的已达者，旧passer达标并保留实际间距后安全结束，不强迫多走或伪造空路径方向；[planner.py:229](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/planner.py:229)、[passing.py:207](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/passing.py:207) |
| P1 | wander先从全图取128个，再筛1–4m；250000格默认seed42首tick误失败。只修局部筛选后，14738个局部候选中只有44个可达的隔墙例仍失败 | 先本地annulus，再随机；保留第一个规划快路，失败后用实际认证起点的格图可达分量筛余下候选。保留128次预算和连续direct兜底，无seed时helper返回None而非“无路”；[solver.py:977](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/solver.py:977)、[planner.py:317](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/planner.py:317) |
| P2 | 声明.002m/s持续前进，12秒只走约2.3cm，固定5cm检查点导致假失败；合法1e−8m/s错初始朝向也被固定“零速”门槛冻结 | 检查点取5cm与一秒名义行程中较小者，真实零移动仍按原12秒失败；精确非零速度/引导按原角速及C候选方向更新，不新增速度下限，不改ORCA半平面或数值保护；[solver.py:674](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/solver.py:674)、[solver.py:323](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/solver.py:323) |
| P1 | 固定stand、短walk绕过地图验证，非法出生点或身体重叠能进入人物构造；全fixed时也跳过当前动态实体检查 | 先检查floor、静态圆盘、当前物理轮廓及身体重叠，再创建actors/ROS。不投影点；物理接触或仅舒适净距不足仍按原合同允许；[runtime.py:75](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/runtime.py:75)、[runtime.py:42](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/runtime.py:42) |
| P1 | 成功command后Path发布失败会再发同序号error；未发布成功却记录signature，破坏只读监控与控制隔离 | Path/Marker独立告警，签名仅成功发布后提交；观察故障不替换成功控制回复，不重积分；[node.py:82](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/node.py:82)、[node.py:137](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/node.py:137) |
| P1 | 新seq重复同time仍能让引擎积分；time不变的6次请求移动约.1767m。客户端也会接受正确seq但time=999的错回复 | 校验有限数值时间/正dt，新请求严格推进时间且delta匹配dt；成功回复绑定seq和本次time，明确错误保持即时传递和旧无time错误兼容；[node.py:82](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/node.py:82)、[transport.py:101](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/ros2_navigation/transport.py:101) |

所有修复都沿用当前架构。没有用场景名、walker ID、特定坐标分支修复运行功能，没有降低身体尺寸/净距/地图边距，没有重排用户行路点；真实几何和特殊输入只用于测试。

## 保留的规划与控制合同

- 实际位置不被投影到格点；扩展起点搜索只产生经过原segment_clear认证的连接。
- A*八邻域顺序、禁止切角、unknown阻挡、目标容差、路径简化与完整运动弦证书保留。8个普通/边界查询的新旧路径逐值一致。
- 可达分量分析只在首个漫游目标规划失败后触发；开放场景首个direct路径成功时不做全图遍历。没有增加BFS缓存或公共API。
- 缺少栅格种子不代表连续空间不可达；有界搜索、保守inset和128次候选预算不构成连续空间完备性/全局最短证明。采样耗尽的错误现明确限定为已采样格图候选。
- 已验证的SFM/ORCA速度求解、接触精化、原实际净空guards、坐标变换和动画根所有权保留。AST对照确认 `_solve_crowd`、`_guard_chords`、`_repair_contact_roundoff`、`_limit_peer_rays` 等数值方法与本轮前一致。

## ROS时序与兼容

新请求的time/dt必须是有限非布尔数值，time非负、dt正数；后续新序号需严格推进time，dt与上次成功请求的时间增量一致。首个请求允许非零启动origin，不要求time==dt。重传须复用原seq和完整原快照，返回缓存且不再积分。

DDS层拒绝时间回退，重置需新会话；纯Python NavigationEngine原有重置API保持不动。成功command同时绑定seq/time。新版错误echo合法request time，旧版明确error缺时间戳仍及时传递，不转成无谓超时。Path/Marker仍是只读诊断，不作运动控制。

自建DDS客户端若原先靠“新序号、同一时间”推进，应改为推进仿真时间或复用原序号重试；无需修改现有工程YAML。Path发布的是原目标规划剩余路线，临时指令与实际运动由guidance/markers分别展示。

## 验证与性能

| 检查 | 实际结果 |
|---|---|
| 完整CPU回归 | **843 passed**，0 failure/error/skip；覆盖787项既有回归和新增边界 |
| 回归反证 | 起点、已达、漫游、低速、固定spawn和协议故障均先记录旧实现失败；封闭/零运动/真实错误负例维持原拒绝语义 |
| 真实CPU DDS | 同时刻拒绝后下一合法请求正常；latejoin Map/Path/Marker/TF收到；不可达保持failed-agent、顶层异常保原错，均未转成timeout |
| 250000格、5漫游者首请求CPU | 开放图.02293–.04322s（0次分量分析）；断连小隧道.02784–.02923s；大可达成分隔墙.94787s，均无error |
| 新120秒原生 | 完整通过，21600个实际物理根观测；最小逻辑净距约.25m，物理胶囊约.249999956m，场景/临时组合/ROS进程清理均成功 |
| 原60秒自动流程 | **7/7原目标到达**，坐标表误差0m，完整流程及退出清理通过；未开启新录像 |
| 正常场景轨迹保留 | 新旧完整7199批次位置、朝向、时间、phase差异均0；[逐项比较](../tests/output/navigation_review_20261105/baseline_trajectory_comparison.json) |
| 避让回归 | 同一冻结统计脚本下普通已前向反向采样0、guide回指0、角速超限0、移动方向/身体heading错配0；[本轮统计](../tests/output/navigation_review_20261105/heading_metrics.json) |

CPU回归包含真实C RVO2与USD数据，也包含SDK/协议替身；真实DDS、真实Kit/PhysX和CPU注入故障的证据分别保存，不能互相替代。大地图计时是CPU导航首请求，不包括Kit/地图烘焙/DDS/GPU，也不是整体实时率。未安装新依赖、修改SDK或降低60Hz控制。

本轮原生审计RTF约0.366（120仿真秒，播放墙钟328.082s）；60秒工作流RTF约0.345（播放墙钟173.988s）。当前仍非实时，本轮不作整体实时率提升结论。

[故障回归/独立计时证据目录](../tests/output/navigation_review_20261105/evidence) · [证据哈希清单](../tests/output/navigation_review_20261105/evidence_manifest.json) · [本轮汇总](../tests/output/navigation_review_20261105/summary.json)

## 复验

从仓库根目录使用既有环境；自动工作流创建新输出，不覆盖用户输入：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python -m pytest \
  src/pedstrain_simulation/simulation/tests -q

UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py workflow \
  src/pedstrain_simulation/simulation/configs/third_3 \
  --no-record --require-route-goals
```

本轮原生覆盖当前third_3及其原60秒配置。其他场景/高度/动态对象应绑定自己的输入重新运行；本轮没有证明任意连续几何、所有用户场景或无限时长均无问题。上轮视频和验收仍保留其冻结源码边界，本轮未新录视频。
