# 近点追逐转圈：修复机制与回归方法（2026-11-05）

后续全片反馈及新发现的拐角／微速度／通行问题见[2026-11-06复审](turning_followup_20261106.md)。本文保留本轮冻结源与录像结果，不代表后续源码验收。

本轮处理的是行人追逐附近引导点时，在很小范围内反复转向、绕圈或缓慢逼近的问题。安全到达与转向观感分别验收：一次原生运行完成，不能单独证明已经消除局部转圈。

本轮从2026-11-05开始，最终原生与录像复验于2026-11-06进行。本文机制对应冻结的 `solver.py` SHA256 `f03f25032b7f853fe5cebbe7aee63d7b7be782be0c87f777135aa14f8b153b6b`、`passing.py` SHA256 `8bdd902a75bba8d128b6e65ba5103816a8465c6a7f203317cd320210a23b8b14`、`goal_occupancy.py` SHA256 `682eb91f4ab12d990bcb4bff3928e4c7c351aac4492fd37fd4c64ce36501c53c`。后续源码变化须重新绑定测试、轨迹和视频证据。

## 已确认的问题与反证

在空地图上，仅一名行人、固定原目标、无障碍和无阶段切换，也能复现近点转圈。目标距离为0.03m、到达容差为0.005m时，原实现从60°、90°、180°初始朝向到达目标前，累计绝对转角分别约747°、871°、964°；单步角速度仍满足限制。角速度有界不能排除在小区域内持续绕圈。[原实现失败回执](../tests/output/turning_fix_20261105/evidence/independent/regression/old_red.xml)保留了这些公开闭环用例。

首版曲率减速将上述三个开放场地用例的累计转角分别降至约97°、140°、231°，但未覆盖全部交会条件；这些数字可用[v2冻结求解器](../tests/output/turning_fix_20261105/evidence/independent/v2_turning/v2_solver_source.py.txt)和同一公开用例复核，不代表本轮最终源结果。v2的120秒原生业务与安全检查通过，转向质量复核仍未接受；例如一段固定 `goal_approach` 引导点在2.43秒内没有移动，行人累计转向约248.66°。不得把这次业务通过标为转圈修复完成。状态区分和完整证据见[v2清单](../tests/output/turning_fix_20261105/evidence/independent/v2_turning/manifest.json)、[固定引导点复核](../tests/output/turning_fix_20261105/evidence/independent/v2_turning/walker03_guide_stability.json)。

新增贴近静止行人的公开用例继续证伪了仅靠首版曲率减速即可解决问题的判断：静止者位于 `(0.78, 0)`、移动者原目标为 `(0.03, 0)` 时，v2仍累计转向约619°；另一微偏移用例约639°。这些是固定的合法几何反例，没有缩小身体或降低原0.25m净距。[新增用例及RED回执](../tests/output/turning_fix_20261105/evidence/independent/regression/v2_contact_receipt.json)与[失败JUnit](../tests/output/turning_fix_20261105/evidence/independent/regression/v2_contact_red.xml)单独保存。

逐层排查还发现，缓存的到达端点即使仍通过可见性检查，也可能接近另一行人的约束圆，导致ORCA持续压低法向进度；当引导点始终非空时，原入场逻辑不会重新选择端点。另一类问题是已经走完的零长度可见顶点继续占据引导位置。它们需要分别处理，不能用放宽到达容差或延长等待预算掩盖。

动画侧的只读检查没有证实独立的整圈躯干旋转。源动作剥离根运动后的髋部摆幅较小，连续画面中的主要身体转动跟随导航根朝向；但旧录像没有保存SDK重定向后的完整目标姿态缓存，因此不把源动作检查当作目标骨骼动画的完整证明。视频验收仍需同时观察身体、脚步和根轨迹。

## 冻结实现的修复机制

### 转弯能力与近点速度一致

[`NavigationEngine._approach_speed`](../ros2_navigation/solver.py)按目标距离和当前朝向误差限制执行速度。设距离为 `d`、朝向误差为 `e`、最大角速度为 `ω`、控制步长为 `dt`，执行上限为：

```text
min(原速度上限, d / max(dt, 2 * abs(sin(e)) / ω))
```

该几何限制让行人在接近目标时有空间完成转弯。SFM原有朝目标的驱动力继续保留；没有把所有引导段都改成按距离指数减速，避免经过中间顶点时不必要地反复刹停。

只有在附近目标的ORCA原始结果沿引导方向的进度低于执行上限的25%时，恢复查询才使用纯位置误差产生的偏好速度，去除该次恢复中的横向惯性。正常SFM偏好、普通交会偏好和非退化运动保留原流程。恢复结果仍经过原RVO2约束、路线前半平面、可达转向扇区及完整位移段安全检查；它不是绕过避碰的直线位置控制。

当近点安全求解得到精确零速度、当前仍有有效引导时，朝向按原角速度界转向目标方向。远处保持原RVO引导方向，保留绕墙和行人侧避的脱困能力。任何非零执行速度均使用其实际运动方向作为朝向，避免以“小速度视作零”造成身体朝向与位移不一致。已认证的有限让位射线继续使用其原有先转身再沿射线移动的控制流程。

对已足够接近且本步能够对齐的极小末段，偏好允许完成有限的最后一步，随后仍经过全部执行检查。这没有改变原目标、到达容差或实际到达判断，也没有直接写入目标位置。已完成的精确零长度顶点会被跳过，继续查询后续有效引导。

已经由零速度求解建立的停车转向会保持承诺的朝向，完成前真实平移速度为零，防止微小非零候选反复打断锁存而左右摆头。
完成帧扣除已经使用的角度，后续运动只能使用剩余角预算；如果外部动态障碍使下一步原地等待不安全，则取消锁存、使用原RVO避险，不强迫停在来袭障碍前。

同一近点的实际执行解在转向投影和安全检查之后出现低进度时，会保持位置纠偏到引导点或控制归属切换，避免原SFM惯性与纠偏偏好逐帧来回切换。
这一保持模式不改变ORCA约束、角速度、身体净距或原目标；后续真实场景还需验证通行调度，不能仅凭近点CPU通过宣称整段通过。

停步时，原软墙体偏好有时建议先向一侧大转，随后行人约束又建议反向。此时控制器会在同一RVO2约束集内查询更贴近当前朝向的候选，要求有可表示、足够有意义的位移、非反向运动、更小转角及完整静态／动态／行人扫掠证书。
候选已在当前执行转向扇区内时才允许实际执行，并再次检查联合位移段；否则仅作后续承诺转向的引导。壁力模型、原速度和角速限制保持，不通过事后改身体yaw掩盖侧移。

### 到达区域内的端点刷新

[`GoalOccupancyCoordinator._refresh_arrival_endpoint`](../ros2_navigation/goal_occupancy.py)仅在接近缓存末点、尚未实际进入原到达球时尝试少量解析候选。候选平衡原到达球内的余量与对其他行人的净空；只有比旧端点改善超过既有数值阈值，并通过原三维到达容差、静态地图、实际行人、动态轮廓和预留区检查，才替换末段。

存在活动占用者时，新端点还必须通过原出口证书；此处使用 `replan=False`，不会为每个候选重新执行A*。候选失败保留旧路径，原来无可用引导时的完整规划及重试规则继续保留。统一的 `_guide` 维护剩余路径、跳过零长度点，并遵循受控门的规划与入场所有权；实际门叶仍接受执行层检查。

### 原驻留完成后的有限资源交换

曲率修正改变交会时序后，曾暴露“新来者没有可认证等待点、原占用者也无法退出”的资源环。只有占用者具有有限的原定离开时间和后续目标，且新来者当前位置满足完整静止安全检查时，协调层才允许等待原转向与驻留完成，然后使用既有的占用者让位事务。

让位起点和截止时间固定，不随重试续期；如果此前已有门控让位，沿用其起点，整个临时位移仍受0.4m预算约束。候选必须同时认证新来者到达原目标和占用者继续原路线，两条路径的证书均保留。未知或永久占用、原地不安全、没有可认证有限让位时仍明确等待或失败。该机制没有提前结束原驻留，也没有把暂时让位当作业务到达。[资源交换机制与失败反例](../tests/output/turning_fix_20261105/evidence/control/goal_owner_swap/README.md)保留了独立来源和边界。

上述生产逻辑没有行人编号、场景名或用户路点坐标分支。原waypoint、出生点、指定yaw、等待、名义速度、到达容差、身体净距和用户60秒配置均保留；当前场景60Hz控制、120Hz物理及0.4m临时让位上限没有因转圈修复而降低。

原驻留已经结束、占用者已进入离开状态时，也进入同一有界流程；首次以当前时刻固定一次恢复预算，不再要求已经失效的未来驻留提示。
真实失败快照中无需额外位移即可认证双方原路线。[已离开状态回归](../tests/output/turning_fix_20261105/evidence/control/already_departing_goal/README.md)保留了旧版拒绝、新版事务释放与实际进展的证据。

### 已批准的自动返程朝向

用户已允许自动生成的返程路点朝向跟随返程路线。共享ROS配置显式使用 `auto_return_heading: follow_route`，由该设置生成的新工程也继承此值。
该策略只将自动反向追加的路点副本 `yaw` 设为 `None`，由实际返程运动决定朝向，避免先执行复制的反向yaw、再立即转回路线。
原YAML的去程点、显式闭合路线的全部指定yaw、所有位置与逐点停留不变；自动回到出生点的原有语义保留。
原生审计和离线评估读取同一策略，仍逐点验证真实到达、原停留和原始目标覆盖；不会用取消去程朝向检查取得通过。

代码与外部省略字段的设置保留 `preserve` 默认，兼容旧调用。需要旧复制朝向行为时可显式写 `auto_return_heading: preserve`。
配置严格拒绝其他值。配置校验、工程生成和评估回归见[配置证据](../tests/output/turning_fix_20261105/evidence/auto_return/config/README.md)与[审计证据](../tests/output/turning_fix_20261105/evidence/auto_return/evaluation/manifest.json)。

### 已建立通行通道的第三人准入

后续长时复验曾在93.9167秒失败：第三人走入已认证通行路径，导致三人互锁；单纯原地重试不能在原期限内解开。
现在只在路线会冲突、当前站位仍保持通行证书且实际原地等待安全时，第三人才进入 `passing_wait`。无冲突路线、已有高优先级控制和不安全站位不被强制排队。
父交易实际通过或结束后释放，父期限不续期；等待者通过 `waiting_actors` 保持协调层归属，门安全仍保留优先权。
失败快照的公开闭环和48项Passing检查见[准入证据](../tests/output/turning_fix_20261105/evidence/control/passing_admission/README.md)。

## 以后修改导航时如何回归

先运行最近的多参数公开闭环回归，再运行Goal、门控和Passing组合。命令从仓库根目录执行，JUnit写入独立临时目录：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python -m pytest -q \
  src/pedstrain_simulation/simulation/tests/test_ros2_navigation.py::test_near_goal_arrival_does_not_circle_while_preserving_motion_limits \
  -o cache_dir=/tmp/pedstrain_turning_pytest_cache \
  --junitxml=/tmp/pedstrain_turning_near_goal.xml

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python -m pytest -q \
  src/pedstrain_simulation/simulation/tests/test_ros2_navigation.py \
  src/pedstrain_simulation/simulation/tests/test_ros2_goal_occupancy.py \
  src/pedstrain_simulation/simulation/tests/test_ros2_door_interactions.py \
  src/pedstrain_simulation/simulation/tests/test_ros2_passing.py \
  -o cache_dir=/tmp/pedstrain_turning_pytest_cache \
  --junitxml=/tmp/pedstrain_turning_navigation.xml
```

近点回归当前有10组参数，覆盖初始60°/90°/180°、距离、名义速度、角速度、30/60/120Hz，以及远处和贴近的静止者。这里改变频率是便携回归覆盖，不是修改用户运行频率。用例检查原目标实际到达、总绝对转角不超过初始偏角加90°、速度/角速度/步长上限、任意非零速度的身体朝向对齐、静态完整位移段和身体净距；静止者必须保持原位。还应保留缓存端点改善、完成顶点继续引导、原驻留后门内外交换、真实回程队列和原始切向微爬快照的现有回归。改变修复时，应验证相应反例在撤去修复后失败，而不是仅检查新实现通过。

CPU用例不能替代用户场景的完整原生轨迹和画面。完成全量测试后，应以独立包装运行120仿真秒，覆盖首次到达、回程、后续循环、门关闭归位和目标占用交换；包装时长不写回用户60秒配置。每轮记录实际加载的源码与配置指纹、控制/物理频率、终止原因、安全与到达评估、清理结果，再单独记录视频媒体规格、解码结果和录制开销。[通用工作流](workflow.md)说明现行执行和输出入口。

全阶段转向统计脚本已随[v2反证证据](../tests/output/turning_fix_20261105/evidence/independent/v2_turning/manifest.json)归档，可复用于新runtime或视频元数据：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/tests/output/turning_fix_20261105/evidence/independent/v2_turning/analyze_turns_any_run.py \
  --runtime NEW_RUNTIME_DIRECTORY --output /tmp/pedstrain_turning_new_analysis
```

录像分析使用同一脚本的 `--metadata VIDEO_JSON` 替代 `--runtime`。比较时使用归档的 `compare_turn_reports.py --baseline BASELINE_ANALYSIS_DIRECTORY/turning_report.json --current NEW_ANALYSIS_DIRECTORY/turning_report.json --output /tmp/pedstrain_turning_comparison.json`；每个报告同目录须有对应的 `actor_series.jsonl`。这些是随验证证据归档的分析工具，具体输入与定义见其源码，不能直接重跑依赖旧 `/tmp` 状态的历史控制器试验来代替现行验收。

统计保留所有phase，包括业务转向、原路返程、门控、Goal和Passing临时运动，分别报告累计绝对转角、带符号净转角、低速转角、局部窗口的路径长/净位移及转向反复。合法业务转向仍计入总量，通过阶段和目标上下文解释；不通过整体排除阶段来制造改善。原始全片窗口之外，还应比较双方都完成的同一原路线段，避免因新运行完成更少循环或被截断而看起来转得更少。

候选异常窗口需回看连续画面中的躯干、脚部和根轨迹。根朝向投影箭头仅表示根预测，不能作为独立的视觉朝向测量。新增场景应使用其实际单位、up-axis和map变换，继续检查原目标覆盖、实际身体净距、门控归位、等待原因和退出清理；不承诺任意几何都可通行或所有未来输入零缺陷。

## 本轮最终原生与录像结果

最终采用 **native120_v9** 与 **recording120_verified**，中间失败、中止或质量不足的运行均不计为最终通过。

[机器汇总](../tests/output/turning_fix_20261105/summary.json) · [完整改动](../tests/output/turning_fix_20261105/evidence/final_changes.patch)

| 验证 | 最终结果 |
| --- | --- |
| 全量回归 | **928 passed**，0 failures、errors、skips；Ruff与106文件格式检查通过 |
| 120秒Isaac Sim／真实ROS2 | 完整通过，**21,600个实际物理根位姿观测**，原目标7/7到达 |
| 身体净距 | 最小逻辑净距 `.2500000198 m`，物理胶囊净距 `.2499999664 m`；保持原约 `.25 m` 配置，差异为浮点读回量级 |
| 同业务段反复 | 中间V5发现的03两秒20次反向片段，最终为0次；所有比较保留双方已完成的相同路线阶段 |
| 全阶段峰值 | 相同冻结口径的两秒raw反向峰：原视频12次、V5为20次、最终4次；残余峰在门前清场，未隐去 |
| 原末段远目标 | 最终115.050秒进入原 `.2 m` 容差，115.0667秒推进下一点；V7在120秒仍未到达该点 |
| 路线覆盖公平性 | 最终完成26个业务上下文，原视频24个、V7为22个，并非少运行路线造成改善 |
| 坐标／清理 | 坐标核对最大误差0m；stage关闭、临时组合删除、ROS退出均通过 |
| 录像 | H.264、1280×720、12FPS，**1440帧／120.000秒**；完整软件解码、全部PNG CRC／尺寸验证通过 |
| 录像隔离 | 与独立原生全部7199个控制批次的位置、朝向、速度、时间和上下文完全一致，源／输入指纹无漂移 |

[全量JUnit](../tests/output/turning_fix_20261105/unit_regression_v9.xml) · [最终原生](../tests/output/turning_fix_20261105/native120_v9/result.json) · [转向汇总](../tests/output/turning_fix_20261105/evidence/independent/v9_turning/summary.json) · [媒体QA](../tests/output/turning_fix_20261105/evidence/media_verified/README.md) · [工作流](../tests/output/turning_fix_20261105/recording120_verified/workflow_report.json)

[最终两分钟视频](../tests/output/turning_fix_20261105/recording120_verified/video.mp4)，SHA256：`d2ab6037a50fb0c5f3827bccf250f4e12f5e8c94ebadebb53045026e47a8de07`。
原场景YAML仍为60秒，录像通过独立120秒包装运行；原路点、人数、速度和控制／物理频率未改。

边界明确保留：门前清场仍有小幅方向修正；连续画面中01约110.52–111.50秒有一次有限反转（朝向跨度83.6°、移动约9.4cm），不称为完全零回摆。
指定yaw、往返折返和门关闭归位仍需要正常转身。未将所列抽样画面冒充所有骨骼或所有未来场景的完整视觉保证。
无条件强制开门清场射线的候选虽通过旧回归，却会使来袭移动墙失去安全下一步，因此[未集成](../tests/output/turning_fix_20261105/evidence/control/door_yield_ray_not_integrated/README.md)。

本轮原生播放墙钟约298.36秒，RTF约0.402，仍非实时；录像另有冻结读回开销。证据覆盖当前输入及已列回归，不是任意几何或无限运行时间零缺陷证明。
