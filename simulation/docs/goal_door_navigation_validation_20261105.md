# ROS2行人导航、门控与监控验证（2026-11-05）

> 历史验收快照：本报告绑定当次回执中的source/input指纹、运行模式与时限。正文的“当前”指该冻结版本；后续整理后的源码须重新验收。现行操作与版本索引见[文档索引](index.md)，整理后结果见[维护记录](project_maintenance_20261105.md)。

冻结源已通过770项CPU测试、120秒原生压力验证及用户原配置60秒录制工作流。普通无录制模式也已完成独立测量，各模式的条件和RTF分别记录。
本报告覆盖当前输入、便携几何/协议回归及本机原生运行，不承诺任意场景均可通行，也未作与IRA效果高低的对照结论。

## 验证范围与输入保留

以先前v2工作流保存的声明输入和完整计划为基准，当次7/7声明输入SHA256全部相同，3名行人的解析后完整计划逐字段相同。
路点、出生点、朝向、等待、开始时间和往返均保留。

| 合同 | 当次值 |
| --- | --- |
| 人数、名义速度、原到达容差 | 3人、各0.9m/s、各0.2m |
| 用户时长、控制、场景物理频率 | 60仿真秒、60Hz、原USD声明120Hz |
| 身体半径、身体间净距、地图边距 | 0.25m、0.25m、0.03m |
| 用户路点坐标 | `stage_world`；通过既有StageFrame转为米制Z-up canonical世界及ROS map |
| 源USD | m/Z-up；文件hash、TCPS60及FPS24均未变 |
| 录像配置 | 12fps、warmup0、1280×720；新工作流使用独立输出目录 |

120秒是验证harness的时长覆盖，未写回用户60秒配置。声明输入包括用户YAML/TOML、运行Kit/导航YAML、源USD、运动库及v2的生成包装文件。
V8还绑定了10项实际配置/角色/运动资产，验收结束时逐项hash均匹配。

运行时源指纹相对v2由107文件变为110文件：新增3、修改10、无删除。此数量由逐文件指纹比较得出，未把Git未跟踪状态当作变更基线。
新增模块为`goal_occupancy.py`、`monitor.py`、`visualization.py`；变化集中于ROS规划/门控、求解仲裁、观测评估和监控入口。
该次冻结的110个运行时文件全部匹配V8实际加载的源指纹。`frames.py`、`mapping.py`、`actors.py`、IRA兼容链和vendored RVO2仍保留原指纹。
本轮功能和结构逻辑没有新增场景名、行人编号或用户路点坐标分支；验证报告中的场景和角色名仅用于定位证据。

## 已确认根因与修复

| 根因 | 修复及安全边界 |
| --- | --- |
| 到达位置位于门扇中途sweep内，直接关门会扫过身体 | 同室内最多0.4m让位，实测门关闭并稳定后归位，恢复原朝向和剩余驻留 |
| 相近目标的来人阻挡已到达者出口；短退出段过早释放 | 保留原目标区域和完整退出范围，持久保存可验证绕行，实际清出后按序接纳 |
| 门内退出与门外目标占用形成交换环；外圈没有安全点 | 固定起点内加入小圈候选，保持门/Goal/Passing所有权和0.4m预算，保留无解时明确失败 |
| 距候选点0.015m即停步，实际仍未让出足够净距 | 实际当前位置没有通过完整`_exit`证书时继续yield，避免用接近候选点代替通行ready |
| 原静态路线忽略未接管的实体门叶；临时状态清空路线 | 全局路线和引导段纳入当前非受控物理轮廓，临时token切换保留并重验原路线 |
| 小圈逐候选A*反复计算相同目标边界，ROS callback超5秒 | 先用physical baseline做完整证书快查，再保留原A*回退；仅对充分远离动态约束的静态端点用单目标缓存 |

静态地图来自PhysX；动态数据继续读取启用碰撞器的实时pose及保守投影。实心门叶和把手不能按“门框”名称忽略。
受控门继续使用实际开度和完整sweep interlock；其实体碰撞、局部约束和完整执行guard保持启用。

端点缓存仅限标准immutable mask及exact已知覆盖层，保留端点顺序/数值；近peer、保留区、观测障碍、未知子类或不一致clearance均原算。
全局动态mask、A*、原容差、速度/转向和static/dynamic/peer完整位移段证书继续生效。没有提高5秒ROS超时、缩小身体或放宽目标容差。

## 全量与原生结果

CPU全量为 **770 passed、0失败、0错误、0跳过**；Ruff检查106文件通过。
原生结果与监督器均为passed，退出码0，实际推进至119.99999999999447秒；IRA、Navigation Core与Behavior Core扩展均为false。

| V8原生指标 | 结果 |
| --- | --- |
| 每人实际物理审计采样 | 7200；有效控制trace为7199批，包含初始化测速度前的物理样本 |
| 逻辑整段最小身体净距 | 0.250000000m |
| 审计物理胶囊整段最小净距 | 0.249999976m；与0.25m相差约2.4×10⁻⁸m |
| 最大实际单步 | ≤0.0150000012m；名义0.9/60=0.015m |
| 原路点覆盖 | 7/7首次覆盖，44.4667秒前全部达到；随后继续原往返循环 |
| 既有25秒拥堵 | 03于32.7667秒接纳，38.3333秒达到原目标，38.3500秒队列清空 |
| 回程拥堵 | 03于59.5667秒接纳，62.7167秒达到回程原目标，62.7333秒队列清空 |
| 后续队列 | 62.7333至120秒没有新queue；最终queue及Passing事务为空 |
| 最长连续walking且速度＜0.03m/s | 01/02/03分别0.60/1.60/0.85仿真秒；不将合法停留或短时低速称为零等待 |
| 清理 | stage、临时组合、ROS进程均清理；监督器确认native和RViz停止 |
| provenance | 运行时source/input均未变；验收结束时冻结源与V8加载源匹配 |

首个闭门episode的独立分析为PASS：

| 项目 | 结果 |
| --- | --- |
| 首次实测关闭并稳定 | 18.1333秒；角度0.614°、角速度-4.776°/s，满足原闭合准则 |
| 返回原锚点 | 20.7500秒，锚点误差4.97×10⁻¹⁰m |
| 原8秒驻留 | 实际8.016667秒；临时运动和闭门等待不计入 |
| 原驻留朝向误差 | 0rad |
| 最大临时让出 | 0.386882m，低于0.4m限额 |
| 恢复原业务路线 | 28.7667秒 |

全部闭门episode的只读复核确认3次实际闭门和归位完成，blocked观测为0：

| Episode开始[s] | 实测关闭[s] | 归位[s] | 最大让出[m] | 已观察原驻留[s] | 驻留覆盖 |
| --- | --- | --- | --- | --- | --- |
| 15.6167 | 18.1333 | 20.7500 | 0.386882 | 8.016667 | 原8秒完成 |
| 62.1833 | 64.6833 | 67.3000 | 0.384788 | 8.016667 | 原8秒完成 |
| 109.0000 | 111.4500 | 114.0333 | 0.377974 | 5.966667 | 120秒截断，未覆盖完整8秒 |

三次归位锚点误差均小于5×10⁻¹⁰m、恢复后驻留yaw误差小于5×10⁻¹⁶rad；所记录的闭门返回、驻留闭门、关闭门叶路段及全sweep检查均无失败。
第三次只确认闭门/归位及截断前的驻留，不宣称它完成原8秒或已经离开。

V8播放墙钟为274.247831秒，RTF约 **0.43756**。该运行包含逐步物理审计和同步RViz，尚未达到墙钟实时率1.0；
不能与普通GUI、无审计或录像模式直接作提速比例比较。CPU独立计量中，v7真实阻断callback为5.983秒；最终两阶段下切回原端点计算为0.423秒，严格缓存后约0.292秒。
这些是计算工作量证据，不是原生RTF倍数，也不证明所有大地图A*长尾均已消失。

02最长1.6秒walking低速发生在84.55–86.15秒：位置净变化约0.000651m，累计绝对转向约3.950rad，附近03中心距从1.797降至1.041m；之后3秒行走约1.622m。
该片段包含大角度转向及近身交会，不将其称为完全无视觉顿挫，也不把所有低速都判成永久停滞。

## RViz与坐标校验

活动会话支持Static map、Observed scene、原目标Route、当前请求引导点、动态轮廓、door portal/sweep及实际门状态分层显示。
Marker按10Hz仿真时间更新，60Hz控制保持原频率；Map/Path/Marker/参考TF支持晚加入订阅。
只读参考TF为`map → <唯一会话namespace>/world_m`，采用Origin/yaw的逆向关系；该子帧是归一米制Z-up canonical世界，Map/Marker/Path仍使用map。
它不替代raw USD单位/up-axis转换，不被运动控制读取，也不覆盖用户map父变换。

从仓库根目录，在新会话仍运行或暂停时使用其runtime目录或状态引用：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py monitor RUN_DIR_OR_STATUS
```

具体数据含义、`--dry-run`、TF数学例与依赖见[RViz监控](rviz_monitoring.md)。发布者退出后，Transient Local历史随节点结束；已清理的V8输出用于检查证据及配置。
monitor入口不启动Kit/Nav2，不发送导航控制。独立TF数学与真实DDS晚加入验证均已通过。

V8监督器记录了本次live会话的[8秒](../tests/output/goal_door_native120_20261105_v8/rviz_after_8s.png)、
[55秒](../tests/output/goal_door_native120_20261105_v8/rviz_after_55s.png)、
[65秒](../tests/output/goal_door_native120_20261105_v8/rviz_after_65s.png)、
[100秒](../tests/output/goal_door_native120_20261105_v8/rviz_after_100s.png)截图。
初版视口裁切已通过单dock、1600×1000及完整地图尺度修正；此前独立实测快照GUI也确认全图和Global Status绿色。
快照guidance图不冒充全局实测轨迹，旧空白/裁切图不计监控通过。

## 场景适配与结论边界

功能从用户计划、地面/身体配置、启用物理形状和声明式门元数据取数。普通地图支持既有米/厘米及Y/Z-up归一化；
当前门schema仍要求米制Z-up、直立无剪切父变换、Z轴铰链和有效室内/门洞描述。全局规划采用单个平面地图，并非任意多层或楼梯规划。
共享门可按元数据选择共享进入；几何无安全让出点或目标长期占用时会报告等待/失败，保留原路点和安全要求。

V3–V7仅作为故障/回归证据：V4存在25–40秒长等，V6在7/7到达后仍于约53秒回程safe-fail，V7在25秒callback超时。
这些记录及旧视频不计入本次最终源冻结后的通过证书。V8证明该次输入下120秒压力运行通过；其他用户场景仍需按自己的碰撞/元数据/任务执行同类工作流。

## 用户60秒新录像与工作流

[本次60秒H.264录像](../configs/third_3/output/review_recording_20261105_v1/video.mp4)由用户原配置生成，
运行目录为`configs/third_3/output/review_recording_20261105_v1`，runtime为`ros2_20261105_011009_35a6b6`。
工作流的preflight、native execution、坐标对齐、轨迹评估及door admission全部通过；source及声明输入未变，stage/临时组合/ROS进程清理均为true。

| 录制与评估项目 | 实测结果 |
| --- | --- |
| 视频 | H.264、1280×720、12fps、720帧、60.000秒 |
| 文件大小 | 1,685,172字节 |
| 视频SHA256 | `4617c4fddebc278ae5898c5961e3d1f8ba662f470f945b2512ea749a28e7852d` |
| 媒体解码 | 单独ffmpeg完整解码检查退出0；媒体规格与文件hash另行核对 |
| 原目标覆盖 | 7/7，原等待/朝向评估通过；配置repeat保留 |
| 逻辑最小身体净距 | 0.25m |
| 最大速度/步长/转向率 | 0.9000000627m/s / 0.01500000105m / 2.500003992rad/s |
| 坐标转换最大误差 | 0m；按stage_world→canonical world→ROS map检查 |
| 与V8前60秒比较 | 同仿真时刻3599批的position/yaw最大差均0，phase差0 |

速度、步长和转向率的浮点尾差均在评估器既有SI slack内：各量为其SI单位的1e-4，原到达容差仅额外允许1e-6m。
录制评估覆盖逻辑轨迹及occupancy安全；实际PhysX胶囊、回程至62.7167秒及后续循环的证据由120秒V8补充，未把60秒片尾等同于完整回程结束。
同时间比较确认录制没有插入额外导航步或改变命令序列。

[25秒人形与状态根投影核对图](../tests/output/goal_door_native120_20261105_v8/recording_frame25_root_projection.png)对应真实人形位置；该单帧核对支持区分合法外侧让位、资产视觉offset与实际根位姿，用户路径没有改动。
它不代替全部720帧的动画质量判断。

### 按运行模式分别记录性能

| 模式 | 仿真时长[s] | 播放墙钟[s] | RTF | 结论边界 |
| --- | --- | --- | --- | --- |
| V8逐步物理audit + live RViz | 120 | 274.247831 | 0.43756 | 压力审计模式 |
| 本次12fps同步渲染/截图录制 | 60 | 248.826220 | 0.24113 | 总墙钟309.973438秒，含setup |
| 旧v2相同录制规格基准 | 60 | 249.274224 | 约0.24070 | 历史性能参考，不计当前功能通过 |
| 普通60秒无录制模式 | 60 | 133.822914 | 0.44835 | headless=True、render_viewport=True；无record、逐帧物理harness或额外RViz GUI |

新旧录制壁钟基本持平，不能声称总体录像RTF明显提升。已证实的性能改进是消除原6秒单tick计算阻断及相应通行问题；
同步渲染/截图链仍是录像主要开销，不能将数值solver优化比例套用到媒体吞吐率。

[普通无录制工作流](../tests/output/normal_no_record_20261105/workflow_report.json)同样strict7/7通过，source/声明输入未变、清理全部成功，
总墙钟194.572038秒。其条件为headless=True、render_viewport=True、60Hz控制/原场景120Hz物理，Marker发布保持开启；
没有录制、逐帧物理审计harness或额外RViz GUI。0.44835仅代表该受测配置，不能推广到任意GUI视角；
没有条件匹配的旧普通模式基线，因此不计算普通模式提升百分比。普通、物理audit+RViz及录制三种模式分别报告。

## 本地证据入口

- [V8原生结果](../tests/output/goal_door_native120_20261105_v8/result.json)、[监督器](../tests/output/goal_door_native120_20261105_v8/supervisor.json)、[首个闭门分析](../tests/output/goal_door_native120_20261105_v8/closure_analysis.json)。
- [全部闭门episodes](../tests/output/goal_door_native120_20261105_v8/all_closure_episodes.json)、[队列与回程摘要](../tests/output/goal_door_native120_20261105_v8/goal_queue_summary.json)、[02低速转向分析](../tests/output/goal_door_native120_20261105_v8/low_speed_turn_analysis.json)。
- [770项JUnit](../tests/output/goal_door_native120_20261105_v8/unit_regression.xml)、[最终缓存callback对比](../tests/output/goal_door_native120_20261105_v8/solver_callback_profile.json)、[非受控实体碰撞器核验](../tests/output/goal_door_native120_20261105_v8/uncontrolled_collider_audit.json)。
- [补充证据来源及SHA256](../tests/output/goal_door_native120_20261105_v8/supplemental_evidence.json)保存上述复制件原始来源和hash。
- [60秒工作流报告](../configs/third_3/output/review_recording_20261105_v1/workflow_report.json)、[媒体回执](../configs/third_3/output/review_recording_20261105_v1/video.json)、[同仿真时间比较](../configs/third_3/output/review_recording_20261105_v1/native_recording_comparison.json)。
- [普通无录制状态与计时](../tests/output/normal_no_record_20261105/status.json)、[普通模式完整工作流](../tests/output/normal_no_record_20261105/workflow_report.json)。
- 源/input完整指纹见V8与新工作流；声明输入基准为[v2工作流回执](../configs/third_3/output/review_recording_20261104_v2/workflow_report.json)，运行时文件差异由该基准逐文件计算。
- 该次冻结关键源：Goal`84d29aa1…56601c`、planner`39b5b140…17ae74`、passing`ac31a151…0de00b`；完整指纹以原生/工作流回执为准。
