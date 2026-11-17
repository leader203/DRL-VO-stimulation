# ORCA、SFM 与避让朝向验证（2026-11-05）

> 本页为避让修复完成时的冻结验收。后续路径规划、漫游与ROS2协议更新及新源码验收见[路径导航审查](ros2_navigation_review_20261105.md)，本页数值仍对应其原回执。

本轮修正了普通避让的无必要反向、缓存引导回指、让位候选的大转角，以及实测接触边界的微速停滞。最终源码通过 **787 项 CPU 回归、120 秒原生物理验收和原用户配置的 60 秒录制工作流**。普通已前向后的明显反向移动采样由旧 120 秒记录的 107 个降为 **0**，普通引导点回指由 4 次降为 **0**。

[机器汇总](../tests/output/avoidance_heading_20261105/summary.json) · [完整新旧统计](../tests/output/avoidance_heading_20261105/evidence/avoidance_heading_final_summary_20261105.json) · [新视频](../tests/output/avoidance_heading_20261105/recorded60/video.mp4)

## 输入和参考边界

项目改动限于 simulation。当前用户 YAML 逐字节未改，出生点、路点、朝向、速度、停留、开始时间、往返、人数和到达容差保留；全部 configs 文件也与本轮修改前哈希一致。当前 third_3 仍为 3 人、0.9m/s、0.2m 原到达容差、60 仿真秒、60Hz 控制、原 USD 120Hz 物理频率；身体半径0.25m、最低行人净距0.25m、地图边距0.03m未放宽。

对照了本机 `sfm_orca_pack_20261029` 的实际 AgentManager 与 ORCA glue：保持同一状态快照计算社会速度、求解安全速度、再单次积分的顺序。该包不包含完整 LightSFM/行为树源码，因此不声称本项目与其全部行为逐项等价。本轮未改随附 RVO2 的13个已登记核心文件，bridge 修改已更新 provenance 哈希。

模型分工参考原始 [SFM 论文](https://arxiv.org/abs/cond-mat/9805244)与 [ORCA 研究说明](https://gamma-web.iacs.umd.edu/ORCA/)：本项目用社会力产生软偏好，由 ORCA 约束速度，再加执行朝向和实际完整运动弦检查。路径占用、门与有限退让属于项目协调层；不把它们冒称为原论文自带的通行调度保证。

## 已确认问题与修复

| 严重性 | 可复现原因 | 通用修复与证据 |
|---|---|---|
| P1 | 合法0.8m中心距、0.2m/s两人相向时，软排斥将偏好推到原路线后约152°；执行层会锁存该方向原地转身 | 消除软偏好的负纵向，ORCA候选同时满足路径前半平面和本步可转扇区；旧锁按当前路径重验。旋转0/1.07rad闭环旧2失败→新通过，保留原目标、净距和角速 |
| P1 | 已认证可见捷径没有持久提交，前方新障碍出现后回选被捷径取代的旧拐点 | 普通路径、目标准入/离开与Passing均保留剩余路径并提交已认证捷径；阻挡时重新规划。连续行走60tick后新障碍用例旧回到x=2<实际x=2.064，修复后向前重新规划 |
| P1 | Passing仅试侧向得分最高4个点，再偏爱较短通路，漏掉少转角的可行候选 | 保留原24方向和全部净空/通路约束，优先可见路径前向及小转角，惰性验证。真实17.15s快照从160.20°转向降至4.80°，仍只让出0.398m |
| P1 | 距安全点1.2cm就假定可放行，实际位置仍无完整通路，19.033s原生退出 | 以实际位置的完整通路证书决定ready；未到位继续原有界段，端点受阻在原期限内等待。同几何失败重试0.5s，身体/障碍/目标变化即时重验，失败不续退让预算 |
| P1 | 极小非零速度被C bridge当零向量跳过朝向检查，零速cap也可能漏出残余速度 | 每个非零候选均检查朝向；零速cap只允许精确零速度。5e−8m/s独立用例旧2失败→新通过 |
| P1 | 横向墙面软力主导偏好，排除反向解后重复投影同一偏好，形成切向微速；随后正向微速又不触发恢复 | 普通且接近其他行人的低进展状态使用带原侧向惯性/让行偏置的路径驱动重询；门/目标临时指令保留原偏好。真实44s和48s几何闭环分别覆盖反向与正向微爬，速度、角速及全部净空不放松 |
| P1 | RVO float32接触切线略向内，双精度严格扫掠检查只能把同一速度射线缩成微速 | 仅对float量级残差尝试极小向外射线，经同一ORCA过滤、原执行角域、微小速度变化界、静态和全部受影响peer的exact弦检查；动态仍整批复验。失败保留原缩速/停步，不增用户净距，不推动静止者 |

实现：[solver.py](../ros2_navigation/solver.py)、[passing.py](../ros2_navigation/passing.py)、[goal_occupancy.py](../ros2_navigation/goal_occupancy.py)、[bridge.cpp](../ros2_navigation/vendor/bridge.cpp)。回归和小型、无pickle的真实几何输入见 [navigation tests](../tests/test_ros2_navigation.py)、[passing tests](../tests/test_ros2_passing.py)、[goal tests](../tests/test_ros2_goal_occupancy.py)及[fixture来源](../tests/fixtures/README.md)。

前向进展判据只是选择软偏好的条件，不是最低执行速度。受限空间仍可停步；没有ORCA/角域交集的停步由实际下一步净空检查保障，不声称该零速度满足整个预测时间内的全部ORCA半平面。

## 坐标和视觉前向

当前舞台是1m/unit、Z-up、map原点0/yaw0，原输入为已组合的 `stage_world`，所以当前Isaac世界坐标数字与ROS map相同。工作流新生成的坐标表最大误差为 **0m**。Y-up一般变换仍为 `(x,-z,y)*metersPerUnit`，随后 `p_map=Rz(-map_yaw)*(world_m-origin_m)`，没有改这条链。

当前三个真实资产的forward为MINUS Y、up为Z，Visual子节点校准后与控制根的+X一致。896组资产/场景Y/Z-up、厘米/米、不同前向轴、非零map yaw与±π附近heading组合，方向误差最大约1.01e−15；相关144项测试通过。出生yaw是相对资产作者朝向的旋转，路点yaw是绝对身体heading，两者语义保留。

新120秒的21600个实际PhysX根姿态，按当前或前一个60Hz控制tick匹配，未匹配0；最大位置误差约2.61e−7m、朝向误差3.41e−7rad。此一tick读回时序没有被当作坐标偏移。所有12935个非零移动命令（阈值1e−12m/s）的速度方向与身体朝向均一致。

[坐标组合证据](../tests/output/avoidance_heading_20261105/evidence/coordinate_heading_review_20261105.json) · [新物理审计](../tests/output/avoidance_heading_20261105/native120_v4/audit.jsonl) · [新工作流坐标表](../tests/output/avoidance_heading_20261105/recorded60/alignment/coordinate_alignment.json)

## 最终验证

| 检查 | 结果 |
|---|---|
| 全套回归 | 787 passed，0 failure/error/skip；[JUnit](../tests/output/avoidance_heading_20261105/unit_regression_complete.xml) |
| 风险边界 | 旧实现/关闭对应修复的真实回归分别失败，修复后通过；包含索引子集/重排、严格零cap、接触切向、真实44/48s几何与门/目标控制隔离 |
| 原生120s | 完整通过，3人各7200物理样本；首轮7/7原目标均已进入原0.2m范围，最晚49.2s；[result](../tests/output/avoidance_heading_20261105/native120_v4/result.json) |
| 身体净距 | 最小逻辑扫掠净距0.250000000000m；实测物理胶囊约0.249999956m |
| 控制隔离/清理 | 原生IRA/NavMesh扩展关闭；场景关闭、临时组合删除、ROS子进程退出均成功 |
| 原配置录像 | 60秒、1280×720、12FPS、720帧H.264；严格7/7原目标到达，完整ffmpeg解码通过 |
| 录像时序 | 3599个控制批次逐项匹配120s验证的前60s；位置、朝向、时间和phase差异均0；[比较](../tests/output/avoidance_heading_20261105/recording_native_comparison.json) |
| 最终指纹 | 原生121项、录像工作流117项source/声明input哈希复核均无不匹配；[核验](../tests/output/avoidance_heading_20261105/final_provenance_verification.json) |

原生和录制都在当前代码、真实PhysX地图及DDS反馈上运行，不以纯CPU或冻结几何测试代替。未安装或升级依赖，未修改SDK、系统IOMMU或源USD。原有路线与实际到达容差没有为获取通过而放宽。

## 同口径反向与等待统计

冻结的同一分析脚本把用户返程、到达后指定转向、门安全返回、目标/通道协作、入口重新定向和临近内部引导点单列。material指标要求速度大于0.03m/s、离内部引导点足够远；“已前向后反向”还要求同一原路线段已建立前向。它是可复现诊断口径，不把所有大角度转向都判成避让错误。

| 指标 | 整理前60s | 旧V8 120s | 本轮v4 120s |
|---|---:|---:|---:|
| 已前向后的普通明显反向移动采样 | 75 | 107 | **0** |
| 普通引导点回指事件 | 2 | 4 | **0** |
| 转向速率超限 | 0 | 0 | **0** |
| 移动速度/身体heading错配 | 0 | 0 | **0** |

新场景普通行走低于0.03m/s的最长连续段，包含合法初始重新定向/返程时按原生观察为0.75s；进一步剔除这些条件、近guide及协作恢复后，最长为0.30s。不能宣称全程零等待：原目标占用的最长 `goal_wait` 为8.716667s，按原指定停留和离开通路规则保留。

有限恢复没有超过0.4m：门控实测最大0.385169m，目标队列0.388942m。原路线返程、用户指定到达朝向、门关闭后的返回和受限空间必要退让仍是合法动作，不能为了表面“永不回头”而取消安全通行。

[同口径完整统计](../tests/output/avoidance_heading_20261105/evidence/avoidance_heading_final_summary_20261105.json) · [统计脚本](../tests/output/avoidance_heading_20261105/evidence/avoidance_heading_metrics_20261105.py)

## 性能与失败记录

17.15s同一CPU快照的新候选选择为0.105–0.115s、一次完整规划，旧策略约0.63s/五次规划；这只是该快照的计算测量，不是整体实时率提升。[候选/证书和计时](../tests/output/avoidance_heading_20261105/evidence/passing_choose_final_benchmark_20261105_after.json)。

本轮120秒物理审计播放墙钟329.605s、总墙钟390.756s，RTF约0.364；录像播放墙钟290.719s、总墙钟351.891s，RTF约0.206。控制保持60Hz，当前仍未达到实时；不同审计、渲染和录像模式不混算性能。

前三次原生尝试分别在19.033s提前ready、45.433s横向软力微爬、49.133s正向微爬退出，均保留为false的中间证据；只有v4完整120s和最终60s工作流计为通过。这些故障均转成独立可复现回归，未通过改时间预算、路径点或最低净距掩盖。

复验从仓库根目录使用既有环境：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python -m pytest \
  src/pedstrain_simulation/simulation/tests -q

UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py workflow \
  src/pedstrain_simulation/simulation/configs/third_3/recording.toml \
  --record --require-route-goals
```

更换USD、路线、角色、尺寸或目标占用关系后仍需按新输入运行验收；本次没有证明任意场景、任意动态障碍或无限时长均无问题，也未作与IRA效果高低的对照结论。

![修复后60秒视频](/project/EXP/IsaacLab/src/pedstrain_simulation/simulation/tests/output/avoidance_heading_20261105/recorded60/video.mp4)
