# 项目复审与按用户配置实录 — 2026-11-04

> 历史验收快照：本报告绑定当次回执中的source/input指纹、运行模式与时限。正文的“当前”指该冻结版本；后续整理后的源码须重新验收。现行操作与版本索引见[文档索引](index.md)，整理后结果见[维护记录](project_maintenance_20261105.md)。

当次复核配置、工作流、录像、退出清理及已有导航回归，修复发现的确定问题并重新录制。
本轮未修改原始路点、人数、速度、开始时间、停留、到达容差、身体尺寸或导航算法。

## 本轮修复

| 问题 | 修正 |
|---|---|
| 工作流不能真正接受录制 TOML 入口 | 原生成包装又引用 TOML，被有意仅接受 YAML 的加载器拒绝。现引用原始 YAML，并展开全部生效覆盖参数；不放宽全局递归加载规则。 |
| 展开配置可能遗漏用户覆盖 | 对仿真、录像、导航、替换人口、机器人及附加层做往返验证，保留绝对引用；附加层不重复拼接，机器人清空也保持。 |
| 取消后缺少最终报告 | 保留 failed/cancelled 报告再传播原 KeyboardInterrupt，不将取消写成成功。 |
| 超时仅等待主进程 | 清理本次自建进程组中的后代，覆盖主进程先退出而子进程忽略 TERM 的实际反例。 |
| 非 BMP 字符路径生成非法 TOML | 按 UTF-8 Unicode scalar 写路径，真实回读含中文和 emoji 的项目/输出路径。 |
| 截图异常后的状态读取错误跳过清理 | 分别尝试状态核对、恢复播放、销毁窗口、移除相机、解绑层和恢复原视口；次生错误附注到原错误。 |
| 相机投影或裁剪外写未被识别 | 新增 projection/clipping range ownership 检查，并保存实际读回与单位。 |

新回归均针对实际复现问题完成修复前失败、修复后通过。源码修改集中在工作流与录像视口，场景资源及用户配置不回写。

## 采用的用户配置

入口为 [third_3/recording.toml](../configs/third_3/recording.toml)，引用原 `simulation.yaml`：

- 三名行人，60 仿真秒；录像 1280×720、12 FPS、预热 0 秒。
- 未提供显式相机，使用覆盖完整路径与人物高度的自动全景机位，保留原光照与遮挡。
- 控制 60Hz、当前场景物理设置保留；录制输出重定向到全新目录，旧视频不覆盖。
- 原 `stage_world` 输入及 SI 速度、净距、容差保持，坐标契约没有变更。

## 验证结果

| 检查 | 结果 |
|---|---|
| 全量 CPU 回归 | **670 项通过** |
| simulation 范围 Ruff | 通过 |
| 完整自动工作流 | passed=true，全部原始路点 **7/7** 到达 |
| 成片 | H.264、1280×720、12 FPS、720 帧、60.000 秒 |
| 独立全片解码 | ffmpeg 无错误完成 |
| 帧时钟 | 全部第 k 帧时间与 k/12 仿真秒一致 |
| 该次输入/源码指纹 | 与录制报告一致，用户输入计划与原配置一致 |
| 最大控制步位移 | 0.015000002m 以下 |
| 最小执行身体扫掠净距 | 约 0.25m，维持原配置 |
| 门首次准入 | 5 条记录，零超距准入 |
| 坐标往返 | 最大误差 0m |
| 清理 | stage、临时目录、自有 ROS 进程清理通过 |

本次总墙钟约 310.55 秒，包含初始化、渲染、写盘与编码；成片播放时长为 60 秒。
轨迹和地图的扫掠核验不是逐帧 PhysX 接触深度测量；已有坐标、门控和物理专项范围见[通用性验收记录](generality_workflow_validation_20261104.md)。

## 文件与复录

- [新视频](../configs/third_3/output/review_recording_20261104_v2/video.mp4)
- [工作流报告](../configs/third_3/output/review_recording_20261104_v2/workflow_report.json)
- [录像元数据](../configs/third_3/output/review_recording_20261104_v2/video.json)
- [最终独立复核](../configs/third_3/output/review_recording_20261104_v2/final_verification.json)
- [本次站位图](../configs/third_3/output/review_recording_20261104_v2/alignment/coordinate_alignment.svg)
- [本次轨迹评估](../configs/third_3/output/review_recording_20261104_v2/evaluation.json)
- [CPU JUnit](../tests/output/review_recording_20261104_final_cpu.xml)

从仓库根目录复录，省略输出参数会创建新目录：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py workflow \
  src/pedstrain_simulation/simulation/configs/third_3/recording.toml \
  --record --require-route-goals
```

工作流自动准备子进程 ROS 环境，不安装依赖。产物为本机输出，默认不随源码提交；原失败输出保留用于回归追踪。
