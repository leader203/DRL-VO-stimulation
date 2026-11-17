# ROS 2 录制审查与实录 — 2026-11-03

> 历史验收快照：本报告绑定当次回执中的source/input指纹、运行模式与时限。正文的“当前”指该冻结版本；后续整理后的源码须重新验收。现行操作与版本索引见[文档索引](index.md)，整理后结果见[维护记录](project_maintenance_20261105.md)。

已按当前 `third_3` 用户配置录制完整视频。主 YAML、原始行路点、人数、速度、朝向、开始时间、停留及到达容差均未修改。
新增 [recording.toml](../configs/third_3/recording.toml) 仅引用主 YAML 并启用录像，控制仍为 60 Hz。

## 成片

| 项目 | 实测结果 |
|---|---|
| 视频 | [third3_ros2_20261103.mp4](../configs/third_3/output/third3_ros2_20261103.mp4) |
| 分辨率 / 编码 | 1280×720，H.264 |
| 播放时长 / 帧率 / 帧数 | 60.000 秒，12 FPS，720 帧 |
| 预热 | 0 秒，保持配置默认值 |
| 采样时间 | 第 k 帧为 k/12 仿真秒，全部与控制时钟对齐 |
| 最大仿真步长 | 0.016666667 秒 |
| 三人原始路点覆盖 | walker01 2/2；walker02 2/2；walker03 3/3 |
| 3599 个控制批次复核 | 静态扫掠、速度、角速与行人间距检查通过 |
| 最大单步位移 | 0.015000002 米以下 |
| 最小逻辑身体扫掠净距 | 0.2499806 米，名义配置 0.25 米 |
| 源码 / 输入指纹 | 运行中和录制后均保持一致 |
| 退出 | scene、临时目录、自有 ROS 进程全部清理 |

成片经 ffprobe 核验，并由 ffmpeg 完整解码通过。视频大小为 1,783,988 字节；
SHA256 为 `fe4b7eb1e414731dff2e58efcdcdc1100a8c6a225959322be1f7fd18ddf4e559`。
人工抽检了初始、中段画面；自动相机覆盖完整路线和人物高度，保留原场景光照及真实墙体遮挡。

记录的静态与身体扫掠结果来自实际控制轨迹及物理生成的地图，不将其冒称逐帧物理接触深度测量。
本轮没有变更导航算法；前一轮实际 PhysX 人物根与胶囊观测见[导航专项记录](ros2_validation_20261103.md)。

## 本轮审查修复

- 接通 ROS 2 独立录像，修复合法配置能通过检查、运行却拒绝录制的不一致；旧 IRA 录像入口保持。
- 补全预检中的相机、预热与停滞设置，检查 FPS 与控制频率整除，复用 fresh 路径保护和现有 ffmpeg/ffprobe。
- 使用独立 camera、RenderProduct 和 viewport；明确相机的规范世界坐标与样本的 ROS map 坐标，支持舞台单位与轴转换。
- 捕获时冻结时间及人物状态，恢复后允许只提交播放状态而未推进时间的更新，不产生控制跳步。
- 修复 SDK 截图 future 完成但 PNG 尚未写完的实际竞态：等待异步写盘，验证尺寸、chunk CRC 和 IEND。
- 报告写入、场景清理和应用关闭独立尝试；次生错误附于原始异常，避免报告失败留下应用运行。

CPU 全量回归 **571 项通过**，simulation 范围 Ruff 检查通过。新增回归包含配置拒绝、预热采样、相机单位/父变换、延迟写盘、损坏帧、超时及最终清理失败。
首次 8 秒试录发现写盘竞态，修复后的新输出试录成功生成 96 帧，之后才执行完整录制。
未安装依赖、修改 SDK 或改变系统 IOMMU 设置。

## 复录与产物

从仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py run \
  src/pedstrain_simulation/simulation/configs/third_3/recording.toml
```

成片已经存在，复录前必须在包装中选择新的 `recording.output_path`，已有视频、JSON 和 PNG 帧不会被覆盖。
配置 `simulation.duration` 是视频记录的仿真区间，不是墙钟耗时。本次总墙钟约 325 秒，含初始化、渲染采样和编码，成片播放仍为 60 秒。

- [录像元数据与原计划](../configs/third_3/output/third3_ros2_20261103.json)。
- [独立轨迹、帧时钟与指纹复核](../configs/third_3/output/third3_ros2_20261103_validation.json)。
- [运行与清理结果](../configs/third_3/output/recording_status_20261103.json)。
- [CPU JUnit](../tests/output/ros2_recording_20261103/cpu_writer_final.xml)。

视频、帧与运行报告是本机输出，默认不随源码提交。参数说明见[录像文档](../recording/README.md)。
