# 仿真效率与模块职责

日常配置使用场景项目中的 `simulation.yaml`，更换场景无需修改性能控制代码。
独立 ROS 节点负责规划与避碰，Isaac 负责物理地图、根位姿与距离动画。

## 无窗口、无视口的纯仿真

无需查看主视口或录制视频时，在已有 YAML 的 `simulation` 中增加：

```yaml
simulation:
  headless: true
  render_viewport: false
```

`headless` 隐藏窗口，`render_viewport: false` 进一步关闭主视口更新。
后者默认是 `true`，普通 GUI 与录像配置保持正常渲染。
只关闭视口不会主动降低时间轴、行人更新或物理步进频率。

会话在物理地图和角色准备阶段保留必要更新，再关闭视口。
配置同时要求 `headless: true` 和录像关闭；冲突在启动前报错。
可用 `--headless --no-render-viewport` 临时覆盖，日常推荐保存到 YAML 后继续使用 `run PROJECT`。

依赖额外相机、传感器或外部图像输出的工程，应验证自己的采集链。
无视口测量不能代替人物可见移动和动画的验收。

`third_3` 已提供可选包装 [headless.toml](../configs/third_3/headless.toml)。从仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py run \
  src/pedstrain_simulation/simulation/configs/third_3/headless.toml
```

它复用原路线、时长与频率，关闭窗口、主视口和录像，并使用独立状态报告路径。
其他场景可采用同样的 `scenario` 包装方式，
文件只需维护运行差异，无需复制行人计划。默认 GUI 配置保持原样。

## 代码职责

ROS 的地图 Observer、交互、求解与显示分工见[控制责任](ros2_navigation.md#控制责任与协议)。
动态位姿逐帧读取，碰撞拓扑按变更失效。配置校验在一次加载中复用同一模型来源与性别组合的候选发现，避免逐人重复扫描目录。
每次加载重新校验资源，不使用跨加载的文件系统缓存。
USD/PhysX 场景操作在 Kit 主线程执行；报告与诊断查询不作为新的运动控制器。

## 如何比较性能

比较同一资产、人数、路线、种子、分辨率和渲染模式，排除启动及着色器编译阶段，并重复运行。
分别记录整帧时间、仿真时间、墙钟时间和实际路线进展；纯 Python 基准提速不等于整体 FPS 同比例提升。
重渲染场景应分别比较普通视口模式与无视口模式，并标注输出用途。

`simulation.duration` 是时间轴上的仿真秒数，不是现实计时，也不包含初始化。
例如实时率 RTF 为 0.5 时，60 秒仿真约需 120 秒播放墙钟时间，另加初始化与退出。
ROS普通运行每约5秒墙钟输出`ROS2_SIMULATION_PROGRESS`，包含会话时间和状态；最终摘要提供
`wall_time_s`及`playback_wall_time_s`，RTF按仿真推进量除以播放墙钟单独计算，录制/普通/物理audit模式分别比较。
`third_3` 保留 60 Hz 更新与场景原有的 120 Hz 物理步，`duration: 60.0` 表示 60 秒仿真。
性能更改不能通过缩小碰撞体、跳过错误检查或降低更新频率获得表面上的提升。
运行前后应比较实际位置、原目标到达、转向、停留与物理步进，不能只检查时间轴增长。

## 视觉卡顿的检查

60 Hz 仿真更新不等于现实时间每秒显示 60 帧。RTF 为 0.37 时，逐仿真帧显示的状态
实际约每秒更新 22 次；动作周期在墙钟时间中也会拉长。
应同时检查墙钟帧间隔、实际身体位移、所选后端的控制状态与可见动画，区分帧率不足、避让起停和资产问题。
同一场景的原定停留、目标占用或身体无法通过的通道也可能产生等待，应按路点和实际几何定位原因。
