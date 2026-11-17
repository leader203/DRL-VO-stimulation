# 仿真录像

录像观察配置选定的仿真，生成 MP4、PNG 帧和 JSON 报告，不发出导航或行为指令。
录制使用 `recording.ros2_recorder.record_ros2_simulation(session)`。

## ROS 2 录像

完整自动流程可直接运行 `run.py workflow PROJECT --record [--output NEWDIR]`。
父进程只为原生子进程加载配置的 ROS setup，保留用户 FPS、预热与相机，将视频写入新目录，
随后输出坐标 JSON/SVG、轨迹评估和 `workflow_report.json`；`--no-record` 跳过视频。
不安装依赖、不修改原 YAML。直接 `run` 的手动录制方式如下。

在已有工程的 YAML 中配置 `recording`，或新建 TOML 包装引用原 YAML：

```toml
scenario = "simulation.yaml"

[simulation]
headless = true
render_viewport = true
start_paused = false
console = false
report_path = "output/recording_status.json"

[recording]
enabled = true
output_path = "output/ros2_demo.mp4"
fps = 12
warmup_s = 0.0
# camera_view = "overhead"
```

运行 `run.py run PATH/TO/recording.toml`；ROS 工程需先加载其 `ros_setup_path`。
包装只覆盖显式参数，原场景、行路点、开始时间、速度、朝向与停留保持。
未覆盖时使用原工程的仿真时长和渲染宽高；`check` 显示有效 FPS、预热、相机和停滞限制。

ROS 录像必须开启 `render_viewport`、自动播放，并关闭初始暂停与控制台；不能使用关闭视口的 headless 包装。
`fps` 不得超过 `control_hz` 且须整除它，例如控制 60 Hz、录像 12 fps，每五个控制步采样一次。
捕获等待期间冻结仿真时间，恢复后继续原控制步，不用墙钟延时改变视频中的运动速度。
截图完成通知后继续等待 SDK 异步写盘，并检查 PNG 的尺寸、chunk CRC 与 IEND，再接受该帧。
视频帧数为 `round(simulation.duration * recording.fps)`；完整秒数与整数 FPS 相乘时，播放时长等于配置的仿真秒数。
`warmup_s` 是不录入视频的额外仿真区间，人物计划仍使用原仿真时钟。
含指定站立或等待的场景宜保留 `max_stationary_s: null`。

ROS 人物位姿批次须保持各自既有的平面高度；超出数值容差的高度变更会整批拒绝，不更新人物根或步态相位。
录制期间也检查完整人物根变换：外部写入的位置、朝向、缩放、剪切或倾斜超出容差时失败，不自动复位或争抢控制。
模型子节点的骨骼动画和单位换算缩放不属于根变换冲突。

`camera_eye_m` / `camera_target_m` 在 ROS 模式下始终使用规范米制 Z-up 世界坐标，经舞台单位/轴转换后布置相机，不能填写 ROS map 坐标。
它们也不跟随 `navigation.waypoint_frame: stage_world` 改成舞台单位；该选择只影响出生点和路点。
两字段同时省略时使用自动取景。默认 `camera_view = "oblique"` 保留斜视相机。
ROS 2 可设 `camera_view = "overhead"`，使用正上方的透视相机，沿规范世界 `-Z` 方向拍摄，画面上方对应 `+Y`。
俯视自动取景覆盖已知占用地图、原路线和完整人物高度。使用该选项时省略 `camera_eye_m` 和 `camera_target_m`。
手动相机同时填写这两个坐标，保留默认 `camera_view`。
ROS 录像在 `camera.camera_view` 记录实际视角，手动相机记为 `custom`。

输出要求全新名字；已有 MP4、同名 JSON、帧目录或悬空链接会在启动前拒绝，编码复用已安装的 ffmpeg，不安装依赖。
ROS 录像另需已有 ffprobe，在编码后核验实际帧数、分辨率、帧率及时长；缺失时启动前报错。
ROS 的每帧人物样本与运动统计使用 `ros_map` 米制坐标；元数据同时保存 map 的规范世界原点与偏航。
录像 `pedestrian_plans` 是规范米制 Z-up，`input_plans` 保留用户原框架数值，
`coordinate_contract` 标注框架、舞台指标与变换，不能把两组位置数值混用。
`record_ros2_simulation` 从新建会话的零时刻开始；重录使用新会话和新输出路径。
录制完成后需验证帧数、分辨率、帧率和时长，并保留实际时间戳与配置记录；截图捕获本身不等于视频已编码成功。
