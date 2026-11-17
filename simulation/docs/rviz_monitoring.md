# RViz2 运行监控

ROS行人会话会生成专用RViz2配置，显示静态PhysX OccupancyGrid、当前动态轮廓、原目标规划路线及实测行人根位姿。
行人的规划、SFM/RVO2避碰和执行继续使用现有ROS控制链；监控入口只启动RViz2。

每个新会话在实际运行输出目录中生成：

- `monitor.json`：会话namespace、ROS domain、setup脚本、话题和配置位置。
- `monitor.rviz`：绑定该会话UUID的话题，Fixed Frame为`map`，初始视图覆盖整张地图。

用另一个终端连接仍在运行或暂停的会话。`workflow --output NEW_RUN`运行期间，在该目录的
`ros2_*`子目录可找到`monitor.json`；可用`rg --files --no-ignore NEW_RUN -g monitor.json`列出路径。
普通状态报告通常在退出时落盘，已有status可能属于旧运行；活动监控优先选本次新生成的manifest。

在仿真保持运行或暂停时，从仓库根目录执行：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py monitor /absolute/runtime/output
```

参数也可使用`monitor.json`或包含`output_directory`的运行状态JSON：

```bash
UV_PROJECT_ENVIRONMENT=env_isaaclab uv run --no-sync python \
  src/pedstrain_simulation/simulation/run.py monitor /absolute/status.json --dry-run
```

`--dry-run`打印实际RViz命令、domain和namespace。入口为子进程准备该会话声明的ROS环境，并传入
`use_sim_time:=true`；不需要修改用户YAML。普通仿真启动方式见[工作流](workflow.md)。
`rviz2`和标准ROS消息包需已存在于声明的ROS安装中；工具不安装依赖。

## 图层含义

| 显示 | 来源与单位 |
| --- | --- |
| Static map | PhysX采样的静态`nav_msgs/OccupancyGrid`；原点及分辨率使用ROS地图坐标[m] |
| Route: 行人名 | 每名行人当前原目标的全局计划`nav_msgs/Path`；规划结合静态图与非受控实测物理轮廓，临时让行可能偏离它 |
| 彩色圆盘及同色朝向箭头 | 控制批次采样的实际行人根位姿[m, rad]；圆盘半径采用该身体的配置值[m] |
| 半透明目标球及YAML编号 | 用户原路点和三维到达容差[m]，按bootstrap中的ROS坐标显示 |
| 橙色箭头 | 已发出的速度[m/s]投影到1秒后的方向和长度，命名为`issued/velocity_1s` |
| 黄色线及端点 | 当前请求的引导点，包含普通导航和临时让行；来自已发出的guidance诊断 |
| 橙色动态轮廓 | 当前实测障碍物的平面投影；高度采用其真实最低高度[m] |
| 门标签 | 实测角度[deg]及`fully_closed`/`fully_open`状态；无有效观测时显示unobserved |
| 青色portal线 | 门描述符中的doorway segment[m]，用于核对门洞的地图位置 |
| 低透明度青色轮廓 | 门描述符中的完整安全sweep包络[m]；当前实测门扇由橙色动态轮廓表示 |
| Queue与安全点线 | 目标占用队列、临时等待点或闭门让位目标 |

Marker的`observed/`、`requested/`、`issued/`命名分别区分实测、目标和指令。
行人标签中的`control phase`来自控制器调度诊断；圆盘位置和朝向箭头来自采样的根位姿。
门参考线仅在已有描述符提供相应几何时显示。安全sweep表示门扇运动保护范围，门的接近触发距离仍取原门配置。
已接管的门继续通过门控协议处理；门洞处的实际通行应同时观察门状态、安全sweep及当前请求引导点。
原路点来自Isaac场景定位时，已有`stage_world → canonical world → ROS map`转换在运行准备阶段完成；
监控直接使用同一bootstrap及同一批次的ROS数据，不再进行轴交换、缩放或yaw补偿。
Map、Marker和Path均使用`map` frame。监控自身提供下述只读参考TF，无需外部TF发布器。

默认仅打开Displays dock，窗口为1600×1000，TopDown视图按地图范围完整取景。
Move Camera和Focus Camera工具可调整视图；需要Views dock时可从RViz的Panels菜单自行添加。

## Canonical world参考TF

监控向`/tf_static`发布Reliable / Transient Local参考关系：parent为`map`，child为
`<实际会话namespace>/world_m`，TF frame名称去掉开头的`/`。准确名称保存在`monitor.json`的
`canonical_world_frame`字段，唯一会话后缀避免与用户已有的world/odom帧重名。
这条参考关系添加map的子帧，不覆盖map已有的父变换；导航、避碰与控制都不读取它。

该子帧表示归一化的米制Z-up canonical世界，并非厘米或Y-up的原始USD坐标。
与已有`StageFrame`转换相同，设置的地图原点为`origin_m`[m]、地图yaw为`yaw_rad`[rad]时：

```text
p_map = Rz(-yaw_rad) * (p_world_m - origin_m)
translation(map -> world_m) = -Rz(-yaw_rad) * origin_m
rotation(map -> world_m) = Rz(-yaw_rad)
```

例如origin=[1,2,3]m、yaw=90°，该TF平移为[-2,1,-3]m，四元数z=-√0.5、w=√0.5。
原始USD坐标须先按场景单位及up-axis转成canonical世界，再用此关系核对ROS map坐标。
标准消息为`tf2_msgs/TFMessage`中的`geometry_msgs/TransformStamped`；参考TF发布一次并可供晚加入的RViz读取。

Markers 最多按 10 Hz 仿真时间更新，行人控制按导航设置的 `control_hz` 运行。暂停时保留最后完整快照；重启仿真后应连接新会话的监控文件。圆盘表示根位置和水平身体范围，骨骼动画可在 Isaac 视口观察。

## 晚启动与会话隔离

Map、每名行人的Path及完整MarkerArray均使用Reliable / Transient Local / Keep Last 1。
RViz配置订阅相同QoS，晚启动或在暂停后打开仍能读取当前图、计划和标记。
MarkerArray每次先清理该专用话题上的旧标记，再发送完整快照，因此移除的障碍不会残留。

话题位于实际`<ros_namespace>/session_<UUID>/...`之下；配置同时保留该次运行的ROS domain。
每个运行目录有自己的配置，切换运行时应选择对应目录。
发布节点退出后，DDS的Transient Local历史随发布者结束；离线录像/轨迹查看使用保存的验证文件。
旧版运行目录没有monitor文件时，需要启动一次带监控发布的新会话。

预设工具仅包含相机、选择和测量操作，不含导航目标、初始位姿或点发布工具。
`monitor`入口没有启动Kit、Nav2或其他运动控制器的逻辑，也不会发送行人控制请求。

## 接线与消息依据

`node.py`的`NavigationMonitor`在控制命令发布后读取原始观测和诊断；重复序列仍走既有缓存返回路径。
`transport.py`在发现真实ROS地图后生成会话文件，通过`monitor_manifest`暴露位置。
监控消息生成失败单独记录，不改变已发出的控制批次或序列。

消息常量与字段依据本机Jazzy的
`/opt/ros/jazzy/share/visualization_msgs/msg/Marker.msg`和`MarkerArray.msg`；
RViz的Map/Path/MarkerArray/TopDownOrtho插件名依据本机`rviz_default_plugins/plugins_description.xml`。
发布/订阅的QoS字段及Late Joiner语义也可核对
[RViz Jazzy QoS属性源码](https://github.com/ros2/rviz/blob/jazzy/rviz_common/src/rviz_common/properties/qos_profile_property.cpp)。
默认视图属性可核对本机`nav2_bringup/rviz/nav2_default_view.rviz`，使用该文件作格式参考不启动Nav2。
