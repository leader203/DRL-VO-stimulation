# 单位与坐标边界

ROS 输入路点由 `navigation.waypoint_frame` 明确解释；内部调度、距离与观测统一使用 SI。
`metersPerUnit` 表示一个舞台长度单位对应多少米，缩放变换还会影响最终物理尺寸，不能只检查该元数据。

| 量 | 公共单位与坐标 | 边界处理 |
|---|---|---|
| ROS 出生点、路点 | `stage_world`：舞台 WORLD 单位/上轴；`world_m`：规范世界 [m] | runtime 只规范化一次，再转换 ROS map；不改原输入 |
| 到达容差 | [m] | 不因 `waypoint_frame` 或舞台厘米制而缩放配置值 |
| 速度、距离、逻辑避让半径 | [m/s]、[m] | 由 ROS 求解器使用，应用根位姿时转换为舞台坐标 |
| 物理胶囊尺寸 | [m]，包含世界缩放 | 径向取物理圆截面的有效尺度，轴向另算；半径倍率调整保持实际总高 |
| 出生 yaw | [deg]，相对模型原朝向 | 模型前向基底校正后转为 ROS 根朝向 |
| 路点 yaw | [deg]，绕场景上轴的世界heading | 0°朝 +X；规范 90°朝 +Y，Y-up stage 对应 −Z；ROS 内部用 rad |
| 机器人 position / yaw_degrees | 父坐标系平移 [m]、场景上轴角度 [deg] | 平移按目的舞台单位换算，父层级变换继续生效 |
| `start_time`、`wait`、`duration`、物理步长 | 仿真时间 [s] | 与暂停、时间回退的处理一并核对 |
| ROS `request_timeout_s`、工作流 `--timeout-s` | 墙钟时间 [s] | 分别限制请求等待和原生子进程运行 |
| USD 动画采样 | time code | 秒与时间码使用该组成舞台的 timeCodesPerSecond，引用的时间缩放/偏移由 USD 组合处理 |
| 录像 fps | 帧/s | 依据实际仿真更新间隔检查采样能力，不以 USD TCPS 或显示 FPS 作为输出上限 |
| 相机 eye / target、裁剪范围 | 规范 Z-up 世界 [m] | 经 `world_to_stage` 转换，相机隔离父变换 |
| 相机焦距、孔径 | 物理毫米；USD 属性为舞台单位十分之一 | 24 mm 焦距写入 `0.24 / metersPerUnit`，36 mm 水平孔径写入 `0.36 / metersPerUnit` |

逻辑避让半径与物理胶囊半径是不同的量，两者均换算为米后各自验证。不能因数值不同就认定为单位错误。
实际运动速度应由世界位移和仿真秒差分核对。

## 输入路点与 ROS map

```yaml
navigation:
  backend: ros2
  waypoint_frame: stage_world
```

`stage_world` 接收 Isaac 场景中已经组合全部父变换的 WORLD 位置，采用实际 `metersPerUnit` / `upAxis`。
父节点存在旋转或缩放时，属性面板的 local translate 不能直接当 world 位置。
`world_m` 接收规范米制 Z-up 世界位置；旧配置省略该字段时保留 `world_m`。
CLI `init` 新 ROS 项目默认明确 `stage_world`，Python 创建 API 的同名参数默认仍为 `world_m`。

舞台长度单位为 `u` 米时，Z-up 位置规范化为 `(x,y,z)*u`，Y-up 为 `(x,-z,y)*u`。
例如 Y-up 厘米场景 WORLD `[100,0,-200]` 对应规范世界 `[1,2,0] m`。
随后 `p_map = Rz(-yaw_rad) * (p_world - origin_m)`；`origin_m` 始终是规范世界 [m]，
`bounds_m`、`floor_height_m` 始终是 ROS map [m]。`camera_eye_m` / `camera_target_m` 始终是规范世界 [m]。
`speed: 0.9` 和 `arrival_tolerance: 0.2` 仍表示 `0.9 m/s`、`0.2 m`，不跟着路点改成厘米数值。

`third_3` 已明确 `stage_world`，其源舞台为米制 Z-up，所以原出生点和路点数字保持不变。
运行 bootstrap 中 `input_plans` 保留原值，`world_plans` 是规范米制，`plans` 是 ROS map；
自动工作流输出的 `coordinate_alignment.json` / SVG 逐点核对这三者与原生舞台 WORLD 的往返误差。
SVG 的虚线只连接原始路点，不表示规划或实际轨迹。

ROS 的 SI 和右手系约定见官方 [REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)。
舞台指标适用于完整世界变换后的长度，不能据原始 translate 数值推断，见
[OpenUSD 长度单位](https://openusd.org/release/api/group___usd_geom_linear_units__group.html)。

地图数组行号沿 ROS +Y 增长，列号沿 +X 增长；cell `(0,0)` 的左下角为 `origin_xy_m`。
PGM 图片和 SVG 屏幕坐标向下，因此输出时翻转纵向；map 自身已经在 ROS 框架中，不再叠加一次世界 map 偏航。
消息语义依据官方 [OccupancyGrid](https://github.com/ros2/common_interfaces/blob/jazzy/nav_msgs/msg/OccupancyGrid.msg)
和 [MapMetaData](https://github.com/ros2/common_interfaces/blob/jazzy/nav_msgs/msg/MapMetaData.msg)，
本项目采用 `-1/0/100` 表示 unknown/free/occupied。

## 出生旋转与路点朝向

角色级`yaw`是相对模型作者朝向的初始旋转；路点`yaw`是到达后的绝对世界heading。
ROS按角色`controlRig:forwardAxis`和`upAxis`归一化视觉，再计算
`root_yaw_map = radians(spawn_yaw) + authored_forward_yaw - map_yaw_rad`。
例如作者前向为−Y的Z-up模型，出生`yaw:0`对应根heading −90°；资产基底校正不平移出生点。
出生旋转数字不能直接当成归一化根的+X heading。

路点heading不加该资产前向偏置：canonical `0°`朝+X、`90°`朝+Y。
Y-up stage的右手正90°对应stage −Z方向；位置按`(x,-z,y)*metersPerUnit`归一化。
`navigation.waypoint_frame`只选择位置输入单位/上轴，角度始终为deg，速度和到达容差仍为SI。
路点`yaw:null`保留到达方向，内部ROS heading使用rad。

## USD 组合与资产

ROS 角色资源通过独立重定向和视觉包装处理资产单位/上轴；根运动只接收 ROS 命令，
步态按实际移动距离推进，静止过渡到 Idle，不回写源角色或动作库。

机器人 `asset_path` 的有效 `metersPerUnit`、`kilogramsPerUnit`、`upAxis` 必须与环境一致。
`scene.layers` 与机器人 `layer_path` 的已声明指标也必须一致；未声明指标的覆盖层按目的舞台坐标编写。
不匹配会在创建运行层之前报出源文件及两侧数值。请准备已转换的独立资产/覆盖层，再将配置指向它。
只改元数据不会自动转换几何、关节、质量和惯量；不能把厘米资产的元数据改成米来冒充完成转换。
`source_prim_path` 选择的模型根应带有需要的作者变换；USD 子树引用不会自动带入未被引用的祖先变换。

运行层保留环境的长度、质量和时间元数据。没有单位元数据的完整机器人资产使用 USD 的默认指标，
不能按文件名推断其单位；覆盖层与完整资产的规则不同。
相关单位定义见上文 OpenUSD 官方说明；组合校验不等于自动几何/物理转换。

## 时间与导航设置

`timeCodesPerSecond`、`framesPerSecond`、Kit 时间轴设置、PhysX 步频和录像 FPS 各有用途。
实际更新间隔从连续仿真秒数计算。录像按整除控制频率的网格采样，并记录 `maximum_simulation_step_s`。

相机物理单位见 [NVIDIA 相机单位说明](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html)。
常规视场角由孔径与焦距比例决定，因而两项同时错相同倍数时画面仍可能看似正常；验证须比较物理量。

## 检查范围

单位回归覆盖米/厘米/毫米边界、缩放与轴方向、时间码以及真实 USD 物理形状解析。
ROS 的位置转换覆盖米/厘米和 Y/Z-up，仍受单层物理地图、资产装配与门 schema 各自限制；
坐标组件通过不能推广成任意单位、门格式或资产都能运行。
组合检查不是通用资产转换器，也不替代新角色、斜坡或动态碰撞的原生验收。
