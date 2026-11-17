# 行人 YAML 参考

简洁 YAML 用于设置场景和每个行人的模型、初始位姿、行为与时间安排。
新工程和编辑步骤见[配置工作流](workflow.md)；此页说明字段及其运行语义。
坐标示例需要替换为自己场景中的实际位置，不能视为已验证路线。

按需要查阅：[角色字段](#角色字段与覆盖顺序)、[路点停留与朝向](#导航点停留与身体朝向)、
[行人性别](#行人性别与模型选择)、[运行与机器人](#运行机器人和录像字段)、[导航参数](ros2_navigation.md#配置)。
创建/运行命令只在工作流维护；路径、默认来源和历史配置只在 [配置索引](../configs/README.md) 维护。

## 完整混合示例

```yaml
schema_version: 1
navigation:
  backend: ros2
  waypoint_frame: stage_world
  settings_path: ./navigation_settings.yaml
scene:
  asset_path: ./room.usd
simulation:
  duration: 60.0
  seed: 42
pedestrians:
  assets: ./characters
  defaults:
    speed: 0.9
    yaw: 90.0
    wait: 0.5
  actors:
    commuter:
      position: [1.0, 1.0, 0.0]
      yaw: 0.0
      start_time: 3.0
      repeat: true
      waypoints:
        - position: [1.0, 3.0, 0.0]
          wait: 2.0
          yaw: 90.0
        - position: [3.0, 3.0, 0.0]
          wait: 4.0
          yaw: -90.0
    visitor:
      asset_path: ./characters/visitor.usd
      position: [3.0, 1.0, 0.0]
      behavior: stand
      yaw: 180.0
    roamer:
      asset_path: ./other_characters
      behavior: wander
      start_time: 5.0
```

本例有三人：`commuter` 沿路线往返；`visitor` 用指定模型站立；`roamer` 从自己的模型目录选角并随机出生漫游。
人数由 `actors` 条目自动计算。`commuter` 有非空路线且未指定默认行为，因此自动使用 `walk`。
`defaults.yaw` 提供初始出生旋转，角色自己的 `yaw` 覆盖它；导航点的 `yaw` 是独立的到达后世界朝向。

所有路径相对本 YAML。请提供实际场景、角色目录/文件和导航设置文件；资产和纹理不会被复制。
角色文件需要可独立重定向的人体骨架和 ControlRig 标记。
ROS 的 `stage_world` 接收已组合父变换的 Isaac 世界坐标，单位与上轴由源 USD 读取；不是选中 Prim 的局部 Translate。
`world_m` 接收规范米制 Z-up 世界位置，省略该字段的旧文件仍采用此兼容约定。
速度、身体尺寸、到达容差始终使用米制，不随输入位置单位一起缩放。
场景根节点名称可不同；人物引用节点不会再次继承父容器的平移、旋转或缩放，世界出生坐标和人物自身尺寸保持一致。

## 顶层结构与公共资源

`schema_version` 必须为整数 `1`，`scene.asset_path` 必填。
`scene.layers` 可追加 USD 层。`simulation`、`robots`、`recording` 使用现有会话字段，
例如运行时长、窗口、报告、额外机器人和录像。`simulation.duration: 0` 等待退出。
地图、人群和通信参数由 `navigation.settings_path` 选择。

| `pedestrians` 字段 | 含义 |
|---|---|
| `count` | 随机生成角色的正整数人数；有 `actors` 时可省略，填写时须严格一致 |
| `actors` | 非空的具名角色映射，每个条目是一人；名字须为 ASCII 标识符 |
| `assets` | 公共本地角色 USD、模型池目录或角色 YAML 清单；每名角色都指定 `asset_path` 时可省略 |
| `motion_library` | 公共本地动作库；省略时由导航设置的本地资产根解析默认动作库 |
| `defaults` | 角色级公共参数；支持 `behavior`、`speed`、`yaw`、`start_time`、`repeat`、`arrival_tolerance`、`wait`、`gender` |

省略 `actors` 时必须提供 `count`，生成 `walker_01`、`walker_02` 等角色，默认从占用地图采样安全出生点并漫游。
以下只是完整场景中的行人块：

```yaml
pedestrians:
  count: 7
  defaults:
    speed: 0.9
    yaw: 0.0
```

场景与角色使用本地资源，默认动作库由导航设置的本地资产根解析。
自己的动作库须包含 `BuiltinActions/MoveWalk/WalkForward` 和 `BuiltinActions/Idle/Idle` 两个 SkelAnimation 片段，分别用于行走和站立；缺失时初始化报错。
公共模型由 `pedestrians.assets` 提供。每个角色可通过自己的 `asset_path` 覆盖为本地角色文件、目录或清单。
`defaults.asset_path` 和 `defaults.waypoints` 均不支持；路线逐人定义。

### 角色资源清单

场景文件中的 `pedestrians.assets` 和 `actors.NAME.asset_path` 可指向本地 YAML 清单。清单按顺序列出相对于导航设置 `asset_root` 的角色 USD/USDA 路径。例如：

```yaml
assets:
  - Isaac/People/Characters/F_Medical_01/F_Medical_01.usd
  - Isaac/People/Characters/M_Medical_01/M_Medical_01.usd
```

场景到清单的路径相对场景文件，清单内的 USD 路径相对资产根。清单须有非空的 `assets` 列表，条目必须存在，并且不能重复或包含 `..`。资产根可使用本地目录或本地 `file://` URI，相对资产根以启动目录为准。预检在启动 Kit 前检查有效人口的资源。选模保留清单顺序，并按下述性别标签筛选。内置场景共用[角色清单](../configs/runtime/characters.yaml)。

## 行人性别与模型选择

`gender` 是可选的模型资产标签筛选，可填写 `male`、`female`、`any`。省略时为 `any`，保持原有不限性别的选模方式。
它不改变角色行为、骨骼或外观；`any` 也不保证男女各半。

统一设置所有人的性别，在已有 `pedestrians.defaults` 中增加字段，例如：

```yaml
pedestrians:
  count: 4
  defaults:
    gender: female
    speed: 0.9
```

这是行人块示例。已有工程保留自己的 `assets`、顶层 `navigation` 和路线。逐人设置时，角色值覆盖公共默认值：

```yaml
pedestrians:
  defaults:
    gender: female
  actors:
    walker_01: {}                 # 继承 female
    walker_02: {gender: male}
    walker_03: {gender: female}
    walker_04: {gender: any}      # 取消这名角色的性别限制
```

此例省略位置和路线，四人随机出生并漫游。若已有具名角色，只在对应角色下增加 `gender`，不要覆盖其 `position`、`waypoints` 等内容。
`check` / `run --dry-run` 的 `pedestrian_plans[*].gender` 显示最终筛选条件。启动时，ROS 2 会话按场景种子与计划顺序从每个角色的候选池抽取模型。

筛选依据是 USD 文件名或直接父目录的明确标签：独立的 `male` / `female` 单词，或 `M_` / `F_` 前缀，大小写不敏感。
例如 `original_female_adult_police_01`、`F_Medical_01` 为 female，`male_adult_police_04`、`M_Medical_01` 为 male；
不会把 `female` 中的 `male` 子串误当男性。项目共用角色库已使用这些命名规则。

- 指定目录直接含 USD/USDA 时，优先使用这些入口；否则扫描一级非隐藏角色子目录。预览和筛选用的发现器按文件名排序取各子目录首个入口。
- 指定 `male` / `female` 时，无标签或文件名与父目录标签冲突的模型不参与选择。自定义模型可以放在明确标记的目录中，例如 `female/Alice.usd`。
- `asset_path` 仍优先于公共模型池；`gender` 只在选定来源中筛选。指定文件性别不匹配、无法识别或没有候选时直接报错，不改选其他模型池。
- `any` 允许未标记模型。不要用 `null`、中文值或布尔值代替上述三个字符串。

角色显示通过独立重定向处理模型单位和上轴。

## 运行、机器人和录像字段

以下为**省略字段时的类型默认值**；生成器或所选配置可以明确设置其他值，例如新工程的 `duration: 60`、`seed: 42`。
路径字段必须是非空字符串，省略不等于填写 `null`。数值须有限，布尔值不能代替数字。

| 顶层/场景字段 | 省略时 | 约束与用途 |
|---|---|---|
| `schema_version` | 必填 | 整数 `1` |
| `scene.asset_path` | 必填 | 现存本地 USD；运行时读取实际单位和上轴 |
| `scene.layers` | `[]` | 现存本地 USD 层列表；按配置参与组合 |
| `robots` | `[]` | 额外机器人列表，字段见下表 |

### simulation

| 字段 | 默认值 | 约束与用途 |
|---|---|---|
| `headless` | `false` | 布尔值；隐藏窗口 |
| `render_viewport` | `true` | 是否更新主视口；false 仅用于 headless 且录像关闭的独立 Session，场景和角色准备就绪后生效 |
| `duration` | `0.0` | 非负仿真秒；0 等待退出，正数为时间轴退出阈值 |
| `start_paused` | `false` | 初始化就绪后暂停 |
| `width` / `height` | `1280` / `720` | 正整数像素；录像时必须为偶数 |
| `physics_hz` | 保留场景值 | 正整数 Hz，或 null 保留 |
| `solver` | 保留场景值 | `PGS` / `TGS`，或 null 保留 |
| `seed` | `null` | 整数0～4294967295；未指定时不固定随机种子 |
| `extensions` | `[]` | 非空扩展名组成的列表；加载阶段验证可用性 |
| `console` | `false` | `play`、`pause`、`status`、`quit` |
| `report_path` | 不输出报告 | 本地状态 JSON 路径，不得覆盖输入或录像数据 |

控制台命令、暂停和报告生命周期见 [工作流](workflow.md#运行控制与报告)。
无窗口纯仿真的选择与验证边界见[仿真效率](performance.md)。

### robots 中的每个条目

| 字段 | 省略时 | 约束与用途 |
|---|---|---|
| `asset_path` / `layer_path` | 必须二选一 | 引用机器人模型或已配置层，不可同时填写 |
| `prim_path` | `/World/Robot` | 绝对目标 prim；多个机器人不能使用相同或相互嵌套的目标 |
| `source_prim_path` | 模型 defaultPrim | 资产内被引用的绝对 prim，仅用于 asset_path |
| `position` | 保留模型位姿 | 三个有限坐标，目标 prim 父坐标系中的米值 |
| `yaw_degrees` | 保留模型位姿 | 绕场景上轴的有限角度 [deg] |

### recording

| 字段 | 默认值 | 约束与用途 |
|---|---|---|
| `enabled` | `false` | 启用当前会话录像 |
| `output_path` | 未设置 | 启用时必填，后缀 `.mp4`，须使用新文件名 |
| `fps` | `12` | 整数 1～60，且须整除 `control_hz` |
| `warmup_s` | `0.0` | 非负仿真秒，不计入视频时长 |
| `camera_view` | `oblique` | 自动视角，可选斜视 `oblique` 或俯视 `overhead` |
| `camera_eye_m` / `camera_target_m` | 自动取景 | 使用手动相机时保持 `oblique` 并同时填写这两个点，采用规范 Z-up 世界坐标 [m] |
| `max_stationary_s` | 不检查停滞 | 可选正秒数；只对要求持续行走的验收启用 |

录像还要求正 duration、至少一帧、偶数画面尺寸，关闭初始暂停和控制台，使用原生自动播放。
已有视频、相邻 JSON 或帧目录不会覆盖；相机恢复、失败报告及观测范围见 [录像说明](../recording/README.md)。

### 组合 TOML 专用选择

这些字段不加入简洁 YAML 的 `pedestrians` 块：

| 字段 | 省略时 | 含义 |
|---|---|---|
| `scenario` | 不允许省略 | 显式选择本项目自定义 YAML；不推断历史场景或人口 |
| `pedestrians.enabled` | `true` | 为 false 时不加载行人，也不读取未用的人口/参数来源 |
| `pedestrians.config_path` | 使用当前场景人口 | 替换人口 YAML/包装 TOML，不继承其环境、机器人或控制 |

TOML 的 scene、simulation、robots、recording 使用上面的字段；scene.asset_path 可省略以保留所选 YAML 的环境。
覆盖顺序与旧格式迁移见[配置迁移与复用](custom_scenes.md)。

## 角色字段与覆盖顺序

| 字段 | 默认值 | 含义 |
|---|---|---|
| `asset_path` | `pedestrians.assets` | 该角色专用的本地 USD 文件、目录或角色清单 |
| `gender` | `any` | 可选 `male`、`female`、`any`；角色值覆盖 `defaults.gender`，仅筛选模型标签 |
| `position` | 未设置 | 初始世界位置 `[m or stage unit]`，由 `navigation.waypoint_frame` 决定；随机出生省略字段，`walk` 必填 |
| `yaw` | `defaults.yaw`，最终为 `0.0` | 初始旋转[deg]，相对模型作者朝向；ROS根heading另含资产前向基底校正 |
| `behavior` | 有非空路线时 `walk`，否则 `wander` | 明确值可为 `walk`、`stand`、`wander` |
| `speed` | `1.0` | 请求行走速度 `[m/s]`，须为正数 |
| `start_time` | `0.0` | 行为开始的绝对时间轴时刻 `[s]`，须非负 |
| `repeat` | `false` | 仅 `walk` 可用，沿返程回到实际出生点后重复 |
| `arrival_tolerance` | `0.15` | `walk` 到达检查的三维距离容差 `[m]`，须为正数 |
| `wait` | `0.0` | 该角色所有导航点的默认停留时间 `[s]`，须非负 |
| `waypoints` | 空 | 该角色的有序目标；`walk` 必须至少一项 |

可设为 `defaults` 的字段采用“角色明确值 → defaults → 类型默认值”的优先级。
仅当角色和 `defaults` 均未指定 `behavior`，才按路线推导行为。
明确设置 `stand` / `wander` 后再填写非空路线会报错；从待机预设改路线时须先移除默认 `stand`。
`stand` 不会因 `speed > 0` 移动。实际行走速度受所选导航后端的净空、避让和制动约束。

## 出生、目标和碰撞边界

所有角色在初始化时生成。`start_time` 只推迟行为执行，等待期间角色已在场景中站立并占位。
ROS 启动时验证显式出生点的占位图净空、当前实体轮廓和行人身体间距。有效点原样使用。
无效点默认报错。将 `spawn_tolerance_m` 设为正值时，可在该水平半径内寻找通过同样检查的出生点，原始 YAML 坐标仍作为请求值保留。
两个固定点的平面中心距至少要达到两人身体半径之和，即 `2 * body_radius_m`。
随机出生还使用 `pair_clearance_m` 作为附加间距。选点失败会终止初始化。

行走目标以包含高度的三维 `arrival_tolerance` 核对实测到达。栅格终点选择须同时满足原到达容差和可选的 `goal_projection_tolerance_m`，以及占位图净空。

## 导航点停留与身体朝向

每点可写 `[x, y, z]`，或写含 `position`、`wait`、`yaw` 的映射：

```yaml
waypoints:
  - position: [1.0, 3.0, 0.0]
    wait: 2.0
    yaw: 90.0
  - position: [3.0, 3.0, 0.0]
    wait: 4.0
    yaw: null
```

`wait` 优先级为点值、角色值、`defaults.wait`，最后 `0.0`。
到达后先平滑转向，再开始停留计时；转向时间不计入 `wait`，即使 `wait: 0` 也会执行明确的转向。
ROS 按仿真控制时钟执行等待。整控制步的等待按该步结束，非整步等待在第一个达到时限的控制步结束。
安全关门、临时让位及归位单独计时。工作流 `evaluation.json` 的 `waypoint_visits` 记录每次路点的实际驻留和截止时尚未完成的驻留。
省略点 `yaw` 或填 `null` 保留到达朝向。导航点无全局朝向默认值。

点`yaw`是绕场景上轴的世界heading；在canonical Z-up世界中，`0°`朝`+X`、`90°`朝`+Y`、`180°`朝`-X`、`-90°`朝`-Y`。
Y-up stage的正90°对应stage −Z，详见[出生旋转与路点朝向](units.md#出生旋转与路点朝向)。
它只控制到达后的转向，不锁定移动过程中的朝向。
这与 `defaults.yaw` / 角色 `yaw` 的出生旋转语义不同，后者取决于模型作者朝向。

`wait` 只控制指定路线停留。非 `walk` 角色明确设置非零 `wait` 会报错；
`defaults.wait` 可供混合角色共享，但只对路线角色生效，不控制漫游短暂停留。

## 时间、往返与重置

`start_time` 采用绝对仿真时间，达到阈值后的更新开始行为，不按加载耗时、墙钟或每次恢复播放重新计时。
暂停不消耗行走调度的等待时间。恢复后继续进度，重置路线或时间轴时需重建会话。

设实际出生点为 `S`，路线为 `A`、`B`：

- `repeat: false`：`S → A → B`，完成转向和等待后在终点待机。
- `repeat: true`：`S → A → B → A → S → A → B …`。

也可将声明的出生坐标 `S` 作为最后一个显式点：`A → S`。
这是一条显式闭环，`repeat: true` 时按 `S → A → S → A → S …` 重复，
保留最后 `S` 点自己的 `wait` 与 `yaw`，不再追加合成返程；`repeat: false` 只执行一次。
闭环按输入坐标精确匹配，不会把容差内的不同路点合并。

`S` 使用实际出生位置。合成返程经过 `A` 时复用其停留，朝向由 `auto_return_heading` 决定，项目共享设置为 `follow_route`。
返回 `S` 的合成目标使用 `wait: 0`、`yaw: null`，不继承默认停留、不恢复初始朝向，然后立即进入下一轮。

出生坐标只能作为含其他目标的路线的最后一个显式点；同一条路线中不能重复列出相同坐标。
不同角色可以共享目标，需实际验证时序、路线和避让。待机、开始前等待、逐点等待和终点驻留使用固定位置的站立动画。

## 校验与高级格式

未知/重复字段、布尔值充当数字、非有限值、负等待、非正速度、人数不符、缺失本地资源和歧义路线均报错。
解析错误带来源文件，角色和路点错误附标识或索引。离线检查后仍需实际运行验证骨架、动画、物理地图与运动。
编辑完成后，用 [逐人计划预览](workflow.md#检查逐人计划) 核对默认值合并后的完整路点、停留与朝向，
以及角色的模型覆盖和行为启动时间。
行人导航失败时，状态保留本次运行时间、错误及最后收到的角色命令。命令中的 `phase`、`route_phase` 和 `waypoint_index` 标记调度进度；位姿、目标与引导点可结合运行目录中的 trace 核对。位置使用 ROS map 米制坐标。
