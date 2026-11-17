# THIRD_2 场景

`simulation.yaml` 定义行人和路线，已显式选择 `navigation.backend: ros2`，
引用[共享 ROS 参数](../runtime/ros2_navigation.yaml)。
人数由 `actors` 条目推导。ROS 执行路线、避碰与积分，Isaac 负责物理地图与视觉；
原计划、时长、种子、物理和场景设置保留。
运行入口见[配置工作流](../../docs/workflow.md)，性别与路线字段见[行人 YAML](../../docs/pedestrian_yaml.md)。
启动 shell 须先 `source /opt/ros/jazzy/setup.bash`。
当前后端选择不代表本场景已完成通行验收，运行边界见[ROS 导航](../../docs/ros2_navigation.md)。

## 入口通行修正

`entry_door_overrides.usda` 通过 `scene.layers` 加载：关闭入口门扇和把手碰撞，并将活动门扇网格从静态 NavMesh 烘焙中排除。
门框、墙体和场景其他碰撞保留，原始 `THIRD_2.usd` 不被改写。
ROS 地图按实际碰撞开关采样，不使用覆盖层中的 NavMesh 排除标记，也不主动恢复已关闭的碰撞。

修改覆盖层后重新启动场景，让 ROS 重新生成物理地图。
只继续播放已装配的旧场景不会应用覆盖层变更。
当前路线、目标和停留设置保留；不同角色目标过近仍可能产生拥堵，需结合实际身体尺寸验证。

## 故障记录

当前配置将状态写入 `output/status.json`，路径相对此目录。CLI 的 `--report` 可以覆盖目的地。
ROS 报告中的 `output_directory` 指向本次地图、完整 trace 与错误上下文。
