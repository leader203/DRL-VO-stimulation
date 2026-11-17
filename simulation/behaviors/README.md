# 机器人可视模型

`robots/visuals.py` 提供显式绑定的 `RobotVisuals` 行为，用于创建随 PhysX 刚体运动的临时可视模型。`behaviors/robot_visuals.py` 仍是已有脚本的入口。它只操作机器人的显示层。行人的导航与显示由 [ROS 2 会话](../docs/ros2_navigation.md) 管理。

行为需要已启动的 Kit，由选择该脚本的场景加载。创建、更新和清理可视模型均在 Kit 主线程执行。
