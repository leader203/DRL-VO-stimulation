# 运行生命周期

`runner.run_simulation(config)` 启动 ROS 2 会话并执行主循环。Kit、DDS、控制时钟、场景与临时文件由 `ros2_navigation/runtime.py` 和 `transport.py` 管理。

启动时组合场景、应用物理设置，并在启用行人时采样地图、创建角色和启动 ROS 节点。Isaac Sim 将实测状态按控制时间发送给 ROS，收到完整有效的批次后应用角色位姿。暂停保留当前时间与路线进度，重置路线需创建新会话。

`console` 支持 `play`、`pause`、`status` 和 `quit`。普通状态报告记录仿真时间、输出目录、ROS namespace 和退出结果。带窗口、无视口与录像的选项见[运行工作流](../docs/workflow.md)。

退出时释放 ROS 进程、采样器、时钟设置及角色层，只关闭本次拥有的 USD 舞台。舞台实际脱离后才删除临时组合层。运行和清理同时失败时，报告保留原始错误和清理失败信息。

自动工作流另在独立子进程边界处理超时和中断，并清理该次进程组。录像使用同一会话，详见[录像说明](../recording/README.md)。
