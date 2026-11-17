# 代码组织与扩展位置

行人运动由 [ROS 2](ros2_navigation.md) 控制。Isaac Sim 持有 USD、PhysX、人物根位姿和动画，ROS 节点持有规划、避碰、目标调度和运动积分。

```text
run.py / __main__.py
  → cli.launcher
      → configuration                 YAML/TOML、路径、角色计划及参数校验
      → projects                       init/check、独立子进程工作流与离线评估
      → runtime.runner                 启动并运行会话
          → ros2_navigation.runtime    Kit、USD、PhysX、控制时钟、资源清理
              → mapping / actors / doors
              → transport ↔ node       DDS 时钟、状态与有序控制批次
                            → planner / solver / passing / goal_occupancy / interactions
          → recording                  相机、PNG 采样及视频编码
```

`configuration/` 维护配置类型、严格 YAML 加载、场景计划与资源引用。TOML 包装通过 `scenario` 选择完整场景，再覆盖运行选项和可选人口。文件内路径统一由 `configuration.paths.resolve_config_reference` 处理。项目检查、报告保护与运行共用人口和资产选择规则。

`projects/workflow.py` 在未导入 SDK 的父进程中预检并创建输出目录，仅给本次子进程加载 ROS setup。仿真结束后，`alignment.py` 核对场景世界、规范米制世界与 ROS map，`evaluation.py` 评估原始路点到达、等待和净距。最终报告记录输入和源码指纹。

`ros2_navigation/runtime.py` 负责场景和会话所有权。它把原 `input_plans` 规范化为米制 `world_plans`，再转换为 ROS map 的 `plans`。场景组合与物理规则由 `scene/composition.py`、`physics.py` 和 `namespaces.py` 维护。USD 和 PhysX 操作在 Kit 主线程执行。

`mapping.py` 从实际碰撞体采样地面与占用格，逐帧读取动态几何。`transport.py` 将实测状态发送给 ROS 节点，完整批次经校验后由 `actors.py` 应用角色位姿并按位移推进步态。`doors.py` 读取实际开度并驱动关节。门元数据、路线与几何覆盖层由各场景声明。

`monitor` 通过 `ros2_navigation/monitor.py` 启动 RViz。`visualization.py` 发布实测状态、请求路径和只读世界坐标参考。录像同样消费会话状态，用受控仿真时刻采样视口。

## 修改位置

| 需求 | 位置 |
|---|---|
| 场景、角色、路线、停留、机器人 | 场景 YAML；字段定义见 [行人 YAML](pedestrian_yaml.md) |
| 地图、控制频率、身体半径与通信 | `configuration/navigation.py` 和所选导航设置 |
| 资源选择、依赖预检和报告输入保护 | `configuration/resources.py`、`population.py`、`projects/inspection.py`、`recording/output.py` |
| 增加 CLI 覆盖 | `cli/arguments.py` 声明，`cli/overrides.py` 合并 |
| 路径、避碰、目标与会车 | `ros2_navigation/planner.py`、`solver.py`、`goal_occupancy.py`、`passing.py` |
| 动态门协作 | `ros2_navigation/doors.py`、`interactions.py` |
| 新增机器人组合或物理参数 | `scene/composition.py`、`physics.py` |
| 相机、采样、编码 | `recording/ros2_viewport.py`、`ros2_recorder.py`、`output.py` |
| 自动验收和 ROS 运行验证 | `projects/workflow.py` 和[测试说明](../tests/README.md) |

场景名称、对象路径与固定坐标属于用户配置。向配置添加字段时，同步维护模型、加载器、预检、模板和最近的公开契约测试。组件测试与整场景的实际证据范围见[测试说明](../tests/README.md)。
