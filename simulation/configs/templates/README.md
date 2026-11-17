# ROS 2 场景模板

`scenario.yaml.in` 是 `init` 和 `create_simulation_project` 的生成输入。运行已有项目只读取已生成的 YAML 和它引用的资源。

`init PROJECT --scene USD --characters DIR --count N` 生成 `simulation.yaml` 与项目内的 `navigation_settings.yaml`。`--navigation-settings FILE` 选择已有参数文件时直接引用。资产路径优先相对新项目，不复制场景、模型或纹理。
`--characters` 也可指定一个角色 USD 文件。使用[角色 YAML 清单](../../docs/pedestrian_yaml.md#角色资源清单)时，将 `--characters` 指向清单，并通过导航设置选择相应的本地资产根。

CLI 生成明确的 `stage_world` 坐标框架，可用 `--waypoint-frame world_m` 改为规范米制世界。默认人口随机出生并漫游，`--behavior standing` 生成站立计划。

生成器在写入前核对资源、角色候选、导航设置和目标目录。创建与运行命令见[工作流](../../docs/workflow.md)，参数来源见[配置索引](../README.md)。
