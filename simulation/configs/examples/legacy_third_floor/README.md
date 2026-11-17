# 三楼历史场景数据

`scenarios/standing.yaml`、`scenarios/walking.yaml` 和 `scenarios/recording.yaml` 保存历史场景的角色数量、位置与路线。它们供复核历史结果和迁移用户路线使用。现行入口与配置方式见[迁移指南](../../../docs/custom_scenes.md#从旧格式迁移)。

在 ROS 2 工程复用这些计划时，先核对场景资产、角色库与原坐标框架，再运行物理地图与原路线验收。
