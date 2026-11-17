# 自定义场景编译

本包把 `schema_version: 1` YAML 转换为 ROS 2 会话配置和逐人计划。字段参考见[行人 YAML](../../docs/pedestrian_yaml.md)，运行步骤见[工作流](../../docs/workflow.md)。

| 文件 | 职责 |
|---|---|
| `models.py` | `Waypoint`、`PedestrianPlan` 的数值、路线与资产类型约束 |
| `loading.py` | 严格字段、默认值合并、人数与文件相对引用 |
| `compilation.py` | 构造会话配置和角色资源信息 |
| `assets.py` | 本地模型候选发现与性别标签筛选 |

```python
from pathlib import Path
from pedstrain_simulation.simulation.configuration.scenarios import load_scenario_config

config = load_scenario_config(Path("simulation.yaml"))
plans = [plan.to_dict() for plan in config.pedestrian_plans]
```

`load_simulation_config` 另外支持通过 TOML 的 `scenario` 引用场景。公共加载入口校验最终选择的资源和计划。重复键和非字符串键由严格 YAML 读取器检查。

`PedestrianPlan.to_dict()` 返回可 JSON 编码的请求数据，供配置预览和录像使用。Python 编译在构造时固定本地输入的绝对路径，YAML 则相对声明文件解析。角色资源可使用本地模型文件、目录或 [YAML 清单](../../docs/pedestrian_yaml.md#角色资源清单)。清单中的 USD 相对有效导航设置的资产根；解析时保留清单顺序。角色在启动时按有效来源、场景种子和性别标签选择。执行计划、实际位姿与动画由 ROS 2 会话管理。

增加字段时，同步更新模型、加载、编译、字段参考和公开契约测试。
