# 配置处理

本包维护可离线使用的类型、解析、合并和校验。用户格式见[行人 YAML](../docs/pedestrian_yaml.md)，参数来源见[配置索引](../configs/README.md)。

| 文件 | 职责 |
|---|---|
| `models.py` | 场景、机器人、会话类型及有效资源校验 |
| `loading.py` | 严格 YAML/TOML 加载与入口合并 |
| `navigation.py` | 输入位置框架和 ROS 2 地图、控制、通信参数 |
| `scenarios/` | 自定义场景、角色计划与资产候选 |
| `population.py` | 有效人口的选择 |
| `recording.py` | 录像配置和采样约束 |
| `paths.py` | 包级资源、项目入口与声明文件相对路径 |
| `resources.py` | 声明资源解析，供预检、运行和报告保护使用 |

先解析和合并来源，再选择有效人口并校验。禁用行人时跳过未使用的角色资源；替换人口只读取一层，其点位按最终工程的 `navigation.waypoint_frame` 解释。跨单位或上轴复用时，先将点位换到目标项目所选坐标框架。

TOML 通过 `scenario` 选择自定义 YAML，再覆盖显式字段；`scene.layers` 追加，`robots=[]` 清空继承列表。完整读取使用 `load_simulation_config(FILE)`，临时覆盖可用 `dataclasses.replace`。

`resolve_simulation_profile` 选择目录内唯一的 `simulation.yaml` 或 `simulation.toml`。`resolve_config_reference` 处理文件相对引用与用户目录展开。项目文件中的路径相对声明文件；CLI 路径相对当前工作目录。报告和录像使用独立的输出路径。

导航选择 `navigation.settings_path` 时，自定义文件的遗漏字段取类型默认值。省略该路径时使用包级 ROS YAML。默认动作库和角色清单条目由导航设置中的本地 `asset_root` 解析。场景也可直接提供角色目录、角色 USD 和动作库文件。

## 固定出生点预检

`check` 和 `run --dry-run` 的 `fixed_spawn_check` 按平面中心距离提示身体重叠。`world_m` 直接使用米制 XY。`stage_world` 在 OpenUSD 和本地舞台指标可用时换算；无法读取指标时返回 `not_checked`。地面、占位图和动态障碍净空由实际启动验证。
随机出生的目标中心间距另含 `pair_clearance_m`。`check --dependencies` 专用于核对声明资源。

公共调用从 `simulation` 或 `simulation.configuration` 导入。入口迁移见[配置迁移](../docs/custom_scenes.md#从旧格式迁移)。
