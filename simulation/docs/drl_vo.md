# 冻结 DRL-VO 策略接入

可选的 DRL-VO 控制层把模型包中的 `frozen_policy.pt` 接到当前 ROS 2 行人仿真和 EVT 机器人上。默认不启用；启用后策略在仿真进程内以 20 Hz 运行，通过 `/cmd_vel` 驱动底盘。

## 启用

```bash
PYTHONPATH=/tmp/pedstrain_root_isaac /home/stardust/isaacsim61_env/bin/python \
  run.py run configs/third_3 \
  --navigation-settings configs/runtime/ros2_navigation.yaml \
  --drl-policy-config configs/runtime/drl_vo.yaml --dry-run
```

去掉 `--dry-run` 并用 `--headless` 或图形窗口运行即可闭环：

```bash
PYTHONPATH=/tmp/pedstrain_root_isaac /home/stardust/isaacsim61_env/bin/python \
  run.py run configs/third_3 \
  --drl-policy-config configs/runtime/drl_vo.yaml
```

示例配置默认 `dry_run: true`；验证观测和动作后改为 `false` 或加 `--no-drl-dry-run`。

## 目标与设备

- `goal_xy_m`：ROS map 米坐标；运行时可用 `--drl-goal X Y` 覆盖。
- `--drl-package-root PATH`：覆盖模型包路径。
- `--drl-device cuda:0`：使用 GPU；默认 CPU（约 9–16 ms/次，20 Hz 预算 50 ms）。
- 模型包默认路径在 `configs/runtime/drl_vo.yaml`，可整体移动，只要同步修改 `package_root`。

## 观测

策略输入固定为 19,202 维：

| 块 | 形状 | 来源 |
|---|---|---|
| 行人速度图 | `2 × 80 × 80` | 机器人基座系 20 m × 20 m、0.25 m/cell；4 个行人的绝对速度旋转到基座系 |
| 前向扫描历史 | `10 × 720` 压缩为 6,400 | 从静态占据栅格合成的 180° 前向扫描，角度 `-π/2 + j·π/720`，0 表示无回波 |
| 局部目标 | 2 | 全局 A* 路径上的 0.5 m pure-pursuit 目标，基座系 |

扫描从 `map.npz` 的静态栅格合成，并在每次扫描前把行人和动态障碍（门、非机器人刚体）的平面圆盘叠加为占据格；机器人自身的刚体被排除。仍然不模拟双 lidar 安装点，中心投射的近距离/自遮挡行为与训练时不同。

## 控制与安全

- 60 Hz 控制循环中每 3 tick 推理一次（20 Hz），每 4 tick 更新扫描（15 Hz）。
- 首个局部目标前先原地对齐（v=0，|ω|≤0.8 rad/s，5° 容差，稳定 0.15 s）；对齐期间不推理。
- 安全保护会覆盖策略输出并发布零速：
  - v ∈ [0, 0.5]、|ω| ≤ 2.0；
  - 扫描最近回波 < 0.25 m 停车，> 0.35 m 释放；
  - 前向 0.25 m 扫掠净空不通过；
  - 行人当前/0.8 s 预测表面净距 < 0.15 m 停车，> 0.25 m 释放；
  - 目标到达/超时、路径不可用宽限、无效观测锁停。
- 前向净空保护使用机器人矩形 footprint（`robot_length_m` / `robot_width_m`）的半宽胶囊，而不是 0.37 m 外接圆；行人净空使用 `robot_clearance_radius_m`（默认 0.51）。
- 当前进被安全保护否决时，控制器进入恢复：
  - 行人阻挡且后方净空足够 → `RECOVERY_BACKOFF`，以 `escape_reverse_m_s` 后退；
  - 否则 → `RECOVERY_TURN`，v=0，朝扫描左右净空较大的一侧原地转向（`escape_angular_rad_s`）。

## 输出

会话目录新增：

- `drl_manifest.json`：模型、权重哈希、设备、起点/目标、初始路径、限制。
- `drl_policy.jsonl`：每个策略 tick 的观测摘要、raw/clipped 动作、目标距离、局部目标、保护状态、推理耗时。
- `drl_scan.jsonl`：每次扫描的位姿、有效束数、覆盖率、最小/最大距离。
- `status.json` / `result.json` 的 `drl_vo` 字段：阶段、策略步数、目标/路径、最后动作、安全原因。

阶段：`WARMUP`、`ALIGNING`、`DRIVING`、`RECOVERY_TURN`、`RECOVERY_BACKOFF`、`DRY_RUN`、`GOAL_SETTLE`、`GOAL_REACHED`、`GOAL_TIMEOUT`、`PATH_UNAVAILABLE`、`SAFETY_STOP`、`PAUSED`、`FAILED`。

## 限制

- 模型在 NewMetro 场景训练，训练时使用 oracle 行人速度；与当前 `third_3` 场景/资产不同，行为未做安全验收。
- 模型没有近人主动减速；安全保护是独立的保守兜底。
- 扫描是静态栅格加动态圆盘的中心投射，不是训练时的双 PhysX lidar；近距离/自遮挡行为会有差异。
- `/cmd_vel` 为全局话题（domain 37），同一 domain 多会话会冲突；机器人图内部对线/角速度有缩放，实际运动依赖底盘标定。
- 单目标，无 episode 重置/目标队列。
