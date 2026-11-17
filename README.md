# DRL-VO + ROS 2 行人仿真

本仓库包含两个项目：

| 目录 | 内容 |
|---|---|
| `simulation/` | ROS 2 行人仿真（SFM/ORCA、占据栅格、自动门）与冻结 DRL-VO 推理接入；包含 `third_3` 四人场景、评估配置和启动脚本 |
| `DRL-VO/` | NewMetro 四行人 480k 步冻结模型包、配套推理/观测代码、原始参考网络和 `reference/python` 运行依赖 |

## 模型

- 权重：`DRL-VO/frozen_policy.pt`
- SHA256：`bef18b93c76f5eef245942b0aa1bede03faaf14efcc7a0fbc38e972a989996ca`
- 来源：480,000 steps / 937 updates，4 名物理行人，0.5 m 前瞻，冻结 eval 模式、确定性均值

## 已排除的生成内容

- `simulation/**/output/`：运行记录、`trace.jsonl`、日志、视频、地图
- `__pycache__/`、`.pytest_cache/`、`*.pyc`
- `simulation/.package_root/`：本机 import 软链，启动脚本会在首次运行时自动重建

仓库仍包含约 11 MB 模型和约 64 MB USD 场景/机器人资产。

## 运行

本机评估入口：

```bash
cd simulation
./run_model_eval.sh --headless --duration 25          # 无行人，模型全权
./run_model_eval_peds.sh --headless --duration 40     # third_3 四行人，模型全权
```

脚本会自动设置 Isaac Sim 自带的 ROS 2 Jazzy 环境，不依赖 `/tmp`。

## 注意

- 配置和脚本里保留了大量原始机器的绝对路径（例如 `/home/stardust/...`、`/home/user/workspace/...`），换机器需要同步修改。
- `DRL-VO/run.sh` 仍是原分享包入口，硬编码 `/home/user/...`；本地仿真接入使用 `simulation/ros2_navigation/drl/`。
- `DRL-VO/SHARE_README.md` 说明了原分享包的边界：不是可独立安装的一键部署包，也不是真机 ROS 部署包。
