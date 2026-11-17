# 分享说明

这是 NewMetro 卫生间四行人场景训练的 DRL-VO 固定演示版本。
模型：frozen_policy.pt；来源 480000 steps / 937 updates；前瞻距离 0.5 m。
入口：run.sh -> run_demo.py；code/ 为配套代码；ui/ 为选点和 RViz；scene/ 为场景配置；reference/ 为原始网络及参考代码。

本包保留原脚本，未改动硬编码路径。原环境入口：
/home/user/workspace/social_navigation_runtime_20261010/python.sh
原运行：bash run.sh
原 Isaac Sim 为 6.0.0.1。交互入口还使用系统 Python、ROS 2 Humble/RViz。
迁移到其他电脑需要调整 /home/user/... 路径，并另备 Isaac 运行环境、完整 USD/材质、AMP/SMPL 人体资源及相关工程依赖。本包不是可独立安装的一键部署包，也不是真机 ROS 部署包。

frozen_policy.pt 是自定义 PyTorch checkpoint，不能当 SB3 ZIP 直接加载；配套加载逻辑见 code/。
这是历史演示模型；未观察到近人主动减速，到达率不等于接触安全。
本次仅核对来源、模型 SHA256 和压缩包完整性，没有重启仿真或训练。
