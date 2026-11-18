# 角色资产（Isaac Sim 6.1 People）

本目录是 `simulation/configs/runtime/characters.yaml` 所需的行人角色资产，使用 Git LFS 存储。

```text
Isaac/People/
  Characters/       # characters.yaml 列出的 21 个角色 + Biped_Setup.usd + filter.json
  MotionLibrary/    # HumanMotionLibrary.usd 及动作片段
  Animations/       # 原 6.1_people 中指向 4.5/Isaac/People/Animations 的软链目标
```

- 已排除 `Characters_61/Reallusion` 和 `Characters_61/biped_demo`（不在 `characters.yaml` 中）。
- `SOURCE.json` 引用的 AMP/SMPL/TRACE 权重在本机不存在，未包含在本仓库。
- 克隆后：

```bash
git lfs install
git lfs pull
```

将 `simulation/configs/runtime/ros2_navigation_local.yaml` 的 `asset_root` 指向
**包含 `Isaac/People/...` 的那一层**，即本目录 `Character-assets/`。
