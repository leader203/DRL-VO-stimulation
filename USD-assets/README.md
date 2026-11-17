# USD 资产

本目录包含仿真的场景/机器人 USD 与贴图，使用 Git LFS 存储。

- `Sence/Rest_Room_Third_Floor/`
- `Sence/Rest_Room_HongKong_NewMetro/`
- `Sence/Rest_Room_HongKong_YOHO/`
- `Sence/Rest_Room_Second_Floor/`
- `rest_room_EVT-EXP_V10/`

克隆后先拉取 LFS 对象：

```bash
git lfs install
git lfs pull
```

注意：`simulation/configs/third_3/textures_local.usda` 中的贴图路径仍是原始机器的绝对路径
（`/home/stardust/Desktop/simulation_EXP_extracted/Sence/...`），换机器使用前需要改成仓库内的
`USD-assets/Sence/...` 路径。
