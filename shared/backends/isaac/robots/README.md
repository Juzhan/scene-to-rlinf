# 可复用机器人资产

这里汇总本地维护和 Isaac Lab 原生提供的机器人配置，不包含任务、环境或训练循环。
统一入口是 `shared.backends.isaac.assets.FACTORIES`，资产目录与 Blender 共用它。
目录查询不启动 Isaac；只有真正构建资产时才导入原生配置。

| 目录名称 | factory | 关节 | 默认状态 / 控制 |
| --- | --- | --- | --- |
| Piper 六轴机械臂 | `piper` | 六轴＋双指，共 8 个 | 固定基座，原抓取预备姿态 |
| Piper X 机械臂 | `piper_x` | 六轴＋双指，共 8 个 | 固定基座，原夹爪与 D435 模型外观 |
| M20 轮足机器人 | `m20` | 12 个腿关节＋4 个轮关节 | 自由基座，根部初始高度 0.58 m |
| Unitree Go2 | `go2` | 12 个腿关节 | Isaac Lab 原生驱动配置，根部初始高度 0.4 m |
| ANYbotics ANYmal-C | `anymal_c` | 12 个腿关节 | Isaac Lab 原生 ANYdrive 配置，根部初始高度 0.6 m |

保留原 USD、驱动分组、驱动力限制、阻尼、刚度和初始关节姿态。
M20 的腿使用位置控制，轮使用速度控制，配置动作时分开选择相应关节。
Piper X 的 D435 外观不会自动产生图像观察；通过传感器安装和 RGB 勾选配置采集。

在 Blender 的 **Add Asset → 机器人** 中选择资产，观察候选会按新增实体生成，
默认不勾选。原生预览解析真实关节/连杆并显示初始姿态。也可直接配置：

```yaml
entities:
  - name: arm
    factory: piper_x
    kind: robot
  - name: mobile_base
    factory: m20
    kind: robot
    pose:
      position: [2.0, 0.0, 0.58]
  - name: inspection_dog
    factory: anymal_c
    kind: robot
```

本地 Piper、Piper X 和 M20 工厂接受可选 `usd_path`（本地路径）和 `fix_root_link`。
Go2 与 ANYmal-C 直接引用 Isaac Lab 原生资产配置。
安装对话框对支持该参数的子机器人自动解除原先的世界固定，随后创建指定固定连接。
M20＋Piper 等组合不注册为新资产；在项目内组合独立实体即可。

已有项目的 `assets.py` 重新导出这里的配置，保持原导入路径可用而不复制参数。
原来的 `M20_USD`、`RP_PIPER_USD` 环境变量仍然有效。
`projects/loco_manip` 的专用复合资产保留在该项目内，不进入公共资产目录。

本次验证记录：`outputs/assembly_framework/asset_library_20261010/`。
验证覆盖原生加载、关节/连杆枚举、短物理步进、用户定义固定连接和 Blender 导入。
这不表示某个行走/抓取任务已配置完成或已有训练成功策略。
