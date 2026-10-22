# 场景后端接入

## 目标与当前进度

项目继续编辑公共 scene.yaml、对象式观察、planner/executor 和 Python Task。共享层把公共场景翻译为各仿真器原生配置；不向用户暴露第二套任务配置，也不改 RLinf 源码。

## 公共定义与后端职责

| 内容 | 公共层 | 后端层 |
| --- | --- | --- |
| 观察 | entity_information、readers、Signals、ObjectView、生成的 observation.py | 原生 Tensor 的读取及坐标参考点映射 |
| planner/executor | control 配置校验、policy_runtime 调度/冻结模块/缓存/复位 | 挂入原生 ActionTerm/ObservationTerm |
| control.output | 命名终点、解码、批次/设备/浮点/有限值约束 | 原生关节解析、执行器模式核验、实际目标写入 |
| Task | python_task 的惰性读取、每步评价一次、结果校验、TaskMemory 复位 | 原生奖励/终止/重置 manager 的挂接 |
| train.yaml | 同一 state_ppo 配方、ObjectPolicy 训练协议 | 环境构建和 RLinf rollout 连接 |

上述公共逻辑已由 Isaac 使用，mjlab 已有状态读取、原生控制项和 Task manager 配置适配；不复制两套任务或策略调度。共享 runtime 不导入仿真 SDK。底层驱动维持原生 manager 步进，未增加自写仿真训练循环。

`env.train.init_params.backend` 是公共后端选择（默认 isaac）。写在 train.yaml 或在命令行覆盖都可以，评估都随训练后端切换；只有显式写 `env.eval.init_params.backend` 时评估才单独选择。选 mjlab 时 run.py 自动改用 `.venv-mjlab` 重新启动（主 `.venv` 没有 mjlab）；显式设置了 `RLINF_PYTHON` 时不替换，缺 mjlab 则直接报错。`run.py check <project> --backend mjlab` 对完整项目检查同一配置及后端能力；也可用标准 Hydra override。没有第二份 mjlab train.yaml。

mjlab 已接入原生 ManagerBasedRlEnv 和未修改的 RLinf PPO，可通过统一入口 check/inspect/smoke/train/evaluate。stock RLinf 中 env_type 仍使用现有 isaaclab 注册槽位，它负责进程和批次传输；项目工厂按 backend 构建真实 mjlab 环境，不启动 Isaac Sim。内部名称不代表仿真后端，项目无需另写 worker 或 runner。

`inspect/view <project> --backend mjlab` 仍是独立的物理场景检查/原生查看器：编译真实 MuJoCo 模型，但不运行 planner/executor 或 Task。场景编译支持 go2、ground_plane、box、dex_cube、clear_sky、USD 转换机器人/物体、父子摆放、初始关节和抽象体的静态/跟随读数；在线运行已验证两个 GPU 并行环境。公共字段可存在于 SceneInfo，未实现的 mjlab 字段在选中或读取时明确拒绝，不生成假数据。

mjlab 已接入：
- 挂载传感器：`camera`（mujoco_warp 原生相机，rgb 与 distance_to_camera/distance_to_image_plane，视场按 Isaac 针孔换算，远平面裁剪同 Isaac 的 max/zero/none）与 `frame_transformer`（MuJoCo framepos）。深度已在 GPU 上对照几何验证；mujoco_warp 返回射线距离，像平面深度按像素余弦换算。
- 物理实体的随机根复位（`reset.region/yaw_range/interval`，按环境采样）；关节力矩读数 `joint_effort`。
- `view` / `record`：用 `env.train.init_params.backend=mjlab` 运行项目 policy（或零/随机动作）；`view` 在 MuJoCo 自带的被动窗口（`mujoco.viewer.launch_passive`）显示 env 0，`record` 离屏录 env 0 的视频。两者都由共享 preview 循环推进策略，窗口只同步显示、不推进物理；不使用 mjlab 自带 viewer 的策略循环。关闭窗口即结束，需要桌面 DISPLAY。

与 Isaac 的已知差异：近平面不裁剪；相机原点落在所挂本体的几何内部（如转换后的 D435 外壳盒）时，光线追踪会从内部击中它，因此构建场景时把这类几何移到不渲染的 geom group 3（碰撞不变，录像中也不显示）；未命中处 rgb 为黑色而非天空。物理连接、抽象体随机复位和抽象体显示尚未接入。

速度参考点沿用已有 Isaac 语义：position 为根连杆原点，linear_velocity 为根刚体质心速度；四元数 wxyz。各后端同名资产的原生关节数组顺序和动力学不保证相同，跨后端权重迁移还需名称/顺序和动力学验收。

资产目录的 supported_backends 表示存在资产工厂，不表示任意配置均可运行。项目自定义双后端工厂通过 assets.py 的 BACKEND_FACTORIES 按后端声明；原 FACTORIES 仅作为 Isaac 工厂，不会错误用于 mjlab。

独立的 [RAX 桌面项目](../projects/rax_tabletop/README.md) 已通过上述项目工厂编译 Panda 与被动抽屉，补充单环境接触、相机、seeded reset 和 Task 信号；额外字段可进入生成的对象类型，禁止覆盖已有字段。它通过 CPU MuJoCo 工具驱动执行，不属于上面的原生 GPU manager／RLinf 训练路径；不能据此宣称公共在线运行已支持任意机械臂、相机或抽屉任务。

## 环境

本机 `.venv-mjlab` 使用 Python 3.11，继承现有 isaaclab 环境的 Torch 2.7.0+cu128，但把以下依赖安装到独立 venv 中，未升级原有 Isaac 环境：

```text
mjlab==1.1.1
mujoco==3.5.0
mujoco-warp==3.5.0
torch==2.7.0
warp-lang==1.12.1
```

这是现有机器上的隔离增量环境，不是完整可搬运锁文件。已补齐原生 RLinf PPO 运行依赖，版本记录在 shared/backends/mjlab/requirements.txt。Warp 1.17 与当前 mjlab 的运行接口不兼容，实际 GPU 验证使用 1.12.1；PyArrow 使用 20.0.0 兼容现有 NumPy。没有升级现有 Isaac 环境。

Go2 模型保留上游 LICENSE、README 和固定提交来源。MuJoCo 保留 Menagerie 的原生力矩电机；本阶段没有将其冒充为 Isaac 的关节位置控制器，也没有添加站立策略。XML 中的原有 home keyframe 在构建时替换为项目初始状态，原始资产文件不修改。

## 资产双格式

`run.py asset convert` 让同一资产同时具备 USD（Isaac）与 MJCF（MuJoCo/mjlab）：USD→URDF 用 Isaac Sim 自带的 `isaacsim.asset.exporter.urdf`（nvidia.srl UsdToUrdf，仅需 pxr），URDF→MJCF 用 MuJoCo 编译器，URDF/MJCF→USD 用 Isaac Lab `UrdfConverter`/`MjcfConverter`。目前没有成熟的直接 USD→MJCF 工具，所以经 URDF 中转。

- 对齐比较直接读 USD（pxr，静止位姿）与编译后的 MJCF，不经过转换代码，容差记录在 ASSET.json；不一致时命令返回非零并逐项列出，不静默修正。
- UsdToUrdf 在关节 body1 位姿非单位时把连杆坐标系移到关节坐标系，却未变换质心与惯量主轴（Piper X `gripper_link2` 质心 y 反号、z 差 1.3 mm）。因此 USD 源转换后按 USD MassAPI 在 MJCF 本体坐标系中重新写入质量、质心与惯量，ASSET.json 的 `inertial_from_usd_mass_api` 列出受影响的本体。
- UsdToUrdf 还可能把连杆坐标系整体转动（Piper X `gripper_link2` 绕自身轴转 π）。相机、TCP 等挂载写在 USD 本体坐标系中，所以对齐比较也检查本体坐标系朝向；USD 源转换时把这类本体改回 USD 朝向，几何、关节、子本体保持世界位姿不变，ASSET.json 的 `body_frames_from_usd` 列出受影响的本体。
- mjlab 工厂 `piper_x`、`usd_robot`、`usd_object`、`urdf_robot` 经 `mjcf_for` 取 MJCF：资产目录有 ASSET.json 时用记录的转换（源文件变化即报错，并重新比较对齐），否则自动转换并缓存到 `.cache/asset_mjcf/`（键含源文件与转换代码哈希）。`usd_robot` 的增益取自 USD DriveAPI（角度单位按每弧度换算）；`usd_object` 仅写了质量时按碰撞几何求惯量，与 PhysX 一致。
- 未比较：碰撞形状（MuJoCo 网格取凸包，PhysX 按 USD 碰撞近似）、接触与摩擦、执行器。执行器增益由后端工厂分别声明，等价性另需动态验收。
- Piper X：`assets/piper_x/mjcf/` 由 `assets/piper_x/piper_x.usda`（Isaac 实际加载的 USD）生成，与直接从厂商 URDF 编译的结果在 50 组随机关节角下连杆位置差 1e-7 m、质量矩阵差 1e-7。厂商仓库自带的 `scripts/urdf_to_mjcf.py` 写死路径且把执行器增益（120/80）写进 MJCF，与 Isaac 配置不一致，未采用。
- URDF/MJCF→USD 方向已接入命令，但尚未在本机实测（需要启动 Isaac，避开正在运行的训练）。

## 后续顺序

1. 扩展接触/固定连接、抽象体随机复位，补充 Blender 的后端能力提示。
2. 对齐具体资产关节顺序和动力学，再开展跨后端权重迁移与策略学习验收。公共配置一致不等于物理行为一致。

仍有部分编辑器目录与历史 term 元数据使用 Isaac 的静态目录，未宣称已完全后端无关；新 runtime 和 mjlab 状态读取不依赖这些 SDK 实现。后续按实际能力抽离，不为尚未实现的功能造空适配。

## 验证

- test_scene_backends.py：真实 mjlab/MuJoCo 模型的初始关节、父子旋转、动态刚体和短 CPU 参考步进。
- test_portable_project.py：同一项目两后端配置一致；对象读取与惰性 Task；原生 mjlab ActionTerm 配置/实例调用公共执行链、实际单位解码、部分环境复位；错误执行器模式拒绝。
- 原有场景、控制链、对象接口和 Python Task 配置回归测试。

- test_mjlab_runtime.py：可选 RUN_MJLAB_GPU=1，验证 GPU 并行推进、单环境截断/复位、复位前最终观察和返回 Tensor 独立快照；普通测试验证后端接口缓存隔离。
- test_stock_rlinf.py：实际原生训练及评估的观察合并路径；共享边界处理评估的字典列表，不改上游源码。
- 实机已完成 2 环境 rollout smoke 和原生 RLinf 两次 PPO 迭代，两个检查点间有参数变化。该示例只验证接口与优化链路，不能当作学会站立或爬楼。

后端只记录在快照与回放报告中（`backends`、`backend_changed`），不进入策略接口签名；用 `--snapshot <run> env.train.init_params.backend=mjlab` 可在另一后端回放同一 checkpoint（评估后端随之切换，除非显式另设）。回放时仍在实际后端上逐项核对观察、动作与关节顺序；各后端原生刚体列表顺序不同且策略不读取，跨后端时不比较。这只说明接口对得上，不保证动力学、接触与渲染一致，结果属于 sim2sim 对照。Isaac 与 mjlab 分别保存原生关节/动作缓存，防止同一项目切换后端误用缓存。

实机验收记录：`outputs/mjlab_ppo_verified_20261019/runs/20261019_111313_4452e77f029a/`。
包含 training.log、evaluation.log、weight_update_check.json 与 validation.json。两次 PPO 迭代后 7 个参数张量变化；从匹配快照加载第二次检查点，经原生 RLinf 评估完成 4 个长度为 3 的回合。success=0，不能解释为训练成功的导航或爬楼策略。
