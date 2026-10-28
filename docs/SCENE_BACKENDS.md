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

### 后端钩子

上层（run.py、tools、shared/assembly、实验快照）不按后端名分支，只经 `shared.assembly.assets` 的
`BACKENDS` / `DEFAULT_BACKEND` / `selected_backend` / `backend_module(后端, 模块)` 找到 `shared/backends/<后端>/` 下的同名实现：

| 位置 | 内容 | 调用方 |
| --- | --- | --- |
| `__init__.py` | 可选 `SDK`、`INTERPRETER`：当前解释器缺 SDK 时改用的工作区解释器 | run.py、tools/inspect_scene（接口解析子进程）、scene_backend |
| `assets.py` | `FACTORIES` 资产工厂；`FREE_SENSORS`：可不挂载、固定在世界的传感器（isaaclab3 为空，mjlab/isaac 为 camera）。其余传感器只存在于连杆上：编辑器添加时不提供“固定在世界”，删除其挂载实体时随之删除 | 场景加载、资产目录、web_editor |
| `project.py` | `PROJECT_TERMS`（是否接受项目 terms.py 的原生 term）、`UNKEYED_FACTORIES`（项目 assets.py 无后端键的 `FACTORIES` 是否属于该后端）、`validate_scene`、`validate_project`；可选 `inspect_scene`（显式 `--backend` 的场景单独检查/查看，未提供时明确报错） | prepare_scene、backend_scene（网页编辑器每次写入后经它提示）、scene_backend |
| `control.py` | `install_terms`：挂接 planner/executor 原生控制项 | compile_control |
| `environment.py` | `make_environment`；预览窗口可提供 `window_closed` | Project、preview |
| `robot.py` | `robots(env, settings)`：返回 `shared.assembly.robot.Kinematics` 子类实例，实现 `joint_names`/`joint_position`/`body_names`/`body_pose`/`body_jacobian` 五个原语 | 本后端 control.py、python_task.py 注入到固定组件、decode/initial 与 Task 的观察对象 |
| `visuals.py`（可选） | 编辑器网格导出：`SOURCES`（使导出网格缓存失效的源文件 glob）、`launch_options`/`launch`（启动原生场景）、`snapshot`（返回各连杆位姿，wxyz）。没有该模块的后端（目前 mjlab）在编辑器中只显示示意代理。导出是增量的，只生成未缓存的实体；编辑器导出的是未保存的工作副本 `.scene.yaml.<pid>.editing`，网格缓存仍记在 `scene.yaml` 名下；只剩静态几何（桌子、无质量 box、地形地块）时不启动物理求解器（Newton 的 MuJoCo 求解器拒绝没有关节的模型）。关节体与刚体都记录 `body_names`/`root_body`，传感器可挂在刚体上（没有连杆网格帧的刚体，如 USD Cube，以实体自身位姿作 root_body 预览）；已导出却没有 body 的静态物体不列为挂载目标。导出在后端的 `INTERPRETER` 中运行，可用 `RLINF_EXPORT_PYTHON` 覆盖；编辑器本身也在同一解释器中运行（需装 viser），可用 `RLINF_EDITOR_PYTHON` 覆盖 | shared/assembly/visuals.py、tools/export_scene_visuals.py、web_editor |

机器人能力接口（`shared/assembly/robot.py`）：生成的 `ExecutorObs`/`PlannerObs`/`SceneInfo` 中 `kind: robot` 的节点继承 `RobotView`，提供 `joint_names()`、`frame_pose(frame)`、`jacobian(frame)`、`differential_ik(frame, position, orientation=None, *, damping, max_step)`，按环境批量计算，环境坐标系、米、wxyz。`frame` 取本体 body 名或命名坐标系；命名坐标系由资产默认（`ASSET_FRAMES`，如 Franka、Piper X 的 `tcp`）加机器人实体的 `frames`（同名覆盖）组成，名称不得与方法、信息字段或子实体重名。命名坐标系同时是观察候选 `<机器人>.<坐标系>.position|orientation`，由公共层经 `frame_pose` 读取，后端无需编译传感器（mjlab 因此允许机器人实体声明 `frames`，非机器人实体仍报错）；`frame_transformer` 传感器保持为独立读数，不算机器人坐标系。运动链由该坐标系决定，不配置关节列表，scene.yaml 也不出现 IK。命名坐标系偏移、雅可比参考点平移和阻尼最小二乘步在公共层计算；后端只提供原语，也可覆盖方法改用原生求解。雅可比在 body 原点处，世界轴，列序同 `joint_position`：Isaac 读 PhysX `get_jacobians()`（其线速度行参考连杆质心，后端平移到连杆原点），mjlab 调 `mujoco_warp.jac` 并在 body 原点取值。这些方法只在环境进程可用；训练组件在 policy worker 只收到张量，调用时明确报错。`differential_ik` 返回一步后的绝对关节目标，不施加关节限位，大误差时用 `max_step` 限制单步。

新后端实现这些模块后加入 `BACKENDS`；暂不支持的能力在钩子内明确报错，不在上层补分支。
仍有少量非分支的 Isaac 依赖留在上层：check 报告的 native_key、历史公式 Task 校验、观察选择解析（`isaac.task_config._selection`）与 Isaac 导出的可视化缓存键。

`env.train.init_params.backend` 是公共后端选择（默认 isaac）。写在 train.yaml 或在命令行覆盖都可以，评估都随训练后端切换；只有显式写 `env.eval.init_params.backend` 时评估才单独选择。选 mjlab 时 run.py 自动改用 `.venv-mjlab` 重新启动（主 `.venv` 没有 mjlab）；显式设置了 `RLINF_PYTHON` 时不替换，缺 mjlab 则直接报错。`run.py check <project> --backend mjlab` 对完整项目检查同一配置及后端能力；也可用标准 Hydra override。没有第二份 mjlab train.yaml。

mjlab 已接入原生 ManagerBasedRlEnv 和未修改的 RLinf PPO，可通过统一入口 check/smoke/train/evaluate。stock RLinf 中 env_type 仍使用现有 isaaclab 注册槽位，它负责进程和批次传输；项目工厂按 backend 构建真实 mjlab 环境，不启动 Isaac Sim。内部名称不代表仿真后端，项目无需另写 worker 或 runner。

`view <project> --backend mjlab` 仍是独立的物理场景原生查看器：编译真实 MuJoCo 模型，但不运行 planner/executor 或 Task。场景编译支持 go2、ground_plane、box、dex_cube、clear_sky、USD 转换机器人/物体、父子摆放、初始关节和抽象体的静态、随机复位、Task 移动与跟随读数；在线运行已验证两个 GPU 并行环境。公共字段可存在于 SceneInfo，未实现的 mjlab 字段在选中或读取时明确拒绝，不生成假数据。

mjlab 已接入：
- 挂载传感器：`camera`（mujoco_warp 原生相机，rgb 与 distance_to_camera/distance_to_image_plane，视场按 Isaac 针孔换算，远平面裁剪同 Isaac 的 max/zero/none）与 `frame_transformer`（MuJoCo framepos）。深度已在 GPU 上对照几何验证；mujoco_warp 返回射线距离，像平面深度按像素余弦换算。
- 挂载射线传感器：`height_scanner`，`scene.yaml` 写法、参数与 Isaac 相同（`mount` 实体+本体、`meshes: [<目标>]`、`parameters.pattern/ray_alignment/max_distance/update_period`），观察同为 `<名>.ray_hit_positions`（`[B, 射线数, 3]`，米，环境坐标系，未命中为 inf）与 `<名>.position`。由 mjlab `RayCastSensorCfg` 实现（`shared/backends/mjlab/assets.py`、`scene.py`），射线按 Isaac GridPattern 的顺序和偏移生成；因 mujoco_warp 只按 geom group 过滤，目标实体的可见几何被复制到专用 group 5（无碰撞、无质量），扫描器只看这一组，与 Isaac 只对目标网格求交一致。限制：目标恰好一个且须为静态实体（不复位、非自由体），`ground_plane` 须水平且在 z=0；`yaw` 对齐只用于竖直射线；`update_period` 不大于决策步；不支持 drift、history、offset、多网格。已验证：GPU 上平地/方块的命中与按 Isaac RayCaster 公式算出的值一致（yaw/world/base 三种对齐）；rough_40m、stairs、star、random_grid、stepping_stones 五块地形逐射线与 MuJoCo CPU `mj_ray` 一致（≤3.3e-6 m，射线恰落在几何棱上时两侧都算合法命中）；同一平地场景两后端 smoke 读数相同。
- 物理实体的随机根复位（`reset.region/yaw_range/interval`，按环境采样）；关节力矩读数 `joint_effort`。
- `view` / `record`：用 `env.train.init_params.backend=mjlab` 运行项目 policy（或零/随机动作）；`view` 在 MuJoCo 自带的被动窗口（`mujoco.viewer.launch_passive`）显示 env 0，`record` 离屏录 env 0 的视频。两者都由共享 preview 循环推进策略，窗口只同步显示、不推进物理；默认取景由 `shared/assembly/framing.py` 计算，两后端分别用 Isaac `ViewerCfg(origin_type=asset_root)` 与 MuJoCo 自由相机跟随同一机器人（窗口中手动旋转缩放保留，只随机器人平移）；不使用 mjlab 自带 viewer 的策略循环。关闭窗口即结束，需要桌面 DISPLAY。
- 抽象体显示（`visualization.abstract`）：两后端用同一份规格（`shared/assembly/abstract.py` 的 `MARKER_COLOR`、`POINT_RADIUS`、`EDGE_WIDTH`、`ARROW_*` 与 `marker_edges`）：点为半径 4 cm 的球，矩形/长方体为直径 8 mm 的圆柱棱线框，方向为沿本地 +X、长 0.3 m 的绿色箭头（圆柱杆 + 末端 1/3 圆锥头）。都走原生的命令项 debug_vis：Isaac 为 `VisualizationMarkers`（箭头由 Cylinder/Cone 生成，不再联网加载 Nucleus 的 arrow_x.usd），mjlab 为 `CommandTerm._debug_vis_impl`，由预览窗口/视频调用 `ManagerBasedRlEnv.update_visualizers` 绘入 MuJoCo 场景（`shared/backends/mjlab/abstract_terms.py`）。MuJoCo 3.5 的箭头连接件只画出起止距离的一半，绘制时传入两倍长度，由离屏渲染测试对照圆柱核验。标记只在项目预览 `run.py view/record <project>`（含评估视频）中创建，训练、smoke 和无画面评估不创建；mjlab 的场景相机（mujoco_warp）本就看不到它们，Isaac 的场景相机是否会拍到标记尚未实测（标记只对次级光线隐藏），因此 view/record 中的 Isaac 相机观察可能包含标记。mjlab 只画 env 0，Isaac 画全部克隆环境。同一入口下 mjlab 传感器自身的 debug_vis（如 `height_scanner` 的射线与命中点）也在预览中画出，颜色是各后端原生样式，不统一。绘制超出 MuJoCo 场景几何容量（录像 10000）时报错，而不是越界写入。显式 `view <project> --backend mjlab` 是另一条只编译物理场景的查看路径（CPU MuJoCo 原生查看器，不建命令项、不跑策略和 Task），不画标记；Isaac 的显式 `--backend` 只支持 check。

与 Isaac 的已知差异：近平面不裁剪；相机原点落在所挂本体的几何内部（如转换后的 D435 外壳盒）时，光线追踪会从内部击中它，因此构建场景时把这类几何移到相机传感器不渲染的 geom group 4（碰撞不变；录像与查看窗口打开 group 4，仍会显示）；未命中处 rgb 为黑色而非天空。物理连接尚未接入。

物理上的对齐处理：MuJoCo 关节限位是软约束，PhysX 是硬限位。位置目标超出关节范围时，执行器会一直顶着限位推（25 g 的夹爪手指在 10 N 下会越界 1–2 cm），因此 mjlab 控制项把 joint_position 目标裁剪到关节范围内（无限位关节不裁剪）。代价是顶在限位时 joint_effort 读数约为 0，而 Isaac 读到的是驱动力上限。每个世界预分配 600 行约束（mujoco_warp 默认估计值在双指夹物时会溢出，并静默丢弃约束行）。mjlab 复位时会把执行器目标清零，所以复位后第一帧的 joint_effort 是按零目标算出的值；Isaac 这一帧读到的则是上一回合残留的值。两边这一帧都没有物理意义。

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

Go2 两后端共用 `assets/go2/`：Isaac 5.1 `go2.usd`（`source/` 内 URL vendor）及由它 `asset convert` 得到的 `mjcf/go2.xml`，质量与 USD 一致（15.02 kg）。mjlab 执行器按 Isaac Lab UNITREE_GO2_CFG 写 DC 电机（kp 25、kd 0.5、力矩 23.5 N·m、速度 30 rad/s），不在 MJCF 中写增益。旧的 Menagerie 模型（`assets/go2_mujoco/`）不再被工厂使用。

## 资产双格式

`run.py asset convert` 让同一资产同时具备 USD（Isaac）与 MJCF（MuJoCo/mjlab）：USD→URDF 用 Isaac Sim 自带的 `isaacsim.asset.exporter.urdf`（nvidia.srl UsdToUrdf，仅需 pxr），URDF→MJCF 用 MuJoCo 编译器，URDF/MJCF→USD 用 Isaac Lab `UrdfConverter`/`MjcfConverter`。目前没有成熟的直接 USD→MJCF 工具，所以经 URDF 中转。

- 资产目录统一为 `assets/<名>/`：ASSET.json、`mjcf/`，有本地覆盖时再加入口 USD；引用的上游文件按原仓库（或 URL）相对路径原样复制到 `source/`，`source/SOURCE.json` 记录 `{repository, commit}` 或 `{url}`，以及逐文件 sha256。`run.py asset check` 先核对这些 sha256。
  - Git 来源：`run.py asset vendor <上游 USD/URDF/MJCF> assets/<名>`。带上它加载的全部文件（MJCF 跟随 `<include>` 与 compiler 的 meshdir/texturedir/assetdir）、所用 ROS 包的 package.xml（`package://` 由此在 `source/` 内解析），以及从模型目录到仓库根各级的 LICENSE/COPYING。只接受已提交、未修改的上游文件。
  - URL 来源：`run.py asset vendor <URL> assets/<名> --base <根 URL>`。递归下载 USD 引用的图层/资产与 MDL 引用的贴图，保存为 `source/<URL 在 base 下的路径>`，相对引用因此原样可用。绝对引用、base 之外的引用和服务器上不存在的文件（Kit 内置的 OmniPBR.mdl 等）不下载，命令列出它们。Isaac 资产用 `--base https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1`（`shared.asset_formats.NUCLEUS_URL`）。
  - 代码用 `nucleus_copy(资产名, Nucleus 相对路径)` 取本地副本；文件缺失时报错，并给出对应的 vendor 命令，不回退到联网读取。
  - 已迁移：Piper X、Piper、SO-101、Go2（USD 及其转换 MJCF）、ANYmal-C（含 ANYdrive LSTM 与导航用 Blind 策略）、Franka Panda、Seattle 桌、DexCube、白色大理石材质、天空 HDR、UR5e、Robotiq 2F-85、FrankaSim Panda（RLinf/serl）。这些资产及引用它们的项目不再读 `third_party/`、其他工作区的模型文件或在线 Nucleus。
  - 尚未迁移：直接使用 Isaac Lab 原生预设的旧项目（Go2/M20 粗糙地形环境、Piper lift、OpenArm 等）中，预设内部仍按 `ISAAC_NUCLEUS_DIR` 联网加载地面材质、天空、桌子等，只有项目显式引用的机器人与物体改为本地副本。`piper_phone_sim` 保留项目内修改过的 Piper X 副本。m20、m20_piper_pendulum、franka_reference 没有上游来源记录，无法 vendor。
- 对齐比较直接读 USD（pxr，静止位姿）与编译后的 MJCF，不经过转换代码，容差记录在 ASSET.json；不一致时命令返回非零并逐项列出，不静默修正。
- UsdToUrdf 在关节 body1 位姿非单位时把连杆坐标系移到关节坐标系，却未变换质心与惯量主轴（Piper X `gripper_link2` 质心 y 反号、z 差 1.3 mm）。因此 USD 源转换后按 USD MassAPI 在 MJCF 本体坐标系中重新写入质量、质心与惯量，ASSET.json 的 `inertial_from_usd_mass_api` 列出受影响的本体。
- UsdToUrdf 还可能把连杆坐标系整体转动（Piper X `gripper_link2` 绕自身轴转 π）。相机、TCP 等挂载写在 USD 本体坐标系中，所以对齐比较也检查本体坐标系朝向；USD 源转换时把这类本体改回 USD 朝向，几何、关节、子本体保持世界位姿不变，ASSET.json 的 `body_frames_from_usd` 列出受影响的本体。
- mjlab 工厂 `piper_x`、`usd_robot`、`usd_object`、`urdf_robot` 经 `mjcf_for` 取 MJCF：资产目录有 ASSET.json 时用记录的转换（源文件变化即报错，并重新比较对齐），否则自动转换并缓存到 `.cache/asset_mjcf/`（键含源文件与转换代码哈希）。`usd_robot` 的增益取自 USD DriveAPI（角度单位按每弧度换算）；`usd_object` 仅写了质量时按碰撞几何求惯量，与 PhysX 一致。
- PhysX 关节链中子连杆的位姿由父连杆经关节坐标系推出，不取 USD 中写的位姿。固定关节两端坐标系有微小不一致时（ANYmal-C 的 LH/RH_FOOT 约 1 mrad），UsdToUrdf 会以 1e-3 容差拒绝转换。`physx_stage` 只在会话层把这类子连杆放到 PhysX 实际模拟的位姿（阈值 0.01 rad / 1 mm，超出仍报错；源文件不改），转换与对齐比较都读它，ASSET.json 的 `fixed_joint_children_posed_by_parent` 列出移动的本体与偏差。
- UsdToUrdf 会丢弃 USD 胶囊碰撞体（M20 腿部因此缺碰撞，站高偏低），转换后按 USD 补回（`capsule_colliders_from_usd`）。对齐比较还逐本体比较各类碰撞体数量。
- 质量按 PhysX 实际语义比较：UsdPhysics 中 mass ≤ 0 或未写表示由碰撞体按密度积分；既无正质量又无碰撞体的刚体，PhysX 给默认 1 kg。旧检查直接比较写入的 0，漏掉了 Piper X 导出 URDF 留下的空 `world` 连杆（浮动安装时臂多 1 kg，MJCF 为 0）。现在这类本体按 1 kg 计入，只在 USD 中存在时列为不对齐；`assets/piper_x/piper_x.usda` 给 `world` 写了 0.001 kg 与极小惯量，保留该本体（连接焊接在它上面）。按碰撞体推算质量的本体（Isaac 5.1 Panda 全部连杆）列在 `masses_from_colliders`，两引擎各按自己的碰撞形状积分，不逐项比较：Panda 实测 PhysX 18.78 kg、MuJoCo 19.73 kg。
- 未比较：碰撞体尺寸与位姿（MuJoCo 网格取凸包，PhysX 按 USD 碰撞近似）、由碰撞体推算的质量、接触与摩擦、执行器。执行器由后端工厂分别声明，等价性另需动态验收。
- Piper X：`assets/piper_x/mjcf/` 由 `assets/piper_x/piper_x.usda`（Isaac 实际加载的 USD）生成，与直接从厂商 URDF 编译的结果在 50 组随机关节角下连杆位置差 1e-7 m、质量矩阵差 1e-7。厂商仓库自带的 `scripts/urdf_to_mjcf.py` 写死路径且把执行器增益（120/80）写进 MJCF，与 Isaac 配置不一致，未采用。
- URDF/MJCF→USD 方向已接入命令，但尚未在本机实测（需要启动 Isaac，避开正在运行的训练）。

已转换且两后端都有工厂的资产（`assets/<名>/` 下 USD、`mjcf/`、ASSET.json，`run.py asset check` 全部对齐）：

| 资产 | 来源 | mjlab 执行器（对应 Isaac 配置） |
| --- | --- | --- |
| piper、piper_x | 项目内 USD | 内置位置执行器，增益同 PIPER_CFG / PIPER_X_CFG |
| m20 | 项目内 USD | Isaac DelayedPDActuator → 理想 PD 外包延迟（0–1 物理步，复位后首步抽取，回合内不变） |
| franka_panda、seattle_table | Isaac 5.1 资产（`source/` 内 URL vendor） | 增益同 FRANKA_PANDA_CFG；桌子为静态体 |
| go2 | Isaac 5.1 资产（`source/` 内 URL vendor） | 默认 DC 电机，参数同 UNITREE_GO2_CFG；`actuators: supermimic` 为带延迟的 DC 电机（见 IL3 一节） |
| anymal_c | Isaac 5.1 资产（`source/` 内 URL vendor），含 ANYdrive LSTM 网络 | `shared/backends/mjlab/actuators.py` 的 LstmActuator，与 Isaac ActuatorNetLSTM 同一网络、同一输入与按环境复位的隐状态 |

- Isaac 工厂同样加载 `source/` 中的副本，不再联网读取 Nucleus。`assets/` 不在 Git 中，换机器时按 SOURCE.json 的 url 与路径重新运行 vendor（sha256 不符时命令拒绝覆盖）。
- Isaac ActuatorNetLSTM 不记录关节速度，DC 电机限幅始终按零速计算，即力矩只裁剪到 ±effort_limit（80 N·m）；mjlab 侧按相同结果实现。
- 原地站立 2 s（dt 5 ms，从 0.6 m 落下）：Isaac 机身高 0.538 m；mjlab 默认 pyramidal 摩擦锥下，低增益的 LSTM 执行器使足端缓慢蠕滑外撇，机身高 0.406 m。改用 `cone='elliptic', impratio=10` 时为 0.515 m，关节偏差方向与 Isaac 一致。摩擦锥是仿真设置，目前公共配置未暴露。
- Go2 背负 Piper X、零动作保持默认姿态 3 s（go2_piper_wholebody）：Isaac 机身高 0.247 m、腿部最大偏离默认角 0.51 rad；mjlab 0.258 m、0.44 rad。两边都因 kp 25 承载机械臂而下沉，属于同一现象，不等于动力学一致。
- Web 编辑器资产架的网格（`.cache/assembly_visuals/`）与缩略图（`.cache/asset_thumbnails/`）已为以上资产、21 块地形地块及各项目预先生成，打开即可加载。

## 关节顺序

PhysX 按广度优先列出关节（Go2：4 个 hip、4 个 thigh、4 个 calf），MuJoCo 按深度优先（FL hip、thigh、calf，再 FR……）。单链机械臂两者恰好相同，多腿机器人不同。各后端原样暴露原生顺序时，同一权重换后端会静默读错关节。

- `asset convert` 在 ASSET.json 写入 `joint_order`：取 USD 的 PhysX 顺序（从根连杆逐层广度优先，同层按连杆名排序；已在本机 Isaac 上对 go2、m20、anymal_c、piper、piper_x、franka 实测一致），无 USD 时取 MJCF 顺序。即规范顺序 = Isaac 现有顺序，Isaac 已有 checkpoint（含冻结 Go2 235 特征行走网络）不受影响。
- `asset check` 要求每种格式的关节集合与 `joint_order` 完全相同，缺失或多出都报错；没有 `joint_order` 而格式中有关节时也报错，提示重新 convert。
- 后端读数（`shared/backends/<后端>/readers.py`）把 `joint_position`、`joint_velocity`、`joint_effort`、`default_joint_position`、`default_joint_velocity` 在设备上按 `joint_order` 取列；机器人接口的 `joint_names()`、`joint_position()` 和雅可比列同样换序；接口解析报告（快照回放核对的 `articulations.joint_names`）也是规范顺序。每个后端经 `robot.joint_layout` 钩子提供，上层不分支。
- 控制目标在两后端都按关节名解析，动作不需换序。policy.py、task.py 无需任何处理。
- 引擎关节集合与记录不一致时直接报错，不做部分映射。没有 ASSET.json 记录的资产保持引擎原生顺序。
- mjlab 下 Go2 的读数顺序因此改变；此前在 mjlab 上训练的 Go2 权重（且当时是 Menagerie 模型）语义已变，新训练写独立目录。

## 程序化地形

两条路线，名称与参数相同（Isaac Lab 子地形配置的同名字段，见 `shared/backends/isaac/assets.py` 的 `TERRAIN_VARIANTS`，未知名称直接报错并列出可选项；列表自动转为元组）。

**1. 地形资产（两个后端通用，推荐用于固定场地）。** 参数化物体尚未支持，所以先生成一次，再当普通静态物体用：

```bash
python run.py asset terrain stairs assets/terrains/stairs --size 8 --difficulty 0.6 \
    --param '{"step_height_range": [0.1, 0.2], "step_width": 0.3, "platform_width": 2.0}'
```

```yaml
entities:
  - name: ground
    factory: usd_object
    parameters: {usd_path: assets/terrains/stairs/usd/stairs.usda}
```

- 不给 `--param` 时使用 `shared/terrain_assets.py` 的 `PRESETS`（即下面验证所用参数）。`python run.py asset terrain all assets/terrains` 按预设把 21 种地块写到 `assets/terrains/<地形名>/`，已有 ASSET.json 的跳过；本机已生成全部 21 块，`asset check` 均通过。
- Web 编辑器资产架自动列出 `assets/terrains/*/ASSET.json` 中的地块（类别“地面 / 地形”，名称取 ASSET.json 的 `terrain.label`），添加后就是指向该 USD 的 `usd_object`。21 块的网格与缩略图已预先生成到 `.cache/`；新生成的地块首次添加时由 Isaac 后台导出一次网格。
- 生成在 Isaac 解释器内由 Isaac Lab 完成（启动无界面 Kit，约数秒），写出 `usd/<名>.usda`、`mjcf/<名>.xml` 和 ASSET.json；`usd_object` 与 `mjcf_for` 按普通资产的方式找到记录的 MJCF，不需要地形专用工厂。
- 地块以实体原点为中心、边长 `--size`。ASSET.json 的 `terrain.spawn_origin` 是原 Isaac 子地形给出的出生点（相对地块中心，例如楼梯顶平台 z=1.5），可作为机器人初始位置的参考。
- 每环境一份的静态物体（不填 `mass` 的 `box`/`usd_object`、地块、桌子等）在 Isaac 中由克隆一次放好：写的位姿加环境原点，之后不再移动，复位只重放机器人和带质量的物体。Isaac Lab 在克隆前为它们建 `XformPrimView`，视图里只有 env_0；早先的实现在启动与每次复位时按全部 env_ids 写它，GPU 上越界（加了 `height_scanner` 后表现为崩溃），且 Fabric 写入并不移动 PhysX 静态碰撞体。同理，使用 `factory: terrain`（TerrainImporter 自带环境原点，与克隆网格不同）时不能再有每环境静态物体，启动时报错：改为 `scope: global`（全场只有一份，按世界坐标放置），或改成带质量的刚体（`box`/`usd_object` 填 `mass`；`seattle_table` 没有质量参数）。预览读数 `<名>.pose` 取 env_0 的局部位姿用于所有环境。
- USD 保留原始三角形作为一个静态网格碰撞体。MuJoCo 的网格碰撞取凸包，非凸地形不能整体转成一个网格，因此 MJCF 按可精确表示的部件重建：轴对齐封闭块 → box；其他封闭凸体 → 凸包网格；规则高度网格的开放曲面 → hfield；平坦开放地块 → 薄板。封闭非凸块或不规则曲面直接拒绝，不做近似。
- 写出后对 MJCF 与生成的三角形在 41×41 个点上做竖直射线比较，最大高度差记入 ASSET.json（`max_height_difference`），超过 2 mm 则报错且不写 ASSET.json。MuJoCo 与 Isaac Lab 高度场用同一条网格对角线，两者吻合。
- 21 种均已在 Isaac Lab 2.3.2 上生成并通过上述比较：网格型落在 box 和凸包（`stairs` 41 个 box、`random_grid` 294 个、`star` 7 个 box 加 3 个凸包、`repeated_*` 2 个 box 加 7 个凸包），高度场型落在 hfield。MuJoCo 中小球落在 stairs、slope、star、repeated_pyramids、stepping_stones 上的静止高度与 USD 表面一致（误差 ≤ 0.5 mm）；Isaac 侧对 21 块逐一做了落体验证（每块 9 个小球，从 3 m 高处落下，静态网格碰撞体）：球静止时底部与原始三角形表面重合，没有穿透（`rough` 因球落在粗糙面上高 1–4.8 mm）；`inverted_slope`、`waves` 的球在碗形地面上仍在滚动，只确认了未穿透，没有静止数据。
- 限制：摩擦两侧都是默认值；离开地块边缘即坠落；`height_scanner` 两后端都可对地块做高度扫描（`meshes: [world/<地块>]`，mjlab 限制见上文“mjlab 已接入”）；Isaac 侧是每个环境克隆一份，`env_spacing` 须不小于地块边长，mjlab 各环境独立世界无此限制；所有环境用同一块地形，没有课程难度。

**2. `terrain` 工厂（仅 Isaac，用于课程难度的多地块训练）。** 不在 Web 编辑器资产架上，只能在 scene.yaml 中手写。`terrain_generator.sub_terrains` 每项写 `factory` 与 `parameters`（`proportion` 控制占比）。mjlab 后端没有该工厂（没有 `terrain` 导入器和按环境分配地块的机制）。

- 网格型（几何精确）：`flat`、`stairs`、`inverted_stairs`、`random_grid`、`rails`、`pit`、`box`、`gap`、`floating_ring`、`star`、`repeated_pyramids`、`repeated_boxes`、`repeated_cylinders`。
- 高度场型（按网格采样，适合粗糙地面）：`rough`、`slope`、`inverted_slope`、`hf_stairs`、`hf_inverted_stairs`、`obstacles`、`waves`、`stepping_stones`。
- `repeated_*` 的 `object_params_start/end` 写成字典（`num_objects`、`height`，以及 `radius` 或 `size`）。
- 来自 Isaac Lab 的约束：`random_grid` 的 `size` 不能被 `grid_width` 整除（余量作边界，如 8 m 用 0.45）；`stepping_stones` 的 `holes_depth` 默认 -10 m，石间是深坑；高度场型用 `border_width`、`horizontal_scale`、`vertical_scale` 控制精度。
- 只验证能由真实 `terrain` 工厂构造并生成网格，没有验证机器人能否在其上行走或训练。

## Isaac Lab 3（Newton / OvPhysX）后端：实验接入

`backend: isaaclab3`（`shared/backends/isaaclab3/`）在 Isaac Lab 3.0（免 Kit）上运行同一份 scene/policy/task/observation。只做公共定义到原生配置的翻译，尽量直接用 IL3 自带实现：

- 环境、步进、自动复位与 manager 均为 IL3 原生 `ManagerBasedRLEnv`；由 `isaaclab.app.launch_simulation` 判断是否需要 Kit（无 Kit 相机/可视化时不启动），`Owner.close` 退出它。
- 物理引擎由训练配方 `env.*.init_params.physics` 选择（命令行如 `+env.train.init_params.physics=ovphysx`），不写进 scene.yaml，同一项目两种引擎都能跑。两者都直接取 IL3 自带运动任务预设 `RoughPhysicsCfg()`：
  - `newton`（默认）：`newton_mjwarp`（NewtonCfg：2 子步、MJWarp njmax 1000/nconmax 300、pyramidal 锥、implicitfast）。`simulation.newton` 覆盖 NewtonCfg 字段，其中 `solver` 覆盖 MJWarpSolverCfg 字段；未知字段报错。
  - `ovphysx`：`ovphysx`（免 Kit 的 PhysX）。`simulation.physx` 只能设 OvPhysxCfg 暴露的字段（GPU 缓冲等）；isaac 后端 PhysxCfg 里其余的场景选项 OvPhysX 不写入 USD，出现时报 NotImplementedError，不静默忽略。
  - 各引擎忽略对方的块（与 mjlab 忽略 `physx` 相同）；dt/gravity 仍写在 `simulation`。
- 渲染器由 `env.*.init_params.renderer` 选择（`+env.train.init_params.renderer=ovrtx`），与物理引擎独立组合，评估同样跟随训练的选择：`newton`（默认）预览用 Newton GL、场景相机用 Newton Warp；`ovrtx` 预览用 IL3 的 `NewtonRTXVisualizerCfg`（OVRTX 路径追踪，带软阴影），场景相机换成 IL3 的 `OVRTXRendererCfg`。两者都免 Kit，不需要 isaacsim；`ovrtx` 需在 `.venv-il3` 另装 `ovrtx`（约 2.2 GB，见 requirements.txt）。其他值报错。
- 资产：Go2 用 IL3 `UNITREE_GO2_CFG`（DC 电机，同 Isaac 5.1 配置，但小腿力矩上限与 Isaac Lab 2.x 不同，行为与权重不可互换）加载工作区 vendored 的 `assets/go2` USD；ground_plane、dome_light、box、ball 用 IL3 原生 spawner（地面为 IL3 默认的浅色米制棋盘，不同于 isaac 后端的深色网格，见下文预览）。地面 USD 与贴图 vendor 在 `assets/ground_plane`（Isaac 6.2 资产，`source/SOURCE.json` 记录 URL 与 sha256），运行时不联网；IL3 只对它的远程默认地址按 2 m 一格缩放 UV，后端的 `_spawn_vendored_ground` 对本地副本做同样的缩放，保持 1 m 棋盘格。piper、piper_x、clear_sky、terrain、seattle_table、dex_cube、usd_robot、usd_object、contact_sensor 直接复用 isaac 后端工厂：这些 Isaac Lab 2 配置经 IL3 保留的 PhysX schema 别名仍可用，且不含四元数（根位姿由 scene.py 按 xyzw 写入）；其中 PhysX 专有字段（求解迭代、combine mode）在 OvPhysX 上生效，Newton 忽略，与 IL3 自带预设一致。
- anymal_c 用 IL3 `ANYMAL_C_CFG`（ANYdrive LSTM 执行器网络），franka_panda 用 IL3 `FRANKA_PANDA_LEGACY_CFG`，USD 与执行器网络都换成工作区 vendored 的 Isaac 5.1 副本；m20 复用 isaac 工厂。IL3 的按引擎 articulation 属性片段没有 `fix_root_link`，固定基座由 spawner 的 `UsdFileCfg.fix_root_link` 设置（isaac 工厂使用的旧别名 `ArticulationRootPropertiesCfg` 仍带该字段，所以 piper/piper_x/m20 无需改动）。临时项目（各机器人站在地面上、随机动作）在 Newton 与 OvPhysX 上 `run.py smoke` 均通过契约检查，仅为接口跑通。
- 执行器路径：IL3 默认 `SimulationCfg.use_newton_actuators=True`，由 Newton 原生执行 DCMotor 等配置，但原生 DC 电机是隐式 PD，与 isaac/mjlab 后端的显式力矩计算不同，且不支持 IL3 自带之外的执行器类。后端在 `environment.build_config` 中固定设为 `False`，所有场景都走 Isaac Lab 的 Python 显式执行器（OvPhysX 读同一开关）。依据：go2_piper_wholebody 的 mjlab 权重在 Newton 原生路径上平均约 64 步即失败，在 Python 路径上稳定（见项目 README）；go2_chase 的 supermimic 电机（`DelayedDCMotorCfg`）本就只能走 Python 路径。
- go2 工厂参数 `actuators`：`native`（默认，IL3 `UNITREE_GO2_CFG`）或 `supermimic`（go2-supermimic 发布权重所用的显式 DC 电机：hip/thigh 与 calf 两组增益、力矩与饱和、转子 armature、0–2 物理步随机指令延迟，以及站姿 hip ±0.1/thigh 0.9/calf −1.8）。数值只写一次（`shared/assembly/assets.py` 的 `GO2_SUPERMIMIC`），IL3 用 `DelayedDCMotorCfg`（`shared/backends/isaaclab3/actuators.py`，DCMotor 加 IL3 `DelayBuffer`，按环境复位时抽延迟），mjlab 用原生 `DelayedActuatorCfg(DcMotorActuatorCfg)`；isaac 后端没有对应实现，选 supermimic 时报错。go2_chase 的 flat.pt 用 supermimic 在 Newton/OvPhysX/mjlab 平地上 100%/100%/98.6% 碰到球（见项目 README）；切换执行器路径前，同一权重在 Newton 原生 DC 电机上 16 环境 20 s 摔倒 23 次。
- 未提供 `urdf_robot`：IL3 免 Kit 环境没有 URDF 导入器。URDF 机器人先用 `run.py asset convert` 转成 USD，再用 `usd_robot`；在 IL3 上选 urdf_robot 时 check 报未知工厂。
- Newton 对 USD 组合错误零容忍：Isaac Lab 2 / OvPhysX 只警告的无效引用，Newton 的 `parse_usd` 直接报错退出。`assets/piper_x` 的源 `base.usd` 原有一条指向不存在 `</visuals/world>` 的空引用（`/agx_arm/world/visuals`，对合成结果无贡献），已删除（原文件备份于 /tmp/piperx_backup）；go2、粗糙地形、dex_cube 已检查无组合错误。新资产应在 `run.py asset convert` 环节排除此类引用。
- 碰撞体位置：Isaac Sim 旧 URDF 导入器把 `CollisionAPI`/`MeshCollisionAPI` 施加在装碰撞网格的 Xform（`collisions/<link>/node_STL_BINARY_`、M20 的 `collisions/collision_N`）上。Kit PhysX（isaac 后端）碰撞其下几何，而 UsdPhysics 只在 Gprim 上定义碰撞体，Newton 与 OvPhysX 都跳过它们，于是 piper、piper_x、M20 整机没有碰撞（夹爪穿过方块、末端可穿地）。后端把 USD 资产的 `spawn_from_usd` 换成 `spawners.spawn_from_usd`：在 env_0 生成后、克隆前把这些 API（含 collisionEnabled 与 approximation）移到其下 Gprim，只对途经的 instanceable 祖先取消实例化；资产文件不改，isaac 后端不受影响。go2、方块等本就在 Gprim 上，不受影响。
- 关节顺序：IL3 的 `ArticulationCfg.joint_ordering` 原生支持按名重排（读数、动作、雅可比列同步），后端把 ASSET.json 的 `joint_order` 填入，因此读数不再换序；顺序不符时 `robot.joint_layout` 报错。
- IL3 四元数为 xyzw，只在 `frames.py` 边界换成公共 wxyz；读数、抽象体命令项、区域复位与机器人接口均按 wxyz 对外。
- 控制链、Python Task、抽象体（静态/随机复位/Task 移动/跟随）沿用 isaac 后端的结构，分别接到 IL3 原生 ActionTerm、Reward/Termination/Event、CommandTerm。IL3 原生 JointAction 类直接作为控制终点。
- IL3 的 manager 会在 cfg 上写 `debug_vis`，所以各类 term 放在属性容器而非 dict；IL3 复位全部环境时传 `slice(None)`，原生环境入口换为索引张量，共享复位只收到张量。
- 复位前最终观察沿用共享 `TerminalObservationMixin`（IL3 观察管理器内部字段与 2.3 相同）；IL3 自带的 `compute_final_obs` 未用，因为执行链观察有状态，需要快照/恢复。

未接入（在 `check` 阶段明确拒绝）：非传感器的 mount/meshes、非机器人实体的命名坐标系、上列以外的传感器工厂。

传感器（都挂在 mount 指定的刚体上，实体 pose 作为 IL3 offset，xyzw 只在后端内转换）：
- `contact_sensor`：复用 isaac 工厂，即 IL3 原生 ContactSensor；reset 后读数为上一物理步的接触，与 isaac 语义相同。
- `frame_transformer`：IL3 原生 FrameTransformerCfg，目标 `{SOURCE_ASSET}/<target_body>`。OvPhysX 上每次读取刚体位姿视图；Newton 上它复制 Newton `SensorFrameTransform`，而后者只在物理步末尾计算，所以 reset 后读到的是复位前的坐标系（初次 reset 为零）。后端在 `_reset_idx` 之后做一次 FK 并用 Newton 公开的 `SensorFrameTransform.update(state_0)` 重算（`scene.refresh_newton_frames`；IL3 无公开的“reset 后刷新传感器”接口，传感器列表取自 `NewtonManager._newton_frame_transform_sensors`，与 IL3 自身 FrameTransformer 的取法相同）。已核对 Newton/OvPhysX/mjlab 在初次 reset 和回合中途单环境复位后的 TCP 读数一致。
- `camera`：IL3 `CameraCfg` 配免 Kit 的 Newton Warp 渲染器（`NewtonWarpRendererCfg`），Newton 与 OvPhysX 都可用，无需 isaacsim/OVRTX。参数与 isaac/mjlab 工厂同一签名（宽高、data_types、焦距、clipping_range、depth_clipping，水平孔径 20.955 mm）。相机 prim 生成为挂载刚体的子 prim（`<body>/<实体名>`，与 IL3 自带腕部相机相同；直接挂在刚体 prim 上时 IL3 会改到 `<body>/camera`），offset 用 `opengl` 约定；`update_latest_camera_pose=True`，否则 IL3 只在初始化时取位姿，Warp 渲染器就不跟随刚体。开启背面剔除，与 Isaac 光栅化一致，相机在自身外壳内仍能看到外部。读数为原始形状 (N, H, W, C)：rgb uint8、depth（米）、normals。深度未命中：Warp 写 0；`max` 由渲染器原生换成远平面值，`none` 由读数换成 inf（同 isaac），`zero` 保持 0。限制：Warp 渲染器不应用近裁剪面，比 near 更近的几何仍可见。选 `renderer: ovrtx` 时同一相机改由 OVRTX 渲染（后端 `assets.use_renderer`）：未命中写 inf（同 Isaac RTX），但 OVRTX 不做 max/zero 裁剪，所以由读数换成远平面值或 0，`none` 保持 inf。远平面附近两者略有差别：同一画面里 Warp 判为未命中的像素比 OVRTX 多约 6%，推测 Warp 按射线距离裁剪、OVRTX 按像平面深度裁剪（未核实）。OVRTX 的 rgb 亮度与 Warp 不同（光照/曝光不同），基于图像的权重不能在两种渲染器间直接互换。check 阶段与 mjlab 相同地核对观察所选图像类型是否由相机产出。

物理连接（`connections`）：IL3 两种引擎都按资产各自导入再复制，资产之间的 USD 关节会丢失，所以按引擎分别接到原生机制：
- Newton：与 mjlab 相同，每个连接在每个世界里成为显式 MuJoCo weld（`newton_builder_world_hook`，eq_solref 时间常数 2·dt），并过滤两侧刚体组之间的接触（Newton 导入 USD 时保留经 FIXED 关节相连的本体，例如 piper_x 的空 `world` 根与 base_link 是两个刚体，而 MJCF 编译会合并它们，所以过滤沿 FIXED 关节扩展到整组）；子实体保留自由根，由复位事件放到连接位姿。需要 MuJoCo-Warp 求解器，否则报 NotImplementedError。weld 有柔性，实测旋转偏差约 10°（mjlab 约 8°）。
- OvPhysX：在 `_init_sim` 内、首次 `sim.reset` 序列化 stage 之前写 PhysX `FixedJoint`（`ExcludeFromArticulation`、关闭两体碰撞）。IL3 原生按资产各自克隆（地面、机器人、附件分别复制），跨资产的关节不属于任何一份克隆而丢失；所以当所有环境持有相同资产时，后端只在 env_0 写关节，用 IL3 公开的 `ovphysx_replicate` 把整个 env_0 一次克隆到全部环境，替换已排队的逐资产克隆（IL3 没有撤回接口，故直接清空其队列，见 `_whole_environment_clone`）。关节锚在世界或各环境资产不同时，退回为每个环境写关节并整 stage 导入（`OvPhysxManager.require_full_stage`），该路径每环境约 28 MB 主机内存，目前没有项目用到。转换得到的固定基座资产把 ArticulationRootAPI 放在通往世界的锚关节上，fix_root_link=False 时 IL3 只禁用该关节，PhysX 会失去 articulation，所以后端与 isaac 后端相同地把 root schema 及其 PhysX 属性移到被连接的刚体上并停用锚关节（`_float_articulation`）。实测（go2_piper_wholebody，随机动作 100 步，抽查 env 0、n/2、n−1）位置误差 ≤0.6 mm、旋转偏差 <1°。建环境加复位的主机内存增量：整 env 克隆 64/512/2048 环境约 3.3/3.7/4.5 GB；整 stage 导入 64/512 环境为 5.0/17.4 GB；同场景 Newton 为 1.7/1.7/1.5 GB，isaac 后端（有连接时关闭 replicate_physics）64/512 环境为 7.3/14.1 GB。
- 连接只能连接同一环境内的实体（与 isaac 相同，check 时拒绝全局实体）。

接触材料（`world.contact_materials`）：直接沿用 isaac 后端的校验与原生启动事件（IL3 在 Newton 和 OvPhysX 上都保留该事件）。`simulation.physics_material` 与 IL3 自带任务相同：PhysX 读取 combine mode，Newton 读取共享字段。

高度扫描器（`factory: height_scanner`）：直接用 IL3 的 `RayCasterCfg`，在 Newton 上由 IL3 映射为形状 BVH 的 `NewtonRaycastSensor`；网格、偏移、`ray_alignment`、`max_distance`、漂移等字段和 isaac 后端相同。传感器跟踪 mount 指定的机器人刚体（`spawn=None`，不再生成已弃用的子 Xform），实体 pose 作为 IL3 offset（位置平移射线起点，旋转只转方向；xyzw 只在后端内转换）。读数 `position` 是挂载刚体根部，`ray_hit_positions` 为环境内坐标，未命中为 inf，与 isaac/mjlab 相同。限制：Newton 射线不按 `mesh_prim_paths` 选网格，只能选“全局世界”（`global_world_only=True`）或“全局 + 本环境全部形状”（会打到机器人自身）。只有前者忠实对应 `meshes`，所以 `meshes` 必须恰好列出全部带形状的全局实体（如 `scope: global` 的地面/地形）；指向按环境复制的物体（如 dual_backend_scene 的台阶方块）在 check 阶段报 NotImplementedError。terrain 工厂复用 isaac 后端实现。

预览（`run.py view/record <project>`）：用 IL3 自带 Newton GL 可视化器（`isaaclab_visualizers.newton.NewtonGLVisualizerCfg`，经 `sim.visualizer_cfgs` 配置；`renderer: ovrtx` 时换成同族的 `NewtonRTXVisualizerCfg`，相机、取景与读帧接口相同），不启动 Kit。view 打开它的交互窗口（需 DISPLAY），record 走离屏 EGL，帧取自其 `render_rgb_array`。只显示 env 0；默认取景与其他后端相同（`shared/assembly/framing.py`，跟随第一个机器人）。IL3 只给 Kit 可视化器提供跟随模式，所以后端每步自行处理：录像时按机器人根部重设视角；窗口模式只平移相机，用户的旋转和缩放保持不变。可视化器按 `sim.render_interval` 重绘，默认每决策步一次（`simulation.render_interval` 可改）。IL3 只在命令行可视化器或连续渲染时才导入视觉几何，配置文件里的离屏可视化器不触发，所以预览时显式设 `load_visual_shapes=True`，否则画面只有碰撞方块。地面上规则排列的黄绿小方块来自 IL3 默认地面贴图（`default_ground_plane_albedo.png` 四角的 NVIDIA 绿），不是抽象体标记。IL3 默认地面只靠 UV 贴图，Newton GL、NewtonRTX 与 Newton Warp 都能正确平铺；曾试过改用 isaac 后端的深色网格，但它靠 MDL 自发光和世界坐标投影画线，Newton 渲染器不执行这些材质（GL 与 Warp 全黑，NewtonRTX 为蓝底白线），故保留 IL3 默认。抽象体标记（`visualization.abstract`）沿用 isaac 后端的同一套规格和 IL3 原生 `VisualizationMarkers`，由 Newton 可视化器绘制（只画 env 0）。IL3 可视化器不连续渲染时不分发标记回调，所以每帧先按 IL3 自带 VideoRecorder 的做法 `sim.forward()` 加 `dispatch_callbacks()`。IL3 的 Newton 标记总把圆柱和圆锥建在 Z 轴上，忽略 `axis`，所以后端把原型改为 Z 向，再给每个实例乘上到 X/Y 的固定旋转（`shared/backends/isaaclab3/abstract_terms.py`）。已用临时项目录像核对矩形线框、点和箭头的形状、朝向与位置。场景相机的画面由上文的 camera 传感器产出，不经可视化器。

已验证预览（dual_backend_scene，2 环境，随机动作，默认跟随 Go2）：`record` 生成 1280×720、60 帧视频，Go2、地面和台阶可见；`view` 在桌面 DISPLAY 上打开窗口，跑完 120 步后正常退出（窗口画面未逐帧核对）。

OVRTX 渲染器（ovrtx 0.5.1）：piper_pick 权重 `record` 在 Newton 与 OvPhysX 上各 150 步，生成 1280×720 路径追踪视频，取景与 GL 相同，可见抓起方块，回合均因成功终止；`view`（DISPLAY=:1）打开 NewtonRTX 窗口运行 200 步后正常退出（未逐帧检查画面）。与 isaac 后端画面对照（临时探针脚本，未改项目）：piper_pick 加 1280×720 场景相机，按默认预览取景（Kit 视口焦距 18.147562 mm）。临时把地面换成 isaac 后端的同一深色网格时，OVRTX 相机渲染我们的 USD 舞台，与 isaac 后端 record 首帧逐像素平均差 2.1/255（Newton 与 OvPhysX 相同），材质、阴影、贴图一致；同场景 Newton Warp 相机为 37.7/255（只读几何、基础颜色与 UV，不执行 MDL 材质、不画 DomeLight 天空、无阴影）。换成只用标准 UsdPreviewSurface 的地面后，纯色盒子地面：OVRTX 1.7/255，Warp 46.0/255；UsdUVTexture 贴图加 `st` UV 的瓷砖地面：OVRTX 3.9/255，Warp 51.0/255。两种地面在 Warp 下颜色与贴图都完整，只是缺天空光与阴影，整体偏暗（平均亮度约 147，isaac 为 197）。因此要三种渲染器都能用，项目材质应只用标准颜色或贴图加 UV，不用 MDL 自发光或世界坐标投影。即 Newton 读取 USD 物理，渲染只覆盖基础外观；要与 Isaac 画面一致须用 OVRTX。NewtonRTX 窗口也按 Newton 模型重建场景，不读我们 USD 的材质与灯光。相机（临时探针：piper_pick 加 2 环境 160×120 腕部相机，用后已删除）：Newton 与 OvPhysX 上 OVRTX 的 rgb/深度形状和类型与 Warp 相同，同一位姿下深度与 Warp 相差约 1e-6 m；远平面设为 0.2 m 时 none/max/zero 三种裁剪读数与 Isaac 语义一致。

短训练（各 3 次 PPO 迭代、每次存档，对比 step 1 与 step 3 的权重，17/17 张量都有变化；只说明接口跑通、权重更新，不代表学会任务）：piper_pick 在 Newton 与 OvPhysX 上用项目默认 256 环境；go2_piper_wholebody 在 Newton 与 OvPhysX 上都用默认 2048 环境（OvPhysX 为整 env 克隆后重测，主机内存峰值比基线多 14 GB；此前整 stage 导入时只能降到 512 环境）。

环境：`.venv-il3`（isaaclab 28.0.0、isaaclab_newton 9.1.4、isaaclab_tasks 25.2.0、newton 1.6.1、mujoco/mujoco_warp 3.12.0、warp 1.17.0、torch 2.11.0+cu130；源码 IsaacLab release/3.0.0 @961760b），与 `.venv`、`.venv-mjlab` 互不影响。选择该后端时 run.py 自动改用此解释器。原生 RLinf PPO 依赖为增量安装，版本记录在 shared/backends/isaaclab3/requirements.txt；未装 wandb（其依赖与 IL3 的 protobuf 冲突，仅在配置 wandb 日志时才需要）。唯一改动的已有包是 hydra-core：IL3 不限版本而装到 1.4.0.dev10，它拒绝 RLinf 入口的 `version_base="1.1"`，故换为 RLinf 自身固定的 1.4.0.dev1（IL3 自己的脚本用 version_base 1.3，同样可用）。OVRTX 渲染器为可选的增量安装，只加了 `ovrtx` 一个包。

已验证（dual_backend_scene，2 环境，GPU，未经 RLinf）：
- `run.py check dual_backend_scene env.train.init_params.backend=isaaclab3` 通过；生成的 observation.py 不变。
- 原生环境构建、复位、30 步推进；观察形状与其他后端一致（关节 12、朝向 4、planner 输出 3），初始位姿与初始关节角符合 scene.yaml，20 步截断及复位前最终观察正常，Task 奖励有限。
- 与 mjlab 同一场景对照（5 步 0.1 动作后）：位置、朝向、速度、关节位置/速度/力矩、抽象体读数一致到约 1e-2；FL_foot 雅可比形状与范数一致（4.115 / 4.114）。零力矩下落静止高度：地面 0.257 / 0.245 m，0.24 m 台阶顶 0.491 / 0.48 m，说明静态方块碰撞生效。
- 这些只说明接口与读数对齐，不代表动力学一致，更不代表可训练或策略成功。

现有项目直接切换（只在命令行加 `env.train.init_params.backend=isaaclab3`，OvPhysX 再加 `+env.train.init_params.physics=ovphysx`；项目文件不改，各项目 observation.py 运行前后一致）。`run.py check` 与 `run.py smoke`（4 环境 × 16 步，契约检查、干净退出）结果：

| 项目 | Newton | OvPhysX |
|---|---|---|
| planner_executor | 接口跑通 | 接口跑通 |
| assembly_go2_navigation | 接口跑通 | 接口跑通 |
| go2_chase | 完整回合成功（go2-supermimic flat.pt） | 完整回合成功（同左） |
| assembly_go2 | 接口跑通 | 接口跑通 |
| go2_piper_wholebody | 部分回合完整（mjlab 训练的 wholebody.pt，63% 轨迹翻倒） | 完整回合成功（mjlab 训练的 wholebody.pt） |
| piper_pick | 完整回合成功（mjlab 训练的 pick.pt） | 完整回合成功（mjlab 训练的 pick.pt） |

piper_pick 用 mjlab 后端训练的权重（运行 `20261022_124529` 的 `global_step_1500`，项目内 `weights/pick.pt`，由 `build_executor(weights=...)` 加载，无需 `--allow-legacy-checkpoint`）做 `run.py evaluate`（16 环境、确定性动作）：Newton 150 回合 success_once 1.0、平均 51 步、抬升 0.23 m；OvPhysX 108 回合 1.0、69 步、0.31 m；训练后端 mjlab 当时评估 153 回合 1.0 / 50 步。修正碰撞体前两引擎均为 0（夹爪无碰撞）。只是单一初始分布下的评估，不代表多条件可靠成功。
go2_piper_wholebody 用 mjlab 训练的 `weights/wholebody.pt`（运行 `20261023_185843` 的 `global_step_6000`）做 `run.py evaluate`（16 环境、确定性动作、1000 步回合）：OvPhysX 16 条轨迹 0 失败、末端误差 5.6 cm（mjlab 5.2 cm）；Newton 19 条、失败 63%、平均 680 步、末端误差 26.6 cm，失败均为倾斜超过 1 rad（机身未触地）。两项修正后才得到这一结果：连接接触过滤扩到 FIXED 关节刚体组（此前 piper_x 的 base_link 与 Go2 机身相互穿插，平均 2.4 步即失败）；改走显式 Python 执行器（Newton 原生隐式 DC 电机上平均约 64 步）。剩余差距来自 Newton 较软的接触与 weld，需在 Newton 上训练才能消除，目前未做。
其余项目在该后端上都没有做训练（无权重更新）或回合成功验收；Go2 电机配置与 Isaac Lab 2.x 不同，Newton/OvPhysX 的接触与连接刚度也与 PhysX 不同，旧权重不能直接认为可迁移。
相机（临时探针：piper_pick 加 160×120 腕部相机，rgb + distance_to_image_plane，用后已删除）：Newton 与 OvPhysX 上相机位置等于 camera_link 并随动作跟随，画面可见方块与夹爪；4 个环境在复位后和 3 步后的逐行深度中位数与 mjlab 一致；三种后端的 check/smoke 均通过。未用相机做训练。

已验证（经原生 RLinf，统一入口）：
- `run.py smoke dual_backend_scene env.train.init_params.backend=isaaclab3` 通过（契约检查、干净退出，动作形状 [2,1,12]）。
- `run.py train ... runner.save_interval=1`：两次 PPO 迭代，step1/step2 检查点间 7/7 参数张量变化（最大 2.5e-3）。
- `run.py evaluate --snapshot <run> env.eval.max_episode_steps=8 runner.ckpt_path=<step2>`：从匹配快照加载，完成 8 条长度 8 的回合，Python Task 指标（task/height_error_m）经原生评估输出；success=0。配方默认评估 32 步而回合长 100，不缩短回合时轨迹数为 0。
- 验收记录：`outputs/dual_backend_scene/runs/20261026_074326_79941716fe4d/`（training.log、evaluation.log、weight_update_check.json、validation.json）。只说明接口与优化链路接通，不是学会任务的策略。
- 高度扫描器（临时探针项目：dual_backend_scene 加挂在 base 上、z=20、yaw 对齐、0.1 m 网格 1.6×1.0 的扫描器，用后已删除）：isaaclab3 187 条射线全部命中地面 z=0，网格以机器人为中心；与 mjlab 同场景逐射线对照，相对机器人的 xy 偏移最大差 4.5e-6 m，z 相同（机器人未转动，未单独核对转动时的 yaw 对齐）。把读数接入 executor 观察后 `run.py smoke ... backend=isaaclab3` 通过，`ray_hit_positions` 作为 (187, 3) 观察组传入模型，未展平。未验证 debug_vis 射线显示，也未用扫描器做训练。

## 后续顺序

1. 扩展接触/固定连接，补充 Blender 的后端能力提示；物理实体的 Task 事件（按 interval 事件钩子接入）。
2. 关节顺序已按 ASSET.json 统一（见“关节顺序”）；继续对齐动力学，再开展跨后端权重迁移与策略学习验收。公共配置一致不等于物理行为一致。

仍有部分编辑器目录与历史 term 元数据使用 Isaac 的静态目录，未宣称已完全后端无关；新 runtime 和 mjlab 状态读取不依赖这些 SDK 实现。后续按实际能力抽离，不为尚未实现的功能造空适配。

## 验证

- test_scene_backends.py：真实 mjlab/MuJoCo 模型的初始关节、父子旋转、动态刚体和短 CPU 参考步进。
- test_portable_project.py：同一项目两后端配置一致；对象读取与惰性 Task；原生 mjlab ActionTerm 配置/实例调用公共执行链、实际单位解码、部分环境复位；错误执行器模式拒绝。
- 原有场景、控制链、对象接口和 Python Task 配置回归测试。
- test_isaaclab3_backend.py：isaaclab3 静态布局与 isaac 一致且不导入 SDK、四元数边界、未接入能力在仿真前拒绝（含 contact_materials 校验）、标记轴向旋转、高度扫描器 meshes 只接受全部全局形状、连接只接受环境内实体及 weld 相对位姿、相机工厂与 isaac 同签名且参数在 SDK 前校验；URDF 导入器式 instanceable Xform 碰撞体移到其下 Mesh 并保留 approximation；渲染器只接受 newton/ovrtx；装有 isaaclab_newton / isaaclab_ov 时检查 `simulation.newton` / `simulation.physx` 覆盖原生预设。

- test_mjlab_runtime.py：ANYmal-C 的 LSTM 执行器每步输出与 Isaac 计算方式逐项一致、只复位指定环境的隐状态；可选 RUN_MJLAB_GPU=1，验证 GPU 并行推进、单环境截断/复位、复位前最终观察和返回 Tensor 独立快照；普通测试验证后端接口缓存隔离。
- test_stock_rlinf.py：实际原生训练及评估的观察合并路径；共享边界处理评估的字典列表，不改上游源码。
- 实机已完成 2 环境 rollout smoke 和原生 RLinf 两次 PPO 迭代，两个检查点间有参数变化。该示例只验证接口与优化链路，不能当作学会站立或爬楼。

后端只记录在快照与回放报告中（`backends`、`backend_changed`），不进入策略接口签名；用 `--snapshot <run> env.train.init_params.backend=mjlab` 可在另一后端回放同一 checkpoint（评估后端随之切换，除非显式另设）。回放时仍在实际后端上逐项核对观察、动作与关节顺序；各后端原生刚体列表顺序不同且策略不读取，跨后端时不比较。这只说明接口对得上，不保证动力学、接触与渲染一致，结果属于 sim2sim 对照。Isaac 与 mjlab 分别保存原生关节/动作缓存，防止同一项目切换后端误用缓存。

实机验收记录：`outputs/mjlab_ppo_verified_20261019/runs/20261019_111313_4452e77f029a/`。
包含 training.log、evaluation.log、weight_update_check.json 与 validation.json。两次 PPO 迭代后 7 个参数张量变化；从匹配快照加载第二次检查点，经原生 RLinf 评估完成 4 个长度为 3 的回合。success=0，不能解释为训练成功的导航或爬楼策略。
