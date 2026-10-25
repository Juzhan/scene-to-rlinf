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
| `__init__.py` | 可选 `SDK`、`INTERPRETER`：当前解释器缺 SDK 时改用的工作区解释器 | run.py、inspect、scene_backend |
| `assets.py` | `FACTORIES` 资产工厂 | 场景加载、资产目录 |
| `project.py` | `PROJECT_TERMS`（是否接受项目 terms.py 的原生 term）、`UNKEYED_FACTORIES`（项目 assets.py 无后端键的 `FACTORIES` 是否属于该后端）、`validate_scene`、`validate_project`；可选 `inspect_scene`（显式 `--backend` 的场景单独检查/查看，未提供时明确报错） | prepare_scene、backend_scene、scene_backend |
| `control.py` | `install_terms`：挂接 planner/executor 原生控制项 | compile_control |
| `environment.py` | `make_environment`；预览窗口可提供 `window_closed` | Project、preview |
| `robot.py` | `robots(env, settings)`：返回 `shared.assembly.robot.Kinematics` 子类实例，实现 `joint_names`/`joint_position`/`body_names`/`body_pose`/`body_jacobian` 五个原语 | 本后端 control.py、python_task.py 注入到固定组件、decode/initial 与 Task 的观察对象 |

机器人能力接口（`shared/assembly/robot.py`）：生成的 `ExecutorObs`/`PlannerObs`/`SceneInfo` 中 `kind: robot` 的节点继承 `RobotView`，提供 `joint_names()`、`frame_pose(frame)`、`jacobian(frame)`、`differential_ik(frame, position, orientation=None, *, damping, max_step)`，按环境批量计算，环境坐标系、米、wxyz。`frame` 取本体 body 名或命名坐标系；命名坐标系由资产默认（`ASSET_FRAMES`，如 Franka、Piper X 的 `tcp`）加机器人实体的 `frames`（同名覆盖）组成，名称不得与方法、信息字段或子实体重名。命名坐标系同时是观察候选 `<机器人>.<坐标系>.position|orientation`，由公共层经 `frame_pose` 读取，后端无需编译传感器（mjlab 因此允许机器人实体声明 `frames`，非机器人实体仍报错）；`frame_transformer` 传感器保持为独立读数，不算机器人坐标系。运动链由该坐标系决定，不配置关节列表，scene.yaml 也不出现 IK。命名坐标系偏移、雅可比参考点平移和阻尼最小二乘步在公共层计算；后端只提供原语，也可覆盖方法改用原生求解。雅可比在 body 原点处，世界轴，列序同 `joint_position`：Isaac 读 PhysX `get_jacobians()`（其线速度行参考连杆质心，后端平移到连杆原点），mjlab 调 `mujoco_warp.jac` 并在 body 原点取值。这些方法只在环境进程可用；训练组件在 policy worker 只收到张量，调用时明确报错。`differential_ik` 返回一步后的绝对关节目标，不施加关节限位，大误差时用 `max_step` 限制单步。

新后端实现这些模块后加入 `BACKENDS`；暂不支持的能力在钩子内明确报错，不在上层补分支。
仍有少量非分支的 Isaac 依赖留在上层：check 报告的 native_key、历史公式 Task 校验、观察选择解析（`isaac.task_config._selection`）与 Isaac 导出的可视化缓存键。

`env.train.init_params.backend` 是公共后端选择（默认 isaac）。写在 train.yaml 或在命令行覆盖都可以，评估都随训练后端切换；只有显式写 `env.eval.init_params.backend` 时评估才单独选择。选 mjlab 时 run.py 自动改用 `.venv-mjlab` 重新启动（主 `.venv` 没有 mjlab）；显式设置了 `RLINF_PYTHON` 时不替换，缺 mjlab 则直接报错。`run.py check <project> --backend mjlab` 对完整项目检查同一配置及后端能力；也可用标准 Hydra override。没有第二份 mjlab train.yaml。

mjlab 已接入原生 ManagerBasedRlEnv 和未修改的 RLinf PPO，可通过统一入口 check/inspect/smoke/train/evaluate。stock RLinf 中 env_type 仍使用现有 isaaclab 注册槽位，它负责进程和批次传输；项目工厂按 backend 构建真实 mjlab 环境，不启动 Isaac Sim。内部名称不代表仿真后端，项目无需另写 worker 或 runner。

`inspect/view <project> --backend mjlab` 仍是独立的物理场景检查/原生查看器：编译真实 MuJoCo 模型，但不运行 planner/executor 或 Task。场景编译支持 go2、ground_plane、box、dex_cube、clear_sky、USD 转换机器人/物体、父子摆放、初始关节和抽象体的静态/跟随读数；在线运行已验证两个 GPU 并行环境。公共字段可存在于 SceneInfo，未实现的 mjlab 字段在选中或读取时明确拒绝，不生成假数据。

mjlab 已接入：
- 挂载传感器：`camera`（mujoco_warp 原生相机，rgb 与 distance_to_camera/distance_to_image_plane，视场按 Isaac 针孔换算，远平面裁剪同 Isaac 的 max/zero/none）与 `frame_transformer`（MuJoCo framepos）。深度已在 GPU 上对照几何验证；mujoco_warp 返回射线距离，像平面深度按像素余弦换算。
- 物理实体的随机根复位（`reset.region/yaw_range/interval`，按环境采样）；关节力矩读数 `joint_effort`。
- `view` / `record`：用 `env.train.init_params.backend=mjlab` 运行项目 policy（或零/随机动作）；`view` 在 MuJoCo 自带的被动窗口（`mujoco.viewer.launch_passive`）显示 env 0，`record` 离屏录 env 0 的视频。两者都由共享 preview 循环推进策略，窗口只同步显示、不推进物理；不使用 mjlab 自带 viewer 的策略循环。关闭窗口即结束，需要桌面 DISPLAY。

与 Isaac 的已知差异：近平面不裁剪；相机原点落在所挂本体的几何内部（如转换后的 D435 外壳盒）时，光线追踪会从内部击中它，因此构建场景时把这类几何移到相机传感器不渲染的 geom group 4（碰撞不变；录像与查看窗口打开 group 4，仍会显示）；未命中处 rgb 为黑色而非天空。物理连接、抽象体随机复位和抽象体显示尚未接入。

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
| go2 | Isaac 5.1 资产（`source/` 内 URL vendor） | DC 电机，参数同 UNITREE_GO2_CFG |
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
- 后端读数（`shared/backends/<后端>/readers.py`）把 `joint_position`、`joint_velocity`、`joint_effort`、`default_joint_position`、`default_joint_velocity` 在设备上按 `joint_order` 取列；机器人接口的 `joint_names()`、`joint_position()` 和雅可比列同样换序；`inspect` 报告（快照回放核对的 `articulations.joint_names`）也是规范顺序。每个后端经 `robot.joint_layout` 钩子提供，上层不分支。
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
- USD 保留原始三角形作为一个静态网格碰撞体。MuJoCo 的网格碰撞取凸包，非凸地形不能整体转成一个网格，因此 MJCF 按可精确表示的部件重建：轴对齐封闭块 → box；其他封闭凸体 → 凸包网格；规则高度网格的开放曲面 → hfield；平坦开放地块 → 薄板。封闭非凸块或不规则曲面直接拒绝，不做近似。
- 写出后对 MJCF 与生成的三角形在 41×41 个点上做竖直射线比较，最大高度差记入 ASSET.json（`max_height_difference`），超过 2 mm 则报错且不写 ASSET.json。MuJoCo 与 Isaac Lab 高度场用同一条网格对角线，两者吻合。
- 21 种均已在 Isaac Lab 2.3.2 上生成并通过上述比较：网格型落在 box 和凸包（`stairs` 41 个 box、`random_grid` 294 个、`star` 7 个 box 加 3 个凸包、`repeated_*` 2 个 box 加 7 个凸包），高度场型落在 hfield。MuJoCo 中小球落在 stairs、slope、star、repeated_pyramids、stepping_stones 上的静止高度与 USD 表面一致（误差 ≤ 0.5 mm）；Isaac 侧对 21 块逐一做了落体验证（每块 9 个小球，从 3 m 高处落下，静态网格碰撞体）：球静止时底部与原始三角形表面重合，没有穿透（`rough` 因球落在粗糙面上高 1–4.8 mm）；`inverted_slope`、`waves` 的球在碗形地面上仍在滚动，只确认了未穿透，没有静止数据。
- 限制：摩擦两侧都是默认值；离开地块边缘即坠落；Isaac 的 `height_scanner` 与射线 `meshes` 在 mjlab 不支持，所以 mjlab 上只能是盲策略；Isaac 侧是每个环境克隆一份，`env_spacing` 须不小于地块边长，mjlab 各环境独立世界无此限制；所有环境用同一块地形，没有课程难度。

**2. `terrain` 工厂（仅 Isaac，用于课程难度的多地块训练）。** 不在 Web 编辑器资产架上，只能在 scene.yaml 中手写。`terrain_generator.sub_terrains` 每项写 `factory` 与 `parameters`（`proportion` 控制占比）。mjlab 后端没有该工厂（没有 `terrain` 导入器和按环境分配地块的机制）。

- 网格型（几何精确）：`flat`、`stairs`、`inverted_stairs`、`random_grid`、`rails`、`pit`、`box`、`gap`、`floating_ring`、`star`、`repeated_pyramids`、`repeated_boxes`、`repeated_cylinders`。
- 高度场型（按网格采样，适合粗糙地面）：`rough`、`slope`、`inverted_slope`、`hf_stairs`、`hf_inverted_stairs`、`obstacles`、`waves`、`stepping_stones`。
- `repeated_*` 的 `object_params_start/end` 写成字典（`num_objects`、`height`，以及 `radius` 或 `size`）。
- 来自 Isaac Lab 的约束：`random_grid` 的 `size` 不能被 `grid_width` 整除（余量作边界，如 8 m 用 0.45）；`stepping_stones` 的 `holes_depth` 默认 -10 m，石间是深坑；高度场型用 `border_width`、`horizontal_scale`、`vertical_scale` 控制精度。
- 只验证能由真实 `terrain` 工厂构造并生成网格，没有验证机器人能否在其上行走或训练。

## 后续顺序

1. 扩展接触/固定连接、抽象体随机复位，补充 Blender 的后端能力提示。
2. 关节顺序已按 ASSET.json 统一（见“关节顺序”）；继续对齐动力学，再开展跨后端权重迁移与策略学习验收。公共配置一致不等于物理行为一致。

仍有部分编辑器目录与历史 term 元数据使用 Isaac 的静态目录，未宣称已完全后端无关；新 runtime 和 mjlab 状态读取不依赖这些 SDK 实现。后续按实际能力抽离，不为尚未实现的功能造空适配。

## 验证

- test_scene_backends.py：真实 mjlab/MuJoCo 模型的初始关节、父子旋转、动态刚体和短 CPU 参考步进。
- test_portable_project.py：同一项目两后端配置一致；对象读取与惰性 Task；原生 mjlab ActionTerm 配置/实例调用公共执行链、实际单位解码、部分环境复位；错误执行器模式拒绝。
- 原有场景、控制链、对象接口和 Python Task 配置回归测试。

- test_mjlab_runtime.py：ANYmal-C 的 LSTM 执行器每步输出与 Isaac 计算方式逐项一致、只复位指定环境的隐状态；可选 RUN_MJLAB_GPU=1，验证 GPU 并行推进、单环境截断/复位、复位前最终观察和返回 Tensor 独立快照；普通测试验证后端接口缓存隔离。
- test_stock_rlinf.py：实际原生训练及评估的观察合并路径；共享边界处理评估的字典列表，不改上游源码。
- 实机已完成 2 环境 rollout smoke 和原生 RLinf 两次 PPO 迭代，两个检查点间有参数变化。该示例只验证接口与优化链路，不能当作学会站立或爬楼。

后端只记录在快照与回放报告中（`backends`、`backend_changed`），不进入策略接口签名；用 `--snapshot <run> env.train.init_params.backend=mjlab` 可在另一后端回放同一 checkpoint（评估后端随之切换，除非显式另设）。回放时仍在实际后端上逐项核对观察、动作与关节顺序；各后端原生刚体列表顺序不同且策略不读取，跨后端时不比较。这只说明接口对得上，不保证动力学、接触与渲染一致，结果属于 sim2sim 对照。Isaac 与 mjlab 分别保存原生关节/动作缓存，防止同一项目切换后端误用缓存。

实机验收记录：`outputs/mjlab_ppo_verified_20261019/runs/20261019_111313_4452e77f029a/`。
包含 training.log、evaluation.log、weight_update_check.json 与 validation.json。两次 PPO 迭代后 7 个参数张量变化；从匹配快照加载第二次检查点，经原生 RLinf 评估完成 4 个长度为 3 的回合。success=0，不能解释为训练成功的导航或爬楼策略。
