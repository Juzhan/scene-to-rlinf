# 场景后端接入

## 目标与当前进度

项目继续编辑公共 scene.yaml、对象式观察、策略各层（`policy.layers`）和 Python Task。共享层把公共场景翻译为各仿真器原生配置；不向用户暴露第二套任务配置，也不改 RLinf 源码。

## 公共定义与后端职责

| 内容 | 公共层 | 后端层 |
| --- | --- | --- |
| 观察 | entity_information、readers、Signals、ObjectView、生成的 observation.py | 原生 Tensor 的读取及坐标参考点映射 |
| 策略各层 | policy / control 配置读取与校验（layers.py、control.py）、policy_runtime 调度/固定层/层间结果/复位 | 挂入原生 ActionTerm/ObservationTerm |
| 控制目标 | 按机器人自动生成（或旧 control.output 命名终点）、解码、批次/设备/浮点/有限值约束 | 原生关节解析、执行器模式核验、实际目标写入 |
| Task | python_task 的惰性读取、每步评价一次、结果校验、TaskMemory 复位 | 原生奖励/终止/重置 manager 的挂接 |
| train.yaml | 同一 state_ppo 配方、ObjectPolicy 训练协议 | 环境构建和 RLinf rollout 连接 |

上述公共逻辑已由 Isaac 使用，mjlab 已有状态读取、原生控制项和 Task manager 配置适配；不复制两套任务或策略调度。共享 runtime 不导入仿真 SDK。底层驱动维持原生 manager 步进，未增加自写仿真训练循环。

### 后端钩子

上层（run.py、tools、shared/assembly、实验快照）不按后端名分支，只经 `shared.assembly.assets` 的
`BACKENDS` / `DEFAULT_BACKEND` / `selected_backend` / `backend_module(后端, 模块)` 找到 `shared/backends/<后端>/` 下的同名实现：

| 位置 | 内容 | 调用方 |
| --- | --- | --- |
| `__init__.py` | 可选 `SDK`、`INTERPRETER`：当前解释器缺 SDK 时改用的工作区解释器 | run.py、tools/inspect_scene（接口解析子进程）、scene_backend |
| `assets.py` | `FACTORIES` 资产工厂；`FREE_SENSORS`：可不挂载、固定在世界的传感器（三个后端都是 camera；isaaclab3 把它生成在所在环境下、位姿即相对环境原点的偏移）。其余传感器只存在于连杆上：编辑器添加时不提供“固定在世界”，删除其挂载实体时随之删除 | 场景加载、资产目录、web_editor |
| `project.py` | `PROJECT_TERMS`（是否接受项目 terms.py 的原生 term）、`UNKEYED_FACTORIES`（项目 assets.py 无后端键的 `FACTORIES` 是否属于该后端）、`validate_scene`、`validate_project`；可选 `inspect_scene`（显式 `--backend` 的场景单独检查/查看，未提供时明确报错） | prepare_scene、backend_scene（网页编辑器每次写入后经它提示）、scene_backend |
| `control.py` | `install_terms`：挂接策略的原生控制项 | compile_control |
| `environment.py` | `make_environment`；预览窗口可提供 `window_closed` | Project、preview |
| `native_terms.py`（可选） | 术语在原生管理器里的实现路径（`observation`/`action`/`reward`/`signal` 四个函数），供场景目录显示；不提供时用该后端自己的模块路径。isaac 返回 `isaaclab.envs.mdp.*`，mjlab/isaaclab3 各自绑定，不伪造 | shared/assembly/catalog.py |
| `randomization.py` | `validate(randomizations, settings)`：静态检查 `init_params.randomization` 经 `shared/assembly/randomization.resolve` 得到的声明，不支持时用 `unsupported` 报错（isaac）；isaaclab3/mjlab 另由本后端 environment 把 `settings['_randomization']` 绑成原生事件项 | prepare_scene、本后端 environment |
| `robot.py` | `robots(env, settings)`：返回 `shared.assembly.robot.Kinematics` 子类实例，实现 `joint_names`/`joint_position`/`body_names`/`body_pose`/`body_jacobian` 五个原语 | 本后端 control.py、python_task.py 注入到固定组件、decode/initial 与 Task 的观察对象 |
| `visuals.py`（可选） | 编辑器网格导出：`SOURCES`（使导出网格缓存失效的源文件 glob）、`launch_options`/`launch`（启动原生场景）、`snapshot`（返回各连杆位姿，wxyz）。没有该模块的后端（目前 mjlab）在编辑器中只显示示意代理。导出是增量的，只生成未缓存的实体；编辑器导出的是未保存的工作副本 `.scene.yaml.<pid>.editing`，网格缓存仍记在 `scene.yaml` 名下；只剩静态几何（桌子、无质量 box、地形地块）时不启动物理求解器（Newton 的 MuJoCo 求解器拒绝没有关节的模型）。关节体与刚体都记录 `body_names`/`root_body`，传感器可挂在刚体上（没有连杆网格帧的刚体，如 USD Cube，以实体自身位姿作 root_body 预览）；已导出却没有 body 的静态物体不列为挂载目标。导出在后端的 `INTERPRETER` 中运行，可用 `RLINF_PYTHON` 覆盖；编辑器本身在装了 viser 的解释器中运行（`shared/assembly/runtimes.py` 查找，同样可用 `RLINF_PYTHON` 覆盖） | shared/assembly/visuals.py、tools/export_scene_visuals.py、web_editor |

机器人能力接口（`shared/assembly/robot.py`）：生成的 `ExecutorObs`/`PlannerObs`/`SceneInfo` 中 `kind: robot` 的节点以及资产能力含 `joint_state` 的带关节物体（如 `usd_articulation` 柜子，`robot_interface(node)` 判定）继承 `RobotView`，提供 `joint_names()`、`frame_pose(frame)`、`jacobian(frame)`、`differential_ik(frame, position, orientation=None, *, damping, max_step)`，按环境批量计算，环境坐标系、米、wxyz。`frame` 取本体 body 名或命名坐标系；命名坐标系由资产默认（`ASSET_FRAMES`，如 Franka、Piper X 的 `tcp`）加机器人实体的 `frames`（同名覆盖）组成，名称不得与方法、信息字段或子实体重名。命名坐标系同时是观察候选 `<机器人>.<坐标系>.position|orientation`，由公共层经 `frame_pose` 读取，后端无需编译传感器（mjlab 因此允许机器人实体声明 `frames`，非机器人实体仍报错）；`frame_transformer` 传感器保持为独立读数，不算机器人坐标系。运动链由该坐标系决定，不配置关节列表，scene.yaml 也不出现 IK。命名坐标系偏移、雅可比参考点平移和阻尼最小二乘步在公共层计算；后端只提供原语，也可覆盖方法改用原生求解。雅可比在 body 原点处，世界轴，列序同 `joint_position`：Isaac 读 PhysX `get_jacobians()`（其线速度行参考连杆质心，后端平移到连杆原点），mjlab 调 `mujoco_warp.jac` 并在 body 原点取值。这些方法只在环境进程可用；训练组件在 policy worker 只收到张量，调用时明确报错。`differential_ik` 返回一步后的绝对关节目标，不施加关节限位，大误差时用 `max_step` 限制单步。

新后端实现这些模块后加入 `BACKENDS`；暂不支持的能力在钩子内明确报错，不在上层补分支。
上层的公共部分已不再导入具体后端包：native_key 与观察选择解析在 `shared/assembly/scene_metadata.py` 与
`shared/assembly/native_terms.py`，历史公式 Task 的校验在 `shared/assembly/legacy/sources.py`；原生实现路径由后端的
`native_terms.py` 声明。Isaac 导出的可视化缓存键仍按后端 `visuals.py` 声明，属后端自身。

`env.train.init_params.backend` 是公共后端选择（默认 isaaclab3；isaac 即 Isaac Lab 2 不再维护）。写在 train.yaml 或在命令行覆盖都可以，评估都随训练后端切换；只有显式写 `env.eval.init_params.backend` 时评估才单独选择。选 mjlab 时 run.py 自动改用 `.venv-mjlab` 重新启动（主 `.venv` 没有 mjlab）；显式设置了 `RLINF_PYTHON` 时不替换，缺 mjlab 则直接报错。`run.py check <project> --backend mjlab` 对完整项目检查同一配置及后端能力；也可用标准 Hydra override。没有第二份 mjlab train.yaml。

mjlab 已接入原生 ManagerBasedRlEnv 和未修改的 RLinf PPO，可通过统一入口 check/smoke/train/evaluate。stock RLinf 中 env_type 仍使用现有 isaaclab 注册槽位，它负责进程和批次传输；项目工厂按 backend 构建真实 mjlab 环境，不启动 Isaac Sim。内部名称不代表仿真后端，项目无需另写 worker 或 runner。

`view <project> --backend mjlab` 仍是独立的物理场景原生查看器：编译真实 MuJoCo 模型，但不运行策略或 Task。场景编译支持 go2、ground_plane、box、dex_cube、clear_sky、USD 转换机器人/物体、父子摆放、初始关节和抽象体的静态、随机复位、Task 移动与跟随读数；在线运行已验证两个 GPU 并行环境。公共字段可存在于 SceneInfo，未实现的 mjlab 字段在选中或读取时明确拒绝，不生成假数据。

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

`run.py asset convert` 让同一资产同时具备 USD（Isaac）与 MJCF（MuJoCo/mjlab）：USD→URDF 用 Isaac Sim 自带的 `isaacsim.asset.exporter.urdf`（nvidia.srl UsdToUrdf，仅需 pxr），URDF→MJCF 用 MuJoCo 编译器，URDF→USD 用 `urdf-usd-converter`（`import` 组）、MJCF→USD 用 `mujoco-usd-converter`（`usd` 组），都不启动 Kit。目前没有成熟的直接 USD→MJCF 工具，所以经 URDF 中转。

- 资产目录统一为 `assets/<名>/`：ASSET.json、`mjcf/`，有本地覆盖时再加入口 USD；引用的上游文件按原仓库（或 URL）相对路径原样复制到 `source/`，`source/SOURCE.json` 记录 `{repository, commit}` 或 `{url}`，以及逐文件 sha256。`run.py asset check` 先核对这些 sha256。
  - Git 来源：`run.py asset vendor <上游 USD/URDF/MJCF> assets/<名>`。带上它加载的全部文件（MJCF 跟随 `<include>` 与 compiler 的 meshdir/texturedir/assetdir）、所用 ROS 包的 package.xml（`package://` 由此在 `source/` 内解析），以及从模型目录到仓库根各级的 LICENSE/COPYING。只接受已提交、未修改的上游文件。
  - URL 来源：`run.py asset vendor <URL> assets/<名> --base <根 URL>`。递归下载 USD 引用的图层/资产与 MDL 引用的贴图，保存为 `source/<URL 在 base 下的路径>`，相对引用因此原样可用。绝对引用、base 之外的引用和服务器上不存在的文件（Kit 内置的 OmniPBR.mdl 等）不下载，命令列出它们。Isaac 资产用 `--base https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1`（`shared.asset_formats.NUCLEUS_URL`）。
  - 代码用 `nucleus_copy(资产名, Nucleus 相对路径)` 取本地副本；文件缺失时报错，并给出对应的 vendor 命令，不回退到联网读取。
  - 已迁移：Piper X、Piper、SO-101、Go2（USD 及其转换 MJCF）、ANYmal-C（含 ANYdrive LSTM 与导航用 Blind 策略）、Franka Panda、Seattle 桌、DexCube、白色大理石材质、天空 HDR、UR5e、Robotiq 2F-85、FrankaSim Panda（RLinf/serl）。这些资产及引用它们的项目不再读 `third_party/`、其他工作区的模型文件或在线 Nucleus。
  - 尚未迁移：直接使用 Isaac Lab 原生预设的旧项目（Go2/M20 粗糙地形环境、Piper lift、OpenArm 等）中，预设内部仍按 `ISAAC_NUCLEUS_DIR` 联网加载地面材质、天空、桌子等，只有项目显式引用的机器人与物体改为本地副本。`piper_phone_sim` 保留项目内修改过的 Piper X 副本。m20、m20_piper_pendulum、franka_reference 没有上游来源记录，无法 vendor。
- SimReady 资产（`tools/simready_mirror.py` 镜像到 `assets/simready/<发布方>/<资产>/`，CC-BY-4.0）只在 isaaclab3 上使用 USD，不转 MJCF：`run.py asset simready <资产>/simready_usd/sm_<资产>_01.usd assets/<名>` 写 `usd/<名>.usda` 包装层，原样引用 SimReady 文件，只加覆盖。源文件引用的图层缺失（镜像未下全）时直接报错，提示重跑镜像，不补造碰撞体。覆盖内容：带 RigidBodyAPI 的实例化 prim 以及实例内的碰撞体取消实例化（Isaac Lab 经 apply_nested 写固定根、质量、材料时跳过实例代理，铰链柜会成为自由体；OvPhysX 不读实例原型里碰撞体的质量，苹果会从 0.157 kg 变成 0.087 kg）；没有转动/平移关节的资产合成一个刚体，放在默认 prim 上：只用固定关节连接或互不连接的多个部件停用关节、去掉各自的刚体与质量，在根部写总质量（只有固定关节的铰链是零自由度，OvPhysX 无法承载；已有单一刚体时保持原处）；有可动关节的资产保留铰链：关节 body0/body1 指向碰撞网格的改指所属刚体（网格与刚体坐标系不一致时报错）；缺 ArticulationRootAPI 时加在唯一不被关节带动的刚体上；根刚体（关节根，或单刚体资产的那个刚体）的坐标系移到资产原点（`_reroot`：其子几何承接原变换，以它为端点的关节坐标系和质心、主轴随之换算，几何与动力学不变）——Isaac Lab 把场景位姿写给根刚体，SimReady 铝梯的根刚体在资产原点上方 2.2 m、大纸箱 0.41 m，不移会在复位时被埋到地下再被挤出（OvPhysX 约一秒升回，Newton 以 120 m/s 弹飞）；isaaclab3 的 `usd_object` 遇到根刚体不在原点的未包装 USD 直接报错。转动/平移关节写 `newton:limitStiffness = inf`（PhysX 的 UsdPhysics 限位本就是硬限位，Newton 默认导入为 1e4 弹簧、几乎无阻尼，5 kg 抽屉会冲过限位约 0.1 m 再弹回）。ASSET.json 记录源图层哈希、SimReady package_id、许可、镜像 revision、改动清单与 `joint_order`；`run.py asset check` 核对源图层与包装层未变。场景中所有导入物体统一用 `usd_object`（仅 isaaclab3 按 USD 结构区分，`kind: entity`）：单个刚体是动态刚体，带可动关节的是没有执行器的铰链（`actuators={}`），关节只受接触与 USD 阻尼/限位作用，可读 `<名>.joint_position/joint_velocity`，`initial_state.joint_positions` 设初始开度（写入时替换资产默认关节表，须列全），`frames` 声明挂在部件 body 上的命名坐标系（如抽屉前板上沿）作为观察；`fix_root_link: true` 把铰链根固定、把刚体设为运动学体。USD 没写质量的动态刚体须填 `mass`，不猜密度。`usd_articulation` 为已有场景保留，等同带关节、默认固定根的 `usd_object`。已用 `cabinet_file_a01`（4 门 6 抽屉）在 Newton 与 OvPhysX 验证固定根、被动关节阻尼、抽屉限位和抽屉内落物；Newton 抽屉撞限位仍反弹约 12%（OvPhysX 为 0）。苹果、易拉罐、平底锅、胶带座（固定关节合并）、玻璃罐（多部件合并）、运动学木箱在两引擎 2 环境下验证了质量一致、落地与复位回到写入位姿。接触质量未全面核对：SimReady 的 PhysX 碰撞用 `approximation = "sdf"`，Newton 不认识该值而用原三角网格；平底锅落在木箱中央时 Newton 上会弹跳甚至滑落，压在箱沿时 OvPhysX 上持续抖动。试过在后端为 Newton 补 `NewtonSDFCollisionAPI`，锅变稳，但柜顶苹果会逐渐沉进柜体（柜体网格不封闭），未采用。示例 `projects/franka_drawer`。 网页编辑器的资产库自动列出 `assets/*/ASSET.json` 中由 `run.py asset simready` 生成的资产，导入 USD 物体也一律添加为 `usd_object`，按文件中的刚体与关节决定动态刚体或被动关节物体，用户不需要知道引擎参数。
- SimReady 按键与旋钮：不少资产带可动部件，例如台灯电源键、血氧仪按键（滑动关节，行程 0.7–1.4 mm）、插座摇杆开关、万用表旋钮。按下即该关节位置变小，可读 `<名>.joint_position`。资产自带的弹簧太弱：0.1–10 N/m 的弹簧压缩 1 mm 只有几毫牛，托不住键帽（台灯键帽 0.1 kg），两引擎上都停在按下的极限；SimReady 写的目标位置又在关节限位之外（本意是预紧），Newton 遇到限位外的目标会改用限位中点（ModelBuilder），按键停在半按下。`usd_object.springs`（关节名 → `{stiffness, damping, target}`，SI 单位，target 缺省 0 即资产静止位，须在限位内）写一个缓存的 USD 覆盖层（`.cache/usd_springs/`，`shared/asset_formats.joint_spring_layer`），两引擎都从 USD 读取驱动。实测（台灯 8000 N/m、血氧仪 20 N/m、开关 0.5 N·m/rad，2 环境，150 步）：静止下沉 = 重量/刚度（台灯 0.12 mm，两引擎一致）；对台灯键施加 6 N 按到 0.87–0.92 mm（理论 0.87 mm），松开后约 0.2 s 回到静止位，Newton 先越过静止位 0.3 mm 再回落。Newton（MuJoCo-Warp）显式积分弹簧刚度：刚度/运动质量须小于约 1/子步²（5 ms 时 4e4），否则发散。“按一下切换并保持”之类的状态在 task.py 里用 `TaskMemory` 按环境记录（读关节位置判断按下、与上一步比较得到按下沿）；灯亮、指针、数字显示这类只影响画面的输出尚未接入。包装时关节根 `ArticulationRootAPI` 写在非刚体上的资产（如血氧仪）会移到不随关节运动的基座刚体上，否则 `fix_root_link` 在 Isaac Lab 中报错。
- 网页编辑器“场景”页是场景列表（`tools/scene_outliner.html`，与资产架同样嵌入、同样经资产库服务）：每行末尾四个开关——锁定（视图中点不中、拖不动，列表中仍可选中编辑；地面/地形默认锁定；写入 scene.yaml 的 `authoring.selectable`，只是编辑器设置）、物理（受力运动 / 不受力即 `kinematic` / 固定根部 `fix_root_link`；静态几何没有刚体，不能切换）、渲染可见（`visible`，抽象体为 `visualization.abstract`）、编辑器中显示（只影响网页，不保存）；行内标出场景运动方式与未保存的修改，双击聚焦，焦点在列表里时 H、F、Esc、Ctrl+S/Z 仍转给编辑器。把一行拖到另一行上即绑定，保持世界位置：传感器安装到对方根连杆（随之运动），抽象体跟随对方根部（仅读数，原有场景运动移除），其他实体只与之共同摆放（布局父子，不建物理约束；物理固定仍在“父物体”栏明确选择）；拖到空白处解除；拖到机器人展开后的连杆行上则装在该连杆上（两端都是物理资产时物理固定，传感器安装）。编辑器界面（`tools/editor_ui/app.js`、`app.css`）由资产库服务提供，编辑器在 viser 页面里用一个隐藏的同源小页面把它注入主文档，隐藏 viser 自带面板并把 3D 画布嵌在左右两栏之间；数据经 `/api/outliner/`（场景树）与 `/api/ui/`（顶栏、选中对象的属性，`Editor.ui_get`/`ui_post`）读写，调用与 3D 视图相同的编辑方法和撤回记录。场景树顶部的 Planner / Executor 开关切换 `Editor.lens`（scene / observe）：在树中标出该组件读取的量的个数，3D 中给被读取的实体（Executor 还包括受控机器人）着该组件的颜色（Planner 蓝、Executor 紫，按 60% 混在模型本色上，选中色仍优先）；逐项选择观察在右栏按选中物体进行，场景外的输入（simulation、组件输出）在不选物体时的场景页勾选。机器人行可展开为可动关节（固定连杆并入父级，串联链不缩进，只在分叉处缩进）。3D 视图的右键先合成一次左键点选再弹出与场景树相同的菜单。视图操作模式 `Editor.gizmo_mode` 为 move / rotate / scale；viser 的变换控件没有缩放手柄，缩放模式只显示轴向箭头，沿轴拖动按 exp(位移/0.25) 改写资产的 `scale` 参数（实时以网格缩放预览，导出的网格仍按保存的尺寸）。右栏按分区显示变换（含缩放、父物体）、观察、控制（写 `control.modes`）与关节初始值、复位、运动、形状、高级（原始参数），分区顺序与折叠存在浏览器。选父物体也可在视图中点：`ui/pick_parent` 让下一次点击决定父物体，`Editor.hit_body` 从点中的网格找到所属连杆。viser 自带面板只在“经典面板”打开时构建（`ui/classic`），否则相机分区不会在每次编辑时渲染。相机画面由浏览器在同一画布上渲染（viser `get_render`），因此只在编辑停下 `CAMERA_SETTLE` 秒后、不在拖动中渲染一次，渲染时暂时隐藏抽象体、传感器和不可见实体；JPEG 经 `/api/ui/camera.jpg` 给右栏。用户导入经 `/api/import/file`（逐个文件的原始字节，保留文件夹内相对路径，zip 解开）与 `/api/import/finish`，`shared/asset_import.py` 转换后登记资产架并加入场景。父物体的变更都经确认对话框：`ui/attach?target=` 给出可选关系、连杆与连杆组和默认值，`ui/attach` 的 `place` 为 keep / origin / align（保持世界位姿 / 移到父连杆原点保持朝向 / 与父连杆坐标系重合）；视图中点选父物体只记下 `parent_choice`，由对话框确认。选中传感器时 `_draw_sensor` 画出它测量的东西（接触力传感器覆盖的连杆与力的方向、高度扫描器按 `pattern`/`ray_alignment` 的射线及其落点、坐标测量器到目标连杆的连线）；`ui/add_sensor` 在实体（及右键点到的连杆）上直接加传感器。传感器参数改动立即并入工作副本，观察目录随之更新（相机的输出图像）；只有影响画面的参数（尺寸、焦距、裁剪面）才重新渲染相机画面。原生网格导出读取 GeomSubset 上绑定的材质（一个 CAD 网格的塑料、金属部分），作为同一部件的多段表面。编辑器的预览网格（`shared/assembly/mesh_preview.py`）：带关节的资产（机器人、柜子）在 `ARTICULATED_TRIANGLES`（100 万）三角形以内按原网格显示，不减面；其余资产按预算减面，其中烘焙贴图（每几个三角形一块贴图岛）的表面焊接后按贴图的几种主色减面，避免沿贴图岛开洞。只改关节初始值时，编辑器继续显示上一次导出的网格并用导出的运动学重新摆位（`geometry_keys(content=True)` 不含初始状态）。尚未迁移的设置在项目菜单“经典面板”中。资产库服务没起来时退回原来的按钮列表。
- 多选（同 Blender 列表）：Ctrl/⌘ 点击加入或去掉，Shift 点击选中两行之间的全部；选中的都高亮，F 聚焦全部，H 一起隐藏/显示，拖动坐标轴一起移动，Delete 一起删除（一次确认、一次撤销）。在场景树或控制面板点一个关节，选中其机器人并点亮该关节带动的连杆。
- 编辑器网格加载：导入的 USD 物体在原生网格出来前画成与实物同尺寸的半透明方块（`asset_formats.usd_bounds` 读 USD 包围盒，几十毫秒，不启动仿真），方块里有一层从底往上填的蓝色进度：导出进程按阶段汇报（建好场景 45%、该物体网格提取完 60%、减面开始 75%、完成 100%），两次汇报之间缓慢上涨，启动导出、构建场景这一步没有计数，只涨到 40%。已缓存的网格直接显示，没有进度。网格缓存键包含后端代码哈希，改了后端代码会让网格失效；此时先显示同一资产上一次的网格（`visuals.previous_visuals`，按不含代码的内容键在整个工作区查找），后台重新导出。
- 网页编辑器“添加”页是资产架（`tools/asset_shelf.html`，嵌入 viser 的 iframe）：工程资产、地形地块、传感器、抽象体与 SimReady 在线/本地资产同一套卡片，打开时停在第一个类别（不默认“全部”），条目不足 5 个的 SimReady 小类并入“其他”，本类无匹配时提示到“全部”中查看。卡片可直接拖进 3D 视图，落点为视线与场景表面的交点；也可选中后点“放置”再在视图里点位置（Esc 取消）；已有物体用变换控件拖动。传感器可选安装到实体及其连杆（跟随运动）。缩略图与“完整库 ↗”页面由资产库服务提供，端口为编辑器端口 + 1（`--library-port` 可改），远程使用时两个端口都要转发。尚未包装的 SimReady 资产添加时，编辑器在后台取本地镜像（没有则从 Hugging Face 下载）并按 `run.py asset simready` 包装，进度显示在资产架底部。添加只改编辑器工作副本，显式保存才写回 scene.yaml。
- 资产工厂按后端注册；生成观察类型、目录与 `Project.register()` 用 `project_registry(package, backend=None)` 的全后端静态视图（同名时默认后端优先），只为某一后端准备场景时该后端缺少的资产仍报错。
- isaaclab3 `franka_panda` 的 `tracking`：`native`（默认，IL3 FRANKA_PANDA_LEGACY_CFG）或 `stiff`（FRANKA_PANDA_LEGACY_HIGH_PD_CFG：臂刚度 400、阻尼 80、去重力，供笛卡尔/差分 IK 动作）。参数只描述跟踪方式，不暴露刚度或引擎字段；isaac/mjlab 的 franka_panda 不接受该参数。
- 去重力按引擎自动对应：资产预设用 PhysX 的单刚体 `disable_gravity` 关重力，而 Newton 只在物理场景上读该标志。后端生成场景时（`scene.compensate_gravity`）给这些刚体同时写 MuJoCo `gravcomp=1`，与 IL3 自带 Franka diffik 预设的做法相同；OvPhysX 忽略 MuJoCo 片段，资产已显式写 gravcomp 时不覆盖。项目和场景不需要写任何引擎专用的重力字段。
- Newton（MJWarp）每环境接触/约束容量按构建好的 Newton 模型估算（`contacts.py`）：取模型的潜在接触数，不低于预设 300 接触/1000 约束行，全部环境合计上限 2^19 接触、2^22 行。SimReady 柜子这类凸分解较多的资产无需手写容量；scene.yaml `simulation.newton.solver` 显式写了 `nconmax`/`njmax` 时以其为准。
- 对齐比较直接读 USD（pxr，静止位姿）与编译后的 MJCF，不经过转换代码，容差记录在 ASSET.json；不一致时命令返回非零并逐项列出，不静默修正。
- UsdToUrdf 在关节 body1 位姿非单位时把连杆坐标系移到关节坐标系，却未变换质心与惯量主轴（Piper X `gripper_link2` 质心 y 反号、z 差 1.3 mm）。因此 USD 源转换后按 USD MassAPI 在 MJCF 本体坐标系中重新写入质量、质心与惯量，ASSET.json 的 `inertial_from_usd_mass_api` 列出受影响的本体。
- UsdToUrdf 还可能把连杆坐标系整体转动（Piper X `gripper_link2` 绕自身轴转 π）。相机、TCP 等挂载写在 USD 本体坐标系中，所以对齐比较也检查本体坐标系朝向；USD 源转换时把这类本体改回 USD 朝向，几何、关节、子本体保持世界位姿不变，ASSET.json 的 `body_frames_from_usd` 列出受影响的本体。
- mjlab 工厂 `piper_x`、`usd_robot`、`usd_object`、`urdf_robot` 经 `mjcf_for` 取 MJCF：资产目录有 ASSET.json 时用记录的转换（源文件变化即报错，并重新比较对齐），否则自动转换并缓存到 `.cache/asset_mjcf/`（键含源文件与转换代码哈希）。`usd_robot` 的增益取自 USD DriveAPI（角度单位按每弧度换算）；`usd_object` 仅写了质量时按碰撞几何求惯量，与 PhysX 一致。
- PhysX 关节链中子连杆的位姿由父连杆经关节坐标系推出，不取 USD 中写的位姿。固定关节两端坐标系有微小不一致时（ANYmal-C 的 LH/RH_FOOT 约 1 mrad），UsdToUrdf 会以 1e-3 容差拒绝转换。`physx_stage` 只在会话层把这类子连杆放到 PhysX 实际模拟的位姿（阈值 0.01 rad / 1 mm，超出仍报错；源文件不改），转换与对齐比较都读它，ASSET.json 的 `fixed_joint_children_posed_by_parent` 列出移动的本体与偏差。
- UsdToUrdf 会丢弃 USD 胶囊碰撞体（M20 腿部因此缺碰撞，站高偏低），转换后按 USD 补回（`capsule_colliders_from_usd`）。对齐比较还逐本体比较各类碰撞体数量。
- 质量按 PhysX 实际语义比较：UsdPhysics 中 mass ≤ 0 或未写表示由碰撞体按密度积分；既无正质量又无碰撞体的刚体，PhysX 给默认 1 kg。旧检查直接比较写入的 0，漏掉了 Piper X 导出 URDF 留下的空 `world` 连杆（浮动安装时臂多 1 kg，MJCF 为 0）。现在这类本体按 1 kg 计入，只在 USD 中存在时列为不对齐；`assets/piper_x/piper_x.usda` 给 `world` 写了 0.001 kg 与极小惯量，保留该本体（连接焊接在它上面）。按碰撞体推算质量的本体（Isaac 5.1 Panda 全部连杆）列在 `masses_from_colliders`，两引擎各按自己的碰撞形状积分，不逐项比较：Panda 实测 PhysX 18.78 kg、MuJoCo 19.73 kg。
- 未比较：碰撞体尺寸与位姿（MuJoCo 网格取凸包，PhysX 按 USD 碰撞近似）、由碰撞体推算的质量、接触与摩擦、执行器。执行器由后端工厂分别声明，等价性另需动态验收。
- Piper X：`assets/piper_x/mjcf/` 由 `assets/piper_x/piper_x.usda`（Isaac 实际加载的 USD）生成，与直接从厂商 URDF 编译的结果在 50 组随机关节角下连杆位置差 1e-7 m、质量矩阵差 1e-7。厂商仓库自带的 `scripts/urdf_to_mjcf.py` 写死路径且把执行器增益（120/80）写进 MJCF，与 Isaac 配置不一致，未采用。
- URDF/MJCF→USD 都已实测：`run.py asset convert` 转 piper_x 与带 `<mimic>` 的夹爪 URDF、go2 与 piper_x 的 MJCF 都跑过，逐项比对（身体、关节顺序、驱动、焊点、对齐）见 [docs/PACKAGING_PLAN.md](PACKAGING_PLAN.md) 的“3.1 实施”。两个转换器都要 `usd-exchange`，因此 `usd-core` 不在同一环境：`import` 组用 usd-exchange 提供 pxr，`usd` 组另有自己的一份。

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

## 室内/室外静态场景

与 SimReady 物体同样的接入方式：`shared/environments.py` 是场景目录（Isaac 5.1 环境：Rivermark、仓库 4 种、简单房间、办公室、医院、数字孪生小仓库；GRScenes-100 已解压的各导航场景），缺文件时下载、包装到 `assets/<名>/`，ASSET.json 的 `environment` 段记录目录键、`pose`（把近原点地面放到 z=0）、许可和包装选项；目录里的选项变了会自动重新包装。

- 命令行：`run.py asset environment --list` 列出全部（已包装 / 本地 / 在线、是否 smoke 验证过）；`run.py asset environment rivermark`（或 `isaac/office`、GRScenes 场景 id、id 前 8 位唯一时可简写）下载并包装；`run.py asset environment grscenes/home`（或 `grscenes/commercial`）下载并解压整包（51 / 24 GB，断点续传），之后各场景出现在目录里。
- 网页编辑器“添加”页多了“场景环境”类别：卡片用发布方的预览图，没有时用从本地 USD 画的俯视平面图（按面积采样网格、取每像素最高点按离地高度着色，室内去掉 1.8 m 以上的顶棚；以碰撞体范围取景，天空球等远景不算；逐网格两遍扫描，一个家居场景约 10–15 s、约 0.6 GB 内存。全机只允许一个绘制进程（文件锁 `.cache/environments/drawing.lock`），降低优先级并设 `oom_score_adj=1000`，内存不足时内核先杀它而不是编辑器或 VS Code；缓存在 `.cache/environments/thumbs/`，也可先运行 `python -m shared.environments` 一次画完）。选中后点“添加到场景”，后台下载/包装，再作为 `usd_object`（`scope: global`，位置取目录 `pose`，不取点击位置）加入；保存后写回 scene.yaml。后台准备期间可以继续添加其他资产（同一资产不能同时准备两次，场景之间排队，名称预留）。GRScenes 整包下载只走命令行。

`run.py asset environment <源 USD> <输出目录> --name <名>` 仍可直接包装任意场景 USD，生成 `usd/<名>.usda` 包装层（只引用源文件，不改源）和 ASSET.json（`environment` 段记录碰撞体数量、包围盒及全部修改）。场景项目用 `usd_object`（scope global）引用，pose 的 z 把地面放到 0。源文件可用 `run.py asset vendor <URL> <目录> --threads 16` 并行下载（断点续传，失败自动重试）。

包装层处理 Newton 不接受的内容：vendored `source/` 内缺失的文件补一个空的占位层（customLayerData 标记 `emptyStubForMissingUpstreamFile`，ASSET.json `stubbed_missing` 列出），只丢失缺的那部分；其他无法解析的引用才停用其父 prim（停用父节点会连带兄弟一起丢失，Rivermark 最初就因此丢了全部建筑）；空 Mesh 改为 Xform；非 Gprim 上的碰撞体移到下面没有碰撞体的 Mesh；刚体 API 移除（Newton 不认 `rigidBodyEnabled = false`，仍会建成自由刚体）、关节禁用、articulation root 移除（场景全部静态，所有环境共享）；零缩放 Mesh 停用，镜像 Mesh 改为正缩放；Y-up/厘米统一为 Z-up/米。处理后仍有组合错误或没有碰撞体时报错。

可选项：
- `--substitutes 映射.json`：缺失文件名 → 替代 USD（相对该 JSON），占位层在子 prim 上引用替代物并换算单位/朝向（放在子 prim 上，因为引用方自己的 xformOpOrder 会覆盖根上的变换）。Rivermark 用 `assets/rivermark/substitutes.json` 把桶上缺的 32 个树木/灌木/草换成 `assets/vegetation`（Isaac Vegetation 库）里的近似品种。
- `--footprint-colliders PRIM`：PRIM 的每个子节点若没有碰撞体（如只有外观的建筑），按其离底 0.3/0.9/1.5 m 处的截面在 0.2 m 栅格上取占据（朝向取主墙方向，封闭内部填实），合并成矩形盒，从底到顶拉伸，写成每个子节点一个不可见（guide）三角网格碰撞体。每个子节点只出一个形状，因为 Newton 转 MuJoCo 时内存随静态形状数近似平方增长（1.5 万个独立盒子超过 22 GB）。
- `--simplify-colliders CELL`：把所有碰撞 Mesh 的顶点吸附到 CELL m 网格（去掉退化、重复三角形），按 8 m 方格合并成少数不可见碰撞网格，原 Mesh 只作外观。GRScenes 家居场景有约 4400 个碰撞 Mesh、两三千万三角形（窗帘、布艺、植物极密），Newton 导入 74 s、finalize 再 120 s、转 MuJoCo 超过 20 GB，smoke 的接口检查 180 s 超时；用 3 cm 网格后 3500 万 → 96 万三角形（6 块），场景创建 9 s。场景目录里的 GRScenes 场景默认开启（3 cm）。属于某个场景物体（见下文“导入场景的组成”）的碰撞体留在该物体下，合成一个 `simplified_collider`，这样移除该物体时碰撞也一起没了；只有不属于任何物体的才合并进方格。GRScenes 家居 MWAX5JYK：639 个物体碰撞体、0 个方格，1460 万 → 28.6 万三角形。目录选项变化后，已包装的场景在下次 `prepare` 时自动重新包装。
- `--texture-size PIXELS`、`--simplify-visuals TRIANGLES`：外观减负，供 Newton GL 预览（view/record）和 Newton 相机使用。Newton GL 给每个 Mesh 单独上传一份带 mipmap 的贴图，不共享：GRScenes 商场 MV4AFHQK-ADY8 有 2 万个 Mesh，每个都带贴图（约 600 px），view 在 16 GB 显卡上加载时显存耗尽。贴图压缩是在 `usd/<名>.textures/` 写缩小的副本（原文件不动）。三角形减负按全场总数定一个单网格上限，超过上限的网格做顶点聚类，只保留原三角形的子集，所以 UV、法线、GeomSubset 材质都能沿用。改动都写在二进制子层 `usd/<名>.visuals.usdc`，不碰碰撞体。GRScenes 默认 128 px、1500 万三角形。上述商场：3670 万 → 1490 万三角形，1195 个网格被简化；15461 个图片中 5104 个被缩小；重新包装 87 s。record 显存峰值 9.2 GB（空闲时 1.8 GB），画面带贴图。
- `--expand-instancers`、`--add-colliders`、`--bare REGEX`：用于没有物理信息的导出（Unreal 的 USD 导出不写任何 UsdPhysics，见 `docs/UE_SCENES.md`）。`--expand-instancers` 把 PointInstancer 展开成每个实例一个 Xform，引用源文件里的原型（Newton 的 USD 导入不读 PointInstancer，UE 的模块化建筑件和植被都是这种形式，不展开就既不显示也不碰撞）；原型自身的变换随引用带入，遮罩和 `invisibleIds` 中的实例跳过，原 instancer 停用，同一原型的副本在编辑器里按重复模型显示。`--add-colliders` 给没有碰撞体、可见、任一方向不小于 5 cm 的 Mesh 加三角网格碰撞体，再由 `--simplify-colliders` 降面。两者都跳过路径匹配 `--bare` 的 prim（默认植被、天空、贴花、粒子；原型全部匹配的 instancer 保持原样）。CARLA Town10HD 裁剪（110 m × 110 m）：展开 118 个 instancer 的 2975 个实例，3172 个网格加碰撞（3450 万三角形，`--simplify-colliders 0.05` 后 263 个物体碰撞体），包装约 80 s、0.9 GB 内存。
- Isaac Lab 的 `UsdFileCfg(scale=...)` 会用 spawn 的缩放替换资产顶层 prim 自带的缩放（包括 `xformOp:scale:unitsResolve`）。isaaclab3 的 `usd_object` 把文件自带的缩放乘回 spawn 缩放；修复前，厘米单位的 GRScenes 包装在仿真里放大了 100 倍。

导入场景的组成：一个场景环境（住宅、仓库）仍是一个 `usd_object` 实体，仿真里整体生成一次；编辑器把它当作一棵物体树（`shared/assembly/composition.py`）。默认 prim 下，Scope 和无类型 prim 是文件夹，路径上第一个 Xform 或几何体是物体（一面墙、一个柜子），除非它自己带了超过 8 个几何子节点（Isaac 场景里装着几千个网格的 `Root`），那它也算文件夹；灯是物体，材质跳过。显示时，超过 40 个物体的文件夹按名称词干分组（`SM_floor ×73`）；GRScenes 的哈希名显示为“文件夹名 + 序号”；只装一个物体的文件夹直接显示成该物体。
- 场景列表中这类实体带展开箭头和物体数；展开后可逐层浏览。每行两个开关：从场景移除（写入 `usd_object.exclude`，路径相对默认 prim，可以是文件夹）和编辑器中隐藏（不保存）。点击选中，双击聚焦；在 3D 视图点中物体时，列表自动展开到该物体。
- `exclude` 在 isaaclab3 中由 `asset_formats.pruned_layer` 写一个缓存层（`.cache/usd_pruned/`），把这些 prim 设为 inactive，两个物理引擎都跳过；编辑器的轻量网格按物体分块（每个物体一块，按路径命名），所以能单独选中、隐藏、移除。移除不改源文件，恢复即从列表里删掉该项。
- 已验证：家居场景移除 `Meshes/Furnitures/table` 后，Newton 模型的形状数 639 → 638；把苹果从桌面上方 1 m 放下（场景平移 3 m，避开叠在一起的其他实体），保留桌子时 1.5 s 后苹果停在约 0.90 m（Newton 0.899，OvPhysX 0.908，桌面约 0.87 m）；移除后落到地面（Newton 0.091 m，OvPhysX 约 0.15 m，落地后仍在滚动）。
- 编辑器网格按物体导出、逐个简化；载入时一个场景中不重复的物体最多拼成 150 块显示（`composition.MAX_PARTS`，浏览器里每块是一个场景节点；重复模型另以批量实例显示），超过时从直接子物体最多的文件夹（含同名分组）开始合成一块：家居场景的墙、地面、杂物、书各成一块，家具逐件可选；室外 Rivermark 3421 个物体按房型、信号灯等合成 146 块；Isaac 办公室平铺 3202 个物体，按名称分组后 196 块。合成块中的物体在列表里照常逐个列出，可单独移除（仿真中生效），视图里整块点选、高亮、隐藏。日常场景以几十到一两百个物体为宜；场景目录卡片标出物体数，超过 1000 个（GRScenes 商场 8243 个、Isaac 办公室、大仓库）标为“性能测试”。商场按物体拆成 8000 多块时，浏览器重连要重放 4.5 万条消息，几分钟画不出来。
- 编辑器网格的生成（`shared/assembly/mesh_preview.py`，版本 4）：每件物体是一个模型，保留自身细节，最多 `OBJECT_TRIANGLES`（2 万）个三角形，不按场景总量分配（水壶、锅不会因为场景大而被压坏）；只有超过 `SCENE_CEILING`（400 万，商场一类性能测试场景）时才整体按比例缩减。超过上限的物体用二次误差边折叠（`fast_simplification`）简化，先焊接接缝处重复的顶点；折叠不能合并分离的小零件（一个柜子由上万个螺丝、把手、装饰条组成），所以小于物体尺寸 3% 的零件在预算不够时从小到大舍去，仍超出的再做顶点聚类。家居场景 1465 万源三角形 → 246 万，简化约 28 s（独立进程，8 个子进程），编辑器读入约 2.5 s。没有 `fast_simplification` 的环境退回顶点聚类或 Blender。旧版本的轻量网格照常显示，下次导出或点“刷新原生网格”时升级。
- 重复模型（`composition.model_key`：引用同一模型文件或同一实例原型、且几何只差一个刚体变换）只导出、简化一份，其余记为 `instance: {of, matrix}`；编辑器把出现两次以上的模型画成一个 viser 批量实例网格，每个副本仍是单独的物体（可点选、高亮、隐藏、移除）。Isaac 办公室 3201 个物体中 2489 个是副本，发到浏览器的三角形从 48.7 万降到 11.3 万；Rivermark 3386 个物体中 2868 个是副本。GRScenes 家居每件家具都是独立模型，没有副本。
- 场景包装（`environments.prepare`，命令行或编辑器添加）后接着在独立进程里生成编辑器网格（`prepare_editor_mesh`，用只含该环境的隐藏场景 `.cache/environments/editor/`），存进共享缓存；缓存键只取决于资产和参数，所以任何项目添加该场景都直接命中。办公室约 30 s，Rivermark 约 3 分钟，都只发生一次。
- 编辑器网格的颜色与贴图取自材质（`asset_formats.base_material`）：绑定材质的基础色贴图与基础色（UsdPreviewSurface、OmniPBR、GRScenes 与 Isaac 场景的 MDL），没有材质时用 `displayColor`。导出时带贴图的网格同时导出 UV（UV 接缝处的角点保留各自的顶点），一个物体的各网格作为“表面”（一段面、各自的贴图或颜色，`mesh_preview.join_surfaces`）合在一起；超出三角形上限的物体逐个表面简化，带贴图的表面在二次误差折叠后按回放的顶点映射取最近原顶点的 UV（球面测试中位误差约 0.002）。编辑器把有贴图或多种颜色的块打包成一个 GLB（每个表面一个图元，贴图缩到 256 px 以内），高亮时叠一层半透明网格；重复模型用批量 GLB 实例。家居场景 2558 个表面中 773 个带贴图（348 张贴图），导出 28 s，轻量网格 29.8 MB，编辑器载入约 6 s、发往浏览器约 42 MB；“项目”页可关掉贴图，只按主色显示。贴图和光照只是编辑器示意，仿真画面以 `run.sh view` / `record` 的原始资产为准。
- 物体的判定：路径上第一个 Xform 或几何体，除非它有 8 个以上带几何的子节点，或有多个子节点且下面共有 1000 个以上网格（Rivermark 的 `main` 下是道具、植被、道路等大分组），这时继续往下拆。长的平铺列表按名称词干分组，分组之外的一次性名称归入“其他 ×N”。分析结果按文件版本缓存在 `.cache/composition/`（Rivermark 首次约 15 s）。
- 双击聚焦某个物体时，从斜上方看它，并临时隐藏该场景中整体高于它 0.3 m 以上的物体（天花板、吊灯），选别的对象后恢复。编辑器网格双面绘制（简化后的网格法线不可靠），所以必须隐藏天花板才能看到下方。
- 移动场景中的物体：选中单个物体（列表或视图中点选）时出现坐标轴，拖动即移动，仍是静态几何；写入 `usd_object.moved`（物体路径 → 位姿，场景坐标，米），isaaclab3 由 `asset_formats.scene_edit_layer` 写缓存层（与 `exclude` 同一层），物体自身及其碰撞体（包括网格物体旁边的简化碰撞体）一起移动；可撤销。原本合并显示的物体在选中时单独画出。已验证：家居桌子移动 1.5 m 后，苹果落在新位置的桌面上（Newton 0.89 m，OvPhysX 0.90 m）。
- 取出为独立物体：选中场景中的单个物体，检查器“取出为独立物体”在原处新建 `usd_object`（`usd_path` 同场景，`prim: <物体路径>`），并把它写进场景的 `exclude`，一步撤销。新实体是静态的，填写 `mass` 后受力运动，也可设 `kinematic`/场景运动、选作观察、在 Task 中使用；删除它时自动放回原场景。isaaclab3 用 `asset_formats.object_layer` 写一个只引用该物体子树的缓存层（`.cache/usd_objects/`）：实体原点就是物体自身的位姿（`object_pose`），单位换算和上层变换放在内层 prim（Isaac Lab 会改写顶层 prim 的变换）；网格物体旁边的简化碰撞体一起带上；有质量（动态）时三角网格碰撞体换成凸包（PhysX 的动态刚体不接受三角网格）。只适用于静态场景文件。
- 已验证：取出家居场景的桌子（原点在 (−0.98, −0.52, 0.50)），场景和桌子一起平移 3 m，把苹果从桌面上方 1 m 放下，1.5 s 后：静态桌子上苹果高 0.907 m（Newton）/ 0.913 m（OvPhysX）；桌子设 15 kg 后立在地面上原处（原点高 0.503 / 0.500 m），苹果 0.890 / 0.895 m。

出生点：场景目录包装后在 ASSET.json `environment.spawn` 记录至多 8 个可放机器人的空地（场景坐标、该处地面高度、周围空余半径）。做法（`environments.spawn_points`）：只看碰撞体（含不可见的、方块等基本形状），取朝上的水平面中最低、面积至少占最大者 5% 的高度为地面（屋顶、楼板底面不算；该层找不到空地就试上一层），栅格上标出地面和离地 0.1–0.8 m 的障碍，留出距障碍 ≥ 0.45 m 的格子，先取最空旷处再依次取离已选点最远处。办公室的窗外城市背景也带碰撞体，目录用 `spawn_exclude` 排除。在 Newton 里把 0.3 m 方块从每个出生点上方 0.5 m 放下，办公室、医院、两个仓库、简单房间、GRScenes 家居共 48 个点都落在记录的地面高度、不偏移。编辑器卡片详情和添加完成提示里会列出这些点。

已验证（仅 IL3 + Newton，Go2 smoke `contract_passed`，4 环境）：

| 场景 | 来源 | 碰撞体 | 说明 |
|---|---|---|---|
| rivermark | Isaac 5.1 室外住宅区（街道、联排住宅、广场、树木） | 5738 | 源 3.3 GB（含 MDL 贴图）；桶上缺 39 个文件，32 个植被用替代品、7 个小道具占位；路面约 z=5.75。建筑/道具原本无碰撞体，`--footprint-colliders /World/main/props --footprint-colliders /World/main/sub_traffic_signs` 为 920 个对象生成 14860 个盒（合成 920 个网格）；树木仍无碰撞体。smoke 约 350 s，场景创建 51 s，24 GB 内存上限内通过 |
| warehouse / full_warehouse | Isaac 5.1 | 781 / 3473 | 地面 z=0 |
| simple_room | Isaac 5.1 | 60 | 目录 pose 把地面抬到 z=0 |
| office / hospital | Isaac 5.1 | 3646 / 2036 | 原点附近无地面，用记录的出生点；办公室地面在 z≈0.75 |
| GRScenes 商业 MWHLEPQK、家居 MV7J6NIK-ADA8 / -ADI8、MWAX5JYK-AAQ8、MWBGLKQK-AAA8 | GRScenes-100 navigation 变体 | 简化后 4–8 块 | Go2 放在第一个记录的出生点；各 smoke 约 80 s，场景创建 4–9 s |

限制：
- 场景是全局共享的，多环境处于同一空间（`env_spacing` 只拉开机器人）；出生点只是建议，项目里仍要自己写进机器人 pose 或复位区域。其余 GRScenes 场景和叉车/多货架/数字孪生仓库未做机器人 smoke。
- mjlab / Isaac Lab 2 未验证；OvPhysX 未单独测。
- Newton 导入全局场景时连同所有可视 Mesh 一起载入（Isaac Lab 的 `add_usd` 未关 visual），可视 Mesh 多的场景加载慢、占内存。
- Newton GL 预览在 Rivermark 上显存不足，录像用 `renderer: ovrtx`（较慢）；MDL 材质随 `asset vendor` 下载后 OVRTX 能显示贴图。
- 大场景的仿真/渲染进程可占 20–30 GB 内存，本机曾因此触发全局 OOM 连带关掉 VS Code；用 `systemd-run --user --scope -p MemoryMax=24G ...` 限制单个任务。
- 许可：Isaac 资产仅限 Omniverse 本地使用；GRScenes 为 CC BY-NC-SA（非商用）。`assets/` 不入库。

## 程序化地形

两条路线，名称与参数相同（Isaac Lab 子地形配置的同名字段，见 `shared/backends/isaaclab3/assets.py` 的 `TERRAIN_VARIANTS`，未知名称直接报错并列出可选项；列表自动转为元组）。

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
- 生成由 Isaac Lab 3 的子地形生成器完成（纯 Python/numpy，在 `.venv-il3` 里直接运行，不启动 Kit 或仿真，约数秒；`RLINF_PYTHON` 可换解释器），写出 `usd/<名>.usda`、`mjcf/<名>.xml` 和 ASSET.json；`usd_object` 与 `mjcf_for` 按普通资产的方式找到记录的 MJCF，不需要地形专用工厂。同样的变体、尺寸、难度与种子在同一 Isaac Lab 版本下逐字节相同（对照 `assets/terrains/` 已生成的地块）。
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
- 网页编辑器“项目”页的“仿真（train.yaml）”栏切换物理引擎与渲染器：可选项来自各后端 `project.SIMULATION_CHOICES`（isaaclab3 为 newton/ovphysx 与 newton/ovrtx，isaac、mjlab 为空），与场景修改一样在保存时写入 `train.yaml` 的 `env.train.init_params.physics/renderer`，只改这一两行，注释与格式不动；`view`、评估都按它运行。性能参考：`franka_apple_crate` 加入 SimReady 铝梯（13 个部件、均为 SDF 碰撞）后，OvPhysX 每物理子步从 2.4 ms 升到 9.1 ms，单环境预览每步约 50 ms（约 20 fps，预览上限为 1/step_dt = 50 fps）；Newton 为 3.4 ms/子步、每步约 19 ms；Newton GL 画面每帧约 10–12 ms。
- 渲染器由 `env.*.init_params.renderer` 选择（`+env.train.init_params.renderer=ovrtx`），与物理引擎独立组合，评估同样跟随训练的选择：`newton`（默认）预览用 Newton GL、场景相机用 Newton Warp；`ovrtx` 预览用 IL3 的 `NewtonRTXVisualizerCfg`（OVRTX 路径追踪，带软阴影），场景相机换成 IL3 的 `OVRTXRendererCfg`。两者都免 Kit，不需要 isaacsim；`ovrtx` 需在 `.venv-il3` 另装 `ovrtx`（约 2.2 GB，见 requirements.txt）。其他值报错。
- 资产：Go2 用 IL3 `UNITREE_GO2_CFG`（DC 电机，同 Isaac 5.1 配置，但小腿力矩上限与 Isaac Lab 2.x 不同，行为与权重不可互换）加载工作区 vendored 的 `assets/go2` USD；ground_plane、dome_light、box、ball 用 IL3 原生 spawner（地面为 IL3 默认的浅色米制棋盘，不同于 isaac 后端的深色网格，见下文预览）。地面 USD 与贴图 vendor 在 `assets/ground_plane`（Isaac 6.2 资产，`source/SOURCE.json` 记录 URL 与 sha256），运行时不联网；IL3 只对它的远程默认地址按 2 m 一格缩放 UV，后端的 `_spawn_vendored_ground` 对本地副本做同样的缩放，保持 1 m 棋盘格。usd_object 由本后端按 USD 物理结构生成（见上文 SimReady）；piper、piper_x、m20（`isaaclab3/robots/`）、clear_sky、terrain、seattle_table、dex_cube、usd_robot 原是 isaac 后端的工厂，2026-11-26 起由本后端自己维护（连同接触材质、初始状态、抽象体标记），不再从 `shared/backends/isaac/` 导入：这些配置是 Isaac Lab 的 RigidBodyPropertiesCfg / ArticulationCfg 写法，经 IL3 保留的 PhysX schema 别名可用，且不含四元数（根位姿由 scene.py 按 xyzw 写入）；其中 PhysX 专有字段（求解迭代、combine mode）在 OvPhysX 上生效，Newton 忽略，与 IL3 自带预设一致。
- anymal_c 用 IL3 `ANYMAL_C_CFG`（ANYdrive LSTM 执行器网络），franka_panda 用 IL3 `FRANKA_PANDA_LEGACY_CFG`，USD 与执行器网络都换成工作区 vendored 的 Isaac 5.1 副本。IL3 的按引擎 articulation 属性片段没有 `fix_root_link`，固定基座由 spawner 的 `UsdFileCfg.fix_root_link` 设置（piper/piper_x/m20 的配置用的旧别名 `ArticulationRootPropertiesCfg` 仍带该字段，所以无需改动）。临时项目（各机器人站在地面上、随机动作）在 Newton 与 OvPhysX 上 `run.py smoke` 均通过契约检查，仅为接口跑通。
- 执行器路径：IL3 默认 `SimulationCfg.use_newton_actuators=True`，由 Newton 原生执行 DCMotor 等配置，但原生 DC 电机是隐式 PD，与 isaac/mjlab 后端的显式力矩计算不同，且不支持 IL3 自带之外的执行器类。后端在 `environment.build_config` 中固定设为 `False`，所有场景都走 Isaac Lab 的 Python 显式执行器（OvPhysX 读同一开关）。依据：go2_piper_wholebody 的 mjlab 权重在 Newton 原生路径上平均约 64 步即失败，在 Python 路径上稳定（见项目 README）；go2_chase 的 supermimic 电机（`DelayedDCMotorCfg`）本就只能走 Python 路径。
- go2 工厂参数 `actuators`：`native`（默认，IL3 `UNITREE_GO2_CFG`）或 `supermimic`（go2-supermimic 发布权重所用的显式 DC 电机：hip/thigh 与 calf 两组增益、力矩与饱和、转子 armature、0–2 物理步随机指令延迟，以及站姿 hip ±0.1/thigh 0.9/calf −1.8）。数值只写一次（`shared/assembly/assets.py` 的 `GO2_SUPERMIMIC`），IL3 用 `DelayedDCMotorCfg`（`shared/backends/isaaclab3/actuators.py`，DCMotor 加 IL3 `DelayBuffer`，按环境复位时抽延迟），mjlab 用原生 `DelayedActuatorCfg(DcMotorActuatorCfg)`；isaac 后端没有对应实现，选 supermimic 时报错。go2_chase 的 flat.pt 用 supermimic 在 Newton/OvPhysX/mjlab 平地上 100%/100%/98.6% 碰到球（见项目 README）；切换执行器路径前，同一权重在 Newton 原生 DC 电机上 16 环境 20 s 摔倒 23 次。
- 未提供 `urdf_robot`：IL3 免 Kit 环境没有 URDF 导入器。URDF 机器人先用 `run.py asset convert` 转成 USD（这一步不启动 Kit），再用 `usd_robot`；在 IL3 上选 urdf_robot 时 check 报未知工厂。
- Newton 对 USD 组合错误零容忍：Isaac Lab 2 / OvPhysX 只警告的无效引用，Newton 的 `parse_usd` 直接报错退出。`assets/piper_x` 的源 `base.usd` 原有一条指向不存在 `</visuals/world>` 的空引用（`/agx_arm/world/visuals`，对合成结果无贡献），已删除（原文件备份于 /tmp/piperx_backup）；go2、粗糙地形、dex_cube 已检查无组合错误。新资产应在 `run.py asset convert` 环节排除此类引用。
- 碰撞体位置：Isaac Sim 旧 URDF 导入器把 `CollisionAPI`/`MeshCollisionAPI` 施加在装碰撞网格的 Xform（`collisions/<link>/node_STL_BINARY_`、M20 的 `collisions/collision_N`）上。Kit PhysX（isaac 后端）碰撞其下几何，而 UsdPhysics 只在 Gprim 上定义碰撞体，Newton 与 OvPhysX 都跳过它们，于是 piper、piper_x、M20 整机没有碰撞（夹爪穿过方块、末端可穿地）。后端把 USD 资产的 `spawn_from_usd` 换成 `spawners.spawn_from_usd`：在 env_0 生成后、克隆前把这些 API（含 collisionEnabled 与 approximation）移到其下 Gprim，只对途经的 instanceable 祖先取消实例化；资产文件不改，isaac 后端不受影响。go2、方块等本就在 Gprim 上，不受影响。
- 导入 USD 的身体路径：转换器保留自己的嵌套层级（`Geometry/<连杆>/<子连杆>`），只由固定关节连接的叶子连杆留成 Xform 帧；场景里仍写 `<实体>/<身体名>`，后端用 `usd_body_paths` 换成文件里的实际子路径（`scene.body_subpath`）：传感器的挂载身体、`frame_transformer` 的 `target_body`、连接（Newton 与 OvPhysX）的两端都按这张表解析，只在有东西挂上去时才读该 USD；名字不在表里时报错并列出可用的身体名；读不到 USD（文件不存在、没有 default prim）时直接报出实体与文件，不落到“没有这个身体”的提示上。接触传感器的身体名模式（如 `.*_wheel`）放在所匹配身体共同的父级下，匹配到的身体不在同一父级（嵌套链上的各节）时报错，要求逐个按名字挂。
- 关节顺序：IL3 的 `ArticulationCfg.joint_ordering` 原生支持按名重排（读数、动作、雅可比列同步），后端把 ASSET.json 的 `joint_order` 填入，因此读数不再换序；顺序不符时 `robot.joint_layout` 报错。
- IL3 四元数为 xyzw，只在 `frames.py` 边界换成公共 wxyz；读数、抽象体命令项、区域复位与机器人接口均按 wxyz 对外。
- 形状读数（`readers._geometry`）：`bounds`、静态物体的 `position`/`orientation` 在第一次读取时从舞台读一份环境副本（每环境克隆一份的实体取它所在环境原点，全局实体按各环境原点换算），之后是常量。参考坐标系与根部读数相同：刚体和铰接体取资产内没有父关节的刚体（根连杆），静态物体取实体 prim；包围盒只含 default/render 用途的几何，按当时舞台上的关节位置（即默认关节位置）。`joint_limits` 读原生 `joint_pos_limits`。已在 zz_dual_arm_microwave 上核对 Newton 与 OvPhysX 读数一致（微波炉 0.34×0.26×0.19 m、门限位 [-2.094, 0]、桌面顶 0.75 m）。isaac、mjlab 后端未实现这三项，选中时 check 报错，Task 读取时报错。
- 控制链、Python Task、抽象体（静态/随机复位/Task 移动/跟随）沿用 isaac 后端的结构，分别接到 IL3 原生 ActionTerm、Reward/Termination/Event、CommandTerm。IL3 原生 JointAction 类直接作为控制终点。
- IL3 的 manager 会在 cfg 上写 `debug_vis`，所以各类 term 放在属性容器而非 dict；IL3 复位全部环境时传 `slice(None)`，原生环境入口换为索引张量，共享复位只收到张量。
- 复位前最终观察沿用共享 `TerminalObservationMixin`（IL3 观察管理器内部字段与 2.3 相同）；IL3 自带的 `compute_final_obs` 未用，因为执行链观察有状态，需要快照/恢复。
- 关节复位：实体 `reset.joint_positions: {关节名: [min, max]}` 每次复位按环境均匀采样这些关节的位置（其余关节回到 `initial_state`，速度清零），绑成原生 reset 事件项；只接受关节实体（机器人、`usd_articulation`），关节名逐个写出。isaac/mjlab 后端选用时报错。
- 运动学实体（`kinematic: true`，`shared/assembly/kinematic.py`）：刚体或静态物体的根改为 UsdPhysics 运动学刚体（Newton 导入为 MJWarp 的 mocap 式自由关节，OvPhysX 为 kinematic actor），参与碰撞但不受力和重力。复位与其他实体相同（`reset` 的 region/pose_range/yaw_range）；回合内由场景运动（`motion`，见下条）或 Task 移动：`evaluate` 返回 `set: {名: [B,3] 位置或 [B,7] xyz+wxyz}`，与抽象体同一入口（同一步 set 优先），后端每个实体一个原生命令项（`kinematic.KinematicPose`），在复位之后、观察之前写入位姿，并把根速度设为位移/step_dt；刚复位的环境保持采样位姿，没有 `set` 也没有 `motion` 时停住并清零速度。注意：位姿每个决策步写一次，子步之间不插值，接触看到的是这一步的位移速度，所以单步位移要小（`projects/go2_crowd` 用转向角速度上限和最高步速约束）；运动学实体之间不互相碰撞，避让要由 Task 自己算。关节实体不能设为 kinematic（构建场景时报错），isaac/mjlab 后端选用时报错；网页编辑器场景列表行尾的物理开关写回同一字段（带关节的物体改写 `fix_root_link`）。
- 场景运动（`motion`，`shared/assembly/motion.py`）：场景自己的运动不经 Task，写在运动学实体或抽象体上。`wander`：在抽象矩形/长方体（`region`，缺省为该实体的 `reset.region`）的平面内直行，每回合按 `speed`（m/s 或 [min, max]）和随机方向取一次速度，越出边缘后以 `turn`（m/s²，缺省使最高速越出 2 cm）折返，回到区域内恢复原速；`path`：沿 `points`（环境坐标）的折线以 `speed` 移动，`loop: true` 绕圈、`false` 往返，`start: first|random` 决定复位时在路径上的位置（因此不能再有 `reset`）。`face: true` 绕竖直轴朝向运动方向（当前朝向视为面朝 +X）。计算只在共享层写一份（批量 torch，每环境独立状态，按环境复位）；isaaclab3 由同一个运动学命令项（`kinematic.KinematicPose`）和抽象体命令项（`AbstractState`）在复位之后、观察之前调用，path 的复位位姿在命令项复位时写入，速度仍为位移/step_dt。游走区域与复位区域一样保持静态。`custom` 调用项目里的函数（`motion: {type: custom, function: .behaviors:keep_away, ...参数}`，点开头相对项目包）：每步对全部环境调用一次 `function(scene, motion)`，`scene` 是与 task.py 相同的 SceneInfo（可读机器人等任意实体，用于避障、跟随、让行），`motion` 带当前 `pose` (B,7)、上一步 `velocity`、`dt`、`active`（本步未复位的行）、其余键组成的 `parameters` 和按环境复位的 `memory`（TaskMemory）；返回 (B,3) 位置或 (B,7) 位姿，形状、设备、有限值逐步校验并报出实体名与函数名。写成类时以参数实例化一次，调用 `step` 和可选的 `reset(env_ids)`。复位位置照常由 `reset` 决定。已用演示项目验证（Newton 与 OvPhysX，2 环境）：方块以 0.4 m/s 离开机械臂底座，到 1.5 m 停下。isaac/mjlab 后端声明 motion 时报错。已验证（Newton 与 OvPhysX，4 环境）：木箱游走速度与区域边界、运动学方块沿路径往返并转向、抽象点随机起点绕圈，两引擎轨迹一致；单环境复位后路径回到起点、停一步再走，游走换新方向。`projects/franka_apple_crate` 的木箱由此移动（原 Task `slide` 已移除，语义相同，回到区域后速度不再漂移）。编辑器“运动”栏选择方式、填速度、在视图中点击添加路径点，并可用同一份代码在视图中预览 8 秒（需要编辑器环境有 torch）。
- 重力补偿（`gravity_compensation: true`，仅 isaaclab3）：实体的刚体属性加 PhysX `disable_gravity`（旧式
  `RigidBodyPropertiesCfg` 按同名字段转成 UsdPhysics / PhysX 片段），`compensate_gravity` 再为 Newton 加 MuJoCo `gravcomp=1`，
  与 Isaac Lab 3 Franka 微分 IK 预设相同；工厂预设不被修改。isaac、mjlab 后端明确报错。微波炉场景的 Franka、Piper X 以零动作录 4 s 不下垂（之前 Franka 明显垂下）。
- 渲染可见（`visible: false`，仅 isaaclab3）：资产 spawner 的 `visible` 置为 False，USD 设为不可见；RTX 与 Newton 渲染都不画，碰撞照常（Newton 模型里该形状的 VISIBLE 标志关闭、COLLIDE 保留，已核对）。用于看不见的挡板、只参与碰撞的代理体。抽象体的对应开关是 `visualization.abstract`。编辑器把不可见实体画成半透明。

未接入（在 `check` 阶段明确拒绝）：非传感器的 mount/meshes、非关节实体（无 joint_state 能力）的命名坐标系、上列以外的传感器工厂。

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

多材质网格（SimReady）：SimReady 每个零件是一个既带碰撞又可见的网格，材质绑在 GeomSubset 上。Newton 只对纯视觉网格按子集拆材质，可见碰撞体整块用网格级的一种材质，所以 Newton GL/NewtonRTX 预览和 Warp 相机里胶带座、柜子抽屉等会整块变成铬或灰。后端 USD 生成函数（`spawners.material_subset_visuals`，env_0 克隆前、Newton 导入前）只改内存舞台，不改资产文件：带子集材质绑定且可见的碰撞网格改为 `render` purpose（仍是碰撞体，RTX 最终渲染照画，Newton 视口语义不画），旁边加一份不带物理 API 的副本（`<名>_visual`，`proxy` purpose，只有 Newton 画，由 Newton 自己按子集拆分）。所需的实例化祖先改为可编辑。项目与上层配置不感知。已核对：Newton 模型碰撞体数、刚体、关节、质量不变；柜顶落方块在 Newton 与 OvPhysX 上都停在柜顶（z 2.050，柜顶 2.0 + 半边长）；OVRTX 相机画面与改动前逐像素平均差 0.47/255（路径追踪噪声）；Warp 相机胶带座恢复红黑、纸箱标签出现、柜子抽屉不再整块发黑。编辑器几何导出跳过 proxy，仍读原网格。Warp 仍无透明、反射、全局光照，背光面偏暗、玻璃不透明。

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
| go2_piper_wholebody | 完整回合成功（Newton 带随机化训练的 wholebody.pt，5 个条件；旧 mjlab 权重 63% 轨迹翻倒） | 完整回合成功（同一权重 5 个条件；旧 mjlab 权重也成功） |
| piper_pick | 完整回合成功（mjlab 训练的 pick.pt） | 完整回合成功（mjlab 训练的 pick.pt） |

piper_pick 用 mjlab 后端训练的权重（运行 `20261022_124529` 的 `global_step_1500`，项目内 `weights/pick.pt`，由 scene.yaml `policy.layers.policy.parameters.weights` 加载，无需 `--allow-legacy-checkpoint`）做 `run.py evaluate`（16 环境、确定性动作）：Newton 150 回合 success_once 1.0、平均 51 步、抬升 0.23 m；OvPhysX 108 回合 1.0、69 步、0.31 m；训练后端 mjlab 当时评估 153 回合 1.0 / 50 步。修正碰撞体前两引擎均为 0（夹爪无碰撞）。只是单一初始分布下的评估，不代表多条件可靠成功。
go2_piper_wholebody 用 mjlab 训练的 `weights/wholebody.pt`（运行 `20261023_185843` 的 `global_step_6000`）做 `run.py evaluate`（16 环境、确定性动作、1000 步回合）：OvPhysX 16 条轨迹 0 失败、末端误差 5.6 cm（mjlab 5.2 cm）；Newton 19 条、失败 63%、平均 680 步、末端误差 26.6 cm，失败均为倾斜超过 1 rad（机身未触地）。两项修正后才得到这一结果：连接接触过滤扩到 FIXED 关节刚体组（此前 piper_x 的 base_link 与 Go2 机身相互穿插，平均 2.4 步即失败）；改走显式 Python 执行器（Newton 原生隐式 DC 电机上平均约 64 步）。剩余差距来自 Newton 较软的接触与 weld。2026-11-02 改在 Newton 上带域随机化从头训练（运行 `20261102_075914`，`global_step_6000`，现为项目 `weights/wholebody.pt`）后，Newton、OvPhysX、mjlab 的 5 个评估条件下均跑完完整回合，仅 Newton 强推扰 1/16 翻倒（单一种子，每条件 16 条轨迹，数字见项目 README）。
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
- test_urdf_conversion.py：`urdf-usd-converter` 转出的嵌套身体路径与固定关节帧、`joint_order`、无驱动关节、
  自由/固定根的焊点处理，固定关节叶子连杆记为 ASSET.json `frames` 并在 `robot_frames` 里可读，以及与 MuJoCo
  编译出的 MJCF 对齐（没装转换器时跳过）。
- test_mjcf_conversion.py：`mujoco-usd-converter` 的同一组契约，对源 MJCF 做对齐（`usd` 组，没装转换器时跳过）。
- test_isaaclab3_backend.py：isaaclab3 静态布局与 isaac 一致且不导入 SDK、四元数边界、未接入能力在仿真前拒绝（含 contact_materials 校验）、读不到的 USD 报出实体与文件、标记轴向旋转、高度扫描器 meshes 只接受全部全局形状、连接只接受环境内实体及 weld 相对位姿、相机工厂与 isaac 同签名且参数在 SDK 前校验；URDF 导入器式 instanceable Xform 碰撞体移到其下 Mesh 并保留 approximation；渲染器只接受 newton/ovrtx；装有 isaaclab_newton / isaaclab_ov 时检查 `simulation.newton` / `simulation.physx` 覆盖原生预设。

- test_mjlab_runtime.py：ANYmal-C 的 LSTM 执行器每步输出与 Isaac 计算方式逐项一致、只复位指定环境的隐状态；可选 RUN_MJLAB_GPU=1，验证 GPU 并行推进、单环境截断/复位、复位前最终观察和返回 Tensor 独立快照；普通测试验证后端接口缓存隔离。
- test_stock_rlinf.py：实际原生训练及评估的观察合并路径；共享边界处理评估的字典列表，不改上游源码。
- 实机已完成 2 环境 rollout smoke 和原生 RLinf 两次 PPO 迭代，两个检查点间有参数变化。该示例只验证接口与优化链路，不能当作学会站立或爬楼。

后端只记录在快照与回放报告中（`backends`、`backend_changed`），不进入策略接口签名；用 `--snapshot <run> env.train.init_params.backend=mjlab` 可在另一后端回放同一 checkpoint（评估后端随之切换，除非显式另设）。回放时仍在实际后端上逐项核对观察、动作与关节顺序；各后端原生刚体列表顺序不同且策略不读取，跨后端时不比较。这只说明接口对得上，不保证动力学、接触与渲染一致，结果属于 sim2sim 对照。Isaac 与 mjlab 分别保存原生关节/动作缓存，防止同一项目切换后端误用缓存。

实机验收记录：`outputs/mjlab_ppo_verified_20261019/runs/20261019_111313_4452e77f029a/`。
包含 training.log、evaluation.log、weight_update_check.json 与 validation.json。两次 PPO 迭代后 7 个参数张量变化；从匹配快照加载第二次检查点，经原生 RLinf 评估完成 4 个长度为 3 的回合。success=0，不能解释为训练成功的导航或爬楼策略。
