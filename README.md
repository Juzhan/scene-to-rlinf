# RLinf 场景与训练项目工作区

目标：在 `projects/<name>/` 内搭建场景、选择观察、编写模型和任务，通过统一入口预览、训练与评估。仿真由 Isaac Lab 推进，训练由 RLinf 执行；Blender 是可选编辑器。

修改或新建项目前先读 [AGENTS.md](AGENTS.md)。

新增 [双后端公共项目](projects/dual_backend_scene/README.md)：场景、对象观察、planner/executor、控制与 Python Task 共用定义及公共运行逻辑，两后端使用同一 train.yaml。mjlab 已接入原生 GPU 环境与 RLinf PPO，完成短训练参数更新验证，并保留独立物理场景预览，详见 [接入边界与计划](docs/SCENE_BACKENDS.md)。

新增 [RAX 桌面实验](projects/rax_tabletop/README.md)：公共 YAML 场景与 Python Task，Panda 堆方块、预开抽屉放入、抓把手打开抽屉再放入；通过官方 RAX MCP 驱动 CPU MuJoCo。当前为手写特权状态基线验收，不是自主学习、UR5 适配或 GPU 训练。

## 新建项目

```bash
bash run.sh new my_project
bash run.sh check my_project
bash run.sh edit my_project
bash run.sh inspect my_project
bash run.sh smoke my_project env.train.total_num_envs=2 env.train.max_episode_steps=8
bash run.sh train my_project
```

`train --board` 在训练期间用 TensorBoard 显示 RLinf 记录的指标（默认 `http://127.0.0.1:6006`，被占用时顺延，`--board-port` 改起始端口）。
页面同时列出该实验以前的运行，便于对比；训练结束时服务随之关闭，并打印重新打开的命令。曲线只反映奖励与损失，实际行为仍用 `view/record/evaluate` 查看。

维护模板：

| 模板 | 用途 |
| --- | --- |
| `planner_executor`（默认） | 固定规划器提供目标，训练执行器控制关节 |
| `assembly_go2` | Go2 速度跟随 |
| `assembly_go2_navigation` | 训练导航规划器，固定行走执行器 |
| `assembly_lift` | Franka 抓取抬升，机械臂与夹爪分终点控制 |

使用 `--template assembly_lift` 等选择模板。其他项目属于独立实验或旧接口示例，可显式复制，不作为当前设计规范。创建不会启动仿真、下载资产或训练。

## 用户需要维护什么

| 文件 | 职责 |
| --- | --- |
| `scene.yaml` | 实体组合（含抽象体）、初始状态、每个组件的观察选择、控制终点 |
| `policy.py` | planner/executor 模型及动作解码；特征处理、缩放和偏移在这里写 |
| `task.py` | `evaluate(obs: SceneInfo)`，返回 reward/success/failure/metrics |
| `observation.py` | 自动生成，不要手改：对象类型与字段说明，以及 policy.py/task.py 函数的返回类型（`PlannerOutput`、`ExecutorOutput`、`ExecutorTargets`、`TaskResult`）；scene.yaml 定不下的维度写作 None |
| `train.yaml` | RLinf 训练配方 |
| `__init__.py` | 绑定共享项目入口 |

模型使用 `obs.robot.joint_position`、`obs.robot.wrist_camera.rgb` 等对象属性；保留张量原始形状和 dtype。模型只看到选中的信息，Task 可以按需读取完整场景基础信息。数学计算与编码由 Python/Torch 完成，不再编写 YAML Task 公式。

机器人节点（`kind: robot`）另有方法：`obs.piper.frame_pose('tcp')`、`jacobian(frame)`、`differential_ik(frame, position, orientation=None, max_step=...)`、`joint_names()`。`frame` 为本体 body 名或命名坐标系（资产默认的 `tcp` 等，可在实体 `frames` 覆盖或新增；也可直接选为观察 `piper.tcp.position`，无需传感器），运动链由坐标系决定，不需要写关节列表或在 scene.yaml 配置 IK。固定组件、`decode_action`/`initial_action` 和 task.py 可调用；被训练组件在 policy worker 中只拿到张量，调用会明确报错。接口与各后端实现见 [后端说明](docs/SCENE_BACKENDS.md#后端钩子)。

共享相机 `factory: camera`（Isaac）的参数：`width`/`height`（像素）、`data_types`（任选 `rgb`、`distance_to_image_plane`、`distance_to_camera`、`normals`，同一分辨率）、`focal_length`（mm，水平孔径固定 20.955 mm）、`clipping_range`（近/远裁剪面，m）、`depth_clipping`（超出远裁剪面的深度：`max` 取远平面值、`zero` 置 0、`none` 保留 inf）。每种输出是相机下的一个观察：`rgb` 为 uint8 `[B,H,W,3]`，两种深度为 float `[B,H,W,1]`（m），`normals` 为 float `[B,H,W,3]`（环境坐标轴单位向量）；不做归一化。只能选择相机已开启的输出，选中的输出从 `data_types` 去掉时，check 与编辑器保存会明确报错。`application.enable_cameras: false` 时相机不构建也不渲染。mjlab 支持 rgb 与两种深度，与 Isaac 的差异见 [SCENE_BACKENDS](docs/SCENE_BACKENDS.md)。

点、方向、矩形、长方体等空间量写成 `entities` 中 `kind: abstract` 的抽象体，与物体一样添加、选中、删除和选作观察，只是没有模型和物理：

```yaml
- name: arena
  kind: abstract
  shape: rectangle          # point | direction | rectangle | box
  size: [6.0, 6.0]          # rectangle 2 个值，box 3 个值
  pose: {position: [0.0, 0.0, 0.4]}
- name: destination
  kind: abstract
  shape: point
  reset: {region: world/arena, yaw_range: [-3.14, 3.14], interval: 8.0}
```

观察字段为 `<名>.position`、`<名>.orientation`（wxyz，环境坐标），矩形/长方体另有 `size`，方向另有 `direction`（本地 +X 的单位向量）。没有 goal 或“随机区域”类型，含义由 `task.py` 和模型代码决定。任何实体的 `reset.region` 可引用静态矩形/长方体抽象体（`world/<名>`）；抽象体的 `reset.interval` 表示按秒重新采样。`mount: {entity: world/<实体>}` 只让抽象体读数跟随该实体根部，不是物理约束，跟随的抽象体不能再随机复位。仿真中显示用 `visualization: {abstract: [名称]}`。mjlab 目前只支持静态和跟随读数。

项目默认保留 planner/executor 两层，`control.train` 选择训练哪层。执行器的 `decode_action(action, obs)` 返回命名 Tensor 字典，对应 `control.output` 的实体、关节与位置/速度/力矩控制模式。训练动作和最终关节目标维度可以不同。框架不会自动替用户编写解码算法。

这些函数的返回类型生成在 `observation.py`，在 `policy.py`/`task.py` 中用作返回注解，键或形状写错时编辑器直接提示：

| 函数 | 返回 |
| --- | --- |
| 固定组件 `forward(obs)` | `PlannerOutput` / `ExecutorOutput`，不保留隐藏状态 |
| 训练组件 `forward(obs)` | `(动作均值, 价值 [B,1])`，另有参数 `actor_logstd`（当前 PPO 接入） |
| `decode_action(action, obs)` | `ExecutorTargets`：键与 `control.output` 一致，实际单位、与 action 同设备的批量浮点 Tensor |
| `initial_action(obs)`（可选 `executor.initial`） | `ExecutorOutput`：启动和每次复位时的解码前动作；未设置为 0 |
| `evaluate(obs: SceneInfo)` | `TaskResult`：reward/success/failure/metrics，按环境输出，缺省 0/False；reward 乘 step_dt 计入回报，success/failure 结束回合，metrics 在回合结束时记入日志 `env/task/<名称>` |

`control.output` 的 `joint_names` 逐个列出关节名，不用正则，列出顺序即该终点的动作维度顺序；`executor.dimension: auto` 只适用于训练动作数等于控制关节总数，类型中直接写出该总数；`executor.output`/`previous_output` 是解码前动作。同一关节被多个终点控制时在环境初始化时报错。Task 跨步统计用 `shared.assembly.task_memory.TaskMemory`（按环境自动复位）：`hold` 计持续时间，`mean` 求平均，`rate` 求逐步变化率（复位后第一步为 inf）。

## 网页、Blender 与终端

`run.sh edit` 默认打开网页编辑器，加 `--blender` 改用 Blender。三者编辑同一份 `scene.yaml`，都经过共享 `SceneDocument` 写回。新建环境默认包含地面、灯光、固定底座的 Franka 机械臂和一个方块（`shared/assembly/starter.py`），可删改。编辑器可以摆放实体、组合安装、调整初始关节状态，添加和编辑抽象体，切换被训练的模块，在各模块页面设置决策周期并勾选观察（模块只能选择前序模块的接口），设置控制终点。planner 输出维度不在网页编辑器中修改，直接写 `scene.yaml`。没有 Task 面板，任务编辑 `task.py`。

网页编辑器（viser，无需 Blender）：

```bash
python3.11 -m venv .venv-editor
.venv-editor/bin/pip install viser==1.1.1 ruamel.yaml==0.18.17 omegaconf==2.3.1 numpy
.venv/bin/python run.py edit <project>                  # 默认编辑器，http://127.0.0.1:8080
.venv/bin/python run.py edit --scene path/scene.yaml --port 8093 --no-native-preview
.venv/bin/python run.py edit <project> --blender        # 改用 Blender（可执行文件取 $BLENDER）
```

界面以左侧“场景”树为中心：地面、机器人、相机、灯光、抽象体等都在树中，也可在 3D 视图中点击选中（鼠标悬停不高亮；拖动坐标轴时不会选中背后的物体）。右侧检查器显示选中对象的位姿、父物体/安装、初始状态和复位范围，这些操作在面板内展开并实时预览，不弹窗遮挡视图，“取消”恢复原状。“父物体”区选择父物体、方式和位置时只预览，点“应用”后才写入（保持当前位置，面板保持展开），坐标轴仍可直接拖动调整相对位置；“方式”为“物理安装（随连杆运动）”（写入连接，两端都是物理资产时默认）或“共同摆放（仅布局）”（写入 `authoring.parents`，不创建物理固定）；“位置”可改为对齐到父物体或自定义偏移，预览后点“应用位置”。物理安装会移除子实体自身的随机复位：只有连接链的根可随机复位，子实体在复位时按连接放到父物体上。删除实体前弹窗确认并列出影响：场景中直接定义的子物体上移一级、保持世界位置（观察名随之变化，需同步 policy.py / task.py）；安装或共同摆放在它上面的实体解除依附并保持原位；挂在它上面的非相机传感器（接触、射线、坐标系等，离开连杆无法工作）随之删除；模板实例内部部件随实例删除。可用 Ctrl-Z 撤回。物体重叠时，在同一处再次点击会在该处的物体之间轮换选中；3D 视图右上角显示世界坐标轴方向。实体的“参数”区按工厂函数签名的类型注解生成字段（整数、数值、开关、单选、多选、2/3 维向量；可空数值留空为 null），未注解的结构值逐项用 JSON 编辑，工厂只接受 `**kwargs` 时仍为整段 JSON；每个字段一步撤回，保存时运行工厂经 `shared.assembly.assets.checked` 附带的无 SDK 校验（与 `run.py check` 相同）。新的参数化资产只需写带注解的工厂并附上校验函数即可获得这些字段。选中相机时检查器显示一张相机视角示意图：位姿取自 `scene.yaml` 的复位姿态，分辨率与焦距取当前参数（含未保存修改，孔径 20.955 与 Isaac 相机一致），由编辑器视图渲染，不是 Isaac 传感器图像；装在机械臂上的相机在仿真开始后会随手臂姿态移动。观察列表按 `obs` 对象分组（`robot`、`height_scanner`、`simulation`、组件接口 `planner` 等），组内显示策略代码读取的属性名（`obs.robot.joint_position` 显示为 `robot` 组内的 `joint_position`），第二行为说明、张量形状与单位；按组件（planner/executor）选择并写回 `control.components.<组件>.observations`；编辑器只编辑带 `control.components` 的当前模板项目，旧 preset 场景的观察与控制不在编辑器中维护。保存键旁的两个图标按钮为撤回 / 重做。快捷键：Ctrl-S 保存、Ctrl-Z / Ctrl-Shift-Z 撤回 / 重做（最多 100 步，跨保存也可撤回；文件被外部修改后历史清空）、Esc 取消预览/选择、F 聚焦、H 隐藏/显示。

“添加”页的陈列栏以卡片列出可用资产。示意图缓存在 `.cache/asset_thumbnails/`：打开场景时从已导出的原生网格自动生成，缺少的可点“生成缺失示意图”，在后台用 Isaac 导出一个临时场景（只含不需要必填参数的共享资产）；无图时显示类别图标。“导入 URDF / USD”把自己的文件登记到 `assets/library.json`（文件留在原处，工作区内的路径按相对路径保存），之后出现在所有项目的陈列栏里。URDF 由共享工厂 `urdf_robot` 在首次加载时转换为 USD（缓存于 `.cache/urdf_usd/`），USD 机器人用 `usd_robot`，USD 物体用 `usd_object`（不填 `mass` 时作为静态资产；填写时 USD 需自带刚体属性）。导入的关节统一使用 `stiffness`/`damping` 隐式 PD 驱动，默认 100/10，可在实体参数中修改；关节、碰撞和质量以文件内容为准，导入后请用 `run.py inspect` 核对。

不要把 viser 装进 Isaac 的 `.venv`（websockets 版本冲突）。网页先显示示意代理和上次导出的网格，需要时在后台用 Isaac 环境导出原生网格；`--no-native-preview` 关闭启动时的自动导出。已导出的网格直接从缓存读取，不启动 Isaac；只有新资产或改了参数时才在后台启动 Isaac 导出，导出后保留 2 分钟供接连导入复用（约 5 GB 内存），空闲或编辑器退出时自动结束；网格按实体缓存，同一资产与参数在各项目和陈列栏间共用，减面由多个后台 Blender 并行完成。点击右上保存按钮（或 Ctrl-S）才写文件并更新 `observation.py`；添加、删除、组合、安装等结构操作会先保存当前修改再应用。浏览器内不运行物理仿真，布局结果仍以 `run.py view/inspect` 为准。

Blender 中，界面保存项目配置时更新 `observation.py`；普通 Ctrl-S 只保存 `.blend`。修改模型输入或新增控制终点后，也要同步项目网络和解码代码。详见 [编辑器说明](tools/blender_addon/README.md)。

## 命令与验证边界

- `check`：静态配置、注册和基础引用检查，不启动仿真，不证明能训练。
- `catalog`：查看场景可用信息；旧实验可能显示历史 term。
- `inspect`：短暂启动仿真，取得实际关节、连杆、观察形状和控制顺序。
- `check` 和编辑器保存时刷新 `observation.py`；实际关节顺序、默认位置看 `inspect` 的输出。
- `view` / `record`：实际场景预览或策略回放。
- `smoke`：短程接口、终止与复位检查，不代表策略学会任务。
- `train` / `evaluate`：调用 RLinf 训练或评估。SFT、MuJoCo/VLA 专用项目以其 README 的能力边界为准。
- `asset convert <源文件> <资产目录>`：由 USD、URDF 或 MJCF 生成另一后端所需格式（USD↔MJCF，经 URDF 中转，全部使用官方转换器），在资产目录写入 `mjcf/` 或 `usd/` 与 `ASSET.json`（来源哈希、转换步骤、对齐报告）；`asset check <资产目录>` 重新比较 USD 与 MJCF 的连杆位姿、质量、质心、惯量、关节轴与限位。自动使用 `.venv-mjlab`（需 mujoco 与 pxr）。外观只求接近：碰撞几何与视觉包围盒相差不超过 5 mm 的连杆在 MJCF 中用碰撞网格并取主要材质颜色；相差更大时（如 piper_x 的相机支架与 D435）保留视觉网格，碰撞几何放到不渲染的 geom group 3；超过 MuJoCo 20 万面上限的网格按顶点聚类简化，记录在 ASSET.json。执行器与增益不在转换文件中，由各后端资产工厂声明。由 URDF/MJCF 生成 USD 会启动 Isaac（Kit）。资产架上的 piper、piper_x、m20、franka_panda、seattle_table、anymal_c 均已转换并通过对齐检查，两个后端都可直接使用（ANYmal-C 在 mjlab 使用同一 LSTM 执行器网络），清单与差异见 [docs/SCENE_BACKENDS.md](docs/SCENE_BACKENDS.md)。
- `asset terrain <地形名|all> <资产目录> [--size --difficulty --seed --param JSON]`：不给 `--param` 时用 `shared/terrain_assets.py` 的 PRESETS；`all` 把 21 种预设写到 `<资产目录>/<地形名>/`，已存在的跳过。`assets/terrains/` 下的地块自动列在 Web 编辑器资产架的“地面 / 地形”类。用 Isaac Lab 地形生成器生成一块以原点为中心的地块，写出 USD（原始三角形静态碰撞体）、MJCF（box、凸包、hfield 重建，MuJoCo 网格碰撞取凸包）与 `ASSET.json`，并把 MJCF 与三角形的高度差校验后记录；之后在两个后端都用 `usd_object` 当普通静态物体引用。非凸封闭块与不规则曲面会被拒绝。详见 [docs/SCENE_BACKENDS.md](docs/SCENE_BACKENDS.md)。

新训练写独立实验目录。快照保存有效配置和项目源码；观察顺序、训练角色、控制周期、解码函数、终点类型与关节顺序变化都属于接口变化。`--allow-scene-change` 不允许绕过模型接口检查。

完整恢复使用 `runner.resume_dir`；`actor.model.initial_checkpoint` 仅加载模型权重，不等于恢复优化器与训练进度。

## RLinf 与安装

默认使用未经修改的 `third_party/RLinf-upstream`，固定提交为 `77ed948d68f07f55b4bc433b7643a409b2cd1adb`。当前本地由 Git worktree 创建，共享旧仓库的 Git 对象，但源码目录独立且干净。换机可独立 clone 并 checkout 同一提交，或用 `RLINF_ROOT` 指定干净安装。

`RLINF_PYTHON` 默认 `.venv/bin/python`。现有本机仿真组合为 Isaac Sim 5.1 / Isaac Lab 2.3.2.post1；虚拟环境仍依赖本机基础安装，不能直接复制到任意机器。无需 Docker。

旧的 `third_party/RLinf` 保留其他实验的源码改动，默认不使用。新模板已移除依赖补丁的 adaptive-KL 设置，使用 RLinf 原生学习率配置；这改变了训练配方，旧成绩不能视为新配方的验收。详见 [RLinf 接入与补丁清单](docs/RLINF_INTEGRATION.md)。

历史训练结果、旧命令与迁移记录保存在 [历史 README](docs/history/WORKSPACE_README_20261019.md)，它不是当前搭建说明。每个具体实验的权重和数据仍在原处。
