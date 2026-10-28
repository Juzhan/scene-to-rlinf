# RLinf 场景与训练项目工作区

目标：在 `projects/<name>/` 内搭建场景、选择观察、编写模型和任务，通过统一入口预览、训练与评估。
仿真由 Isaac Lab 3（Newton 或 OvPhysX 物理）推进，训练由 RLinf 执行。

- 安装、部署、案例与验证状态：[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- 资产（机器人、物体、地形、地面、天空盒）不在 git 中，用 `python3 tools/asset_bundle.py fetch` 取得资产包；
  安装完成后运行不需要联网，所有资产从本地 `assets/` 加载
- 修改或新建项目前先读 [AGENTS.md](AGENTS.md)；后端钩子与能力差异见 [docs/SCENE_BACKENDS.md](docs/SCENE_BACKENDS.md)

## 项目

| 项目 | 内容 |
| --- | --- |
| [`planner_executor`](projects/planner_executor/README.md) | 默认模板：Go2 走向目标点，固定规划器给目标，训练执行器 |
| [`piper_pick`](projects/piper_pick/README.md) | Piper X 抓起方块并抬升，附已训练权重 |
| [`go2_chase`](projects/go2_chase/README.md) | Go2 追逃跑的虚拟球，使用 go2-supermimic 发布的行走权重 |
| [`go2_piper_wholebody`](projects/go2_piper_wholebody/README.md) | Go2 背 Piper X，跟踪机身速度与末端目标，附已训练权重 |

## 常用命令

```bash
bash run.sh new my_task                       # 从模板 planner_executor 生成项目，train.yaml 写明 backend: isaaclab3
bash run.sh edit my_task                      # 网页编辑器
bash run.sh check my_task                     # 静态检查，刷新 observation.py
bash run.sh view my_task --actions policy     # 实时窗口
bash run.sh record my_task --actions policy   # 离屏录像
bash run.sh train my_task                     # RLinf PPO
bash run.sh evaluate my_task env.eval.total_num_envs=16
```

命令后加 `env.train.init_params.physics=ovphysx` 换用 OvPhysX 物理。`train --board` 在训练期间用 TensorBoard
显示 RLinf 记录的指标（默认 `http://127.0.0.1:6006`，被占用时顺延，`--board-port` 改起始端口）；
曲线只反映奖励与损失，实际行为仍用 `view/record/evaluate` 查看。

## 用户需要维护什么

| 文件 | 职责 |
| --- | --- |
| `scene.yaml` | 实体组合（含抽象体）、初始状态、每个组件的观察选择、控制终点 |
| `policy.py` | planner/executor 模型及动作解码；特征处理、缩放和偏移在这里写 |
| `task.py` | `evaluate(obs: SceneInfo)`，返回 reward/success/failure/metrics，可选 set |
| `observation.py` | 自动生成，不要手改：对象类型与字段说明，以及 policy.py/task.py 函数的返回类型（`PlannerOutput`、`ExecutorOutput`、`ExecutorTargets`、`TaskResult`）；scene.yaml 定不下的维度写作 None |
| `train.yaml` | RLinf 训练配方，含后端与物理引擎选择 |
| `__init__.py` | 绑定共享项目入口 |

`policy.py` 与 `task.py` 开头写有“在哪里写”的说明。模型使用 `obs.robot.joint_position`、`obs.robot.wrist_camera.rgb`
等对象属性；保留张量原始形状和 dtype。模型只看到选中的信息，Task 可以按需读取完整场景基础信息。
数学计算与编码由 Python/Torch 完成，不编写 YAML Task 公式。

机器人节点（`kind: robot`）另有方法：`obs.piper.frame_pose('tcp')`、`jacobian(frame)`、`differential_ik(frame, position, orientation=None, max_step=...)`、`joint_names()`。`frame` 为本体 body 名或命名坐标系（资产默认的 `tcp` 等，可在实体 `frames` 覆盖或新增；也可直接选为观察 `piper.tcp.position`，无需传感器），运动链由坐标系决定，不需要写关节列表或在 scene.yaml 配置 IK。固定组件、`decode_action`/`initial_action` 和 task.py 可调用；被训练组件在 policy worker 中只拿到张量，调用会明确报错。

共享相机 `factory: camera` 的参数：`width`/`height`（像素）、`data_types`（任选 `rgb`、`distance_to_image_plane`、`distance_to_camera`、`normals`，同一分辨率）、`focal_length`（mm，水平孔径固定 20.955 mm）、`clipping_range`（近/远裁剪面，m）、`depth_clipping`（超出远裁剪面的深度：`max` 取远平面值、`zero` 置 0、`none` 保留 inf）。每种输出是相机下的一个观察：`rgb` 为 uint8 `[B,H,W,3]`，两种深度为 float `[B,H,W,1]`（m），`normals` 为 float `[B,H,W,3]`（环境坐标轴单位向量）；不做归一化。只能选择相机已开启的输出，选中的输出从 `data_types` 去掉时，check 与编辑器保存会明确报错。`application.enable_cameras: false` 时相机不构建也不渲染。各后端支持的输出见 [SCENE_BACKENDS](docs/SCENE_BACKENDS.md)。

## 抽象体

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
- name: ball
  kind: abstract
  shape: point
  reset: {region: world/arena}   # 每回合初始位置；回合内由 task.py 的 set 移动
```

观察字段为 `<名>.position`、`<名>.orientation`（wxyz，环境坐标），矩形/长方体另有 `size`，方向另有 `direction`（本地 +X 的单位向量）。没有 goal 或“随机区域”类型，含义由 `task.py` 和模型代码决定。任何实体的 `reset.region` 可引用静态矩形/长方体抽象体（`world/<名>`）；抽象体的 `reset.interval` 表示按秒重新采样。`mount: {entity: world/<实体>}` 只让抽象体读数跟随该实体根部，不是物理约束，跟随的抽象体不能再随机复位。

回合内移动抽象体属于任务设定，不在 scene.yaml 中配置：`task.py` 的 evaluate 返回 `'set': {'ball': 位置}`（`[B,3]`，或 `[B,7]` 环境坐标 xyz+wxyz），由后端原生 command 项在本步复位之后、观察之前生效；刚复位的环境保持采样位姿，所以 Task 在第 t+1 步看到的就是策略在第 t 步观察到的位置。决策步长读 `obs.simulation.step_dt`。可移动的是未跟随、且未被任何 `reset.region` 引用的抽象体，它们另有 `<名>.linear_velocity`（位移 / step_dt，未移动或刚复位时为 0）；区域保持静态，对它们 set 会在运行时报错。物理实体暂不能由 Task 移动。仿真中显示用 `visualization: {abstract: [名称]}`：只在项目预览 `view/record` 中画出（点为球，矩形/长方体为线框，方向为 +X 箭头），不进入模型输入。

## 控制与返回类型

项目默认保留 planner/executor 两层，`control.train` 选择训练哪层。执行器的 `decode_action(action, obs)` 返回命名 Tensor 字典，对应 `control.output` 的实体、关节与位置/速度/力矩控制模式。训练动作和最终关节目标维度可以不同。框架不会自动替用户编写解码算法。

这些函数的返回类型生成在 `observation.py`，在 `policy.py`/`task.py` 中用作返回注解，键或形状写错时编辑器直接提示：

| 函数 | 返回 |
| --- | --- |
| 固定组件 `forward(obs)` | `PlannerOutput` / `ExecutorOutput`，不保留隐藏状态 |
| 训练组件 `forward(obs)` | `(动作均值, 价值 [B,1])`，另有参数 `actor_logstd`（当前 PPO 接入） |
| `decode_action(action, obs)` | `ExecutorTargets`：键与 `control.output` 一致，实际单位、与 action 同设备的批量浮点 Tensor |
| `initial_action(obs)`（可选 `executor.initial`） | `ExecutorOutput`：启动和每次复位时的解码前动作；未设置为 0 |
| `evaluate(obs: SceneInfo)` | `TaskResult`：reward/success/failure/metrics，按环境输出，缺省 0/False；reward 乘 step_dt 计入回报，success/failure 结束回合，metrics 在回合结束时记入日志 `env/task/<名称>`；可选 `set: {抽象体名: [B,3] 位置或 [B,7] 位姿}` 在本步复位之后、观察之前移动抽象体 |

`control.output` 的 `joint_names` 逐个列出关节名，不用正则，列出顺序即该终点的动作维度顺序；`executor.dimension: auto` 只适用于训练动作数等于控制关节总数，类型中直接写出该总数；`executor.output`/`previous_output` 是解码前动作。同一关节被多个终点控制时在环境初始化时报错。Task 跨步统计用 `shared.assembly.task_memory.TaskMemory`（按环境自动复位）：`hold` 计持续时间，`mean` 求平均，`rate` 求逐步变化率（复位后第一步为 inf）。

## 网页编辑器

`run.sh edit` 打开网页编辑器（viser，与仿真同一个环境，安装见 [DEPLOYMENT](docs/DEPLOYMENT.md#步骤)），编辑同一份 `scene.yaml`，经共享 `SceneDocument` 写回，终端修改与编辑器修改等价。

```bash
bash run.sh edit <project>                        # http://127.0.0.1:8080
bash run.sh edit --scene path/scene.yaml --port 8093 --no-native-preview
```

### 界面用法

左侧面板有五页：**场景**（实体树）、**添加**、**观察**、**控制**、**项目**；右侧面板上方是保存、撤回、重做，下方的**检查器**显示当前选中实体的属性（在场景树或 3D 视图中点击选中）。编辑器不自动保存：所有改动（含添加、删除、组合、撤回/重做）先写入项目目录下的隐藏工作副本 `.scene.yaml.<pid>.editing`（保存按钮变色），点保存或 Ctrl-S 才写入 `scene.yaml` 并刷新 `observation.py`；“放弃修改并重新加载”丢弃全部未保存改动。保存前若 `scene.yaml` 已在编辑器外被修改，保存会拒绝覆盖并提示重新加载。工作副本在编辑器退出时删除，不进入实验快照和 git。

| 要做的事 | 入口 | 操作 |
| --- | --- | --- |
| 添加物体 | 添加 → 资产 | 按“类别”筛选，在陈列栏点选卡片，填“实体名称”，需要时改“参数”和“初始位置”，点“添加到场景”。传感器在“安装到 / 安装连杆”里选所在实体和连杆。自己的模型用“导入 URDF / USD”。 |
| 设置父物体 | 场景树选中实体 → 检查器“父物体” | 选“父物体”和“方式”：**物理安装（随连杆运动）** 写入连接，**共同摆放（仅布局）** 只记布局、不产生物理固定。再选父连杆、位置（保持当前 / 对齐到父物体 / 自定义偏移）和朝向对齐，视图里实时预览，点“应用”写入；“解除父物体”保留当前位置。 |
| 设置观察 | 观察 | 顶部“训练”选择被训练的模块（planner 或 executor）。每个模块一页“<模块> 的输入”：填“决策周期”，按 `obs` 分组勾选输入，列表显示属性名（如 `obs.robot.joint_position`）、形状和单位；勾选顺序即模型输入顺序。 |
| 设置控制 | 控制 | 每个控制目标一栏（名称 · 类型 · 机器人 · 维度），展开后勾选关节，勾选顺序即动作维度顺序，可改位置/速度/力矩类型，点 ✕ 删除。“＋ 新增控制目标”选机器人、类型、名称后添加。增删目标后同步 `policy.py` 的 `decode_action` 返回值。 |

此外，检查器里还能改位姿、参数、初始关节状态和随机复位范围；**项目**页用于新建或打开环境、放弃修改、刷新原生网格。奖励和成功条件没有界面，直接写 `task.py`。

### 细节

新建环境与 `run.sh new` 相同，复制一个维护模板，因此一开始就有控制终点、`policy.py`、`task.py` 和 `train.yaml`；首次保存前放在临时目录，保存时整体移到 `projects/<名称>`，不覆盖已有项目（`shared/assembly/starter.py`）。编辑器可以摆放实体、组合安装、调整初始关节状态，添加和编辑抽象体，切换被训练的模块，在各模块页面设置决策周期并勾选观察（模块只能选择前序模块的接口），设置控制终点。新建终点默认只选该实体尚未被控制、且与第一个这类关节同类型（转动/移动、有无限位）的关节，例如 Franka 先建的终点是 7 个臂关节，夹爪另建；勾选观察时列表里的位置按观察目录固定，取消再勾选回到原位（已选项的相对顺序即模型输入顺序，不因此改变）。数值框精度 0.001（角度 0.1°）。planner 输出维度不在网页编辑器中修改，直接写 `scene.yaml`。没有 Task 面板，任务编辑 `task.py`。

界面以左侧“场景”树为中心：地面、机器人、相机、灯光、抽象体等都在树中，也可在 3D 视图中点击选中（拖动坐标轴时不会选中背后的物体）。右侧检查器显示选中对象的位姿、父物体/安装、初始状态和复位范围，这些操作在面板内展开并实时预览，“取消”恢复原状。“父物体”区选择父物体、方式和位置时只预览，点“应用”后才写入；“方式”为“物理安装（随连杆运动）”（写入连接，两端都是物理资产时默认）或“共同摆放（仅布局）”（写入 `authoring.parents`，不创建物理固定）。物理安装会移除子实体自身的随机复位：只有连接链的根可随机复位，子实体在复位时按连接放到父物体上。删除实体前弹窗确认并列出影响：直接定义的子物体上移一级、保持世界位置（观察名随之变化，需同步 policy.py / task.py）；安装或共同摆放在它上面的实体解除依附并保持原位；挂在它上面的非相机传感器随之删除；被删实体的控制终点和观察项随之移除，弹窗逐项列出（policy.py 的 decode_action 和 obs 引用需同步）。被删除的实体从高度扫描器的射线目标中去掉；删掉最后一块地面会让扫描器没有目标，弹窗事先提示（Isaac Lab 3 要求恰好列出全部有形状的全局实体，例如 `scope: global` 的地面或地形）。物体重叠时，在同一处再次点击会在该处的物体之间轮换选中。每次结构改动和保存后按项目 train.yaml 的后端对当前编辑内容做一次与 `check` 相同的静态场景检查（如传感器未挂载、IL3 射线目标不符），不通过时仍可保存，但弹出提示并在状态栏保留原因，直到修正。实体的“参数”区按工厂函数签名的类型注解生成字段，保存时运行工厂附带的无 SDK 校验（与 `check` 相同）。选中相机时检查器显示相机视角示意图（编辑器视图渲染，不是仿真传感器图像）。观察列表按 `obs` 对象分组，显示策略代码读取的属性名、说明、张量形状与单位，按组件写回 `control.components.<组件>.observations`。快捷键：Ctrl-S 保存、Ctrl-Z / Ctrl-Shift-Z 撤回 / 重做（最多 100 步）、Esc 取消预览/选择、F 聚焦、H 隐藏/显示。

“添加”页的陈列栏以卡片列出可用资产，`assets/terrains/` 下的地块列在“地面 / 地形”类。添加传感器时在“安装到 / 安装连杆”中选择所在实体和连杆（默认选中的实体，否则第一个机器人的根部），新传感器直接以安装状态写入；接触、高度扫描、坐标变换传感器只能安装在连杆上，相机还可选“不安装，固定在世界中”（Isaac Lab 3 的相机仍需安装）。连杆名来自原生网格导出，导出完成前添加会提示稍候。坐标变换的 `target_body` 默认取安装实体根部，在检查器中从该实体的连杆里选择。新高度扫描器以全部全局地面为射线目标；已瞄准全部地面的扫描器在新增全局地面时自动加入它，删掉地面后再添加的地面也会自动成为目标。示意图缓存在 `.cache/asset_thumbnails/`。“导入 URDF / USD”把自己的文件登记到 `assets/library.json`（文件留在原处），之后出现在所有项目的陈列栏里；URDF 由共享工厂 `urdf_robot` 在首次加载时转换为 USD（缓存于 `.cache/urdf_usd/`），USD 机器人用 `usd_robot`，USD 物体用 `usd_object`。导入后用 `view` 核对关节、碰撞和质量。

网页先显示示意代理和上次导出的网格；新资产或改了参数时在后台用仿真后端的解释器导出原生网格（`RLINF_EXPORT_PYTHON` 可指定），导出后保留 2 分钟供接连导入复用，空闲或编辑器退出时自动结束；`--no-native-preview` 关闭启动时的自动导出。网格按实体缓存，同一资产与参数在各项目和陈列栏间共用。点击保存（或 Ctrl-S）才写 `scene.yaml` 并更新 `observation.py`；添加、删除、组合、安装等结构操作先把面板中的修改并入工作副本再应用，原生网格导出也读取工作副本。浏览器内不运行物理仿真，布局结果以 `view` 为准。

## 命令与验证边界

- `check`：静态配置、注册和基础引用检查，不启动仿真，不证明能训练；刷新 `observation.py`。
- `view` / `record`：实际场景预览或策略回放。未给 `--camera-eye/--camera-target` 时，相机按场景描述自动取景，并跟随 env 0 的第一个机器人根部；给出任一项时按环境原点坐标固定放置。
- `smoke`：短程接口、终止与复位检查，不代表策略学会任务。
- `train` / `evaluate`：调用 RLinf 训练或评估。
- `catalog`：列出场景可读的信息目录。
- `asset convert <源文件> <资产目录>`、`asset check <资产目录>`、`asset terrain <地形名|all> <资产目录>`：维护资产用，生成并比对 USD 与 MJCF、生成地形地块，写 `ASSET.json`；需要 `.venv-mjlab`，详见 [docs/SCENE_BACKENDS.md](docs/SCENE_BACKENDS.md)。

新训练写独立实验目录。快照保存有效配置和项目源码；观察顺序、训练角色、控制周期、解码函数、终点类型与关节顺序变化都属于接口变化。`--allow-scene-change` 不允许绕过模型接口检查。
回放 checkpoint 时只比较运行会导入的项目模块和场景配置内容；Python 按去掉注释与文档字符串的语法树比较，README 等无关文件或只改注释不算差异。不一致时报错列出文件名并给出 `--snapshot <快照路径>`；`policy.py` 的改动属于策略语义，`--allow-scene-change` 也不放行。

完整恢复使用 `runner.resume_dir`；`actor.model.initial_checkpoint` 仅加载模型权重，不等于恢复优化器与训练进度。

## RLinf

默认使用未经修改的 `third_party/RLinf-upstream`，固定提交 `77ed948d68f07f55b4bc433b7643a409b2cd1adb`；也可用 `RLINF_ROOT` 指定干净安装。
模板使用 RLinf 原生 PPO 学习率配置。接入方式与依赖边界见 [RLinf 接入](docs/RLINF_INTEGRATION.md)。

仓库中 Isaac Lab 2（`isaac`）与 mjlab 后端及旧实验的代码仍保留，但不在当前部署范围内；历史记录见 [历史 README](docs/history/WORKSPACE_README_20261019.md)。
