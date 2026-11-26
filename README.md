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
| [`go2_goal`](projects/go2_goal/README.md) | 默认模板：Go2 走向目标点，一层策略 |
| [`planner_executor`](projects/planner_executor/README.md) | 同一任务拆成两层：固定的 planner 定目标，训练 executor 行走 |
| [`piper_pick`](projects/piper_pick/README.md) | Piper X 抓起方块并抬升，附已训练权重 |
| [`go2_chase`](projects/go2_chase/README.md) | Go2 追逃跑的虚拟球，使用 go2-supermimic 发布的行走权重 |
| [`go2_piper_wholebody`](projects/go2_piper_wholebody/README.md) | Go2 背 Piper X，跟踪机身速度与末端目标，附已训练权重 |

## 常用命令

```bash
bash run.sh new my_task                       # 从模板 go2_goal 生成项目，train.yaml 写明 backend: isaaclab3
bash run.sh new my_arm --template piper_pick  # 机械臂模板：Piper X 抓方块，带已训练的权重
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

## 依赖组与按需安装

环境按用途分组（详细设计与实施状态见 [docs/PACKAGING_PLAN.md](docs/PACKAGING_PLAN.md)）：

| 组 | 能做什么 |
| --- | --- |
| `core` | `new`、`catalog`、终端编辑 `scene.yaml`、生成 `observation.py`；`check` 另需项目后端组 |
| `editor` | 网页编辑器（`edit`），不需要仿真器 |
| `import` | `asset import`：网格、USD、URDF/MJCF 的登记与预览；URDF 转 USD 不需要 Isaac Sim（MJCF 转换另需 `usd` 组） |
| `usd` | MJCF 转 USD（`mujoco-usd-converter`）；它自带的 MuJoCo 4.x 与 `import` 组的 3.12 不能同装，所以单独一个环境 |
| `isaaclab3` | `check`、`view`、`record`、`smoke`、`evaluate` 的仿真部分 |
| `train` | `train`、用训练检查点 `evaluate`（RLinf、ray） |
| `deploy` | 真机推理只需要 `torch` |
| `isaac`、`mjlab` | 冻结后端，各自 SDK |

```bash
python3 tools/setup_env.py --check --python .venv-il3/bin/python --with isaaclab3,train  # 缺哪个组
python3 tools/setup_env.py --with core,editor --env .venv-editor                          # PyPI 组，用 requirements/ 的锁文件
python3 tools/setup_env.py --with isaaclab3,train --env .venv-il3 --isaaclab-source ~/repos/IsaacLab-3.0
```

`run.py` 的命令缺组时只打印一行提示（`需要 [train]：tools/setup_env.py --with train`），
并在工作区里找装有该组的解释器自动切换；唯一的环境变量覆盖是 `RLINF_PYTHON`。
测试按组标记：`pytest -m group_core`，或 `WORKSPACE_TEST_GROUP=core pytest tests` 只收集该组。

## 用户需要维护什么

| 文件 | 职责 |
| --- | --- |
| `scene.yaml` | 实体组合（含抽象体）、初始状态，和 `policy`：各层的模型、频率、输入，控制方式的例外 |
| `policy.py` | 策略网络（每层一个 PyTorch 模块）及动作解码；特征处理、缩放和偏移在这里写 |
| `task.py` | `evaluate(obs: SceneInfo)`，返回 reward/success/failure/metrics，可选 set |
| `observation.py` | 自动生成，不要手改：对象类型与字段说明，以及 policy.py/task.py 函数的返回类型（每层的 `<层>Obs`、被训练层的 `Action`、`decode_action` 的 `Targets`、`TaskResult`）；scene.yaml 定不下的维度写作 None |
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

## 策略、控制与返回类型

scene.yaml 的 `policy:` 写策略：一层或两层（`layers`，有序，名字自己取，最后一层驱动机器人），每层是 policy.py 里的一个
PyTorch 模块，写明多久算一次（`period`，秒）和读什么（`inputs`：场景信息名，或另一层的名字 = 读它的结果，自己的名字 = 自己上一步的动作）：

```yaml
policy:
  layers:
    policy: {model: .policy:Policy, period: 0.02, inputs: [robot.joint_position, destination.position, policy]}
  decode: .policy:decode_action
  initial: .policy:initial_action   # 可选
  modes: {piper: joint_velocity}    # 可选：不按关节位置控制（或 none：不受策略控制）的机器人
```

两层时另写 `train: <层名>` 选择训练哪层，另一层固定。分层只在两部分频率不同、一部分是现成的模型，或想单独替换一部分时才需要；
编辑器“策略”里的“增加一层… / 去掉 <层>”改的是同一份配置。层的结果由代码定义：forward 返回 Tensor，或张量字典
`return {'goal': goal,  # [B, 3] 目标位置（m）}`（键名显示在编辑器两层之间的箭头上，每项后的注释在悬停提示里），下一层读 `obs.<层>.<键>`。
动作相对的名义姿态和缩放写成 policy.py 里的常数（如 go2_goal 的 `STANCE`、`SCALE`），不读仿真的默认关节位置。
场景里每台机器人的全部关节自动受策略控制；`decode_action(action, obs)` 返回 `{机器人名: [B, 关节数]}` 的实际单位目标，
训练动作和关节目标的个数可以不同（如末端位移 + 逆运动学）。框架不会自动替用户编写解码算法。旧的 `control:` 写法（planner /
executor、dimension）照常读取运行。

网络大小和自己的超参写在 train.yaml 的 `actor.model` 里：模型的构造参数写 `config` 就收到这一段（`config.hidden_dims`、`config.<自加的键>`），
不同配方可以给不同的值；scene.yaml 里该层的 `parameters` 也作为构造参数传入（如权重文件）。
被训练的层只返回动作，个数就是网络最后一层的大小（构造参数写 `actions` 时传入受控关节总数）。PPO 需要的动作噪声
（初值 train.yaml `actor.model.action_std`）和价值网络由框架补上：模型写了 `value(obs)` 就用它，否则是读本层数值输入的 MLP
（层数 `actor.model.hidden_dims`）；自定义的价值网络放在名为 `value_head` 的属性里，RLinf 才会给它 `optim.value_lr`。
`parameters.weights` 由框架加载。

这些函数的返回类型生成在 `observation.py`，在 `policy.py`/`task.py` 中用作返回注解，键或形状写错时编辑器直接提示：

| 函数 | 返回 |
| --- | --- |
| 被训练层 `forward(obs)` | `Action`：动作 [B, 动作个数]；可选 `value(obs)` 返回 [B, 1] |
| 固定层 `forward(obs)` | Tensor 或张量字典，不保留隐藏状态 |
| `decode_action(action, obs)` | `Targets`：每台受控机器人一项，实际单位、与 action 同设备的批量浮点 Tensor |
| `initial_action(obs)`（可选 `policy.initial`） | `Action`：启动和每次复位时的动作；未设置为 0 |
| `evaluate(obs: SceneInfo)` | `TaskResult`：reward/success/failure/metrics，按环境输出，缺省 0/False；reward 乘 step_dt 计入回报，success/failure 结束回合，metrics 在回合结束时记入日志 `env/task/<名称>`；可选 `set: {抽象体名: [B,3] 位置或 [B,7] 位姿}` 在本步复位之后、观察之前移动抽象体 |

Task 里 `obs.<层>.output` / `previous_output` 是解码前的动作。旧写法的 `control.output` 逐个列出关节名，同一关节被多个终点控制时在环境初始化时报错。
Task 跨步统计用 `shared.assembly.task_memory.TaskMemory`（按环境自动复位）：`hold` 计持续时间，`mean` 求平均，`rate` 求逐步变化率（复位后第一步为 inf）。

常用的物体关系在 `shared/predicates.py`（参考 BEHAVIOR / LIBERO 任务文件的谓词，写成按环境批量的 Torch 函数）：
`inside(物, 容器)`、`on_top(物, 支撑面)`、`near(a, b, 距离)`、`is_open(物, 关节)` / `is_closed`、`upright(物)` 返回 [B] 布尔，用于 success / failure；
`inside_gap`、`on_top_gap`、`distance`、`open_fraction`、`tilt`、`bottom` / `top` 返回米、弧度或 0–1 的程度（关系成立时间隙为 0），用于塑形奖励。
参数直接传 `obs` 节点：实体（用它的 `bounds` 包围盒随之转动）、抽象长方体或矩形（`size`；矩形上下不限）、抽象点或 [B, 3] 位置。
例如 `inside(obs.bowl, obs.microwave, margin=.02) & is_open(obs.microwave, 'microjoint', .5)`；持续一段时间用 `TaskMemory.hold`。
这些判断只看几何：“在里面”指中心在对方包围盒内，对篮子、多层架子这类空心或凹形物体较粗，可在关键位置放一个跟随实体（`mount`）的抽象长方体代替；
`on_top` 不判断接触和支撑。为此 SceneInfo 的实体另有 `<名>.bounds`（[B, 2, 3]，本体坐标下几何的最小角与最大角，默认关节位置时）、带关节实体的
`<名>.joint_limits`（[B, 关节数, 2]），桌子等静态物体也有固定的 `position` / `orientation`；目前由 isaaclab3（Newton、OvPhysX）读取，其他后端读取时报错。

## 网页编辑器

`run.sh edit` 打开网页编辑器（viser，与仿真同一个环境，安装见 [DEPLOYMENT](docs/DEPLOYMENT.md#步骤)），编辑同一份 `scene.yaml`，经共享 `SceneDocument` 写回，终端修改与编辑器修改等价。

```bash
bash run.sh edit <project>                        # 打开 http://127.0.0.1:8080；viser 在 8081，远程使用时两个端口都要转发
bash run.sh edit --scene path/scene.yaml --port 8093 --no-native-preview
```

### 界面用法

界面是编辑器自己的页面（`tools/editor_ui/`），viser 只负责中间的 3D 视图：顶栏是项目名（新建 / 打开项目）与常用命令（导入、重新加载、刷新网格、贴图、简洁风格、经典面板；窗口窄时先收成图标，再收进“⋯”）、左右栏开关（T / N；拖动两栏内侧边缘调宽、双击收起，宽度按浏览器记住）、物理引擎、渲染器、复制预览命令、消息（提示和错误都留在这里，转换工具的输出可展开查看）、撤回/重做和保存，后端检查不通过时顶栏出现提示；视图左上角是移动 / 旋转 / 缩放（G / R / S）与聚焦。左栏是筛选框、每层一个读取开关（一层时叫“策略”，两层时是各层的名字；点一下在场景树和 3D 中标出这一层读取的物体，🔥 表示被训练的层）和场景树；树行末尾常显锁定、物理、渲染、显示四个开关（平常状态淡色，改过的着色），有子项的行都能折叠：机器人展开到连杆，布局组合和装在别的实体上的物体（如机器人上的相机）收在它下面；底部“添加”（Shift+A）打开资产架，“导入”载入自己的文件。右键场景树行或 3D 视图中的物体都会选中它并弹出同一菜单。右栏是选中对象的属性：变换（位置、旋转、缩放；数值框可左右拖动、点击输入、两端箭头步进）与父物体、相机画面、传感器或灯光的参数（单位写在数值框里）、观察（每层一列勾选，列头是层名的首字母）、控制方式与关节初始值（拖动滑轨实时预览，松开写入一步撤回）、复位、运动、形状，机器人和物体的原始参数收在默认折叠的“高级”里；什么都不选时右栏显示场景的策略：读哪些输入、多久算一次，箭头指向它控制的机器人（控制对象：关节数与控制方式）；两层时两张层卡片之间是一个箭头，悬停显示上一层结果的键名与说明（来自 policy.py 里 forward 返回的字典）和对应的代码（forward、decode_action）；层读自己上一步的动作时，输入里就写这一层的名字。分区可折叠、拖动标题排序，下拉选项多时可搜索。项目菜单里的“经典面板”打开旧的 viser 面板（初始速度、传感器安装偏移等尚未迁移的设置），只在打开时更新。编辑器不自动保存：所有改动（含添加、删除、组合、撤回/重做）先写入项目目录下的隐藏工作副本 `.scene.yaml.<pid>.editing`（保存按钮变色），点保存或 Ctrl-S 才写入 `scene.yaml` 并刷新 `observation.py`；“放弃修改并重新加载”丢弃全部未保存改动。保存前若 `scene.yaml` 已在编辑器外被修改，保存会拒绝覆盖并提示重新加载。工作副本在编辑器退出时删除，不进入实验快照和 git。

| 要做的事 | 入口 | 操作 |
| --- | --- | --- |
| 添加资产 | 左栏“添加” | 资产架按类别列出共享资产、SimReady 与场景环境，点选卡片添加。 |
| 导入自己的文件 | 左栏“导入”，或把文件 / 文件夹直接拖进窗口 | 支持 URDF、ROS 的 `.urdf.xacro`、MJCF、USD/USDZ，以及 OBJ、STL、PLY、glTF/GLB、FBX、DAE、OFF、3MF（可打包成 zip）。URDF、xacro、MJCF、USD 可以选它所在的整个文件夹，甚至整个仓库（如 `xarm_ros2-humble/`），导入框以目录树列出其中能单独导入的文件（只定义宏的 xacro、不是模型的 .xml 不列；同名文件按所在目录区分，可搜索），点选要导入的那个：页面先只发送其中的小文本文件（URDF、xacro、`package.xml`、YAML 等），编辑器读出它实际用到的网格和贴图，再只上传这些（`shared/asset_import.py` `needed_files`；xarm7 加夹爪从 89 MB 的仓库里只传约 5 MB）；单独选一个 URDF 时它的网格传不上来，会在转换前提示。选“物体 / 机器人 / 场景”和是否固定根部（xacro 另填参数，如 `dof:=7 add_gripper:=true`），上传后在后台转换：视图左下角的进度卡显示正在做什么、用时，可取消；视图中心的地面上标出它将出现的位置。完成后放到那里，同时列入资产架。提示和错误都留在顶栏“消息”里，转换工具的输出可展开查看。 |
| 设置父物体 | 把场景树行拖到另一行，或拖到机器人展开后的某个连杆上；右键“设父物体：点选…”（P）后在视图或树中点父物体，点到机器人哪个连杆就装在哪；右栏点“父物体” | 都会弹出确认：选关系（安装 / 跟随 / 固定，或仅摆放）、连杆（接触传感器还可选“所有 *_foot”这类连杆组），以及位置“保持原位 / 移到父物体原点 / 对齐父物体（位置与朝向）”。已有父物体的再拖到别处同样弹出；拖到空白处或点 ×（Alt+P）解除并保持原位。 |
| 添加传感器 | 右键机器人或物体（视图里右键会记住点到的连杆），或右键机器人展开后的连杆 | 相机、接触力传感器、高度扫描器、坐标测量器直接装在那里并选中，再在右栏调整。 |
| 看传感器量什么 | 选中传感器 | 接触力传感器在它覆盖的每个连杆上画出力的方向，高度扫描器画出射线和落点，坐标测量器画出到目标连杆的连线；参数（更新周期、历史帧数、扫描网格、射线朝向等）在右栏“参数”，单位写在数值框里。 |
| 布局组合 | 场景树或 3D 视图里多选后 Ctrl+G 或右键 | 只一起摆放，不建物理连接；选中组合右键解散。 |
| 设置观察 | 选中物体看右栏“观察”；场景外的输入在不选物体时的场景页 | 物体的量 × 每层一列；悬停看 `obs.` 属性名、形状与单位。相机的输出图像改了以后观察列表立即跟着变。 |
| 策略读什么、多久一次、分不分层 | 不选物体时右栏“策略” | 一张卡片：输入列表（每个物体一组，点名字选中它，眼睛按钮在场景中标出；场景外的输入有重力、另一层的结果、自己上一步的动作）、“每 … s”、箭头、控制对象；“增加一层… / 去掉 <层>”；两层时顶部“训练”二选一。 |
| 设置控制 | 选中机器人 → 右栏“控制”，或不选物体时策略卡片末尾的“控制对象” | 机器人的全部关节自动受控；只选位置 / 速度 / 力矩 / 不控制（写入 `policy.modes`）。滑轨改初始值并实时预览。导入的带关节资产在“控制”里有“作为机器人”开关（`kind: robot` / `entity`）：打开后由策略控制，场景已有策略时先设为“不控制”。受控机器人还有“重力补偿”开关（`gravity_compensation`，连杆不受重力、关节只管跟踪目标，像真机控制器的重力补偿；动作写成“实测位置 + 增量”时要打开，否则会慢慢下垂）。 |
| 相机画面 | 选中相机 → 右栏“画面” | 编辑器视图按相机位姿与内参渲染的示意（不是仿真传感器图像），改动停下约 0.4 s 后刷新，抽象体、传感器和不可见实体不画进去。 |

奖励和成功条件没有界面，直接写 `task.py`。

### 细节

新建环境与 `run.sh new` 相同，复制一个维护模板（四足 `go2_goal` 或机械臂 `piper_pick`，模板自带的 `weights/` 一并复制），因此一开始就有策略、`policy.py`、`task.py` 和 `train.yaml`；首次保存前放在临时目录，保存时整体移到 `projects/<名称>`，不覆盖已有项目（`shared/assembly/starter.py`）。编辑器可以摆放实体、组合安装、调整初始关节状态（机器人和带关节的物体都可以，如微波炉门、抽屉，场景树里也能展开到关节；带关节的导入物体还能在“关节”栏把一个关节设为按比例跟随另一个，写成 `usd_object.couplings`，仿真里由物理引擎保持，Newton 与 OvPhysX 都按 PhysX 联动关节结构生效，如踏板带动垃圾桶盖、旋钮带动指针），添加和编辑抽象体，切换被训练的模块，设置决策周期、输出维度并勾选观察（模块只能选择前序模块的接口），设置各机器人的控制方式。勾选观察时位置按观察目录固定，取消再勾选回到原位（已选项的相对顺序即模型输入顺序，不因此改变）。用 `control.output` 逐个列出控制目标的旧项目，控制面板只显示、不修改。数值框精度 0.001（角度 0.1°）。没有 Task 面板，任务编辑 `task.py`。

界面以左侧“场景”树为中心：地面、机器人、相机、灯光、抽象体等都在树中，也可在 3D 视图中点击选中（拖动坐标轴时不会选中背后的物体）；按住 Ctrl（macOS 上 Cmd）点击在视图里加减选、Shift 点击加选，和场景树里 Ctrl / Shift 点击得到同一个选择，选中的几个一起显示轮廓，坐标轴、F / H / Delete 都作用于全部，右栏显示最后点中的那个。只有坐标轴能移动物体：在视图里按住任何物体拖动都只转视角（`web_editor.BODY_DRAG` 关闭了按住物体直接拖的旧方式）。工具栏的磁铁（吸附表面，默认开）打开时，沿 X/Y 轴或水平面移动的物体会贴着下方的桌面、地形走，松手落稳并提示重叠；沿 Z 轴移动不吸附。End 键、工具栏下箭头或右键“落到下方表面”把选中的物体放到下方最高表面上。坐标轴在屏幕上大小固定，缩放视图时不变。`--port`（默认 8080）上打开的就是编辑器页面：viser 自带的前端源码打上 `tools/viser_client/rlinf.patch` 后构建的单文件页面 `viewer.html`，页面头部直接带上编辑器界面，由编辑器自己的 HTTP 服务（资产架、场景树和界面接口也在这里，SimReady 资产库在 `/library`）提供，连接下一个端口上的 viser websocket；那个端口上的 viser 原页面打开后会自动跳到编辑器页面。改进版视图：选中物体有被遮挡时也可见的轮廓；两侧栏浮在全宽视图之上，收起或调宽时视角不动。使用者不需要 Node；改补丁后用 Node ≥ 24 运行 `python tools/build_viser_client.py` 重新构建（`--edit` 在 `.cache/viser_client` 准备可编辑的源码，`--diff` 写回补丁），构建记录的 viser 版本与安装的不一致时，`--port` 上是 viser 原页面（界面加载后再注入），编辑器的 HTTP 服务在下一个端口。视图默认是仿 Newton OpenGL 查看器的简洁风格（渐变天空、主光源阴影），项目菜单“简洁风格”可切回 viser 默认的环境光。1 m 棋盘地面只由场景里的 `ground_plane` 实体画出（在场景树中可选中、隐藏）；没有显示中的地面实体时，原点处只有一张细线参考网格，它不是地面，物体在仿真中不会落在上面。网页与 Newton 窗口都是固定光照的示意，亮度不代表相机画面，要看相机实际渲染用下文 `--camera` / `--camera-render`。父物体的关系分两种：物理的（传感器安装在连杆上、抽象体跟随根部、两个物理资产固定连接，写入连接；两端都是物理资产时默认）和“仅摆放”（写入 `authoring.parents`，不创建物理固定），改关系或连杆都保持当前世界位置。物理安装会移除子实体自身的随机复位：只有连接链的根可随机复位，子实体在复位时按连接放到父物体上。删除实体（Delete 键、右键菜单或右栏垃圾桶）前在编辑器自己的对话框中确认并列出影响：直接定义的子物体上移一级、保持世界位置（观察名随之变化，需同步 policy.py / task.py）；安装或共同摆放在它上面的实体解除依附并保持原位；挂在它上面的非相机传感器随之删除；被删实体的控制终点和观察项随之移除，弹窗逐项列出（policy.py 的 decode_action 和 obs 引用需同步）。被删除的实体从高度扫描器的射线目标中去掉；删掉最后一块地面会让扫描器没有目标，弹窗事先提示（Isaac Lab 3 要求恰好列出全部有形状的全局实体，例如 `scope: global` 的地面或地形）。物体重叠时，在同一处再次点击会在该处的物体之间轮换选中。每次结构改动和保存后按项目 train.yaml 的后端对当前编辑内容做一次与 `check` 相同的静态场景检查（如传感器未挂载、IL3 射线目标不符），不通过时仍可保存，但弹出提示并在状态栏保留原因，直到修正。实体的“参数”区按工厂函数签名的类型注解生成字段，保存时运行工厂附带的无 SDK 校验（与 `check` 相同）。选中相机时检查器显示相机视角示意图（编辑器视图渲染，不是仿真传感器图像），“对准当前视角”把相机放到 3D 视图的位置和朝向（装在连杆上的相机记录相对连杆的位姿）；右键给机器人加相机时，装在名字含 camera 的连杆上按相机连杆习惯朝 +x，其余连杆（手、夹爪）朝 +z 向外看，名字取 `<机器人>_camera`；机器人还没导出连杆时先自动导出，完成后再装上。名称框里写字母、数字、下划线组成的名字会真正改名（scene.yaml 里的观察、安装、父物体等引用一起改，policy.py / task.py 里的 `obs.旧名` 需手动改），写中文等只改编辑器里的显示名。缩放模式（S）在改进版视图里是每根轴一条线、末端一个小方块（和 Blender 一样），按住原地拖动缩放，不移动物体；旋转模式在旋转弧旁画出三根轴线。顶栏“复制预览命令”给出带当前视角的 `view` 命令，选中相机时改为 `--camera <名>`，看该相机的真实画面。观察列表按 `obs` 对象分组，显示策略代码读取的属性名、说明、张量形状与单位，按组件写回 `control.components.<组件>.observations`。快捷键：Ctrl-S 保存、Ctrl-Z / Ctrl-Shift-Z 撤回 / 重做（最多 100 步）、Esc 取消预览/选择、F 聚焦、H 隐藏/显示。

“添加”页的陈列栏以卡片列出可用资产，`assets/terrains/` 下的地块列在“地面 / 地形”类。添加传感器时在“安装到 / 安装连杆”中选择所在实体和连杆（默认选中的实体，否则第一个机器人的根部），新传感器直接以安装状态写入；接触、高度扫描、坐标变换传感器只能安装在连杆上，相机还可选“不安装，固定在世界中”（各后端由 `FREE_SENSORS` 声明，Isaac Lab 3 也支持）。连杆名来自原生网格导出，导出完成前添加会提示稍候。坐标变换的 `target_body` 默认取安装实体根部，在检查器中从该实体的连杆里选择。新高度扫描器以全部全局地面为射线目标；已瞄准全部地面的扫描器在新增全局地面时自动加入它，删掉地面后再添加的地面也会自动成为目标。示意图缓存在 `.cache/asset_thumbnails/`。导入的文件属于当前项目：放进 `projects/<项目>/assets/<名称>/source/`，转换结果与记录在同一目录，登记在该项目的 `assets/library.json`，只出现在这个项目的资产架（标“本项目导入”），复制项目或用它作模板时一起带走；新项目要先保存再导入（`shared/asset_import.py`，命令行 `run.py asset import <文件或目录> --project <项目> --kind object|robot|scene`）。不给 `--project` 时放进工作区的 `assets/imported/`、登记在 `assets/library.json`，供所有项目使用，这是维护者扩展公共资产的方式。转换：ROS 的 `.urdf.xacro` 先用 xacro 展开（`$(find 包名)` 是上传文件里写着这个名字的 `package.xml` 所在的文件夹，不需要装 ROS；缺包时写明包名）。URDF 的网格按 `package://` 的包、URDF 旁的相对路径、再按路径结尾在上传的文件里找；缺外观网格或无法替代的碰撞网格时在转换前停止并列出文件（请导入整个文件夹），只缺碰撞网格时用该连杆的外观网格并写明；惯量不满足三角不等式的连杆按 MuJoCo 的做法取主惯量平均值并写明；URDF 的 `<mimic>` 写成 `usd_object.couplings`。URDF 不含关节增益，Isaac 导入器自带的增益在轻的连杆上会发散，所以 URDF 机器人一律给默认 PD。URDF / MJCF 走 `run.py asset convert`（Isaac 导入器生成 USD，MuJoCo 编译 MJCF，记录对齐结果，转换器留下的空引用会去掉；为被别的模型包含而写的 MJCF，如 robosuite / LIBERO 物体用 `inertiagrouprange` 排除了全部几何体，单独导入时去掉该项、按几何体自身质量计算；Isaac 导入器把被动关节阻尼写成力上限为 0 的驱动，包装层把力上限改为不限，否则 PhysX 不施加阻尼、Newton 拒绝该关节；MJCF 的 site 是参考点和区域标记，导入器会把它们变成可见形状，包装层把它们隐藏）；USD 直接使用，按场景导入时包装成静态环境并补碰撞体；网格写成 USD，物体是一个刚体：每个部件另生成封闭凸包作碰撞体（不可见），质量按凸包以 500 kg/m³ 估算（可在“高级”里改 `mass`），质心与转动惯量按凸包写明，下载的网格常不封闭，不让引擎从显示网格推算；场景是静态三角网格碰撞。导入的机器人（`usd_object`）关节由参数 `actuators`（关节名正则 → `stiffness`/`damping`/可选 `effort_limit`，Isaac Lab 隐式 PD）驱动，USD 没有写驱动刚度时导入自动填默认 PD（刚度 100、阻尼 10）并提示，按机器人在“高级”里调整（如 R1 Pro 的虚拟底座关节需要很高刚度）；编辑器给已有策略的项目添加机器人时，新机器人默认“不控制”（`control.modes: none`），已有 `decode_action` 不受影响，要训练它再改为“位置”并补上 decode。glTF / OBJ 按 Y 轴向上转成 Z 向上；网格文件不带单位，尺寸像毫米或厘米时会提示在变换里设缩放。FBX 等需要本机装有 Blender。SimReady 资产是平台提供的公共资产（下载与包装放在工作区 `assets/`）；其中关节有驱动刚度的按机器人添加，如 M20，可展开连杆、设置控制。导入后用 `view` 核对关节、碰撞和质量。从资产架拖入视图、点选位置放置或导入时，物体按自身底面范围向下取样，落在下方最高表面（地形起伏、桌面）上方 3 cm，不按原点高度摆放；放下后若仍与其他物体或静态场景（导入的建筑、地形）相交，状态栏写明对象和相交深度，因为仿真开始时引擎会把相交的物体弹开或让它穿过地形。按坐标填写的位置保持不变。

网页先显示示意代理和上次导出的网格；新资产或改了参数时在后台用仿真后端的解释器导出原生网格（`RLINF_PYTHON` 可指定），导出后保留 2 分钟供接连导入复用，空闲或编辑器退出时自动结束；`--no-native-preview` 关闭启动时的自动导出。网格按实体缓存，同一资产与参数在各项目和陈列栏间共用。点击保存（或 Ctrl-S）才写 `scene.yaml` 并更新 `observation.py`；添加、删除、组合、安装等结构操作先把面板中的修改并入工作副本再应用，原生网格导出也读取工作副本。浏览器内不运行物理仿真，布局结果以 `view` 为准。

## 命令与验证边界

- `check`：静态配置、注册和基础引用检查，不启动仿真，不证明能训练；刷新 `observation.py`。
- `view` / `record`：实际场景预览或策略回放。未给 `--camera-eye/--camera-target` 时，相机按场景描述自动取景，并跟随 env 0 的第一个机器人根部；给出任一项时按环境原点坐标固定放置。不给策略时默认 `--actions hold`：每步重复 `policy.initial` 给的动作（没写时为 0），`zero` 为全零动作；动作写成“实测位置 + 增量”的机械臂要开重力补偿才停得住。`--camera <名>` 改为显示/录制场景中该相机传感器的画面（env 0，小图按整数倍放大）；`--camera-render`（IL3）在窗口视角放一台临时相机，用场景相机的渲染器（默认 Newton Warp 光线追踪，配置 OVRTX 时为 RTX）画出当前视角，位置、朝向与视场角每帧跟随窗口相机。它不画抽象体等预览标记，且渲染比普通窗口慢。场景灯光（`dome_light`、`clear_sky` 等）只在 OVRTX 相机画面中起作用（`renderer: ovrtx` 时的 `--camera` / `--camera-render` 和训练中的相机观察）；`view` 窗口（GL 或 RTX）和 Newton Warp 相机都用各自固定的灯光，不读场景灯。`dome_light` 是环境光，位置和父物体对它没有影响。
- `smoke`：短程接口、终止与复位检查，不代表策略学会任务。
- `train` / `evaluate`：调用 RLinf 训练或评估。
- `catalog`：列出场景可读的信息目录。
- `asset convert <源文件> <资产目录>`、`asset check <资产目录>`、`asset terrain <地形名|all> <资产目录>`：维护资产用，生成并比对 USD 与 MJCF、生成地形地块，写 `ASSET.json`；地形生成用 Isaac Lab 3 的生成器（`.venv-il3`，不需要 Isaac Sim）；URDF 转 USD 用 `urdf-usd-converter`（`import` 组），MJCF 转 USD 用 `mujoco-usd-converter`（`usd` 组），都不启动 Kit；两者都没装时才回退到带 Isaac 导入器的解释器（Isaac Sim 环境）。详见 [docs/SCENE_BACKENDS.md](docs/SCENE_BACKENDS.md)。

新训练写独立实验目录。快照保存有效配置和项目源码；观察顺序、训练角色、控制周期、解码函数、终点类型与关节顺序变化都属于接口变化。`--allow-scene-change` 不允许绕过模型接口检查。
回放 checkpoint 时只比较运行会导入的项目模块和场景配置内容；Python 按去掉注释与文档字符串的语法树比较，README 等无关文件或只改注释不算差异。不一致时报错列出文件名并给出 `--snapshot <快照路径>`；`policy.py` 的改动属于策略语义，`--allow-scene-change` 也不放行。

完整恢复使用 `runner.resume_dir`；`actor.model.initial_checkpoint` 仅加载模型权重，不等于恢复优化器与训练进度。

## RLinf

默认使用未经修改的 `third_party/RLinf-upstream`，固定提交 `77ed948d68f07f55b4bc433b7643a409b2cd1adb`；也可用 `RLINF_ROOT` 指定干净安装。
模板使用 RLinf 原生 PPO 学习率配置。`actor.global_batch_size` / `micro_batch_size` 默认 `auto`：每轮采样（环境数 × 步数）整体作为一个 minibatch，按不超过 4096 的约数分 micro batch，所以环境数可以任意改；写明数值时 run.py 先检查整除关系并用中文说明（`run.py _configure_batches`）。接入方式与依赖边界见 [RLinf 接入](docs/RLINF_INTEGRATION.md)。

仓库中 Isaac Lab 2（`isaac`）与 mjlab 后端及旧实验的代码仍保留，但不在当前部署范围内；历史记录见 [历史 README](docs/history/WORKSPACE_README_20261019.md)。
