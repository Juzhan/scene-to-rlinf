# 项目创建与重建约定

适用于本工作区内新建项目、重构项目及共享接入层的改动。后续 agent 开始工作时先读本文件，
再读目标项目 README。用户在当前会话中的明确要求优先于本文件。
这是项目设计约定，不是要求每次工作都扩建框架；只完成当前需求涉及的部分。

## 1. 项目本质与责任边界

- 目标是让用户在 `projects/<name>/` 内直观配置场景、机器人、观察、动作、任务和模型，
  然后通过统一入口启动训练、预览与评估。优先减少用户必须理解的概念和文件。
- 仿真器及其原生环境负责物理、控制、传感器和环境推进；RLinf 负责它已提供的采样、
  优化、分布式训练和检查点能力。先查现有实现，再决定是否补适配。
- 用户项目不应重复实现 worker、runner、优化器循环、Ray 启动或生命周期管理。
  确有接口差异时，在共享接入层作最小补充，不能把内部运行细节变成用户必填配置。
- shared 负责通用能力；项目负责具体网络、场景定制、任务语义和数据语义。
  不把单个示例的模型名称、物体名称、观察维度、成功阈值写死在共享代码中。
- 底层仿真相关的修改与扩展尽量维护在各自后端代码内（`shared/backends/<后端>/`）；
  上层解耦部分（run.py、tools、shared/assembly、实验快照）只通过通用钩子或后端声明访问能力，
  不在上层写 `if backend == ...` 式分支，经 `backend_module` 调用各后端同名钩子（清单见
  `docs/SCENE_BACKENDS.md` 后端钩子），确保每个后端都能对应同一接口；某后端暂不支持时在钩子内明确报错。

## 2. 项目文件与运行入口

- 默认以根目录的 `train.yaml`、`scene.yaml`、`policy.py` 和简短的 `__init__.py`
  为项目入口，按需要增加文件；不能为了目录看起来完整就创建空壳模块。
- 默认配方放项目根目录。确有多个实验配方时可以扩展，不强制引入 `recipes/` 或 `training/`。
- 继续使用 `run.py check/train/view/record/evaluate/smoke <project>` 的统一命令。
  不要求每个项目自己写 `eval.sh`、测试循环或额外 `eval.yaml` 才能使用。
- `task.yaml` 不是 RLinf 的要求，不要为了包装接口而强制添加。新模板的 Task 直接写在 task.py，
  不为阈值或计算强制增加 YAML 字段。
- 接入现成场景时，优先引用现有环境与资产。共享适配已能完成的操作，不要求项目再写
  `env.py`；有特殊布局或测量需求才增加。`scene.py` 与 `env.py` 不要求强行拆分。
- 普通 Isaac assembly 项目复用原生资产、控制和 manager，不加载一整套原生任务预设。
  新模板明确配置目标、奖励和结束条件；旧实验显式使用的 preset 仅用于保留原实验语义。
  不把 MuJoCo 示例的全部文件复制过去。
  `check` 应保持静态检查；缺少在线能力的后端应明确说明，不假装所有命令都已支持。

## 3. 场景、机器人与编辑

- 场景、机器人、相机及设备沿用已有实体组合和连接表达。机器人加相机、夹具等通过组合
  完成，不为每种组合维护一份独立机器人实现或资产。
- 区分用于摆放的父子坐标关系与物理固定约束；对齐操作不能隐式创建物理约束。
- 用户可通过 Blender 或终端编辑同一份项目配置。编辑器应写回公共配置和编辑接口，
  不能形成另一套只在界面里有效的项目定义；不强制用户安装 Blender 才能训练。
- 新后端可分阶段接入。能读取 MuJoCo 信号，不等于已经支持 Blender 编辑 MuJoCo 场景。

## 4. 对象式观察与生成类型

- 用户模型接收 obs: Obs（或 PlannerObs / ExecutorObs），使用 obs.robot.joint_position、
  obs.robot.wrist_camera.rgb 等属性。项目代码不依赖字段字符串、全局 states 向量或内部传输格式。
- 项目只保留 README.md 一份说明。函数返回约定作为类型生成在 observation.py（PlannerOutput/ExecutorOutput/
  ExecutorTargets/TaskResult），policy.py/task.py 用它们标注返回值；只从 scene.yaml 确定，不确定的维度写 None。
- observation.py 从场景公共信息目录和观察选择生成，提供带 Tensor 注解的嵌套类及形状、单位、
  坐标系说明。生成文件不要手改；check/启动及 Blender 保存时更新，policy.py/task.py 不覆盖。
- Policy 类型只暴露选中的字段；SceneInfo 暴露完整基础信息，Task 按需读取，不预先复制全部图像。
  复用 Signals 的读取与 Information 元数据，不另建仿真读数机制。
- Tensor 保留图像/射线/关节等原始维度、dtype 和批次轴。ObjectView 仅提供属性访问；
  数学、编码、归一化、拼接由用户模型负责。对象属性不可重新赋值，读取张量也不应原地改写。
- 命名必须可表达为 Python 属性。缺少/未选择的字段明确报错，不静默返回 None。
- 面向真实部署的策略仅选择可获得或可估计的信息；Task 可读仿真真值，不自动混入模型输入。

## 5. 模型、组合策略与训练方式

- `build_model` 构造被训练或加载的模型。已有 assembly 用同一个模型完成训练和运行，
  不应为了形式一致强制增加 `build_policy`。
- VLA 加残差等确需多个组件时，可以提供 `build_policy` 组合完整在线策略。
  必须明确哪些参数被训练、哪些组件冻结、观察如何流转、动作如何合并。
- SFT 可以仅加载训练组件并读取离线数据；即便主要做 SFT，仿真任务项目仍应备好场景、
  完整在线策略和 Task，以便训练产物通过统一入口测试。
- 模型训练方法与场景后端分开，不因使用 SFT 就复制一套项目运行系统。
  接通环境并不等于接通 PPO，更不等于任意 VLA 都能直接做 PPO。
- 模型专用 YAML 字段由项目模型解析。共享评估层不读取 `policy.vla`、特定组件名或
  特定模型的权重文件格式；项目向共享层提供必要的准备和产物声明。
- 自定义字段仍是代码与配置之间的约定，不意味着只改 YAML 名字就能生效。
  不把 `vla`、`residual_weights` 等项目字段描述成 RLinf 的固定关键字。

## 6. Python Task 与基础控制

- 场景配置只承担实体、目标、观察选择和基础控制。项目 task.py 自动发现，无需 YAML 注册
  evaluate、parameters 或公式；阈值和计算直接写 Python/Torch。
- task.py 提供 evaluate(obs: SceneInfo)，有状态时提供 Task.evaluate(self, obs: SceneInfo)。
  常用关系（inside/on_top/near/is_open/upright 及对应间隙）用 `shared/predicates.py`，不在项目里重写。
  返回 reward/success/failure/metrics 字典，每项按环境输出；缺省 reward=0、success/failure=False；
  可选 set（抽象体名 -> 位置/位姿）是 Task 发起的每步事件。
  不负责模型加载、窗口、仿真 step/reset 或训练循环。
- 共享接入层将 Task 接到 Isaac 原生奖励/终止项，同一决策步只评价一次；原生奖励仍乘 step_dt。
  需要跨步统计可用 TaskMemory，按环境复位；复杂项目状态可提供 reset(env_ids)。
  不为普通任务强制引入状态类。Task 与观察读取独立，模型反馈不经过 Task 缓存。
- 新 Python Task 项目不接受 scene.yaml 中的 task/variables 评价配置，不保留双重配置入口。
  原公式引擎不是新模板要求，未迁移历史实验的运行记录不重写。
- 控制按机器人自动：每台机器人的全部关节受策略控制，`policy.modes` 只写例外（速度/力矩/不控制）；decode_action 返回实际单位目标，缩放/偏移在模型代码完成。频率由各层的 period 声明。
- targets 是实体的显式固定期望值。点、方向、矩形、长方体等空间量是 `entities` 中 `kind: abstract` 的抽象体：
  与物体同等地添加、选中、编辑、删除和选作观察（`<名>.position/orientation`，另有 size 或 direction），
  只是没有模型和物理。没有 goal/随机区域类型，含义由 Task 或模型代码决定；不再有 `regions:` 段。
  抽象体可用 `reset`（region/yaw_range/interval）随机放置；任何实体的 `reset.region` 引用静态矩形/长方体
  抽象体的场景路径 `world/<名>`。`mount` 只让抽象体读数跟随实体根部，不是物理约束，且不能再随机复位。
  相对量在模型或 Task 代码计算；仿真显示开关为 `visualization.abstract: [名称]`，两后端形状尺寸共用一份规格，仅在项目预览 view/record 画出（显式 --backend 的物理查看器不画）。
  回合内移动分两种来源，经同一后端写入口：场景本身的运动（宠物游走、车沿路径、移动的木箱）写成实体的
  `motion`（`wander` 区域内游走 / `path` 沿路径 / `custom` 调用项目函数，如 NPC 避障；`shared/assembly/motion.py`，编辑器“运动”栏可设，前两种可预览），
  不进 Task；与任务语义相关的事件由 evaluate 返回 `set: {名: [B,3] 位置或 [B,7] xyz+wxyz}`，同一步 set 优先。
  两者都由后端原生 command 项在复位之后、观察之前生效；刚复位的环境保持采样位姿（path 的复位位置在路径上），
  Task 看到的是策略上一步观察到的位置。
  可移动的是未跟随、且不被任何 `reset.region` 或游走区域引用的抽象体，它们另有 `<名>.linear_velocity`（位移/step_dt，复位后为 0）；
  区域保持静态常量，非法 set/motion 报错并写明实体名。决策步长用 obs.simulation.step_dt。物理实体只有声明 `kinematic: true` 的（仅 isaaclab3）能由 motion 或 set 移动，见 docs/SCENE_BACKENDS.md；动力学实体的 Task 事件尚未接入。
  两后端同样支持静态、随机复位、Task 移动和跟随读数；scene motion、`visible: false` 只接入 isaaclab3，其他后端明确报错。
- 是否受力（动态 / 运动学 / 固定根部）、是否出现在渲染画面（`visible`，抽象体为 `visualization.abstract`）
  是实体属性，编辑器场景列表行尾直接切换，写回同一份 scene.yaml；“编辑器中隐藏”只影响网页，不保存。
- 机器人的重力补偿（`gravity_compensation: true`，仅 isaaclab3，编辑器“控制”栏开关）也是实体属性：连杆不受重力（PhysX
  disable_gravity + Newton gravcomp），关节只管跟踪目标，对应真机控制器自带的重力补偿；动作是“实测位置 + 增量”时必须打开，
  否则没有固定设定点，手臂会慢慢下垂。增益仍按机器人设置。
- 导入的场景环境仍是一个 `usd_object` 实体，编辑器从 USD 结构推出物体树（`shared/assembly/composition.py`）展开显示；
  移除物体只写 `usd_object.exclude`（仿真与编辑器共用），不为场景里的每个物体复制实体定义；需要单独移动、受力或给 Task
  用的物体才取出为 `usd_object(prim=...)` 实体（同时写入原场景的 exclude）。日常场景以几十到一两百个物体为宜，上千个的作性能测试。
- 统一入口负责评估/录像/汇总。MuJoCo/VLA 专用项目尚未自动迁移到这套 Python Task 运行接入。

## 7. 历史、数据与权重兼容

- 使用时序输入时，离线数据与在线策略必须共用相同的构造语义：帧顺序、频率、坐标系、
  差分、补齐、有效标记和重置规则。不能随机拼接样本冒充连续历史。
- 历史不得跨回合、跨环境或跨缺失步串接；多环境接入时需要按环境分别维护和重置。
- 先检查 RLinf/仿真后端已有的历史能力是否匹配。奖励模型历史、轨迹存储和策略输入窗口
  用途不同，不能因为都有“history”就直接互换；必要的模型预处理可以保留。
- 可复用的窗口维护可收敛到 shared，项目仅保留自身特征构造；不要求新建框架才能实现历史。
- 修改输入顺序、维度、单位、动作含义或归一化方式时，应更新接口标识并检查权重兼容性。
  不截断、补维或静默解释旧权重；保留原数据和旧结果，新训练写独立目录。
- 数据划分按回合，归一化只从训练集计算。记录数据来源、有效配置及模型版本或文件标识。
- 加载权重初始化与完整续训必须区分；没有恢复优化器、调度器和训练进度时，不称为完整续训。

## 8. 当前实现边界：不能当作所有新项目的模板要求

- 维护中的 assembly 用 `policy.layers` 的模型类；当前共享 SFT 接入仍调用 `build_model(cfg, training_dataset)`。两者尚未统一。
  统一签名是后续改进方向，不能只改项目函数名而不核对共享调用方。
- 当前碗盘项目在线评估使用共享的单环境驱动；不代表已接入 RLinf 分布式评估或在线 PPO。
- 当前 RLinf `HistoryManager` 接在奖励模型历史路径上；它没有自动为此残差构造策略历史。
  碗盘项目的 `history.py` 仍同时服务离线转换和在线推理，尚未抽取通用窗口组件。
- `franka_bowl_vla` 当前采用 6 帧 × 24 维、20Hz、7 维输出，前 10 步只执行 VLA。
  残差输入不包含物体真值、接触真值、任务成功状态、历史残差或最终下发动作。
  这些是该项目的明确选择，不是所有机器人或 VLA 项目的硬性参数。
- 碗盘项目的 39 维旧权重不能用于新的 144 维输入。其当前数据、命令和验证记录以
  `projects/franka_bowl_vla/README.md` 为准。

## 9. 完成改动与交接

- 先找现有组件和参考项目，再增加代码。不要通过复制旧实验的训练脚本来创建新项目。
- 验证与改动相称：配置检查、必要的接口测试、短仿真或训练检查；没有需求时不自行启动
  长时间训练。改纯文档不要求重新跑训练。
- 明确区分“接口跑通”“权重发生更新”“完整回合成功”和“多种条件下可靠成功”。
  几步 smoke 或损失下降不等于策略学会任务。
- 项目 README 写清入口、可修改项、模型输入输出、运行方式和未接入的能力。
  新约定与本文件冲突时先结合用户最新要求明确取舍，再同步文档，不维护互相矛盾的规范。

## 10. 策略（policy 层）与 RLinf 接入

- scene.yaml 的 `policy:` 是唯一的策略配置：`layers`（一层或两层，有序，名字自取，最后一层驱动机器人）、
  `decode`、可选 `initial`、`modes`、两层时的 `train`。每层写 `model`（policy.py 里的类或工厂）、`parameters`、
  `period`（秒）、`inputs`（场景信息名，或另一层的名字 = 读它的结果；自己的名字 = 自己上一步的动作）、可选 `history`。
  `shared/assembly/layers.py` 读成内部 control 映射，运行时、类型生成、编辑器共用；旧的 `control:`（planner/executor、
  dimension）照常读取运行，两者不能同时写。
- 默认模板 go2_goal 是一层；分层只用于不同频率、现成不训练的模型或需单独替换的部分。编辑器“增加一层 / 去掉一层”
  改同一份 scene.yaml，增加时在 policy.py 末尾追加起始代码。场景项目的 train.yaml 必须写明
  `env.train.init_params.backend`（新项目为 isaaclab3）；未写明的旧项目沿用 configs/state_ppo.yaml 的 isaac，check 给出 WARN。
- 层的结果由代码定义：forward 返回 Tensor，或张量字典 `{'goal': ..., 'heading': ...}`（每项后写 `# [B, ...] 说明`），
  下一层读 `obs.<层>.<键>`。键名与注释从 policy.py 里 `return {...}` 静态读出（不导入），形状写不出的为 None，运行时第一步测得并核对；
  编辑器箭头上显示键名，注释在悬停提示里。不为层的输出写维度配置，也没有专门的结果类。被训练层返回一个动作张量。
- 动作相对的名义姿态（站姿、home 位姿）、缩放等是策略自己的常数，写在 policy.py 里，真机部署用同一组数；
  不把仿真的 `default_joint_position` 当作策略输入或动作基准（编辑器不再提供它和“仿真”作为输入；Task 可以读）。
- 模型的设置写在 YAML 里由代码读取，不在编辑器里填：构造参数写 `config` 就收到 train.yaml 的 `actor.model`
  （hidden_dims 及项目自加的任何键，环境进程里经 configs/state_ppo.yaml 的 `init_params.model` 传入，同一份）；
  scene.yaml 该层的 `parameters` 作为构造参数传入（如权重文件）。
- 被训练层 forward(obs) 只返回动作均值 [B, 动作个数]；个数即模型最后一层大小，构造参数写 `actions` 时传入受控关节总数，
  否则用试运行测出（试运行与固定层的首次运行都保存并恢复随机数状态，不改变环境的随机序列）。
  `shared/integrations/rlinf/object_policy.py` 补上 PPO 需要的 `log_std`（初值 train.yaml `actor.model.action_std`）与价值网络：
  模型有 `value(obs)` 就用它，否则是读本层数值输入的 MLP；价值网络参数名须含 `value_head`（RLinf 据此用 value_lr 与预热）。
  forward 返回 (均值, 价值) 并自带 actor_logstd 的旧式模型照常可用。`parameters.weights` 由框架加载
  （被训练层是整个适配器的检查点，固定层取其 `model.` 部分或模块本身的 state dict）；项目自有格式的权重用别的参数名由模型自己读。
- 固定层在环境进程按周期运行、无隐藏状态、不能读自己的输出；被训练层的动作来自 RLinf。共享层复用 Isaac 原生循环，
  不写新的仿真推进循环。LazyLinear 在共享接入层用实际观察形状和 dtype 初始化后才交给 FSDP/优化器。
- RLinf 传输/轨迹内部使用命名张量字典，图像等形状不展平。接收模型调用时再构造 typed view。
  共享适配器将命名 Tensor 字典放在原生 RLinf states 传输字段内，task_descriptions 提供批次元数据；
  forward_inputs 按 RLinf 轨迹批处理要求保存为平铺字典，项目不需要知道其键格式。
- 迁移后旧整策略 checkpoint 不静默加载：适配器参数名已变（如 `model.actor_logstd` → `log_std`；只有 `weights`
  初始权重做这一换名）。迁移项目的输入顺序与动作含义不变，现有权重评估结果与迁移前一致。新训练写独立运行目录。
- 支持多机器人控制；未实现两层同时训练、三层以上或任意控制图。Blender 插件仍只认旧的 `control:` 写法。
  无 Task 面板，任务直接编辑 task.py。

## 机器人控制（按机器人自动）

- 场景里每台机器人（资产能力含 articulation 或 `kind: robot`，非 kinematic）的全部关节都受策略控制，默认关节位置目标
  （`shared/assembly/control.py` `automatic_outputs`）。柜子、门等只有 joint_state 的铰接物体不受控。编辑器里是策略卡片末尾的“控制对象”。
- `policy.modes`（可选）只写例外：`<机器人>: joint_velocity | joint_effort | none`；`none` 表示不受策略控制（保持默认目标，
  如由场景运动驱动的机器人）。modes 写了非机器人或未知模式时报错。
- `policy.decode` 引用 policy.py 中的 decode_action(action, obs)，返回 `{机器人名: Tensor[B, 关节数]}`，每台受控机器人一项、
  不多不少；列序同 `obs.<机器人>.joint_position`（ASSET.json joint_order）。不训练的关节（如夹爪）由 decode 给默认值；
  末端控制等其他动作在 decode 里换算（如 `differential_ik`）。机器人增删后 decode 需同步，运行时报错会列出应有的名字。
- 训练动作与控制目标分离：RLinf 记录采样动作的概率，解码后的目标直接下发各原生控制项，不重新采样或按目标计算概率。
  所以控制写法只影响 decode 的返回形状，不影响训练算法与权重。wholebody 为 18 个训练动作（腿 12 + 臂 6），
  目标为 robot 12 + piperx 8（夹爪取默认值）。
- 控制目标必须是当前设备上的批量浮点 Tensor；不逐环境循环，不调用CPU/NumPy。
- 旧项目和实验快照的 `control.output`（命名终点映射，每项 type/asset_name/joint_names，逐个列关节、不用正则）仍可读取并照旧
  运行；它与 modes 不能同时写。同一机器人需要混用控制模式时暂用这种写法（`policy.output` 同义）。
- `policy.initial` 为可选项目函数，定义复位后第一步之前的动作，默认零；reset 只重置对应环境。
  `obs.<层>.output/previous_output` 是解码前的动作。

## 原生 RLinf 依赖边界

- 默认 `run.py` 使用 `third_party/RLinf-upstream` 的干净固定提交；不修改第三方源码，也不以全局 monkey patch 代替源码补丁。
- 维护模板及 IL3 维护项目使用原生 PPO 学习率设置和 sampling_params；`do_sample` 在共享模型适配器内转换为采样/确定性推理。
- `third_party/RLinf` 暂存历史实验的补丁，不是新项目运行依赖。依赖 adaptive_kl_target、SAC 样本预算等扩展的旧配方须显式选择历史 RLINF_ROOT，不能静默忽略配置。
- 历史 YAML 公式实现收于 `shared/assembly/legacy`；Python Task 主路径不导入它。新项目不使用该包。
- 依赖审计及当前验证边界见 `docs/RLINF_INTEGRATION.md`。不要把 CPU 协议测试或本地梯度更新描述为 RLinf 分布式训练验收。

## mjlab 接入阶段

- `projects/dual_backend_scene` 已具备同一份 scene/policy/task/observation/train 定义；train 的 `env.train.init_params.backend` 选择后端，不复制配方。
- 公共观察字段、双组件调度、动作结果校验和 Python Task runtime 位于 shared/assembly；两后端只绑定原生读数/控制项/managers。不得在 mjlab 复制策略或 Task 计算循环。
- mjlab 已接通原生 GPU 环境及原生 RLinf PPO，支持 check/smoke/train/evaluate/view/record（view 用 MuJoCo 被动窗口显示 env 0）；已验证短 rollout、部分复位、终止观察和参数更新。显式 --backend mjlab 的 view 仍仅查看物理场景，不运行策略。
- `.venv-mjlab` 为增量验证环境，不升级现有 Isaac 依赖；Go2 两后端共用 `assets/go2/`（USD 与其转换 MJCF，来源见 `source/SOURCE.json`）。
- 新机器人/物体资产用 `run.py asset convert` 同时备好 USD 与 MJCF，记录 ASSET.json 并通过对齐检查；不手写转换器，不在转换文件里写执行器增益，对齐不通过时不静默修正。
  用户自带文件经编辑器“导入”或 `run.py asset import --project <项目>` 进该项目的 `projects/<项目>/assets/<名>/`，登记在项目自己的 `assets/library.json`，项目自己管理；不带 `--project` 时进工作区 `assets/imported/<名>/` 与 `assets/library.json`，仅用于维护者扩展公共资产（`shared/asset_import.py`）：URDF/MJCF 仍走 convert（ROS `.urdf.xacro` 先按上传里的 package.xml 展开；网格缺失时停止并要求导入整个文件夹，只缺碰撞网格时用外观网格并写明；URDF 无关节增益，一律给默认 PD），USD 直接用，网格（OBJ/STL/glTF/FBX 等）写成 USD 并估算质量；原文件保留在 source/；物体用封闭凸包碰撞并写明质心与惯量。导入机器人的关节增益写在实体参数 `usd_object.actuators`（不写进转换文件），USD 无驱动刚度时导入给默认 PD；编辑器向已有策略的项目添加机器人时默认 `modes: none`。
- 后续按 `docs/SCENE_BACKENDS.md` 扩展传感器和连接，不把短训练参数更新当作学会任务；关节顺序以 ASSET.json `joint_order`（= Isaac 顺序）为准，由各后端读数换序；同名资产仍不保证动力学/权重兼容。

## 域随机化与多条件评估

- 域随机化写在 train.yaml `env.train.init_params.randomization`（实体 → mass/friction/com/push），评估默认跟随；
  声明由 `shared/assembly/randomization.py` 校验，各后端 `randomization.py` 绑成原生事件项，后端不支持时明确报错。
  摩擦按 PhysX 材料系数理解；MuJoCo 系引擎（Newton、mjlab）给被随机化实体的几何体接触优先级并换算成同等组合结果。
  尺寸等无法按环境随机的量通过评估条件改场景，不在 randomization 里伪造。
- 评估条件写在 train.yaml 顶层 `evaluation.conditions`（不传给 RLinf），`run.py evaluate <项目> --conditions all`
  每个条件独立评估并写 summary.json；条件取自当前项目配方，可用于旧运行快照，每个条件目录记录实际应用内容。
  条件只能替换随机化或改实体字段，不能改训练组件和策略接口。
- “多条件可靠成功”需写明条件、引擎、轨迹数与种子；单一种子、少量轨迹只说明在这些条件下成功。
