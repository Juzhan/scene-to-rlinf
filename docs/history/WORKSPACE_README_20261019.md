# RLinf 项目开发工作区

新建或重构项目前先阅读 [项目创建与重建约定](../../AGENTS.md)。该文件集中记录目录、共享能力、观察、模型接口和训练评估的约定，供开发者与 agent 共同遵循。

场景信息驱动的 Task 已可在 `scene.yaml` 的 `task` 配置；完整计算、判断、时序组合与示例见 [任务规则说明](../../shared/assembly/TASK_RULES.rst)。Task 读取完整信息目录，不受策略 observation 选择限制。

信息目录已区分数值、位置、向量、旋转、位姿、区域等含义，并声明形状、单位和坐标系；Task 可校验并推导计算结果类型，详见 [信息分类与计算设计](../../shared/assembly/INFORMATION_TYPES.rst)。

把场景、机器人、观察、动作、任务和训练配方放在自己的 `projects/<name>/` 目录内，通过统一入口编辑、训练和检查结果。当前配置化场景使用 Isaac Sim 5.1 / Isaac Lab；训练、优化器及检查点沿用 RLinf。训练无需启动 Blender，也不要求 Docker。

2026-10-10 已完成 Go2 楼梯训练闭环：复用官方预训练步态，经 RLinf 微调 100 轮、完整恢复至 120 轮，四个评估种子的上下楼路线 **128/128 成功**。测试为约 4–5 cm、9–10 cm 单级台阶，固定航点控制器提供速度指令，策略控制关节。见 [训练与框架验收报告](../../outputs/assembly_framework/blender_training_20261010/REPORT.rst)、[真实录像](../../outputs/assembly_framework/blender_training_20261010/go2_stairs_success.mp4)及[模型与快照索引](../../outputs/assembly_framework/blender_training_20261010/result.json)。这是已有步态上的微调结果，不是从零训练或自主导航。

## 从自己的项目开始

在 `~/ws/rlinf-workspace` 执行。机器狗楼梯任务可从 [Go2 楼梯模板](../../projects/go2_stairs/README.md) 开始：

```bash
bash run.sh new my_stairs --template go2_stairs
bash run.sh check my_stairs
bash run.sh catalog my_stairs
bash run.sh inspect my_stairs
bash run.sh edit my_stairs
```

不指定模板时，`bash run.sh new my_project` 默认生成 [planner + executor 项目](../../projects/planner_executor/README.md)：planner 透传所选输入，executor 为可训练网络，两个模型均在场景配置中显式选择。

机械臂场景可以改用 `bash run.sh new my_lift --template assembly_lift`。生成器只创建新项目目录，不覆盖已有文件，也不会自动训练或下载资产。`bash run.sh list` 列出可用项目；`bash run.sh new --interactive` 提供终端创建向导。

| 入口 | 用途 | 是否启动仿真 |
| --- | --- | --- |
| `check <项目>` | 检查配置、资产引用和组件名称 | 否 |
| `catalog <项目>` | 列出观察、动作、奖励的含义、依赖、参数与公式 | 否 |
| `inspect <项目>` | 读取真实关节、连杆、观察维度和动作顺序，缓存供编辑器使用 | 是，一个短暂运行的环境 |
| `edit <项目>` | 在 Blender 中编辑场景及训练组件 | 首次/刷新时短暂导出原生几何，编辑期间不运行仿真 |
| `view <项目>` / `record <项目>` | 在 Isaac 中检查真实场景，或录制策略回放 | 是 |
| `smoke <项目>` | 短程检查环境与训练协议，不判断是否学会任务 | 是 |

## Blender 与终端共用同一份配置

`edit` 打开 Blender 的 **RL Project** 侧栏，读取项目的 `scene.yaml`。可摆放实体、调整局部位置和朝向、添加资产与区域、勾选观察、配置基础关节动作，以及启用/禁用奖励或覆盖权重。点击 **Save Project YAML** 写回训练配置；普通 Ctrl-S 只保存 `.blend` 编辑工作区。子装配的实例修改写入 `overrides`，不会改动所有实例共享的源文件。

纯终端用户可以直接编辑 YAML，用 `catalog` 查看组件说明，用 `check` 检查配置。脚本和 agent 也可通过公共编辑接口读取版本并提交指定修改；外部文件发生变化时不会静默覆盖。两种入口可交替使用，具体操作见 [Blender 编辑器说明](../../tools/blender_addon/README.md)。

`assembly_lift` 自带已绑定方块复位的 `spawn_area`：中心 `[0.5, 0, 0.055]`，完整尺寸 `[0.2, 0.5]` 米。可以在 Blender 空间中移动、旋转和调整这块区域；训练每回合按 `object.reset.region` 采样根位置。区域也可以只作为几何描述，单独添加区域不会自动生成任务或奖励。高度表示实体根中心，默认不推断物体尺寸、贴地或避碰。

标准状态策略使用 `obs_dim: auto`、`action_dim: auto`，启动时从实际环境取得维度。调整观察列表或基础动作后，不必手工计算网络输入输出长度。`inspect` 生成的有效关节缓存允许在 Blender 中逐个选择关节；配置或源码变更后需要重新检查并加载。图像、字典或其他自定义模型输入仍由项目模型定义。

Blender 现在缓存并展示 Isaac 原生的 Go2、Franka 和程序化楼梯几何，机器人按原生初始关节姿态显示。首次打开会短暂启动无界面的 Isaac 导出进程，之后移动物体复用缓存；修改几何参数后点击 **Refresh Native Geometry**。编辑器展示 seed42 下一个训练副本的实际地形块，以该副本原点为局部零点；材质使用简化显示色，任务随机复位和运动仍以 `view` 为准。导出不可用的资产才使用有标记的示意代理，Isaac 原始资产引用不会改写。

常用文件是 `scene.yaml`、`train.yaml` 和 `policy.py`。可复用组合放在 `assemblies/`；特殊资产与任务项可由可选的 `assets.py` / `terms.py` 注册。普通项目不需要自己实现仿真循环。详见 [Assembly Lift](../../projects/assembly_lift/README.md)、[Assembly Go2](../../projects/assembly_go2/README.md) 和 [框架规划](../../projects/assembly_demo/ARCHITECTURE_PLAN.rst)。

## 训练、续训与独立验收

编辑后先检查真实场景和短程接口，再启动训练：

```bash
bash run.sh view my_stairs --num-envs 1 --steps 600
bash run.sh smoke my_stairs env.train.total_num_envs=2 env.train.max_episode_steps=8
bash run.sh train my_stairs runner.logger.experiment_name=my_stairs_run
```

`train` 调用 RLinf 原生入口，配置覆盖沿用 Hydra；`--config` 选择项目内的训练配方。楼梯模板默认 1024 个独立环境、1500 轮、每 100 轮保存；所需训练预算取决于任务与实际验收。Lift 模板默认两轮用于接入验证，正式学习需增加预算。多个并行环境是采样副本，不表示多机器人调度。

训练启动时打印独立运行目录与 **Frozen project** 快照路径，保存当次场景、子装配、项目源码、配方及可核对的资产引用。RLinf 继续负责模型、优化器和训练进度的检查点。新实验使用新的实验名，结果位于 [outputs](../../outputs)。

从完整检查点续训仍使用 `train`，没有单独的 `resume` 命令：

```bash
bash run.sh train my_stairs --snapshot /absolute/path/to/run/snapshot \
  runner.resume_dir=/absolute/path/to/checkpoints/global_step_100 \
  runner.max_epochs=300
```

`runner.max_epochs` 是恢复后的总轮数，`runner.resume_dir` 指向完整 `global_step_*` 目录。仅设置 `actor.model.initial_checkpoint` 是加载权重初始化，不代表恢复优化器及进度。重放采用当次冻结项目；相同维度但观察顺序、动作缩放或关节选择已变，也不能当作兼容接口直接加载。

Go2 楼梯项目另有独立路线验收入口，读取项目 `evaluation.yaml`，重新加载权重执行固定路线：

```bash
bash run.sh assess my_stairs --snapshot /absolute/path/to/run/snapshot \
  --checkpoint /absolute/path/to/full_weights.pt \
  --output /absolute/path/to/evaluation.json \
  --video /absolute/path/to/routes.mp4
```

`assess` 当前专用于 Go2 楼梯路线，不是所有任务的统一成功定义。默认测试 seed 43/44、级别 0/4、四个世界方向及上下楼；成功要求真实跨越台阶、到达目标、保持直立且没有机身触地跌倒，并低速保持。速度指令由固定目标控制器产生，十二个关节由学习策略控制；因此评价的是楼梯运动技能，不是自主导航。确切阈值与覆盖范围见 [Go2 验收说明](../../projects/go2_stairs/README.md)。

其他项目可用 `evaluate <项目>` 调用 RLinf 原生评测、用 `view/record --checkpoint ...` 回放策略。训练奖励上涨、smoke 通过或进程正常退出都不是任务成功；最终能力应连同独立评测 JSON 和真实录像判断。本轮开发与训练验证集中在 `outputs/assembly_framework/blender_training_20261010/`，此处不预报最终成功结果。

通用 `state_ppo` 配方默认仅在结束时保存，具体项目可覆盖保存周期。长任务需要设置合适的 `runner.save_interval`；当前 Go2 楼梯模板已使用周期保存。2026-10-09 的[历史检查点清理审计](../../outputs/assembly_lift/history_checkpoint_cleanup_20261009.json)记录了先前删除范围；旧 SAC 权重仍可评测，但没有保留回放缓存的历史实验不能视为完整可续训。

2026-10-12 起，assembly 模板使用显式目标和配置化任务，不再加载整套原生任务预设。Lift 为 27 维观察、Go2 为 223 维，旧 checkpoint 不兼容。原始短程接入记录位于 `outputs/assembly_framework/unified_authoring_20261010/`；[assembly_demo](../../projects/assembly_demo/README.rst) 仍用于独立场景组合验证。

## 不用 Isaac 的走格子

[gridworld](../../projects/gridworld) 复用 Gymnasium 1.2.0 的 FrozenLake：固定 4×4 地图、不打滑，只有到达终点奖励为 1，掉洞则失败，64 步未结束则超时。环境只做 CPU 数组运算，不导入 Isaac，也不需要 Pygame；当前只提供无窗口的检查、训练和评测。

```bash
bash run.sh check gridworld
bash run.sh test gridworld
bash run.sh smoke gridworld
bash run.sh train gridworld runner.logger.experiment_name=my_gridworld_run
```

项目内的[环境](../../projects/gridworld/env.py)负责批量观测和真实终态；[策略](../../projects/gridworld/policy.py)同时定义 actor（16→64→4 个方向的 logits）和 critic（16→64→1）。观测是当前格子的 one-hot，动作是一个整数，不是四维连续动作。地图、模型和 PPO 参数在[配方](../../projects/gridworld/train.yaml)中，训练使用现有 RLinf GPU 后端；不需要为此运行物理仿真。

默认训练 40 轮，每轮 64 个环境各采样 64 步，共 163840 次交互。[实际训练记录](../../outputs/gridworld_native_20261003_141653/metrics.log)和[原生独立评测](../../outputs/gridworld_native_evaluation_20261003/metrics.log)已生成；后者重载第 40 轮检查点，完成 1280 回合，到达率 100%，平均 6 步。另一次[无图形验收](../../outputs/gridworld_native_20261003_141653/acceptance.json)每种策略各跑 1000 回合：均匀随机 1.1%、未训练贪心策略 0%、训练后贪心策略 100%。已学会这张固定地图的最短路线，不代表能泛化到新地图或打滑环境。

重载这次保存的模型执行原生评测：

```bash
CHECKPOINT="$PWD/outputs/gridworld_native_20261003_141653/gridworld_native_20261003_141653/checkpoints/global_step_40/actor/model_state_dict/full_weights.pt"
bash run.sh evaluate gridworld env.train.total_num_envs=128 runner.logger.experiment_name=my_gridworld_eval "runner.ckpt_path=$CHECKPOINT"
```

## MuJoCo 双方块堆叠 SAC

2026-10-08 用户要求保留并暂停 Isaac，改为 MuJoCo 堆方块 SAC。Isaac 停在第 4755 轮，最近完整恢复点为 [global_step_4620](../../outputs/franka_isaac/sac_15000_ready_20261007/runs/isaac_sac_15000/checkpoints/global_step_4620)，场景、配置、日志、曲线与模型均保留；PPO/DINO 也不启动。

新任务 [frankasim_stack](../../projects/frankasim_stack/train.yaml) 复用原 MuJoCo Franka/Robotiq 模型、opspace 控制和相机，增加一块绿色底方块，将紫色方块放到绿色方块上。两块均为 4 cm、0.1 kg，随机位置初始分离至少 10 cm。actor 仍为双路实际 1280x720 RGB 缩至 128x128 加机器人 state7，不接收物体或目标坐标，不用 ManiSkill/VLA。动作仍是累积 XYZ 位移乘 0.1 和夹爪增量乘 1；50 Hz 控制、0.002 s 物理步长，回合增加至 250 步/5 秒。

奖励分项：接近 `0.2*exp(-20d)`；抬升 `0.3*clip(height_gain/0.08,0,1)`；抬升超过 2.5 cm 后，对齐 `0.5*exp(-40*xy_error-20*abs(height_error))`；真实承托放置 `+1`；放置且松爪 `+1`；成功稳定堆叠 `+2`。物体坐标只用于这些奖励和物理评测。成功要求中心偏差小于 8 mm、高度误差小于 4 mm、真实承托接触法向力大于 0.05 N、两块倾斜小于 10 度、底块在地面上；夹爪控制小于 0.1、没有夹爪接触且 TCP 离方块大于 7 cm；两块自由关节速度向量范数小于 0.05，连续保持 0.5 秒。只抬起、悬空对齐或仍夹持均不算成功。

五项局部测试通过，包含实际接触、松爪稳定、破坏后清零、随机复现、无坐标泄漏和原生包装层契约。明确使用特权坐标的控制器基线，在 seed43 完成一次堆叠：松爪稳定 0.98 秒、中心偏差 3.3 mm；这是控制器基线，不是学习策略。两轮真实 SAC 完成、206 个模型张量有限，actor/critic 优化步数为 2/8，checkpoint 和双图像/state7 replay 已核验；短测策略独立评测成功率 0，冻结骨干验证通过，仅表示流程可用。

新的 15000 轮 SAC 已完成，用户服务 `frankasim-stack-sac-15000.service` 已正常退出，没有加载旧抬升策略、优化器或 replay。algorithm、actor、rollout、weight_syncer 与原 MuJoCo SAC 相同。源代码、模型、MJCF 资产和配置已冻结于 [正式计划](../../outputs/frankasim_stack/sac_15000_20261008/plan.json)。最终独立评测 seed43/44 各 32 回合，均为 **0/32，合计 0/64**，冻结骨干验证通过，状态为 `acceptance_failed`，不是运行异常。不能把抬升任务 63/64 或控制器基线成功算作学习到堆叠。

真实奖励曲线有提升，但不等于任务成功：训练 `env/reward` 前十条均值 0.00186，末十条均值 0.36694；训练成功指标全部为 0，72 次周期评测成功指标也全部为 0。独立评测 mean return 分别为 70.985/58.058；最终方块平均抬高约 7.0 cm，而相对底块的 XY 偏差平均约 31.7 cm。64 回合末态承托力均为 0，最长有效松爪稳定保持均为 0 秒。[学习策略真实录像](http://127.0.0.1:18887/sac_15000_20261008/evaluation_seed43/replay.mp4)和[第一个回合 4 秒帧](http://127.0.0.1:18887/sac_15000_20261008/evaluation_seed43/learned_episode0_4s.png)显示抓起后仍持握、没有堆放。

v1 诊断：接近与抬升可持续贡献最高 0.5/步，而 30 cm XY 距离处的对齐项至多约 0.000003（尚未计高度误差），支持策略停留在容易获取奖励的中间行为。它不是奖励为唯一原因的因果证明；v1 源码、奖励和结果保持冻结，新奖励单独作为下面的 v2 实验。

本机入口：[训练曲线](http://127.0.0.1:6027/#scalars)、[训练和最终评测状态](http://127.0.0.1:18887/sac_15000_20261008/status.json)、[场景原图](http://127.0.0.1:18887/scripted_feedback_check_20261008/front_episode0.png)、[控制器基线录像](http://127.0.0.1:18887/scripted_feedback_check_20261008/replay.mp4)。服务只绑定 127.0.0.1；图像已解码，录像为真实 720p，HTTP 已验证，未作浏览器播放验收。

### 堆叠奖励 v2 续训

2026-10-08 用户授权“调整继续”。[v2 项目](../../projects/frankasim_stack_v2/train.yaml)只覆盖奖励，继承 v1 的真实物理环境和成功判据，不降低 8 mm 对齐、真实接触、松爪和连续稳定 0.5 秒门槛。双真实 720p、state7、原四维动作及随机初始位置保持不变；仍不用 ManiSkill/VLA，actor 不接收物体或目标坐标。

设 $L=\operatorname{clip}(\Delta z/0.04,0,1)$。新奖励是以下阶段项的最大值，而不是把接近和抬升长期相加：接近 $0.2\exp(-20d)(1-L)$；搬运 $L(0.1+\exp(-4e_{xy}))$；降低到堆叠高度 $1.5L\exp(-8e_{xy}-12|e_z|)$；真实放置 2；真实放置且松爪 3。原连续稳定成功另加 2，总成功奖励 5。搬运信号在 30 cm 距离仍约 0.40；抬升超过 4 cm 后不能仅靠继续抬高或 TCP 更接近来增加奖励。它仍是稠密阶段奖励，不是学会堆叠的保证；因奖励尺度改变，v1/v2 return 不能直接比较，能力按同一成功判据比较。

沿用 v1 第 15000 轮 actor 及共享视觉/状态编码器，但 **Q、目标 Q、alpha、优化器、replay 和新计数全部重置**，不是完整恢复旧训练。追加 15000 轮新奖励 SAC；除这项初始化元数据外，algorithm、actor、rollout、weight_syncer 与 v1 完全一致。[正式计划](../../outputs/frankasim_stack_v2/sac_15000_20261008/plan.json)冻结源码、资产、奖励、旧 checkpoint SHA 和评测条件，结束自动 seed43/44 各 32 回合，仍需每个 seed 成功率至少 80% 且骨干保持冻结，才标记能力验收通过。

原任务与 v2 共 10 项测试通过，包括奖励排序、宽距离进展、抬高收益平台、原真实接触/稳定成功，以及逐张量验证 actor 精确继承而 Q 保持新初始化。两轮真实 SAC 已生成 206 个有限模型张量，actor/critic 计步 2/8；新 replay 为 6 条轨迹、1536 样本，双 RGB/state7 正确。冒烟独立评测 seed43/44 各 32 回合正常完成、骨干冻结验证通过，仍为 0/64；两个 seed 均有 22/32 曾抬起、4/32 曾到达 3 cm 对齐范围，没有真实放置或有效松爪。两轮冒烟只验证执行链路，不用于评判新奖励能否学会堆叠。

正式追加 15000 轮已由用户后台服务 `frankasim-stack-v2-sac-15000.service` 启动；新 Q/优化器/replay 从零开始、actor 继承 v1 第 15000 轮，不使用两轮冒烟 checkpoint。当前正式策略能力尚待最终独立评测，不把启动或奖励增加计为任务成功。

v2 本机入口：[新训练曲线](http://127.0.0.1:6028/#scalars)、[训练及最终评测状态](http://127.0.0.1:18888/sac_15000_20261008/status.json)、[继承 actor 基线录像](http://127.0.0.1:18888/inherited_actor_check_20261008/replay.mp4)。旧曲线仍保留；新服务只绑定 127.0.0.1。录像元数据确认 1280x720、50 fps，HTTP 已验证，未作浏览器播放验收。Isaac/PPO 继续暂停，DINO 不启动。

## Isaac 视觉 SAC 迁移

2026-10-07 启动的 15000 轮 Isaac SAC 已于 2026-10-08 按用户要求暂停并完整保留，输出为 [实验计划](../../outputs/franka_isaac/sac_15000_ready_20261007/plan.json)，用户后台服务 `franka-isaac-sac-15000-ready.service` 已停止。完成到 4755，完整可恢复点为 4620。PPO 保持暂停，最后完成到 1307，完整可恢复点为 1200；DINO 不启动。

保留 Panda 夹具，使用原任务 4 cm、0.1 kg 方块、随机 XY 范围和 z=0 地面。前视相机严格复用 MuJoCo 的位置、OpenGL 姿态及 45 度垂直视场；实际第二路 `handcam_depth` 的安装关系映射到 Panda 手掌，垂直视场 62 度。两路真实渲染 1280x720，再缩到 128x128；actor 只接收两路 RGB 和七维机器人状态，不接收物体/目标坐标，不使用 ManiSkill 或 VLA。本地缓存官方 Panda 的 14 个 USD/网格依赖，避免 Ray 远程加载失败。

奖励仍为 $0.3\exp(-20d)+0.7\operatorname{clip}(\Delta z/0.2,0,1)$。动作是累积 XYZ 位移增量乘 0.1、按原工作空间裁剪，以及 0..1 的累积夹爪控制；控制频率 50 Hz、物理步长 0.002 s、回合 100 步。SAC 的 algorithm、actor、rollout、weight_syncer 与 MuJoCo 工作区基线一致；没有加载 MuJoCo 策略。PhysX、Panda 夹具、DLS IK/PD 控制和接触力定义与原仿真不同，不宣称物理等价。

迁移检查已通过五项回归测试、双路原始截图与动作/自动重置检查。修正地面后的两轮真实 SAC 生成完整 checkpoint：206 个模型张量有限，actor/critic 优化步数分别为 2/8，replay 保留两路 uint8 图像和 state7。短测 checkpoint 的单独 100 步评测也已验证加载、冻结骨干和录像输出，成功率为 0，仅证明评测链路可用。此前缺地面的短测以及首个正式启动与额外评测实例并发造成的 OOM 均保留，不参与新正式实验；新训练启动前已停止额外评测并释放显存。当前还未验收学习能力。

正式训练冻结源代码、模型、资产和有效配置，结束后自动进行 seed 43、44 各 32 回合独立评测，核对冻结视觉骨干，并分别报告抬升 20 cm 和连续 0.5 s 双侧接触稳定握持。对照 MuJoCo SAC 的 63/64 抬升、31/64 稳定握持；各 seed 抬升至少 80%、合计与基线差距不超过 5 个百分点且稳定握持不低于基线，才标为工程意义上相近。相同异步更新预算并不保证相同采样量，单训练种子不代表统计等价。

本机服务入口：
- [真实训练曲线](http://127.0.0.1:6026/#scalars)
- [训练及最终评测状态](http://127.0.0.1:18886/sac_15000_ready_20261007/status.json)
- [对齐后的前视原图](http://127.0.0.1:18886/reference_cameras_20261007/front.png)
- [对齐后的腕部原图](http://127.0.0.1:18886/reference_cameras_20261007/wrist.png)

服务仅绑定 127.0.0.1；截图已实际解码、HTTP 链接已验证，未进行浏览器播放验收。训练由 [独立驱动](../../projects/franka_isaac/experiment.py)调用原生 RLinf 异步入口，物理评测由 [检查入口](../../projects/franka_isaac/check.py)运行实际 Isaac 环境。

## 官方 FrankaSim 独立复刻

先独立运行[官方视觉异步 SAC 配方](../../third_party/RLinf/examples/embodiment/config/frankasim_sac_cnn_async.yaml)，再经用户确认接入本工作区的项目层。独立阶段不调用 `run.py train`，也不重启旧的自定义视觉 Lift/stack 实验。[复刻驱动](../../tools/frankasim_reference.py)直接调用官方 `train_async.py`，不另写学习算法。

任务是 MuJoCo 的 `PandaPickCubeVision-v0`：随机桌面位置抓取立方体，官方成功判据为比初始位置抬高 20 cm。策略输入是前视和腕部 RGB，以及七维机器人状态；视觉状态不含物体或目标坐标。使用轻量预训练 ResNet10 的冻结骨干和可训练的 SAC 策略/Q 网络，不用 ManiSkill 或 VLA。

每个控制步 0.02 s，动作是三维末端位移增量（缩放 0.1）和夹爪控制增量（缩放 1），旋转由现有控制器保持。原奖励为 $0.3\exp(-20d)+0.7\operatorname{clip}(\Delta z/0.2,0,1)$，其中 $d$ 为夹爪与立方体的距离；没有新增训练奖励。

已记录的适配：两路相机实际渲染 1280×720，再在环境包装层缩为 128×128；官方回放窗口/缓存从 20000 条轨迹缩为 256 条（原实现单个图像字段预分配约 252 GB，本机内存不足）；保存间隔改为 420，以匹配原有 210 轮评测间隔。缩小回放会影响采样分布，所以这是硬件适配复刻，不是完全相同参数的复刻。batch size 256、SAC 更新规则、动作和奖励保持官方配方。

独立验收使用种子 43、44，各 32 回合，分别要求官方成功率至少 80%，并额外要求稳定抓持成功率至少 80%：抬高 20 cm、双侧接触力各大于 0.1 N、速度低于 0.1 m/s、夹爪距离小于 6 cm，连续保持 0.5 s；同时核对冻结骨干权重未改变。两轮短测只验证训练、检查点和评测链路，不代表已学会抓取。正式训练预定 15000 轮，只有独立验收通过才进入工作区复现。

隔离仿真依赖放在 `.venv-frankasim`；通过只读路径复用原工作区的 Ray 等核心依赖，不修改原 Isaac 环境。每次实验使用新的输出目录，保存配置、源文件和模型哈希、日志、曲线、检查点、独立评测 JSON 和录像：

```bash
.venv-frankasim/bin/python -m pytest tests/test_frankasim_reference.py -q
.venv-frankasim/bin/python -m tools.frankasim_reference prepare --epochs 15000 --output outputs/frankasim_reference/my_reference
.venv-frankasim/bin/python -m tools.frankasim_reference execute --output outputs/frankasim_reference/my_reference
```

`status.json` 的 `accepted` 才表示上述物理验收达标；`acceptance_failed` 必须分析，`failed` 是运行异常。独立模式验收通过后驱动仅标记 `awaiting_workspace_replication`，不会自行实现或启动第二阶段。

2026-10-07 的[独立实验结果](../../outputs/frankasim_reference/reference_20261007/status.json)：15000 轮正常完成，种子 43/44 的官方成功率分别为 29/32（90.6%）、27/32（84.4%），合计 87.5%；稳定抓持分别为 22/32、26/32，合计 75%，完整验收未通过。视觉骨干未变。官方抬升指标达标不等于稳定抓持达标。证据：[真实曲线](http://127.0.0.1:6022/#scalars)、[两回合策略录像](http://127.0.0.1:18884/reference_20261007/replay.mp4)、[独立评测](http://127.0.0.1:18884/reference_20261007/status.json)，HTTP 链接仅本机可用。

## FrankaSim 工作区复现

用户确认将官方指标达标的版本迁移后，新增[项目配方](../../projects/frankasim/train.yaml)和[运行预检](../../projects/frankasim/project.py)。环境和 CNN 使用同一原生实现；入口通过 `runner.async_training: true` 调用 RLinf 的异步训练入口，默认同步项目和评测入口保持不变。不复制已训练策略，不加载参考实验的检查点，仍从预训练视觉骨干和新初始化的可训练网络开始。

与独立实验保持相同任务、奖励、四维动作、双路 720p 渲染、七维机器人状态、SAC/DrQ/10 Q 头、batch 256 和回放窗口 256。测试逐项比较环境、模型、算法及分布式配置；共享总测试中旧 `my_robot`/`piper_throw` 视觉用例缺少 RGB 输入仍失败，没有修改这些任务或掩盖测试缺口。

两轮[工作区真实短测](../../outputs/frankasim/smoke_workspace_20261007/status.json)已完成：actor/critic 优化器实际更新，参数与梯度有限，原生检查点及 64 回合评测完成，冻结骨干未变；成功率 0%，只证明端到端链路。正式[工作区实验](../../outputs/frankasim/workspace_20261007/plan.json)重新训练 15000 轮，结束执行同一套独立验收，并在状态中分别记录 `official_accepted` 与完整稳定抓持验收结果。

```bash
RLINF_PYTHON="$PWD/.venv-frankasim/bin/python" bash run.sh check frankasim
RLINF_PYTHON="$PWD/.venv-frankasim/bin/python" bash run.sh train frankasim
.venv-frankasim/bin/python -m tools.frankasim_reference prepare --workspace --epochs 15000 --output outputs/frankasim/my_workspace_run
.venv-frankasim/bin/python -m tools.frankasim_reference execute --output outputs/frankasim/my_workspace_run
```

前两个命令使用项目入口；后两个是包含冻结记录和自动独立验收的替代实验方式，不要同时启动两份训练。新实验用新目录。运行时不要编辑计划中锁定的项目、入口或驱动源码。

## FrankaSim 逐步消融

2026-10-07 用户授权按顺序做消融，先 [PPO + ResNet10 配方](../../projects/frankasim/ppo.yaml)，性能相近或更好后才切换 DINO。当前 DINO 阶段没有启动，旧自定义 Isaac 视觉任务也没有重启。

PPO 保留同一 FrankaSim 任务、奖励、七维机器人状态、四维有界动作、双路实际 720p 相机和冻结 ResNet10。1062 轮 × 256 步 × 1 环境 = 271872 条训练交互，与工作区 SAC 回放记录的累计样本量相近；SAC 的 15000 次更新不能直接视为相同数量的 PPO 轮数。回放中的累计样本不包含可能未消费的异步队列数据，因此不是严格相同的模拟器总交互计数。

这是算法配方对照，而不是仅替换一个字符串：PPO 使用 GAE、clip、价值头和旧策略信息，每轮同步策略（不是 SAC 的每 30 轮），采用原生 PPO 默认的可学习、状态无关标准差，初始值 exp(-0.5)。不使用 SAC 的 Q 集成/回放或 critic DrQ；编译/CUDA graph 暂时关闭。初始学习率仍为 3e-4、gamma 仍为 0.96，PPO 高斯 KL 目标 0.01 控制学习率。全部差异保存在实验计划，不能把结果简单归因于单个算法实现。

[项目策略适配器](../../projects/frankasim/policy.py)只在 PPO 模式生效。实测原生 CNN 对有界动作采样和回算的 logprob 不一致，因此保留压缩前的原始动作，统一概率与 Jacobian 修正，并测试动作饱和情况。原生视觉池化的 0.1 dropout 还会导致未更新参数时 eval→train 的 KL 非零，PPO 模式禁用这两处随机 dropout。冻结视觉骨干、物理动作和原 SAC 策略不变。此问题不是旧 DINO 失败的已证实原因。

两阶段均使用种子 43/44 各 32 回合独立评测。进入 DINO 的工程门槛：两个种子各至少 80% 官方抬升成功率；合计成功率比 SAC 基线 63/64 低不超过 5 个百分点；合计稳定抓持率不低于 SAC 的 31/64；冻结骨干验证通过。`comparison_passed/awaiting_dino_ablation` 只代表可继续消融，不代表完整物理验收或统计显著优于 SAC。原来每个种子稳定抓持率至少 80% 的完整验收仍单独记录。未过门槛时先分析 PPO，不提前换 DINO。

启动前四轮短测确认完整链路运行、冻结骨干不变，实际高斯 KL 为 0.0082/0.0133/0.0082/0.0059，训练没有卡在学习率下限。四轮后的两个独立种子成功率仍为 0%，不能称为策略已经学会。

2026-10-07 的[正式 PPO 结果](../../outputs/frankasim/ppo_ablation_20261007/status.json)：1062 轮、271872 条训练交互正常完成，耗时约 2 小时 1 分钟（含独立评测）。种子 43/44 各 32 回合的抬升与稳定抓持成功率均为 0%；平均回报分别为 18.886/19.977，最高抬升分别为 1.75/1.51 cm，未达到 20 cm。冻结骨干验证通过，最终检查点张量有限。状态为 `acceptance_failed/awaiting_analysis`，`error=null`，不是运行异常。

| 同任务独立评测 | 官方抬升成功 | 稳定抓持 0.5 s |
| --- | --- | --- |
| 工作区 SAC + ResNet10 | 63/64（98.4%） | 31/64（48.4%） |
| 本次 PPO + ResNet10 | 0/64（0%） | 0/64（0%） |

真实 PPO 曲线有 1062 个更新点：训练每步奖励首 10 轮均值 0.00434、末 10 轮 0.18534，但训练与 18 次周期评测的成功率一直为 0。最终高斯 KL 为 0.00846，末 10 轮均值 0.01431；价值解释方差末 10 轮均值 0.854。奖励增加不代表抓取学会，结果与停留在靠近物体的局部行为相符，但尚未证明具体失败原因。压缩后动作的估计熵由约 2.33 降至 -4.79；最终原始高斯标准差仍约 0.50–0.55，不能仅凭熵下降断言原始标准差坍塌。

**本次 PPO 配方在匹配预算下没有达到 SAC 相近性能，DINO 晋级门槛未通过，因此未切换 DINO。** 这不是 PPO 普遍不能学习的证明，也不是旧 DINO 失败原因的结论。下一步应先对 PPO 的探索、动作压缩饱和和采样/更新预算分别做单项对照，保留任务、奖励、视觉骨干与已有失败结果；尚未启动这些后续实验。

证据（仅本机）：[PPO 真实曲线](http://127.0.0.1:6024/#scalars)、[SAC 曲线](http://127.0.0.1:6023/#scalars)、[PPO 两回合录像](http://127.0.0.1:18885/ppo_ablation_20261007/replay.mp4)、[完整评测和比较](http://127.0.0.1:18885/ppo_ablation_20261007/status.json)。视频为 H.264、1280×720、50 FPS、4 秒，媒体 HTTP 200；当前浏览器工具不可用，未完成浏览器播放验收。

```bash
RLINF_PYTHON="$PWD/.venv-frankasim/bin/python" bash run.sh check frankasim --config ppo
.venv-frankasim/bin/python -m tools.frankasim_reference prepare --workspace --config ppo --epochs 1062 --output outputs/frankasim/my_ppo_ablation
.venv-frankasim/bin/python -m tools.frankasim_reference execute --output outputs/frankasim/my_ppo_ablation
```

执行器结束后自动完成同一套独立评测、比较和录像；只标记下一阶段状态，不会自行实现或启动 DINO。进入 DINO 后还需重复实验，才能判断编码器变化是否导致差异；任务、控制自由度、训练预算等旧实验差别需分别测试。

### PPO 延长至 15000 轮

用户于 2026-10-07 授权继续 PPO，目标为**总计 15000 轮**。从原生 `global_step_1062` 完整检查点恢复模型和优化器，追加 13938 轮、3568128 次训练交互；累计预算为 3840000 次交互，而不是新增 15000 轮。任务、奖励、动作、双路相机、冻结 ResNet10 及 PPO 超参数保持不变，仍未切换 DINO。环境重新初始化，不保证与不停机连续训练产生相同轨迹。

这次是用户授权的**训练预算延长**，不再与 SAC 的约 27.2 万条回放样本等交互量，也不是 SAC/PPO 同轮数即同优化量的比较。保持之前失败实验及评测原样，续训在新目录单独保存计划、恢复检查点哈希、日志和独立评测。`--epochs` 表示总目标，`--resume-dir` 必须指向带有 `actor/dcp_checkpoint` 的完整 `global_step_*` 目录，而不是单独的策略权重文件。

```bash
.venv-frankasim/bin/python -m tools.frankasim_reference prepare --workspace --config ppo --epochs 15000 --resume-dir outputs/frankasim/ppo_ablation_20261007/runs/frankasim_ppo/checkpoints/global_step_1062 --output outputs/frankasim/my_ppo_15000
.venv-frankasim/bin/python -m tools.frankasim_reference execute --output outputs/frankasim/my_ppo_15000
```

两轮真实恢复验证已完成：32 个优化器参数的 Adam step 从 4248 增至 4256，日志继续记录第 1062/1063 轮，高斯 KL 为 0.0182/0.0075，模型与完整评测链路正常。正式长训练仍从原第 1062 轮检查点开始，不混入恢复短测。按此前速度预计需要约一天，实际取决于本机负载。最终仍使用种子 43/44 各 32 回合独立评测；成功与稳定抓持门槛不降低，训练正在运行时不能宣称策略达标。

正式[续训计划](../../outputs/frankasim/ppo_15000_resume_20261007/plan.json)已冻结并由用户后台服务 `frankasim-ppo-15000.service` 运行。证据入口（仅本机）：[15000 轮续训曲线](http://127.0.0.1:6025/#scalars)、[续训状态](http://127.0.0.1:18885/ppo_15000_resume_20261007/status.json)。原 1062 轮 PPO 曲线仍保留在 6024，SAC 在 6023。这里只记录已启动，不预报最终能力；当前尚无 15000 轮最终评测或新策略录像。

## 先看任务再训练

新增的 `view` 和 `record` 直接使用项目的 Isaac 任务配置，不创建 RLinf 训练器、不更新模型。当前支持本工程的 Isaac 状态任务，VLA 模板不走这个预览入口。

推荐先检查配置，再打开一个环境观察：

```bash
bash run.sh check cartpole
bash run.sh view cartpole --steps 0
```

`view` 打开原生 Isaac 窗口，默认一个环境、零动作，按接近实时的速度运行。`--steps 0` 表示持续预览，直到关闭窗口、按原生停止按钮或在终端按 Ctrl+C；不传时默认运行 600 个控制步。原生暂停/继续按钮可用。可以在视口调整观察角度，在 IsaacLab 面板查看动作、观测、指令、奖励和终止项的调试可视化；面板遮挡场景时可移动或关闭。

零动作不是关闭物理，也不保证机器人不动：动作仍经过任务定义的缩放与控制器，重力、接触、默认关节目标和自动重置照常生效。它适合检查资产、初始姿态、场景和控制器默认状态，不代表策略已经学会任务。

再用小幅随机动作检查关节响应、动作方向和重置行为：

```bash
bash run.sh view cartpole --actions random --random-scale 0.1 --steps 600
bash run.sh view go2 --config platforms --num-envs 1 --actions random --steps 300
```

随机动作在每步独立采样，默认范围为策略动作空间的 `[-0.1, 0.1]`，之后仍由任务动作项缩放。这不是合理的运动策略，摔倒或任务失败并不必然说明环境配置错误。新项目也使用相同命令，只需替换项目名。

没有桌面、希望保留录像或方便对照时，使用无界面的 `record`：

```bash
bash run.sh record cartpole --steps 600 --camera-eye 5 -7 4 --camera-target 0 0 2
bash run.sh record piper --steps 300 --camera-eye 1.1 -1.2 1.0 --camera-target 0.3 0 0.2
```

视频为 1280×720，帧率取自任务控制频率；运行 600 步不一定是 600 秒。每次输出到对应实验目录下的 `previews/<时间戳>/`，包含视频、首帧和 JSON 报告，不覆盖已有结果。终端会打印完整路径。报告记录动作来源、观测/动作维度、步数、自然终止、超时重置和环境关闭结果，不把诊断结果当成任务成功率。窗口模式也保存报告，但不录视频。

已有策略时，可直接加载检查点观察动作。例如本工程的两轮平衡杆验证模型：

```bash
bash run.sh view cartpole --steps 0 --checkpoint "$PWD/outputs/cartpole_native_validation/cartpole_native_validation/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
```

传入 `--checkpoint` 自动选择策略动作，并严格加载当前项目模型的权重；不兼容的网络或维度会报错。也可将 `view` 换为 `record` 保存回放。仅传 `--actions policy` 而未提供权重时，会使用项目配置的初始权重；如果也没有配置初始权重，则是未训练网络，并明确打印警告。两轮验证模型只证明训练接口可用，实际回放仍会很快失败，不能视为已学会平衡。

完整参数见 `bash run.sh view --help` 或 `bash run.sh record --help`。注意：

- 预览采用 `env.train` 的任务配置，默认不缩短回合；`--steps` 是总预览步数，回合上限由 `env.train.max_episode_steps` 决定。
- 重点检查物体尺寸与位置、初态接触、关节动作方向、奖励响应，以及成功/失败时是否按预期终止和重置；画面能显示只是第一步。
- 修改任务定义后重新启动预览；它不是运行中自动重载的编辑器。
- `--num-envs` 支持 1–16，建议从一个开始。小规模预览使用原生 USD 克隆，避免单环境 Fabric 克隆报错；训练的克隆设置不变。
- `view` 需要桌面会话中的 `DISPLAY`；纯终端或远程无桌面环境使用 `record`。不要同时启动多个重型仿真任务。
- 这两个入口都不会控制真实机器人。为了避开当前 Isaac 的应用关闭问题，完成环境与视频清理后直接退出进程，不宣称已验证 Kit 的完整正常销毁流程。

已实跑验证平衡杆随机动作录制 120 帧、两次重置，检查点回放 60 帧且权重 SHA 不变，以及 Piper 零动作录制 60 帧；全部视频完整解码且非空。原生窗口关闭事件和 Ctrl+C 退出均通过检查。实现位于[预览工具](../../tools/preview.py)。

## 项目开发规范

[CartPole 平衡杆项目](../../projects/cartpole)是最小完整示例。下面的文件布局只是组织约定，不是叠加在 RLinf 之上的新协议。

| 项目文件 | 职责与要求 |
|---|---|
| [project.py](../../projects/cartpole/project.py) | 提供 `register()`，通过官方 `RLINF_EXT_MODULE=projects.NAME.project` 加载；使用 RLinf 原生接口注册模型构造函数和环境。仿真模块应延迟导入。 |
| [train.yaml](../../projects/cartpole/train.yaml) | 原生 Hydra/RLinf 配置，包含任务、模型、维度、PPO、资源分配和实验名。其他配方也放在同一项目内。 |
| [env.py](../../projects/cartpole/env.py) | Isaac 任务定义：场景与资产、动作、观测、奖励、终止、重置和课程。任务类只能在 AppLauncher 启动后加载。 |
| [policy.py](../../projects/cartpole/policy.py) | 通过 `build_model(config, torch_dtype)` 返回模型；可以调用上游的模型专用构造函数，也可以自己实现网络。 |
| [test_policy.py](../../projects/cartpole/test_policy.py) | 项目自己的测试，包括动作采样、对数概率重算、梯度和检查点重载等关键行为。 |

Python 包标记文件由生成器自动创建。资产定义、奖励函数和测试可按需要在项目内拆分。没有强制的项目描述类、发现元数据对象、固定文件数量，也不需要给每个项目编写训练循环。

手写 PPO 策略需要实现 RLinf 的 [BasePolicy](../../third_party/RLinf/rlinf/models/embodiment/base_policy.py) 接口，并使用 `nn.Module` 管理网络。[已验证的示例](../../projects/cartpole/policy.py)实现了 `forward`、`default_forward` 和 `predict_action_batch`，包含高斯策略网络、可训练的对数标准差以及独立价值网络。权重从随机初始化开始，不依赖预训练策略。

当前状态 PPO 配方的观测为 `{"states": [batch, obs_dim]}`；动作和采样时的对数概率形状为 `[batch, 1, action_dim]`；采样时的价值估计形状为 `[batch, 1]`。采样必须提供 `forward_inputs`，使训练阶段能在保留梯度的情况下重算相同动作的对数概率。这些形状只适用于本状态 PPO 示例，不是 VLA 或所有 RLinf 算法的统一限制。

配置化场景使用的 [RLinf 适配层](../../shared/integrations/rlinf/isaac.py)支持 `[batch, H, action_dim]` 的有限值动作，`H >= 1`；场景定义不限制策略的块长。适配层复用原生 RLinf 的动作块循环，返回逐步观测、信息及 `[batch, H]` 奖励与结束标记。每个环境在块内首次结束后，后续奖励填零，结束标记按 RLinf 约定汇总到最后一列；块末 `final_observation` 和回合指标保留该环境的首个真实终态，供超时 bootstrap 和日志使用。Isaac 仍批量推进剩余物理步；已经复位但又执行过剩余动作的环境在块末重新初始化，末步刚完成原生复位的环境不重复复位。此约定不是逐环境暂停仿真，也不改变原生物理步的事件或课程副作用。现有状态策略仍输出单步；多步策略需自行匹配配方的 `num_action_chunks`、动作维度和训练接口。本次仅验证 CPU 传输与回合契约，尚未实测 VLA 或多步 Isaac 物理运行。

机器人项目默认调用 RLinf 的 MLP 专用构造函数。使用 `new ... --policy scratch` 会改为把手写通用网络复制到新项目中。VLA 使用自己的模型专用构造函数；不要在已经注册的模型构造函数中递归调用全局模型分发入口。

## 已有任务

下列九个 Isaac 状态配方均于 2026-10-03 通过真实 GPU 的重置、步进、终态和退出检查：四个环境、运行 16 步，回合上限设为八步，共八次重置。这是未训练策略的集成检查，不代表任务能力或训练收敛。另新增的网格任务使用 CPU 环境，验证记录单独列出。

| 项目 | 配方 | 状态维度 / 动作维度 | 验证记录 |
|---|---|---|---|
| [cartpole](../../projects/cartpole) | `train` | 4 / 1 | [接口检查](../../outputs/cartpole/checks)，另有实际 PPO 训练与评测 |
| [gridworld](../../projects/gridworld) | `train` | 16 / 1 个离散动作，4 种选择 | [CPU 接口检查](../../outputs/gridworld/checks)、[固定地图验收](../../outputs/gridworld_native_20261003_141653/acceptance.json) |
| [go2](../../projects/go2) | `train`, `platforms` | 235 / 12 | [楼梯](../../outputs/go2/checks)、[高平台](../../outputs/go2_platforms/checks) |
| [m20](../../projects/m20) | `train`, `route` | 243 或 245 / 16 | [基础任务](../../outputs/m20/checks)、[路线任务](../../outputs/m20_route/checks) |
| [piper](../../projects/piper) | `train`, `obstacles` | 33 / 7 | [抓取](../../outputs/piper/checks)、[避障抓取](../../outputs/piper_obstacles/checks) |
| [bimanual_reach](../../projects/bimanual_reach) | `train` | 56 / 14 | [双臂到达检查](../../outputs/bimanual_reach/checks) |
| [loco_manip](../../projects/loco_manip) | `train` | 271 / 22 | [移动倒立摆检查](../../outputs/loco_manip/checks) |
| [franka_stack_vla](../../projects/franka_stack_vla) | `probe`，暂无训练配方 | 冻结 SmolVLA：三路图像、6 维 SO-101 状态与动作 | 已完成短回放，尚未成功堆叠 |

Go2 和 OpenArm 的任务基类来自已安装的 Isaac Lab。M20 的任务代码和资产配置不再导入厂商训练包。Piper、地形、奖励和倒立摆定义由旧项目的任务代码迁入。共用的任务函数放在 [shared](../../shared) 中，这一层不是新的训练调度框架。

本次迁移不以兼容旧检查点或保持旧训练分布为目标。例如，双臂 Reach 使用原生 56 维状态任务，不是旧的已训练 54 维相对误差策略；移动操作任务改为继承独立 M20 任务及原生 Isaac MDP，不再继承旧厂商训练基类；观测噪声遵循任务自身配置，不再被全局关闭。旧冻结模型及其成功率不能用来证明这些新配方已经达标。

以下历史实验仍保留在[旧项目目录](../../../robot-platform/source/robot_platform/robot_platform/projects)中，不将其标为已经迁好的 RLinf 配方。该链接仅供查阅历史，不是运行依赖：

| 历史实验 | 单独保留的原因 |
|---|---|
| 视觉装配 | 使用 SB3 循环策略 PPO，包含专门的图像、生命周期和碰撞处理；本工程尚未迁入对应的原生 RLinf 视觉装配适配器。 |
| 视觉插入 | 使用监督式行为克隆、固定工具场景及专门的数据和评测流程；不是新的 Franka VLA 任务。 |
| 双臂插入与真实接触插入 | 包含 IK、抓持与力诊断、分别训练的控制器；并非统一的 RLinf 训练流程。 |
| MuJoCo 与跨仿真器迁移实验 | 使用其他仿真器和残差策略，不在本 Isaac 状态适配器的覆盖范围内。 |

这些实验的源码、原始权重、产物和失败记录均保留，但不是新工程的依赖。旧 M20 厂商配方、回放与检查点转换工具、各历史实验的调参预设，也不属于本工程必须兼容的内容。用户已有的 `my_balance` 项目未修改。

## 运行条件与边界

维护者负责准备兼容的系统级 Isaac 运行环境及模型专用依赖。本目录已经包含自己的 RLinf 源码、项目虚拟环境和三组机器人资产副本，开发者新增常规项目时无需修改旧工程或系统安装。项目独立不意味着任意新模型都不再需要额外依赖或硬件资源。

| 设置项 | 当前目录布局下的默认值 |
|---|---|
| `RLINF_PYTHON` | 本项目的 `.venv/bin/python`，继承系统已安装的 Isaac 软件包 |
| `RLINF_ROOT` | 本项目的 [third_party/RLinf](../../third_party/RLinf) |
| `M20_USD` | 本项目的 [assets/m20/M20.usd](../../assets/m20/M20.usd) |
| `RP_PIPER_USD` | 本项目的 [assets/piper/piper.usd](../../assets/piper/piper.usd) |
| `RP_M20_PIPER_PENDULUM_USD` | 本项目的 [assets/m20_piper_pendulum/M20_Piper_pendulum.usd](../../assets/m20_piper_pendulum/M20_Piper_pendulum.usd) |

这些路径可通过环境变量覆盖，无需修改上游代码；资产定义也由项目维护。三组资产已按完整目录复制，USD 子层和网格引用保持原样，未使用指向旧工程的符号链接。Go2 和 OpenArm 使用 Isaac 原生资产路径与缓存，不依赖旧工程；在其他机器上首次运行时可能需要获取这些原生资产。

项目虚拟环境已重新创建，并离线复制现有依赖、修正命令脚本路径；其基础解释器仍是 `/home/jz/anaconda3/envs/isaaclab`，没有复制整个系统级 Isaac 安装。标准材质 `OmniPBR.mdl` 也由该安装提供。本次没有安装完整 VLA 环境。更换机器或基础解释器时，需要重新部署兼容运行环境；Python 虚拟环境不能视为任意位置都能直接运行的独立系统镜像。

已验证的运行组合：Python 3.11.16、Torch 2.7.0+cu128、Ray 2.48.0、Isaac Sim 5.1.0、Isaac Lab 2.3.2.post1，以及 RTX 5080 16 GB。RLinf 基线固定在 `77ed948d68f07f55b4bc433b7643a409b2cd1adb`。为接入非 Isaac 网格任务，本地副本仅在[环境选择入口](../../third_party/RLinf/rlinf/envs/__init__.py)增加 `gymnasium` 枚举、`REGISTER_GYMNASIUM_ENVS` 注册表和查找分支；这是尚未上游化的本地补丁，不是该提交原有的通用 Gymnasium 支持。PPO、worker、动作转换和采样流程未改动，升级 RLinf 时需要保留或重新适配这个小补丁。

[IsaacStateEnv](../../shared/isaac_state.py) 继承原生 `IsaaclabBaseEnv`，复用 RLinf 的子进程通信，只补充状态张量、重置前的真实终态观测，并防止重复自动重置。它保留任务原有的终止与奖励含义，不把任意正奖励视为任务成功。当前状态适配器要求自动重置、动作块长度为一、保留原生终止信号，且不把奖励转换为相邻步的差值。训练适配器仍默认无界面、不提供图像观测；`view`/`record` 复用其任务构造函数，单独启用旁观窗口或录像，不把状态策略改成视觉策略。

当前 Isaac 版本的原生应用关闭流程可能卡住。状态子进程会先关闭环境，再通过小型 `ProcessExit` 代理退出，而不是调用 `SimulationApp.close()`。退出时仍可能出现原生 CUDA IPC 生产者提前退出警告；进程成功退出不代表完全没有警告，也不代表已验证 Kit 的完整正常关闭流程。启动失败的处理仍受原生通信机制限制。每次只运行一个仿真检查。

## VLA 接入

[SmolVLA 项目](../../projects/franka_stack_vla/README.md)包含冻结模型推理、残差策略接口和原生 Isaac 短回放，目前没有训练配方。通过专用环境运行 `RLINF_PYTHON="$PWD/.venv-smolvla/bin/python" bash run.sh probe franka_stack_vla`，默认使用 SO-101 白桌配置，保存三相机与全景录像及任务结果。已完成真实回放，但尚未成功堆叠；关节标定和相机对齐仍存在限制。

`probe` 是新增的可选项目行为验证钩子，不是所有 policy 的统一评测器。目前只有该项目提供此钩子。代码测试使用 `test`，支持通用预览接口的项目使用 `view`/`record --actions policy`，原生 RLinf 评测使用项目自己的 `evaluate` 配方。统一命令入口不代表 VLA 分布式训练接口已经验收。

## 验证记录

- 工作区测试覆盖所有配方、全新进程中的原生扩展加载、不导入旧框架、终态观测与避免重复重置、调用上游入口、预览参数、零动作录制和窗口关闭竞态；项目测试覆盖手写和上游模型的梯度与重载检查。
- 上表九个状态配方均通过 GPU 检查；不据此宣称新机器人策略已经收敛或通过独立能力验收。
- [手写策略的实际 PPO 日志](../../outputs/cartpole_native_validation/train.log)：两轮训练，64 个环境 × 32 步 × 2 轮，共 4096 次状态转移。两个原生检查点的数值均有限，其中九个策略与价值网络张量发生更新。
- [原生独立评测的实际日志](../../outputs/cartpole_native_evaluation/evaluate.log)：成功重载第二轮检查点，完成 16 个八步回合，没有优化更新。这是短程集成测试，不是完整回合长度下的平衡杆任务验收。
- 恢复后的旧工程通过 64 项相关测试。旧资产、检查点及用户任务定义均未替换。

## 独立搬迁验证

2026-10-03 已从旧工程内部移动到 `~/ws/rlinf-workspace`：

- 整个项目搬移前后内容摘要一致，已有输出和检查点未改写。
- Piper、M20 和复合倒立摆资产副本均与原目录逐文件一致；旧工程原件保留。
- USD 依赖检查无未解析引用；所有项目资源均位于本目录，唯一外部材质依赖是系统 Isaac 自带的 `OmniPBR.mdl`。
- 使用新目录自己的虚拟环境和 RLinf 副本，七项公共测试及一项策略测试全部通过。
- 在禁止读取旧工程的进程中，原生模型构造、动作输出和三组资产默认路径检查通过；新目录内没有符号链接指向旧工程。
- 本次搬迁没有重新启动训练或仿真。上节 GPU、PPO 与评测成绩是此前已经完成的记录，历史日志中的原始绝对路径按原样保留。
