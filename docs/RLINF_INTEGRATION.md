# RLinf 接入边界与历史补丁

2026-10-19：统一入口默认使用 `third_party/RLinf-upstream`，基于固定提交
`77ed948d68f07f55b4bc433b7643a409b2cd1adb`。此目录没有本地源码修改。
本机用 `git -C third_party/RLinf worktree add --detach ../RLinf-upstream HEAD` 创建；
换机可在该目录独立 clone 官方仓库并 checkout 上述提交。运行所需的 Python、Torch、Ray、Isaac 依赖仍需安装。

## 新模板如何使用原生接口

- 环境通过原有 `REGISTER_ISAACLAB_ENVS` 注册，模型通过 `register_model` 注册。
- `shared/integrations/rlinf/observations.py` 将命名 Tensor 字典放入原生 `states` 字段；图像、射线、关节保持形状和 dtype，没有统一展平。
- `task_descriptions` 是原生传输元数据，也让 RLinf 能推断批次数；不进入用户的对象观察。
- `ObjectPolicy` 解包 `states` 并构造 `Obs`。训练轨迹的 `forward_inputs` 仍按原生批处理要求使用平铺命名 Tensor 字典。
- `state_ppo.yaml` 的原生 `sampling_params` 区分训练与评估；共享模型读取 `do_sample`，直接调用也支持显式 `mode`。不修改 rollout worker。
- 默认模板使用原生学习率配置，移除 `adaptive_kl_target`，不另外实现 actor/优化循环。旧实验的自适应学习率结果不可当作新配方结果。
- Isaac 终态快照与自动复位接入仍在共享 adapter 中，复用原生循环；不要求项目维护 env/worker。
- 检查点保留数量使用已有的共享 Runner 子类；启动入口选择该子类，未修改第三方类或训练循环。

## 保留的历史源码改动

旧目录 `third_party/RLinf` 没有被重置。下列变动是历史实验的实际依赖，不是新模板要求；
运行历史配方需要先阅读其 README，并明确设置 `RLINF_ROOT="$PWD/third_party/RLinf"`。
不能把它们简单变成全局 monkey patch 来宣称不依赖补丁。

| 修改位置 | 用途 | 新路径或处理方式 |
| --- | --- | --- |
| data/schema/embodied_types.py | 自定义观察透传、批次推断 | 新模板改用原生 states 字典，无需补丁 |
| workers/rollout/hf/huggingface_worker.py | 自定义模型接收 train/eval mode | 新模板使用原生 do_sample 参数 |
| envs/sim/isaaclab/isaaclab_env.py | 保留原始 infos | assembly adapter 已自行接入 step/metrics，无需补丁 |
| envs/__init__.py | 新增 gymnasium 注册入口 | gridworld 仍依赖历史副本；尚未迁移，不冒充 Isaac 后端 |
| algorithms/utils.py、workers/actor/embodied_fsdp_actor_worker.py | 自适应 Gaussian KL 学习率 | 新模板移除该扩展；旧配方在干净依赖上明确报错 |
| runners/async_embodied_runner.py | 按 replay 样本预算停止 | 旧异步 SAC 实验专用，干净依赖不支持该参数 |
| workers/actor/async_fsdp_sac_policy_worker.py | 每轮等待新样本配额 | 同上，干净依赖不支持该参数 |
| workers/actor/fsdp_sac_policy_worker.py | 恢复 critic 更新计数 | 历史 SAC 续训路径，干净入口拒绝该续训配方，尚未迁移验收 |
| envs/sim/frankasim/frankasim_env.py | 渲染尺寸与图像缩放 | 旧 MuJoCo 图像配方依赖，干净入口明确拒绝这些参数；需后续环境子类适配 |
| tests/unit_tests/test_worker.py | 历史 worker 扩展测试 | 留在历史副本，不当作干净依赖的验收 |
| toolkits/standalone_eval_scripts/openpi/* | 专用评估工具与说明 | 与维护中 assembly 模板无关 |

当前只保证维护模板走干净依赖。SFT、MuJoCo/VLA、历史 SAC 的能力继续以项目说明为准；
本轮没有宣称整个工作区的所有历史实验都已摆脱补丁。

## 观察历史窗口

可选配置 `control.components.<角色>.history: {观察路径: T}`（整数 ≥2，键须是该组件已选观察）。
未配置时行为与之前完全相同。

- **位置**：窗口保存在共享 `ComponentInput`（`shared/assembly/history.py` 的 `HistoryWindow`），
  它就是该字段的 `last` 张量 `[B, T, ...]`。被训练角色的原生观察项直接返回这个窗口，所以窗口随
  `states`/`forward_inputs` 存进 RLinf 轨迹，PPO 重算 logprob 时看到的输入相同，模型里没有隐藏状态。
  固定组件、`decode`/`initial` 和 `<角色>.input.*` 读的是同一份窗口。Isaac 与 mjlab 都经同一个共享
  运行时，不需要后端钩子。
- **顺序与补齐**：轴 1 为旧→新。回合内的第一帧会填满所有槽位（与 Isaac/mjlab `CircularBuffer`
  相同，测试中与 Isaac 实现逐步比对过）。`HistoryWindow.valid` 标出真实观察过的槽位。在线不另加
  有效位字段：被训练角色的真实帧数是 `min(回合步数+1, T)`，回合步数可由已有的 `simulation.time`
  和 `simulation.step_dt` 算出。
- **复位**：按环境清零并重新计数，不跨环境、不跨回合。被训练角色的窗口在一个决策步内只推进一次，
  重复计算观察不会多推。显式的部分复位只推进被复位的环境。终态快照会显示终止帧，但之后会恢复，
  不推进在线窗口。
- **频率**：每个组件按自己的周期推进窗口。planner 窗口里的帧间隔是 planner 周期，executor 窗口里的
  帧间隔是 executor 周期。决策之间的读取（如 `decode`）返回已存的窗口。
- **离线数据**：`history_windows(frames, T, episode=, step=)` 复用同一个类。帧必须按该组件的决策
  频率记录，函数不做重采样。回合变化或步号不连续时重新开窗，绝不跨缺失步拼接。
- **接口标识**：`history` 在场景 `control` 内，`project_signature` 会把它算进 `interface_sha256`。
  改变窗口长度、增减字段都会让旧检查点被拒绝，新训练必须写到独立目录。
- **未选原生 history_length 的原因**：原生窗口只存在于被训练角色的观察组，固定组件和 `decode`
  看不到。mjlab 终态快照用 `update_history=False`，会漏掉终止帧，与 Isaac 不一致。显式部分复位
  会给未复位环境多推一帧。原生窗口也没有有效位，离线数据还要另写一套。
- **尚未提供**：编辑器和 Blender 中没有窗口长度控件（取消选择观察时会同步删除对应条目）。
  也没有在线有效位字段。

## 维护约束

扩展优先顺序：原生配置与注册入口 → 共享环境/模型 adapter → 原生支持的子类或钩子。
若必须替换完整 worker/训练循环，先说明缺少的上游能力，不复制训练框架，也不静默忽略参数。
旧实验权重、快照、源码与训练记录保持原路径；升级依赖必须重新核对接口与运行结果。

## 验证

`tests/test_stock_rlinf.py` 在新进程中确认实际导入干净目录，验证原生 EnvOutput、PolicyInput
路由、图像 dtype/维度、轨迹批处理、确定性评估与一次本地梯度更新。
`tests/test_assembly_native_rlinf.py` 用 CPU 传输替身验证完整原生 adapter 的部分复位和终态保存。
这两者都不是 Isaac 物理测试，也不是 RLinf 分布式 PPO 训练成功。

本轮结果：215 项核心回归通过，60 项控制/编辑入口回归通过（两组有重叠，不相加）；
四个维护模板的 check 均通过。独立新进程测试还调用了原生 MultiStepRolloutWorker.predict，
确认原生采样参数实际到达对象策略。测试记录存于 outputs/framework_cleanup_20261019/。
检查时 GPU 仅余约 0.75 GB，本轮未启动 Isaac 物理测试或分布式训练；当前版本仍待这些实机资源验证。

## mjlab 原生 PPO 验收

mjlab 环境通过现有环境注册入口构建，底层使用 mjlab 原生 manager 循环。已完成 GPU 部分复位与最终观察测试、原生 RLinf rollout、两次 PPO 更新、检查点保存和加载后的 4 回合评估。记录见 [后端说明](SCENE_BACKENDS.md)。原生评估合并器会把 states 字典变成生产者列表，共享 observations.fields 将其按批次合并；训练的递归字典路径保持不变。未修改上游代码，也未安装全局 monkey patch。

该验收不包含策略学会任务、相机输入、策略渲染或历史 MuJoCo/VLA 项目迁移。

## SuperMimic 类世界模型方法

> **方法版本说明（2026-10-26）**：本节只针对公开
> [`Zhengtianjin/go2-supermimic`](https://github.com/Zhengtianjin/go2-supermimic)
> 与 arXiv 2610.28480 摘要所述的 **world model + imagined BPTT** 实现。用户指出
> 另有 `super-priv-humanoid` 的 **privileged teacher + LiDAR estimator** 路线；
> 该链接当前返回 404，未核对代码。若目标是后者，训练循环是“特权信息策略 → 监督
> estimator → student 蒸馏/PPO”，不是世界模型 BPTT，本节“不能直接复刻”的结论
> 不适用；接入评估需在拿到仓库后补写。

结论先说：**按公开 go2-supermimic 的 world-model 实现**，当前 RLinf 不能通过配置或
替换项目模型直接复刻 SuperMimic。
它不是一个 PPO 变体，需要“采集 → 拟合确定性世界模型 → 在想象轨迹上做 BPTT → 更新
actor/critic”的独立训练循环，而现有 RLinf 具身路径没有这个循环。

参考实现：[`Zhengtianjin/go2-supermimic`](https://github.com/Zhengtianjin/go2-supermimic)
提交 `054a13c`。它的训练循环在 `go2_supermimic/supermimic/train.py`（659 行），
配套 `state.py`（270）、`imagine.py`（293）、`buffer.py`（117）、`world_model.py`（97）、
`driver.py`（138），合计约 1.5k 行；单进程 `cuda:0`，不依赖分布式训练。

### 与当前 RLinf 的接口差异

| SuperMimic 需要 | RLinf 现状 | 差距 |
| --- | --- | --- |
| 逐 step 的 env 状态、动作、上一动作、接触/落地标记、地形 relief | `rlinf/data/storage/replay/buffer.py` 的 replay 按 `Trajectory` 存，服务 SAC/DAGGER；`EmbodiedRunner` 的 rollout 轨迹按 PPO 批处理组织 | 可复用存储机制，但没有 SuperMimic 的 `WindowArchive`、k-step 窗口和状态切片；需要新的 collector |
| 确定性世界模型 `s' = f(s, a, a_prev, xi, scan)`，k-step 自回归训练 | 没有 world-model loss 或 dynamics model worker；`rlinf/envs/sim/world_model/env.py` 是 Wan/OpenSora **视频生成**环境，不是可微状态动力学 | 需要新增模型与训练目标，不能把视频 world model 当成动力学模型 |
| 对 H=32 步想象轨迹 BPTT，奖励/终止在 PyTorch 内解析、可微 | 具身 actor worker 消费轨迹并做 PPO 损失（SAC 另有 worker）；项目 Task 是运行期只读评估，返回非可微标量 | 需要独立的 imagination 模块和可微奖励镜像；不能直接把 `task.py` 反传 |
| actor 用想象回报更新，critic 用 TD(λ) 拟合想象数据 | PPO 的 advantage/return 计算与 SAC 的 critic 目标都不是这个目标 | 算法层新增，不是超参差异 |
| 2048–4096 环境、40k 轮、每轮若干次世界模型更新 | 现有模板配方通常是 256–1024 环境、按 epoch 更新 | 计算规模与循环结构都不同 |

`sac`、`dagger`、`offline`、`rlt` 也不能替代：SAC/DAGGER 仍是 model-free；offline 是
监督式离线训练；RLT 不是状态空间世界模型。`WorldModelEnv` 的后端只生成画面并用奖励
模型打分，没有梯度穿过动力学。

### 可行路线

1. **独立接入层（推荐先做）**：在 `shared/integrations/supermimic/`（或项目内独立训练
   入口）实现 collector、世界模型、imagination 和 critic，复用现有 mjlab 环境、
   `run.py` 的项目/场景/快照与 TensorBoard；PPO 路径保持不动。参考实现本身就是单进程，
   16 GB GPU 的官方配置是 2048 环境、4096 想象批、400 窗口；本机 RTX 5080 16 GB 与之
   同级。这条路线能尽快验证方法，但训练循环不等同于“RLinf 已支持”。
2. **RLinf 原生 runner/worker**：新增 `WorldModelRunner` 与 world-model/imagination
   workers，并把逐 step 状态接入 replay。只有确实需要 Ray/FSDP 和统一分布式调度时才值得；
   否则会为了复用框架复制一套训练循环，违背 `docs/RLINF_INTEGRATION.md` 的维护约束。

### 复刻前必须先补的共享能力

- **状态**：70 维 `s`（base 高度、机体系重力/线速度/角速度、q/qd、hip/knee/foot 相对
  位置）、4 维 context（脚摩擦、COM 偏移）、地形 relief/height scan。当前对象观察只暴露
  关节与实体位姿，不暴露 hip/knee/foot 点；collector 可在环境进程内直接读 mjlab 实体，
  或先加共享的 world-model state 适配。
- **可微任务**：chase/dodge 的奖励与终止要在想象中可微。参考仓库也保留两份定义（env 奖励
  用于日志，`imagine.py` 用于训练），需要一致性测试；不能把只读 Task 当作训练目标。
- **历史与复位**：actor 需要 5 帧历史，世界模型需要回合边界。collector 能看到 reset，
  应在采集端维护历史；PPO 路径可用 `history` 窗口（见“观察历史窗口”），但世界模型的回合边界仍需另接。
- **地形查询**：参考的 height scan 由解析地形场生成；我们的地形是烘焙的 USD/MJCF 网格，
  需要射线或高度场查询。两后端都已有 `height_scanner`（`ray_hit_positions`），mjlab 尚无 LiDAR。
- **稀有事件与探索**：OU 探索、driver/expert 策略、rare-event archive、RunningNorm 都是
  采集/训练循环的一部分，不属于现有 PPO 配方。
- **权重与实验目录**：SuperMimic actor 输入是 350/430/735 维，当前 `go2_chase` PPO 是
  49 维；权重不可互换，新实验必须写独立目录并记录接口标识。

本轮没有实现上述训练循环，也没有宣称能复现参考策略。已验证的只是 `projects/go2_chase`
的 PPO 流程；参考仓库自己也说明 flat 从零训练约 7k 轮时仍常被杆击中，terrain 是多次
重启后的结果，因此短程 smoke 或损失下降不能当作方法复刻成功。

