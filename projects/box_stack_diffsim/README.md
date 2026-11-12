# box_stack_diffsim

两立方体堆叠场景 + 一条**不经过 RLinf** 的神经网络训练路径：用 Newton（Isaac Lab 3 环境自带的
`newton` / `warp`）做可微仿真，把整段轨迹的损失直接反传到策略，也就是 BPTT。
思路对应 SuperMimic 的「用动力学提供梯度」，只是把学出来的世界模型换成了 Newton 的可微物理。

场景本身仍用本工作区的公共定义描述：`scene.yaml` + `policy.py` + `task.py` + `observation.py`，
可以 `check`、可以 `view`，也可以交给 isaaclab3 后端训练。可微训练另走 `diffsim/`。

## 目录

```
scene.yaml          公共场景：桌面、两个立方体、Franka、堆叠目标
assemblies/arm.yaml Franka 组合（带 tcp 帧与腕部相机）
policy.py           公共 planner/executor（RLinf 路径用）
task.py             堆叠评价：上层立方体落在下层之上
diffsim/            Newton 可微训练
  scene.py          scene.yaml -> Newton 模型（静态设施 + 自由刚体）
  rollout.py        每个决策步 = 一个由 Warp Tape 支撑的 torch autograd 算子
  policy.py         torch 策略：观测 -> 速度指令
  train.py          闭环 BPTT 训练入口（多世界批量）
  validate.py       闭环梯度 vs 有限差分校验
  baseline.py       脚本控制器基线：确认动作空间与视界本身可达
```

## 公共场景

```bash
bash run.sh check box_stack_diffsim          # 静态检查，生成 observation.py
bash run.sh view  box_stack_diffsim          # 预览（需要 Isaac）
```

- `table` 是静态长方体；`cube_lower` / `cube_upper` 是质量为 0.2 kg 的自由立方体。
- `targets.cube_lower` / `targets.cube_upper` 给出两个立方体的期望位置（堆叠后）。
- 机器人来自 `assemblies/arm.yaml`（`franka_panda`），`control.output` 有两个命名终点：
  `arm`（7 关节位置）与 `gripper`（2 个手指位置）。
- `policy.py` 的 executor 是 8 维动作 -> 7 个臂角偏移 + 开合，`decode_action` 保持与
  `assembly_lift` 相同的语义。

## 可微训练（不需要 RLinf）

```bash
# 先确认任务可达、梯度可信
.venv-il3/bin/python -m projects.box_stack_diffsim.diffsim.baseline --horizon 90
.venv-il3/bin/python -m projects.box_stack_diffsim.diffsim.validate --case contact --epsilon 1e-3

# 训练；结果写入 projects/box_stack_diffsim/runs/bptt_<时间戳>/
.venv-il3/bin/python -m projects.box_stack_diffsim.diffsim.train --iterations 300 --worlds 32 --horizon 90
```

每个决策步是一个 torch autograd 算子，反向时运行该步的 `wp.Tape`。梯度因此同时经过动作和
之后每一步的观测，是完整的闭环 BPTT。`--truncate K` 可每 K 步截断梯度。
评测用 64 个固定起点并行跑；成功 = 上层立方体距下层顶面 ≤ 2 cm、下层位移 ≤ 2 cm、末速度 < 5 cm/s。

### 策略输入输出

- 输入（10 维）：上层立方体相对「下层顶面」的位移、上层立方体线速度、下层立方体相对初始位置的位移、常数 1。
- 输出（4 维）：世界系速度指令 `vx, vy, vz`（上限 0.2 m/s）与 `grip`。
  `grip=1` 时被持立方体跟随速度指令，`grip=0` 时完全交给重力与接触，相当于松手。
- 起点：上层立方体在场景初始位姿基础上 x/y 偏移 ±3 cm、抬高 6–16 cm，距目标约 0.23 m。
- 损失：末态放置误差² + 10×下层位移² + 0.05×末速度² + 0.1×全程均值（误差² + 10×位移²）。

### 动作语义的由来

Newton 1.6.1 里能被有限差分验证的 BPTT 只有「自由刚体 + 软接触 + 每步注入速度」这一条路径；
关节驱动（Featherstone PD）的梯度在 20 步后就与有限差分差 13 倍，不能用来训练。
所以这里的动作是被持物体的笛卡尔速度，Franka 只在预览与 RLinf 路径里出现。
细节、复现命令和限制见 [docs/NEWTON_DIFFSIM.md](../../docs/NEWTON_DIFFSIM.md)。

## 实测结果（RTX 5080 16 GB，2026-11-02）

### 先修掉的问题（旧 `place_*` 运行全部基于有问题的设置，结果作废）

- **物理会炸**：Newton 默认 `kf=1000` 加 `dt=1/240` 时，两个动态立方体只要错位 3 mm 叠放就飞出。
  改为 `kf=10`、`dt=1/960`（60 Hz × 16 子步）后，错位 3/10 mm 静置 2 s 位移 < 2 mm。
- **视界不够**：旧设置 60 步（1 s）、限速 0.2 m/s，而起点距目标约 0.23 m。
  同一仿真里脚本控制器（先悬停对齐再下降松手）60 步成功率 30%，90 步 100%（64 个起点，误差 1.4 mm）。
- **多世界失效**：旧注入逐世界整份拷贝速度，只有最后一个世界生效；初始位姿与桌面间距也不一致。
- **不是闭环 BPTT**：旧实现切断了「状态 → 后续动作」的梯度；现在完整反传，并已通过有限差分校验。

### 训练（32 世界/次，90 步，lr 1e-3，64 个固定起点评测）

| 方案 | 迭代 0 | 100 | 200 | 275 | 300 | 用时 |
| --- | --- | --- | --- | --- | --- | --- |
| 全程 BPTT | 0.227 m / 0% | 0.064 m / 0% | 0.044 m / 0% | **0.024 m / 15.6%** | 0.024 m / 4.7% | 8.6 min |
| BPTT + SGS 起点池（`--sgs-pool 512`） | 0.227 m / 0% | 0.105 m / 0% | 0.196 m / 0% | 0.094 m / 0% | 0.099 m / 0% | 8.5 min |

表中为平均放置误差与成功率。

- 全程 BPTT 的误差持续下降，300 次迭代时仍在下降，但成功率只有 5–16%，**没有学会**。
  中途卡在约 5 cm 处：它直线逼近目标，撞上下层立方体的侧面；接触梯度只告诉它「往里推」，不会提示「先抬高越过去」。
  后来慢慢走出了这个局部极小。反向过程没有出现非有限梯度，但接触瞬间梯度范数会冲到 1e3。
- SGS 起点池在训练起点上成功率很高（30–90%），完整任务却更差。原因是采样权重 `p^t(1-p)^(1-t)`
  在 p=0 时为 0，远处起点从未成功过，只靠 20% 均匀采样碰到，所以训练一直停在靠近目标的起点（平均难度等级 ≈ 0.77）。
  这与 `my_robot` SGS 实验里记录的现象相同：起点阶梯要能逐级「交接」，单纯混合远近起点不够。

**结论**：这条通路现在是可信的（物理稳定、闭环梯度经过校验、多世界有效），但目前只到「权重更新、误差下降」，
没有达到可靠的完整回合成功。脚本基线 100% 说明任务本身可达；要验证纯梯度方法能否学会，下一步是更长的训练
（运行 300 次迭代时误差仍在下降），以及让 SGS 起点池按难度逐级推进。

## 尚未接入

- 机械臂关节动力学不在反向图里，所以学到的策略不能直接下发给 Franka 执行。
- 抓取是「速度伺服式刚体连接」，没有手指接触与摩擦闭合。
- 没有 RLinf 分布式训练：`diffsim/train.py` 是单进程 torch 循环。
- 只有 RTX 5080 单卡规模（32 个并行世界），与论文规模差距很大。
