# Newton 可微 BPTT 实验

本文件记录 `projects/box_stack_diffsim/` 使用的牛顿（Newton 1.6.1 + Warp 1.17）可微仿真边界：
哪些能求导、哪些不能、梯度可信到什么程度。结论来自本工作区 `.venv-il3` 里实测的探针，
不是文档摘抄；复现命令附在每节末尾。

## 结论摘要

| 能力 | 状态 | 证据 |
| --- | --- | --- |
| 自由刚体 + 软接触 + `SolverSemiImplicit` | 可用，梯度与有限差分一致 | 相对误差 4.3e-4（40 决策步 / 160 子步） |
| 每步注入被控刚体线速度（写到新数组） | 可微 | 同上；`validate.py` 默认用例 |
| 直接改写 `state.body_q` / 接触位形 | **不推荐**：接触法向被冻结，位形梯度不可用 | 注入位姿后 `grad == 0`，而有限差分非零 |
| 在原数组上原地写速度 | **不可微**：Tape 版本化后梯度丢失 | 原地写得到 0，改用 `wp.clone` 后正常 |
| Featherstone 关节驱动（机械臂 PD） | 梯度存在但不可信 | 20 步时解析/有限差分比值 13×，梯度量级可达 1e12 |
| Featherstone 接触 + 自由刚体 | 接触梯度为 0 | 与 SemiImplicit 同一场景对照 |
| 持续接触下每步硬覆盖速度 | 会泵能量 | 立方体压桌面时速度反号膨胀 |
| 两个动态立方体错位叠放（默认 `kf=1000`、`dt=1/240`） | **会炸开** | 错开 3 mm 静置即飞出，角速度上百 rad/s |
| 同上，`kf=10`、`dt=1/960` | 稳定 | 错开 3/10 mm 静置 2 s 位移 < 2 mm |
| 闭环 BPTT（梯度经过后续观测回到策略） | 可用 | 参数方向导数 vs 有限差分：无接触 1.3e-4，压在下层立方体上 3.9e-3 |

因此本项目把「策略动作」定义为**被持物体的笛卡尔速度指令**，仿真只用自由刚体和
`SolverSemiImplicit`，这是当前版本里唯一被有限差分验证过的 BPTT 路径。

## 可复现的校验

```bash
# 闭环梯度 vs 有限差分；随机策略跑 30 个决策步，比较损失对策略参数随机方向的导数。
# 退出码 0 表示相对误差 < 5%
.venv-il3/bin/python -m projects.box_stack_diffsim.diffsim.validate --case free --epsilon 1e-3
.venv-il3/bin/python -m projects.box_stack_diffsim.diffsim.validate --case contact --epsilon 1e-3
```

## 关键实现细节

1. **模型与接触参数**：`requires_grad=True` 的 `ModelBuilder.finalize`，碰撞管线同样
   `CollisionPipeline(model, requires_grad=True)`。0.2 kg 立方体取 `ke=2000`、`kd=6`、`mu=0.8`，
   **`kf=10`、`dt=1/960`（60 Hz 决策 × 16 子步）**。SemiImplicit 的摩擦是粘性项
   `min(kf·v_t, mu·f_n)`，默认 `kf=1000` 对 0.2 kg 物体是显式积分不稳定的；只要两个动态立方体
   错位接触就会炸开。物体压在静态桌面上时不暴露这个问题。
2. **形状变换**：静态形状（`body=-1`）用世界位姿；挂在刚体上的形状必须是**体坐标**。
   把世界位姿写给刚体形状会让碰撞体跑到两倍位置，表现为「完全没有接触」。
3. **每步清零**：`state.clear_forces()`。SemiImplicit 会把接触力累加进 `state.body_f`，
   不清零时几十步后发散。
4. **速度注入**：`wp.clone(state.body_qd)` 产生新数组，`wp.launch` 写入被控体，
   再赋回 `state.body_qd`。原地写会丢失梯度。
5. **闭环 BPTT**：每个决策步是一个 `torch.autograd.Function`（`rollout._Step`），前向用独立的
   `wp.Tape` 跑 16 个子步，每个子步新建 State 并使用固定的 Contacts 缓冲；反向把 `body_q/body_qd`
   的上游梯度交给该 tape。这样观测、策略和动力学都在同一个 torch 图里，梯度会经过策略对状态的反馈。
6. **多世界**：`replicate(spacing=0)`，各世界重叠但 Newton 只在同一世界内检测碰撞；被持物体的速度
   注入是一个按刚体并行的 kernel（旧实现逐世界整份拷贝，只有最后一个世界生效）。

## 已知限制

- **接触事件不可微**：接触的产生/消失是离散事件，梯度只覆盖已经存在的接触。
- **法向冻结**：`differentiable_contacts.py` 只对接触点和距离求导，法向不参与求导。
  官方标注为 experimental（`CollisionPipeline` docstring）。
- **非有限梯度**：窗口内出现接触生成/消失时，`Tape` 反向可能给出 NaN。训练脚本统计
  `skipped` 并跳过该次更新；数值不稳定不是策略问题。
- **不覆盖关节动力学**：Franka 的关节 PD、关节限位、齿隙等不在反向图里。原因见上表。
- **反向缓存随视界增长**：每个子步保留一份 Contacts 缓冲与 State，90 步 × 16 子步 × 32 世界约占 4 GB 显存。

## 与 SuperMimic 式世界模型的差别

SuperMimic 用学到的动力学 `f_phi` 提供梯度：梯度平滑但与真实物理有偏差，
需要先训练世界模型。这里用 Newton 的 `f_phys`：梯度来自求解器本身，没有额外训练，
但受上面四条限制。两者在多步反传的形式一致，区别是 `∂s_{t+1}/∂a_t` 的来源。

对 SGS（success guided sampling）而言，这里的实测意味着：

- 可微物理适合做**短视界**的精细调整（放置、接触后的微调），不适合直接替代大规模采样；
- 机械臂级 BPTT 需要 Newton 后续版本修好关节与接触的梯度；
- 现阶段最现实的是「可微物体 + 脚本/采样机械臂」的混合：大范围搬运靠采样，
  接触前后的毫米级修正靠 BPTT。
