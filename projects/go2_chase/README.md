# go2_chase

Go2 追一个会逃跑的虚拟球，用头部（水平距离）碰到它。场景与任务复刻
[`Zhengtianjin/go2-supermimic`](https://github.com/Zhengtianjin/go2-supermimic)（MIT，提交 `054a13c`）的 **Chase** 部分；
executor 直接使用该仓库发布的平地 actor 权重 `flat.pt`（SuperMimic 方法训练，本工作区没有重新训练）。
默认后端 Isaac Lab 3 + Newton。

## 文件与要写的位置

| 文件 | 内容 | 谁来改 |
| --- | --- | --- |
| `scene.yaml` / `scene_terrain.yaml` | 实体、观察选择、5 帧历史、控制目标 | `run.sh edit` 或直接编辑 |
| `policy.py` | `build_planner`、`build_executor`（模型）、`decode_action`（动作→关节目标）、`initial_action` | 换模型时改这里 |
| `task.py` | `evaluate(obs)`：奖励、成功、失败、指标、逃跑球的 `set` | 改任务时改这里 |
| `observation.py` | 由 scene.yaml 生成的观察类型（形状、单位），`check` 时刷新 | 不要手改 |
| `train.yaml` / `train_terrain.yaml` | PPO 配方（平地/粗糙地形） | 微调时改 |
| `weights/` | `flat.pt`（随项目进 git，见“权重”）、`WEIGHTS.json`、许可证 | 只读 |

## 场景

两个配方共用 `scene.yaml`/`policy.py`/`task.py`，只有场景文件不同：

| 配方 | 场景 | 说明 |
| --- | --- | --- |
| `train.yaml`（默认） | 平地 | `ground_plane` + Go2 + 虚拟球 |
| `train_terrain.yaml` | 粗糙地形 | `assets/terrains/rough/usd/rough.usda`（8 m 方块，噪声 2–8 cm，border 0.25 m，难度 0.5）；通过 `env.train.init_params.scene_config` 换场景，其余配置不复制 |

- Go2 初始位姿 (0, 0, 0.29)（地形场景 0.4），`actuators: supermimic` 给出站姿与执行器（见下节）。
- `ball`：`kind: abstract` 的点（无模型、无碰撞、无质量），初始 (2, 0, 0.18)（地形 0.3），
  每回合在 `ball_area`（2×3 m 矩形，中心 (2,0)）内随机复位；`task.py` 的 evaluate
  每个决策步（0.02 s）返回 `'set': {'ball': move_ball(...)}` 让它逃跑。`visualization.abstract: [ball]` 只在 view/record 预览画出。
- 逃跑（`task.py` 的 `move_ball`，参数取参考 `target/generator.py` 课程起点的固定值）：
  - 逃离方向：从机器人水平位置指向球；逃离速度 `0.5 × max(0.85 × 机器人水平速度, 0.5 m/s)`；
  - OU 抖动：随机部分 = 上一步速度 − 本步逃离速度，按 θ=1.5/s 回归、σ=4 加噪，并限幅使总速度不超过 1.25 m/s；
    无隐藏状态，上一步速度读 `obs.ball.linear_velocity`（共享层按位移/dt 给出，复位后为 0），步长读 `obs.simulation.step_dt`；
  - 墙：离原点 3.2 m 以内自由移动，接近 3.5 m 时去掉向外分量，超出后向内推回；高度、朝向保持复位值。
- 时序：Task 的 set 由原生 command 项在本步复位之后、观察之前生效（刚复位的环境不移动）。Task 第 t+1 步看到的球位置
  就是策略第 t 步观察到的位置；复位后的第一帧球不动、速度为 0。
- `head`：`frame_transformer` 挂在 `base` 的 (0.32, 0, 0.1)，用于计算头部到球的距离。
- 地形只是静态网格，没有 height scan 观察。

## 模型、观察与动作

executor 50 Hz（`period: 0.02`）是参考 actor 的原结构：`RunningNorm`（权重内的均值/方差，冻结）+
MLP 350-512-256-128-12（ELU）。`policy.py` 的 `features(obs)` 按权重的约定拼出 350 维输入：
6 项，每项是 5 帧历史（旧→新，scene.yaml `executor.history` 提供，复位后以首帧补齐）依次展平：

| 项 | 每帧维度 | 处理 |
| --- | --- | --- |
| 机体角速度 | 3 | 转到机体系 |
| projected gravity | 3 | (0,0,-1) 转到机体系 |
| 关节位置 | 12 | 换成 actor 顺序后减站姿 |
| 关节速度 | 12 | 换成 actor 顺序 |
| 上一步动作 | 12 | `executor.output`，复位后为 0 |
| 目标槽 ×2 | 28 | 槽 1：球相对位置与球速度（偏航系）、不确定度 0、可见 1、age 0、点目标、active 1；槽 2（躲避）全 0 |

- actor 的关节顺序是逐腿 FL、FR、RL、RR 的 hip/thigh/calf；机器人读数是 hips、thighs、calves
  （ASSET.json `joint_order`），由 `ASSET_TO_ACTOR` 换序。`control.output.legs` 直接按 actor 顺序列关节。
- 动作 12 维，目标 = 站姿 (hip ±0.1, thigh 0.9, calf −1.8) + scale × 动作，scale 为 0.3349（hip/thigh）、0.2853（calf）。
- 没有线速度、LiDAR、height scan 输入（与该 actor 一致）；地形场景对策略不可见。
- 以上排列、顺序、站姿、scale 都由权重决定，改任何一项都需要重新训练，不能只改代码。
- PPO 微调用：`actor_logstd`（初值 −2）与新建价值头（350-`hidden_dims`-1）。权重文件没有 critic。

机器人用 `go2` 工厂的 `actuators: supermimic`：参考仓库的显式 DC 电机（hip/thigh kp 17.69、kd 1.126、
力矩 23.7/饱和 47.4 N·m、vel 30.1；calf kp 39.81、kd 2.534、45.43/90.86 N·m、vel 20.07；含转子 armature）
与 0–2 物理步指令延迟，初始关节角为站姿、初始高度 0.29 m（地形 0.4 m）。Isaac Lab 3 上这类执行器不在
Newton 原生执行器支持范围内，后端自动改用 Isaac Lab 的 Python 执行器路径（Newton 与 OvPhysX 相同）；
用 `actuators: native`（IL3 自带 DC 电机预设）时该权重会摔倒。旧 `isaac` 后端没有延迟 DC 电机，选 supermimic 时报错。

planner 每 0.2 s 固定透传 `ball.position`（3 维），不参与 executor 输入，保留以沿用 planner/executor 模板。

## 权重

`weights/flat.pt` 随项目一起进 git（`.gitignore` 对 `projects/*/weights/*.pt` 例外放行），哈希与来源见 `weights/WEIGHTS.json`。
`build_executor` 默认加载它；scene.yaml `executor.parameters.weights` 可换成别的文件（相对本项目或绝对路径），
写 `null` 则随机初始化。文件缺失时直接报错，不会静默用随机权重。evaluate/view 若另给 checkpoint，在此之上加载。

## 奖励与成功（task.py）

原始值，计入回报时由共享层乘 `step_dt=0.02 s`：

- 追球：`1.5 × clip(v·direction, -5, 5)`；朝向：`0.5 × (1 + cos)`；头部到球的
  `1.5 × exp(-(d/0.3)²)`。
- 姿态：`exp(-g_xy²/0.4²)` 直立奖励，`-0.05 × 角速度 xy 平方和`，`-0.1 × (q-default)²`。
- 约束：出场（离原点超过 4 m）`-2 × 超出距离`，动作变化 `-0.05 × Δa²`，
  `-2e-5 × 力矩²`，低于 0.15 m `-10 × 差值`，摔倒 `-200`。
- 成功：头部到球**水平**距离 <0.2 m 且未摔倒，一次性 +20 回报（原始 1000），回合结束。
  虚拟球没有可碰的高度，与参考一样只看水平距离；`distance_m` 也是水平距离。
- 失败只有摔倒；原来的“球掉出场景”已删除（抽象点不会掉落，逃跑限制在 3.5 m 内）。
- 指标：`distance_m`、`head_distance_m`、`speed_toward_mps`、`tagged`、`fell`。

对照参考：保留 `chase_approach`、`facing_target`、`upright`、`tilt_rate_l2`、
`joint_deviation_l2`、`action_rate_l2`、`joint_torques_l2`、`base_height_low`、
`outside_arena`、`terminated`；省略 `evade_*`、`hit`、`crouch`、`hard_landing`、
`dof_pos_limits`、`joint_acc_l2` 和障碍物接触项（这些属于 dodge/obstacles）。
参考的 tag 是 15 权重的连续奖励、碰到后球继续跑；这里的 tag 同时结束回合，所以
把一次性成功奖励设为 20 回报，避免在球旁悬停刷密集奖励。

## 运行

后端由 `train.yaml` 的 `env.train.init_params.backend/physics` 选择（默认 `isaaclab3` + `newton`）；
命令后加 `env.train.init_params.physics=ovphysx` 换 OvPhysX，评估与预览跟随训练配置。

```bash
bash run.sh check go2_chase                     # 静态检查，刷新 observation.py
bash run.sh check go2_chase --config train_terrain
bash run.sh edit go2_chase                      # 浏览器编辑场景
bash run.sh view go2_chase --actions policy     # 用 flat.pt 实时预览（不给 --actions 时无 checkpoint 即零动作）
bash run.sh record go2_chase --actions policy --steps 300   # 离屏录像
bash run.sh evaluate go2_chase env.eval.total_num_envs=16
bash run.sh evaluate go2_chase --config train_terrain env.eval.total_num_envs=16
bash run.sh train go2_chase                     # 从 flat.pt 出发 PPO 微调（1024 环境；未验证收益）
```

注意：

- `env.train.total_num_envs × max_steps_per_rollout_epoch` 必须能被 `actor.global_batch_size`（8192）整除。
- 评估每轮最多 `max_episode_steps`（300 步）；碰到球即结束回合，所以一轮有多条轨迹。
- 训练后评估/录像用 `--snapshot <运行>` 里的冻结源码。

## 验证状态

2026-10-28，`flat.pt`，`bash run.sh evaluate`，16 环境 × 300 步（碰到即结束，统计完整回合）：

| 后端 | 场景 | 回合数 | 碰到 | 摔倒 | 平均回合步数 |
| --- | --- | --- | --- | --- | --- |
| IL3 Newton | 平地 | 95 | 100% | 0 | 46.6 |
| IL3 OvPhysX | 平地 | 80 | 100% | 0 | 56.0 |
| IL3 Newton | 粗糙地形 | 66 | 95.5% | 4.5% | 67.3 |
| mjlab | 平地 | 69 | 98.6% | 1.4% | 61.1 |

`record --actions policy --steps 300`（1 环境，Newton）：6 s 内 6 次碰到球、0 次摔倒。

即“完整回合成功”：平地上两种物理引擎都稳定碰到球；粗糙地形用的是平地权重（对地形不可见），偶有摔倒。
同一 actor 在 Newton 上换成 IL3 原生 DC 电机（Newton 原生执行器）时，16 环境 20 s 内摔倒 23 次，
所以执行器模型必须与权重一致。没有在 IL3 上做 PPO 微调；“多条件可靠”只覆盖上表这几种设置。

## 已知边界

- 只有 Chase：没有投掷球/杆、dodge、obstacles，Task 的 `set` 只移动抽象体。
- 球的逃跑参数固定在参考课程起点；被碰到后回合直接结束，不复现参考里球加速逃开、继续追的过程。
- Go2 模型是工作区的 Isaac USD（mjlab 用其转换的 MJCF），不是参考的 Menagerie `go2.xml`：总质量 15.02 vs 15.21 kg，
  脚质量、接触参数不同；执行器和站姿已对齐参考。权重仍能直接工作（见上表），但不是逐位复现。
- 地形场景是本工作区 8 m rough 块，不是参考的混合地形课程；初始高度固定 0.4 m，未按局部地形调整。
- 参考仓库另有 `terrain.pt`、`obstacles.pt`，需要 height scan/LiDAR 输入，未接入。
