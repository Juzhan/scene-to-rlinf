# go2_piper_wholebody

Go2 背 Piper X 的全身控制：跟踪随机的机身速度命令和末端目标。复刻
[LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab) 的 GO2-PIPER Flat 任务，地面改为粗糙地块。
默认后端 Isaac Lab 3（`isaaclab3`）+ Newton。

## 文件与要写的位置

| 文件 | 内容 | 谁来改 |
| --- | --- | --- |
| `scene.yaml` | 实体、抽象体、连接、各组件的观察选择、控制终点 | `run.sh edit` 或直接编辑 |
| `policy.py` | `build_planner`（命令采样）、`build_executor`（模型）、`decode_action`（动作→关节目标）、`initial_action` | 换模型、改动作缩放或命令范围时改这里 |
| `task.py` | `Task.evaluate(obs)`：23 项奖励、失败条件与指标 | 改任务时改这里 |
| `observation.py` | 由 scene.yaml 生成的观察类型（形状、单位），`check` 时刷新 | 不要手改 |
| `train.yaml` | RLinf PPO 配方；`env.train.init_params.backend/physics` 选后端与物理引擎 | 训练时改 |
| `weights/` | `wholebody.pt`（随项目进 git，见“权重”）、`WEIGHTS.json` | 只读 |

改了 `scene.yaml` 的实体、观察或控制终点后先 `check` 刷新 `observation.py`，再按其中的类型改 `policy.py`/`task.py`。

## 场景与模型

- Go2 机身上用固定连接 `piperx_attachment` 装 Piper X（挂在 `robot/base`，z 偏移 0.06）。
- 地面为 40 m 粗糙地块 `assets/terrains/rough_40m`（起伏 2–8 cm），每个环境一块，`world.env_spacing: 40`。
  没有高度扫描，策略对地形是盲的。
- 接触传感器 `base_contact`、`feet_contact`；`dex_cube`、`camera` 在场景中但不参与任务。
- 抽象长方体 `ee_workspace` 跟随 piperx 根部（局部中心 (0.57, 0, 0.25)，0.25×0.7×0.5 m），只跟随读数，
  不是物理约束；在编辑器里拖动它即改变末端目标范围。

| 组件 | 周期 | 输入 | 输出 |
| --- | --- | --- | --- |
| planner（固定 `CommandSampler`） | 10 s，复位时也调用 | piperx、ee_workspace 的位姿与尺寸 | 6 维 `[vx, vy, wz, x, y, z]` |
| executor（训练） | 0.02 s | 见 `scene.yaml` 中 executor 的观察 | 18 维动作 |

- 命令：vx ±1、vy ±0.6、wz ±1，10% 环境站立；末端目标在 `ee_workspace` 内均匀采样，表达在 Piper 基座系。
- executor：非对称 actor-critic，MLP [512, 256, 128]，ELU。actor 只用可部署输入（角速度、重力投影、
  关节偏移与速度、上一步动作、命令）；critic 另加机身线速度、TCP 位置、足端接触。
- `decode_action`：腿 = 默认角 + 0.25·a[:12]，臂 joint1..6 = 默认角 + 0.25·a[12:]；夹爪不控制。
  机械臂增益按参考 GO2_PIPER_CFG（joint1–6 kp 50/50/80/30/30/20，kd 3/2/3/3/2.5/1）。
- 奖励：末端位置/姿态跟踪、速度跟踪、相对机身高度（相对最低足端，目标 0.30 m）、力矩/加速度/功率/
  动作变化、足端滑移与腾空、关节偏离与对称。失败为机身触地或倾斜超过 1 rad；没有 success 定义，
  看 metrics 的 `ee_position_error_m`、`lin_vel_xy_error_m_s` 等。
- 训练：2048 环境 × 24 步/轮，update_epoch 5，4 个 minibatch，entropy 0.005，回合 1000 步（20 s），
  lr 1.5e-4 cosine 衰减到 3e-5（参考为 4096 环境、5e-4 加自适应 KL；RLinf 原生 PPO 无自适应 KL，
  固定 5e-4 时约 1430 轮发散）。

与参考的差异：粗糙地块代替平地；机身高度改为相对足端；末端目标 z ∈ [0, 0.5]（参考 [-0.2, 0.5]）；
actor 不堆叠 3 帧历史；腿用各后端原生 Go2 DC 电机（参考为带延迟的 PD）。

## 权重

`weights/wholebody.pt` 随项目一起进 git，哈希与来源见 `weights/WEIGHTS.json`：本工作区 2026-10-23 在 **mjlab**
后端训练 6000 轮的 `global_step_6000`（运行 `outputs/go2_piper_wholebody/runs/20261023_185843_d8380ea45fb3`）。
scene.yaml `executor.parameters.weights` 让 `build_executor` 默认加载它；写 `null` 则随机初始化（从头训练时改成
`null`，否则 train 是在它之上继续训练）。文件缺失时直接报错。evaluate/view 若另给 checkpoint，在此之上加载。

## 运行

```bash
bash run.sh check go2_piper_wholebody                   # 静态检查，刷新 observation.py
bash run.sh edit go2_piper_wholebody                    # 浏览器编辑场景（viser）
bash run.sh view go2_piper_wholebody --actions policy   # 用 wholebody.pt 实时预览
bash run.sh record go2_piper_wholebody --actions policy --steps 400   # 离屏录像
bash run.sh evaluate go2_piper_wholebody                # 评估 wholebody.pt（16 环境，一个完整回合）
bash run.sh train go2_piper_wholebody                   # PPO 训练（从头训练先把 weights 设为 null）
RUN=$PWD/outputs/go2_piper_wholebody/runs/<运行>
bash run.sh evaluate go2_piper_wholebody --snapshot $RUN \
  runner.ckpt_path=$RUN/go2_piper_wholebody/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt
```

- 换 OvPhysX：加 `env.train.init_params.physics=ovphysx`。评估、预览自动跟随训练的选择。
- OvPhysX 下固定连接由后端写 PhysX FixedJoint 并整 env 克隆，2048 环境主机内存比 Newton 多约 3 GB（见
  `docs/SCENE_BACKENDS.md`）。

## 验证状态

2026-10-28，`wholebody.pt`，`bash run.sh evaluate go2_piper_wholebody`（确定性动作，16 环境，每轮 1000 步）：

| 后端 | 轨迹数 | 失败 | 平均回合步数 | 末端位置误差 | 线速度误差 | 机身高度 |
| --- | --- | --- | --- | --- | --- | --- |
| IL3 OvPhysX | 16 | 0 | 1000 | 5.6 cm | 0.21 m/s | 0.315 m |
| IL3 Newton | 19 | 63% | 680 | 26.6 cm | 0.67 m/s | 0.257 m |
| mjlab（训练后端） | 16 | 0 | 1000 | 5.2 cm | — | — |

- OvPhysX：mjlab 权重直接跑完完整回合，跟踪误差与 mjlab 相当。
- Newton：大部分回合能走一段，但会翻倒（倾斜 >1 rad，机身未触地）；同一种子 16 环境 1000 步录像中
  Newton 失败 18 次、OvPhysX 1 次。Newton 上机身平均低约 4 cm，接触与 weld 连接比 PhysX 软（连接旋转偏差约
  10°），策略未在其上训练。要在 Newton 上可靠运行需在 Newton 上继续训练，目前未做。
- IL3 短训练（3 轮，2048 环境，17/17 张量变化）只说明接口跑通、权重更新。
- 各后端 Go2 原生电机预设不同（IL3 小腿 45.4 N·m，mjlab 全部 23.5 N·m）；IL3 全部走 Isaac Lab 的显式
  Python 执行器（见 `docs/SCENE_BACKENDS.md`），Newton 原生隐式 DC 电机上平均约 64 步就失败。
- 只在这一块地形和随机命令下评估，没有 success 定义；不能称为多条件可靠成功。

## 未接入

- 高度扫描（IL3 已支持 `height_scanner`，本项目未选用）、3 帧历史、自适应 KL、执行器延迟。
- 夹爪不控制；`dex_cube` 与 `camera` 不参与任务。
