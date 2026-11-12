# piper_pick

Piper X 在地面抓起方块，抬离地面 4 cm 并稳住 0.5 s 即成功。默认后端 Isaac Lab 3（`isaaclab3`）+ Newton。

## 文件与要写的位置

| 文件 | 内容 | 谁来改 |
| --- | --- | --- |
| `scene.yaml` | 实体、抽象体、各组件的观察选择、控制终点 | `run.sh edit` 或直接编辑 |
| `policy.py` | `build_planner`、`build_executor`（模型）、`decode_action`（动作→关节目标）、`initial_action` | 换模型、改动作缩放时改这里 |
| `task.py` | `Task.evaluate(obs)`：奖励、成功、失败与指标 | 改任务时改这里 |
| `observation.py` | 由 scene.yaml 生成的观察类型（形状、单位），`check` 时刷新 | 不要手改 |
| `train.yaml` | RLinf PPO 配方；`env.train.init_params.backend/physics` 选后端与物理引擎 | 训练时改 |
| `weights/` | `pick.pt`（随项目进 git，见“权重”）、`WEIGHTS.json` | 只读 |

改了 `scene.yaml` 的实体、观察或控制终点后先 `check` 刷新 `observation.py`，再按其中的类型改
`policy.py`/`task.py`。网络、缩放写在 `policy.py`；奖励、阈值写在 `task.py`。

## 场景与模型

- 场景：Piper X 固定在原点；`cube`（4.8 cm 方块）每次复位随机放在抽象矩形 `cube_area` 内
  （中心 (0.32, 0)，16×16 cm）；`tcp_frame` 测 `gripper_base` 前 0.13 m 的指尖中心；`wrist_camera`
  在场景中但未被选为观察，训练时不渲染。
- planner：每 0.2 s 透传 `cube.position`（模板要求保留，不训练）。
- executor（训练，0.02 s）：输入全部关节位置/速度/力、`cube.position`、`tcp_frame` 指尖位置，
  `policy.py` 另算“方块 − 指尖”×10；MLP 输出 7 维动作和价值，`actor_logstd=-0.5`。
- 动作：每回合从 `READY`（指尖朝下、张开、区域中心上方 0.15 m）开始。6 个臂关节目标 =
  实测关节位置 + 0.05 rad × 动作；第 7 维为夹爪开度，截到 [-1, 0] 后镜像成两指目标 `READY + 0.05 m × 动作`。
- 奖励（`task.py`，计入回报时乘 0.02 s）：接近 `1 − tanh(d/0.1)`；近处合爪奖励；夹住后按抬升量
  和静止度加分；成功（夹住、抬升 ≥4 cm、速度 <0.1 m/s 持续 0.5 s）一次性 +50；方块离区 0.6 m
  或掉出地面失败 −5；动作变化惩罚。没有接触传感器，夹住由指尖距离与开度判断。
- 训练：256 环境 × 64 步/轮，1500 轮，batch 4096 / micro 1024，update_epoch 5，lr 3e-4。
- 域随机化（`train.yaml` `env.train.init_params.randomization`，评估默认跟随）：方块质量 ×[0.5, 2]、
  摩擦 [0.6, 1.2]，每个环境创建时采样一次。不需要时删掉这一段。

## 权重

`weights/pick.pt` 随项目一起进 git（`.gitignore` 对 `projects/*/weights/*.pt` 例外放行），哈希与来源见 `weights/WEIGHTS.json`：
本工作区 2026-10-22 在 **mjlab** 后端训练 1500 轮的 `global_step_1500`
（运行 `outputs/piper_pick/runs/20261022_124529_30e82905256d`）。scene.yaml `executor.parameters.weights`
让 `build_executor` 默认加载它；写 `null` 则随机初始化（从头训练时改成 `null`，否则 train 是在它之上继续训练）。
文件缺失时直接报错。evaluate/view 若另给 checkpoint，在此之上加载。

## 运行

```bash
bash run.sh check piper_pick                    # 静态检查，刷新 observation.py
bash run.sh edit piper_pick                     # 浏览器编辑场景（viser）
bash run.sh view piper_pick --actions policy    # 用 pick.pt 实时预览（不给 --actions 时无 checkpoint 即零动作）
bash run.sh record piper_pick --actions policy --steps 300   # 离屏录像
bash run.sh evaluate piper_pick env.eval.total_num_envs=16   # 评估 pick.pt
bash run.sh train piper_pick                    # PPO 训练（从头训练先把 weights 设为 null）
RUN=$PWD/outputs/piper_pick/runs/<运行>
bash run.sh evaluate piper_pick --snapshot $RUN \
  runner.ckpt_path=$RUN/piper_pick/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt
```

- 换 OvPhysX：在命令后加 `env.train.init_params.physics=ovphysx`。评估、预览自动跟随训练的选择。
- 每轮评估跑完整一个回合（`env.eval`）；成功即结束回合，所以一轮有多条轨迹。
- 多条件评估：`train.yaml` 的 `evaluation.conditions` 列出 7 个条件（原始、训练范围、重方块 ×[2, 3]、
  低摩擦 [0.4, 0.5]、小/大方块缩放 0.7/0.9（默认 0.8）、方块区域放大到 24×24 cm）。在上面的 evaluate 命令后加 `--conditions all`
  （或 `--conditions heavy_cube,wide_area`），每个条件单独评估，最后打印一张表并写 `summary.json`。
  条件取自当前 train.yaml，可用于之前保存的运行。

## 验证状态

2026-10-28，`pick.pt`，`bash run.sh evaluate piper_pick env.eval.total_num_envs=16`（确定性动作，每轮 500 步）：

| 后端 | 回合数 | 成功 | 失败 | 平均回合步数 | 抬升 |
| --- | --- | --- | --- | --- | --- |
| IL3 Newton | 150 | 100% | 0 | 51.2 | 0.23 m |
| IL3 OvPhysX | 108 | 100% | 0 | 69.0 | 0.31 m |
| mjlab（训练后端，2026-10-22 运行自带评估） | 153 | 100% | 0 | 50.2 | — |

- 权重在 mjlab 上训练，直接在 IL3 两种物理引擎上评估完整回合成功；条件评估未对 `pick.pt` 做。

2026-11-02，IL3 Newton 从头训练（`weights` 设为 `null`，单一种子），两次训练后各用最后的 checkpoint 做
`--conditions all`（16 环境 × 500 步，确定性动作，单一种子；表中为成功率 / 平均回合步数，所有条件失败率 0，
除注明外）：

| 条件 | A：无随机化，600 轮<br>Newton | A<br>OvPhysX | B：随机化，700 轮<br>Newton | B<br>OvPhysX |
| --- | --- | --- | --- | --- |
| 原始 | 100% / 90 | 100% / 95 | 100% / 63 | 6% / 455（失败 31%） |
| 训练范围 | 100% / 91 | 100% / 95 | 100% / 63 | 38% / 402（失败 6%） |
| 重方块 ×[2, 3] | 100% / 92 | 86% / 163 | 100% / 64 | 31% / 417（失败 31%） |
| 低摩擦 [0.4, 0.5] | 100% / 94 | 100% / 96 | 100% / 75 | 13% / 465（失败 38%） |
| 小方块 0.7 | 100% / 88 | 100% / 92 | 100% / 64 | 61% / 343 |
| 大方块 0.9 | 100% / 93 | 98% / 105 | 100% / 62 | 41% / 403 |
| 方块区域 24 cm | 99% / 104 | 96% / 122 | 100% / 64 | 19% / 412（失败 25%） |
| 轨迹数（每条件） | 67–80 | 35–76 | 97–124 | 16–18 |

- 运行：A `outputs/piper_pick/runs/20261102_073203_6db26978abb6`（训练中约 549 轮首次全部成功，
  `global_step_600`）；B `outputs/piper_pick/runs/20261102_081229_3672d838fb85`（`global_step_700`）。
  评估结果在各运行的 `evaluations/` 下。两次训练后都改过共享代码，因此评估加了 `--allow-shared-change`。
- A 训练时没有随机化，“训练范围”对它等同“原始”。500 步结束时仍在进行的回合不计入，轨迹数少的条件
  （回合长）结果偏向较快完成的回合。
- B 在 Newton 的 7 个条件下都 100% 成功，回合更短；在训练引擎上可称为“这些条件下成功”
  （单一种子、每条件约 100 条轨迹）。
- B 换到 OvPhysX 明显失败：录像中夹爪能到达并夹住方块，但方块在指间翻转、压着地面抬不起来，回合超时。
  A 没有随机化反而能迁移。质量/摩擦随机化不能代替引擎差异；跨引擎是否成功取决于学到的夹法，
  要在两种引擎上都可靠需要在 OvPhysX 上同样训练/评估，或交替训练，目前未做。
- `weights/pick.pt` 仍是 mjlab 权重（在 IL3 两种引擎上都成功），没有替换为 A 或 B。

## 历史要点

- 2026-10-21/22：从图像 + 绝对关节目标起步，3000 轮内 0 成功；加入指尖位置与相对向量、连续抬升奖励、
  一次性成功奖励，改为关节增量动作与 7 维夹爪后，1500 轮训练成功（当时记录：Isaac 运行 `20261022_075652`；
  mjlab 运行 `20261022_124529` 即 `pick.pt`）。更早的 27 维观察、8 维动作权重都不能加载。
- 脚本 IK 抓取在 Isaac 与 mjlab 均 8/8，说明资产夹爪可抓起方块；mjlab 侧的约束行数、目标裁剪、
  手指垫片等修复在 `shared/backends/mjlab/`。

## 未接入

- 没有接触传感器；图像未用于策略（`wrist_camera` 在场景中但未选为观察）。
- 方块尺寸只能通过评估条件改场景，不能在训练中按环境随机；没有在两种物理引擎间同时训练。
- 感知与重建试验（双相机、局部重建）已移出项目，记录与场景在 `outputs/piper_perception/NOTES.md`。
