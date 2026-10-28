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

## 验证状态

2026-10-28，`pick.pt`，`bash run.sh evaluate piper_pick env.eval.total_num_envs=16`（确定性动作，每轮 500 步）：

| 后端 | 回合数 | 成功 | 失败 | 平均回合步数 | 抬升 |
| --- | --- | --- | --- | --- | --- |
| IL3 Newton | 150 | 100% | 0 | 51.2 | 0.23 m |
| IL3 OvPhysX | 108 | 100% | 0 | 69.0 | 0.31 m |
| mjlab（训练后端，2026-10-22 运行自带评估） | 153 | 100% | 0 | 50.2 | — |

- 权重在 mjlab 上训练，直接在 IL3 两种物理引擎上评估完整回合成功；IL3 短训练（3 轮，17/17 张量变化）只说明接口跑通、权重更新。
- 只在当前方块区域内评估，其他位置、尺寸、摩擦未验证，不能称为多条件可靠成功。

## 历史要点

- 2026-10-21/22：从图像 + 绝对关节目标起步，3000 轮内 0 成功；加入指尖位置与相对向量、连续抬升奖励、
  一次性成功奖励，改为关节增量动作与 7 维夹爪后，1500 轮训练成功（当时记录：Isaac 运行 `20261022_075652`；
  mjlab 运行 `20261022_124529` 即 `pick.pt`）。更早的 27 维观察、8 维动作权重都不能加载。
- 脚本 IK 抓取在 Isaac 与 mjlab 均 8/8，说明资产夹爪可抓起方块；mjlab 侧的约束行数、目标裁剪、
  手指垫片等修复在 `shared/backends/mjlab/`。

## 未接入

- 没有接触传感器、没有域随机化；图像未用于策略（`wrist_camera` 在场景中但未选为观察）。
- IL3 上没有从头训练成功的记录。
- 感知与重建试验（双相机、局部重建）已移出项目，记录与场景在 `outputs/piper_perception/NOTES.md`。
