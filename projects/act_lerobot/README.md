# act_lerobot：在 LeRobot 数据集上用 RLinf 训练 ACT

用采集好的示范数据训练 ACT（Action Chunking Transformer）。模型和数据处理直接用 LeRobot 0.4.3 的实现，梯度更新、验证和检查点由 RLinf 的监督训练 worker（经 `shared/integrations/rlinf/sft_project.py`）完成。训练结果可导出为 LeRobot 标准的 `pretrained_model` 目录，交给 `lerobot-record --policy.path=...` 在机器人上运行。

## 文件

| 文件 | 作用 | 何时修改 |
| --- | --- | --- |
| `train.yaml` | 数据集路径、验证回合、ACT 结构与优化参数、训练步数 | 换数据集或调参时 |
| `data.py` | 读 LeRobot 数据集（v3.0），按回合划分训练/验证，每帧配上之后 `chunk_size` 步动作 | 一般不改 |
| `policy.py` | 构建 ACT、归一化统计、训练/验证损失、导出 `pretrained_model` | 一般不改 |
| `piper_real.py` | 在真实 Piper X 上运行导出的 `pretrained_model`（经 `~/ws/robot_toolkit` 的连续执行器） | 换机器人、相机配置或安全参数时 |

## 数据要求

- LeRobot 数据集目录（含 `meta/info.json`，`lerobot-record` 写出的 v3.0 格式），写到 `data.train_data_paths`。
- 必须有 `observation.state`、`action` 和至少一路相机图像。`actor.model.cameras` 为空时使用全部相机，也可以只列部分。
- 其他字段（如 `observation.sim_time`、深度目录）不进入模型。
- `data.episodes` 限定参与训练和验证的回合（编号列表，默认全部），例如去掉相机异常的回合。
- 训练集与验证集按回合划分：默认所用回合中最后 1 个验证（`data.validation_episodes`，也可写回合编号列表）。状态和动作的均值、方差只用训练回合计算；图像用 ImageNet 的均值和方差（ResNet 预训练骨干对应的统计），不从数据计算。

## 运行

```bash
bash run.sh check act_lerobot                      # 读数据集、划分回合，不训练
bash run.sh train act_lerobot experiment.output=runs/<新目录> data.train_data_paths=<数据集目录>
.venv/bin/python -m projects.act_lerobot.policy projects/act_lerobot/runs/<目录>   # 导出 pretrained_model
```

训练目录里：`act_best.pt`（验证误差最低）、`act_last.pt`、`validation.json`（每次验证的误差）、`lerobot/`（ACT 配置与归一化处理器）、`logs/`（TensorBoard）。导出后的 `runs/<目录>/pretrained_model/` 可在 `.venv-piper-record` 等装有 LeRobot 0.4.3 的环境中用 `ACTPolicy.from_pretrained` 与 `make_pre_post_processors(config, pretrained_path=...)` 加载。

微调：把 `experiment.initial_weights` 设为一个 LeRobot `pretrained_model` 目录或本项目的 `act_*.pt`。只载入模型权重，不恢复优化器和训练进度，不是完整续训；数据集的状态/动作维度和相机必须与该权重一致。

## 主要参数（train.yaml）

- `chunk_size`：一次预测的动作步数，按数据集帧率换算时长（10 fps 下 20 步为 2 秒）。`n_action_steps`：部署时每次推理执行的步数。
- 每批 8 帧，学习率 1e-5，权重衰减 1e-4，梯度裁剪 10，与 LeRobot ACT 默认相同。
- `runner.max_steps`（默认 20000）、`runner.val_check_interval`（默认每 500 步验证并保存 `act_best.pt`）。
- `data.num_workers`：视频解码进程数，默认 8。

验证误差是部署方式下（无 VAE 编码器，潜变量为 0）归一化动作块的平均 L1；训练损失另含 VAE 的 KL 项（权重 `kl_weight`）。`validation.json` 的字段名沿用共享层的 `validation_mse`，对本项目而言是 L1。

## 验证状态（2026-11-17）

用已有的 piper_x 仿真手机遥操作数据（`~/Desktop/recordings/20261008_141506_ebbba4/dataset_stacking_corrected`，5 个回合、3597 帧、10 fps、3 路 640×360 相机、7 维状态与动作）：

- `check` 通过：训练 4 个回合 2620 帧，验证 1 个回合 977 帧。
- 60 步短训练（RTX 5080）：验证误差 0.868 → 0.640 → 0.610；每步约 0.146 s；显存峰值约 8.2 GB（含训练前已占用的约 1.8 GB）。
- 导出的 `pretrained_model` 在 `.venv-piper-record`（LeRobot 0.4.3）中加载，`select_action` 在验证回合的帧上逐帧输出 7 维动作。

- `~/ws/robot_toolkit` 的采集导出（`export_lerobot`，LeRobot 官方 writer）：在 `anaconda3/envs/vla`（LeRobot 0.6.1）里用合成的 piper_x 数据（4 回合 × 40 帧、2 路 64×48 相机、10 fps）导出，数据集格式为 v3.0（LeRobot 0.4.3 到 0.7.0 都写 v3.0）；本项目（LeRobot 0.4.3）读取、`check`、20 步训练通过；导出的 `pretrained_model` 在 LeRobot 0.6.1 中加载并 `select_action` 通过。该环境自带 av 18.1.0，而 LeRobot 0.6.1 要求 `av<16`，带相机导出时报 `module 'av' has no attribute 'option'`；临时用 av 15.1.0 覆盖后导出成功（未修改该环境）。

以上只说明接口和训练流程跑通、权重在更新；没有完整训练，没有在仿真或真机上闭环运行。

## 训练记录：piper 主从遥操作数据（2026-11-18）

- 数据：`~/ws/outputs/robot_tasks/piper_master_slave/dataset/merged_20261118`，48 回合、15231 帧、30 fps、2 路 640×480 AV1 相机（wrist/external）、7 维状态与动作（6 关节 rad + 夹爪宽度 m）。
- 命令（`train.yaml` 未改，以下为命令行覆盖）：`bash run.sh train act_lerobot experiment.output=runs/piper_ms_20261118 data.train_data_paths=<上述目录> data.validation_episodes=4 actor.model.chunk_size=100 actor.model.n_action_steps=50 runner.max_steps=50000 runner.val_check_interval=1000 runner.save_interval=10000`。训练 44 回合 14078 帧，验证回合 44–47 共 1153 帧。
- RTX 5080 约 0.11 s/步，50000 步用时 1 小时 42 分。验证误差（归一化 L1）0.724 → 3000 步 0.409，之后在 0.40–0.42 之间波动，最低 0.399（41000 步，即 `act_best.pt`）；训练损失降到 0.073。
- `act_best.pt` 已导出为 `runs/piper_ms_20261118/pretrained_model/`。按实际单位逐步统计的平均绝对误差（rad，夹爪 m）：验证回合第 1 步 0.037、第 30 步 0.083、第 100 步 0.109；训练回合各步都在 0.016–0.019。验证回合前约 15 步不如“保持当前位置”的基线（第 1 步基线为 0.015），之后优于基线（第 100 步基线为 0.221）。训练与验证差距大，说明对训练回合拟合充分、对未见回合的泛化有限。
- 未在真机或仿真上闭环运行；离线误差不能代表抓取成功率。

## 与 LeRobot 原生训练对比（2026-11-18）

用 `.venv-piper-record` 的 `lerobot-train`（LeRobot 0.4.3）在同一数据、同一训练回合（0–43）和同样的 ACT 结构与优化设置下训练 50000 步，输出在 `runs/lerobot_train_piper_ms_20261118/checkpoints/<步数>/pretrained_model`：

```bash
.venv-piper-record/bin/lerobot-train --dataset.repo_id=local/merged_20261118 --dataset.root=<数据集> --dataset.episodes="[0,...,43]" \
    --dataset.video_backend=pyav --policy.type=act --policy.chunk_size=100 --policy.n_action_steps=50 --policy.push_to_hub=false \
    --policy.device=cuda --batch_size=8 --steps=50000 --save_freq=10000 --num_workers=8 --seed=1000 --output_dir=<新目录> --wandb.enable=false
```

唯一差别：LeRobot 用数据集 `stats.json`（全部 48 回合）做状态/动作归一化，本项目只用训练回合。该环境为此以 `--no-deps` 加装了 termcolor 与 pyyaml-include。每步约 0.09 s，50000 步 76 分钟，最终训练损失 0.071（本项目 0.073）。

验证回合 44–47 上的平均绝对误差（7 维平均，rad / m），按动作块步数：

| 模型 | 第 1 步 | 第 10 步 | 第 30 步 | 第 50 步 | 第 100 步 | 1–50 步平均 |
| --- | --- | --- | --- | --- | --- | --- |
| 保持当前位置 | 0.015 | 0.047 | 0.111 | 0.158 | 0.221 | 0.093 |
| LeRobot 50000 步 | 0.041 | 0.053 | 0.078 | 0.090 | 0.109 | 0.070 |
| LeRobot 40000 步 | 0.042 | 0.053 | 0.077 | 0.090 | 0.109 | 0.070 |
| 本项目 50000 步（`act_last.pt`） | 0.037 | 0.053 | 0.085 | 0.096 | 0.110 | 0.074 |
| 本项目 41000 步（`act_best.pt`） | 0.037 | 0.053 | 0.083 | 0.094 | 0.109 | 0.073 |

两者差别在 5% 左右，各只训练了一个种子，不足以说明哪一方更好。

真机（`piper_real.py`，`--steps 10 --ensemble 0.01 --max-speed 0.9 --max-accel 2.5`，30 秒，红块在绿块右后方）运行 LeRobot 50000 步版本：第 2–6 秒伸向红块，第 7 秒闭合（0.018 m），j1 转向绿块，第 10–12 秒张开；第 16 秒再次闭合，之后夹爪停在约 0.044 m 并保持到结束。现场观察：抓取成功，放置偏离绿块，与本项目版本的表现类似。单次运行，未统计成功率。

## 真机运行（Piper X）

`piper_real.py` 读取 `pretrained_model`，相机按 `~/ws/robot_toolkit/configs/tasks/piper_master_slave.json` 打开。动作块交给 robot_toolkit 的 `run_continuous_stream` 执行，执行中会检查关节限位、做 jerk 限制的参考插值、跟踪误差超限时中止，夹爪在臂跟上目标后才动作。需要装有 LeRobot、pyAgxArm、pyrealsense2、ruckig 的环境（本机为 `~/ws/RealSense-Stereo-Trial/.venv-linux`）：

```bash
cd ~/ws && PYTHONPATH=~/ws RealSense-Stereo-Trial/.venv-linux/bin/python rlinf-workspace/projects/act_lerobot/piper_real.py \
    rlinf-workspace/projects/act_lerobot/runs/<目录>/pretrained_model --dataset <训练数据集>            # 只读试运行，不发 CAN 报文
    ... --execute --seconds 25 --steps 50                                                            # 真机执行
```

- 不加 `--execute` 时只读：打开相机，用实时画面推理，打印预测动作与实测关节，并保存两路相机画面。
- 加 `--execute` 后：先以 `--start-speed`（0.3 rad/s）移到数据集各回合首帧状态的平均姿态，夹爪张开到首帧平均宽度，保存两路相机画面（`start_*.png`），再执行策略（`--start-only` 只做到这一步，用于两次试验之间复位）。`--steps` 是每次推理执行的步数，`--max-speed`/`--max-accel` 是关节速度和加速度上限（默认 0.6 rad/s、1.2 rad/s²；示教中速度 p99 约 0.9 rad/s、加速度 p95 约 2–3 rad/s²），`--ensemble` 打开 ACT 的时间集成（覆盖同一动作序号的各个动作块按 `exp(-系数·i)` 加权平均，旧块权重高），`--max-jump` 是目标与参考的最大偏差（默认 0.35 rad，超过则中止）。关节目标裁剪到限位以内，超出 0.05 rad 以上则中止。Ctrl+C 停止并保持当前位置；结束后臂保持使能。
- 主臂必须不在总线上（`piper-scan` 应只看到 1 台臂）。发运动指令会让主从示教失效，恢复方法见 `robot_toolkit/PIPER.md`。
- 每次运行在 `runs/<目录>/real/<时间>/log.json` 记录完整预测动作块、执行的动作、10 Hz 状态和执行器结果。

## 真机试运行（2026-11-18，`piper_ms_20261118`）

- 数据：采集时间较早的第 0–26 回合外部相机几乎全黑（平均亮度 0.03–0.05），第 27–47 回合正常（约 0.33）。实时画面亮度约 0.33，与后半部分一致。
- `--steps 20`、20 秒：臂一直停在起点附近小幅来回摆动（j2 在 0.72–0.90 之间），没有伸向方块。j4 的预测比限位 1.553 rad 高 0.01–0.03 rad（示教时 j4 本来就贴着限位）。
- `--steps 50`、25 秒：伸向方块（j2 到 1.68），约第 10 秒夹爪闭合，但实测宽度只有 0.011 m（示教中夹住方块时约 0.04–0.05 m），说明没夹到；之后反复张开和闭合，没有抬起。动作时钟只走了 11.8 秒（限速 0.6 rad/s 低于示教的 p99 约 0.9 rad/s，臂跟不上时时钟暂停）。
- 两次都没有触发中止，最大跟踪误差 0.037 rad。
- `--steps 50 --max-speed 0.9 --max-accel 2.5`：每个新动作块的第一步都把 j2 拉回 0.65–0.69（不管当前 j2 是 0.82 还是 1.0），换块时目标跳变越来越大，第 5.3 秒因目标与参考相差 0.352 rad（`--max-jump` 0.35）中止。示教数据中动作与状态没有这种偏差（开头 20 帧平均差 ≤ 0.006 rad），是模型对实时画面的预测偏差。
- `--steps 10 --ensemble 0.01 --max-speed 0.9 --max-accel 2.5`：动作连续，没有来回摆动，动作时钟走了 22.0/25 秒。顺序与示教一致：伸向红块（第 3–6 秒）、第 7 秒闭合夹爪、j1 转向绿块、第 11–13 秒张开，之后再次闭合又张开。闭合时实测宽度 0.022 m。现场观察：红块被抓起，放置位置偏离绿块后掉落，随后策略尝试再次抓取。
- 同样设置再跑 30 秒（红块初始在绿块左前方）：第 7 秒闭合（0.021 m），第 12–15 秒张开；第 18 秒、第 27 秒又各闭合一次，最后一次实测 0.009 m（空夹）。结束时红块在绿块右后方，没有叠到绿块上。
- 结论：执行链路已跑通；加时间集成后能抓起红块，但放置位置偏离，两次都没有完成任务。

## 未接入

- 统一入口的在线评估（`run.py evaluate/view`）：数据来自外部的 MuJoCo 采集程序，不是本工作区的场景项目，尚未接到场景中闭环运行。
- 时间集成在 `piper_real.py` 中按动作序号实现（`--ensemble`），不是 LeRobot 的 `temporal_ensemble_coeff`（后者要求每步推理）。
- 只支持单卡训练（共享 SFT 接入的限制）。

## 依赖

`.venv` 中以 `--no-deps` 加装了 lerobot 0.4.3 及其 ACT 与数据集读取所需的 draccus 0.10.0、typing-inspect、mypy-extensions、mergedeep、diffusers 0.33.1、av 15.1.0、pyserial 3.5、deepdiff 8.6.1、orderly-set 5.5.0，没有升级或替换已有包（直接安装 lerobot 会把 torch 换成不支持 RTX 5080 的版本，diffusers 0.35 要求更新的 peft）。ResNet18 的 ImageNet 权重缓存在 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`。
