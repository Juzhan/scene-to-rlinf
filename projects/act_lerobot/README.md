# act_lerobot：在 LeRobot 数据集上用 RLinf 训练 ACT

用采集好的示范数据训练 ACT（Action Chunking Transformer）。模型和数据处理直接用 LeRobot 0.4.3 的实现，梯度更新、验证和检查点由 RLinf 的监督训练 worker（经 `shared/integrations/rlinf/sft_project.py`）完成。训练结果可导出为 LeRobot 标准的 `pretrained_model` 目录，交给 `lerobot-record --policy.path=...` 在机器人上运行。

## 文件

| 文件 | 作用 | 何时修改 |
| --- | --- | --- |
| `train.yaml` | 数据集路径、验证回合、ACT 结构与优化参数、训练步数 | 换数据集或调参时 |
| `data.py` | 读 LeRobot 数据集（v3.0），按回合划分训练/验证，每帧配上之后 `chunk_size` 步动作 | 一般不改 |
| `policy.py` | 构建 ACT、归一化统计、训练/验证损失、导出 `pretrained_model` | 一般不改 |

## 数据要求

- LeRobot 数据集目录（含 `meta/info.json`，`lerobot-record` 写出的 v3.0 格式），写到 `data.train_data_paths`。
- 必须有 `observation.state`、`action` 和至少一路相机图像。`actor.model.cameras` 为空时使用全部相机，也可以只列部分。
- 其他字段（如 `observation.sim_time`、深度目录）不进入模型。
- 训练集与验证集按回合划分：默认最后 1 个回合验证（`data.validation_episodes`，也可写回合编号列表）。状态和动作的均值、方差只用训练回合计算；图像用 ImageNet 的均值和方差（ResNet 预训练骨干对应的统计），不从数据计算。

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

## 未接入

- 统一入口的在线评估（`run.py evaluate/view`）：数据来自外部的 MuJoCo 采集程序，不是本工作区的场景项目，尚未接到场景中闭环运行。
- 时间集成（`temporal_ensemble_coeff`）未在部署测试中使用。
- 只支持单卡训练（共享 SFT 接入的限制）。

## 依赖

`.venv` 中以 `--no-deps` 加装了 lerobot 0.4.3 及其 ACT 与数据集读取所需的 draccus 0.10.0、typing-inspect、mypy-extensions、mergedeep、diffusers 0.33.1、av 15.1.0、pyserial 3.5、deepdiff 8.6.1、orderly-set 5.5.0，没有升级或替换已有包（直接安装 lerobot 会把 torch 换成不支持 RTX 5080 的版本，diffusers 0.35 要求更新的 peft）。ResNet18 的 ImageNet 权重缓存在 `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`。
