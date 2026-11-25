# go2_goal

一层策略：读场景里的量，输出每个关节的动作。新建项目的默认模板。

## 文件

| 文件 | 内容 |
| --- | --- |
| `scene.yaml` | 实体、抽象体，和 `policy`：模型、多久算一次（`period`）、读哪些量（`inputs`）；机器人的全部关节自动受策略控制（`policy.modes` 写例外）。用编辑器或终端修改 |
| `policy.py` | `Policy`（网络）、`decode_action`（动作 → 关节目标）；名义站姿 `STANCE` 和动作缩放 `SCALE` 是策略自己的常数 |
| `task.py` | `evaluate(obs: SceneInfo)`：奖励、成功、失败与指标 |
| `observation.py` | 自动生成，不要手改：输入的类型、形状和单位，以及上面函数的返回类型 |
| `train.yaml` | RLinf 训练配方；`actor.model.hidden_dims` 是网络的层，`action_std` 是动作噪声的初始大小 |

## 要改的地方

- 网络写在 `policy.py` 的 `Policy`：普通的 PyTorch 模块，`forward(obs)` 只返回动作 `[B, 动作个数]`。
  训练时框架自己加上 PPO 需要的动作噪声和价值网络；想自定义价值网络，就在类里写 `value(self, obs)`，
  网络放在名为 `value_head` 的属性里（RLinf 按这个名字给它 `optim.value_lr`）。
- 网络大小和自己的设置写在 `train.yaml` 的 `actor.model` 里（如 `hidden_dims`，或新加的任何键），
  `Policy` 的构造参数写 `config` 就会收到这一段，用 `config.<名字>` 读。不同配方（如另一份 train_*.yaml）可以给不同的值。
- 动作个数就是网络最后一层的大小。构造参数写 `actions` 时，框架传入受控关节总数（这里 12）；
  改成末端控制等其他动作时，直接写成自己的数，并在 `decode_action` 里换算成关节目标（如用 `differential_ik`）。
- 改了 `scene.yaml` 的实体或输入后，先 `check` 刷新 `observation.py`，再按其中的类型改 `policy.py` 与 `task.py`。
- 需要一个不直接控制机器人的上层规划（给出目标点、路径或指令，可以更慢，或是现成的模型、手写规则）时，
  在编辑器“策略”里点“增加一层…”；示例见 planner_executor。

## 运行

```bash
bash run.sh check go2_goal      # 静态检查，刷新 observation.py
bash run.sh edit go2_goal       # 浏览器编辑场景
bash run.sh view go2_goal --num-envs 1 --steps 400
bash run.sh train go2_goal
bash run.sh evaluate go2_goal runner.ckpt_path=$PWD/outputs/go2_goal/runs/<运行>/go2_goal/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt
```

默认后端 Isaac Lab 3 + Newton（`train.yaml` 的 `env.train.init_params.backend/physics`）；换 OvPhysX 在命令后加
`env.train.init_params.physics=ovphysx`，评估、预览跟随训练的选择。
evaluate 不给 `runner.ckpt_path` 时评估的是未训练的策略；每轮跑完整一个回合（`train.yaml` 的 `env.eval`）。
几次 PPO 更新只说明接口跑通，不代表学会任务。

## 当前内容

（新建项目后在这里记录本项目的选择：机器人、输入、动作含义、奖励、已验证到哪一步。）

模板的示例是 Go2 走到 `destination`：策略每 0.02 s 读机身速度、朝向、关节位置与速度、目标点和上一步动作，
输出 12 个数：相对名义站姿 `STANCE` 的偏移（单位 `SCALE` = 0.25 rad）。`decode_action` 返回 `{'robot': STANCE + SCALE × 动作}`
（键是机器人名，列序同 `robot.joint_position`）；复位后第一步之前动作为 0，即保持站姿。
站姿是策略自己定的常数，真机上用同一组数，不读仿真的默认关节位置。
更换机器人或任务后，示例的 `policy.py`、`task.py` 需整体改写，并用本项目的内容替换本段。
