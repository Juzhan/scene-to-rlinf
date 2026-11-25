# planner_executor

两层策略的模板：和 go2_goal 是同一个任务（Go2 走到目标点），拆成两层写，对照着看什么时候、怎么分层。

- `planner`（固定，每 0.2 s）：把 `destination` 换算成目标，返回字典 `{'goal', 'heading'}`。
- `executor`（训练，每 0.02 s）：读机器人状态和 `obs.planner.goal` / `obs.planner.heading`，输出 12 个腿关节动作。

planner 是上层规划：不直接控制机器人，只把目标交给 executor，可以更慢，也可以换成路径规划、预训练模型或手写规则，
executor 的输入接口不变。不需要这样一个上层时，用一层（go2_goal）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `scene.yaml` | 实体、抽象体，和 `policy`：两层（`layers`）各自的模型、频率（`period`）、输入（`inputs`；写层名 = 读那一层的结果），`train` 选择训练哪层；机器人的全部关节自动受控（`policy.modes` 写例外） |
| `policy.py` | `Planner`、`Executor`（网络）、`decode_action`（动作 → 关节目标）；名义站姿 `STANCE`、动作缩放 `SCALE` 是策略自己的常数 |
| `task.py` | `evaluate(obs: SceneInfo)`：奖励、成功、失败与指标 |
| `observation.py` | 自动生成，不要手改：每层输入的类型、形状和单位，以及函数的返回类型 |
| `train.yaml` | RLinf 训练配方；`actor.model.hidden_dims` 是 executor 的层，`action_std` 是动作噪声的初始大小 |

## 要改的地方

- 两层都是普通的 PyTorch 模块。上层返回字典，每项后面写 `# [B, 3] 说明`，编辑器两层之间的箭头上显示这些名字，说明在悬停提示里；
  下层用 `obs.planner.<键>` 读。被训练的层只返回一个动作张量，PPO 需要的动作噪声和价值网络由框架补上。
- 网络大小和自己的设置写在 `train.yaml` 的 `actor.model` 里，`Executor` 的构造参数 `config` 收到这一段。
- 想训练 planner、固定 executor（比如 executor 是训练好的行走网络）：把 `train` 改成 `planner`，executor 用
  `parameters.weights` 加载权重；planner 的输出就是被训练的动作。
- 改了 `scene.yaml` 的实体或输入后，先 `check` 刷新 `observation.py`，再按其中的类型改 `policy.py` 与 `task.py`。

## 运行

```bash
bash run.sh check planner_executor      # 静态检查，刷新 observation.py
bash run.sh edit planner_executor       # 浏览器编辑场景
bash run.sh view planner_executor --num-envs 1 --steps 400
bash run.sh train planner_executor
bash run.sh evaluate planner_executor runner.ckpt_path=$PWD/outputs/planner_executor/runs/<运行>/planner_executor/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt
```

默认后端 Isaac Lab 3 + Newton（`train.yaml` 的 `env.train.init_params.backend/physics`）；换 OvPhysX 在命令后加
`env.train.init_params.physics=ovphysx`，评估、预览跟随训练的选择。
evaluate 不给 `runner.ckpt_path` 时评估的是未训练的策略；每轮跑完整一个回合（`train.yaml` 的 `env.eval`）。
几次 PPO 更新只说明接口跑通，不代表学会任务。

## 当前内容

（新建项目后在这里记录本项目的选择：机器人、输入、动作含义、奖励、已验证到哪一步。）

示例：planner 每 0.2 s 给出目标位置和朝向；executor 每 0.02 s 输出 12 个数，相对名义站姿 `STANCE` 的偏移
（单位 `SCALE` = 0.25 rad），`decode_action` 返回 `{'robot': STANCE + SCALE × 动作}`；动作为 0 时保持站姿。
2026-11-25 从旧写法（`control:`、`build_planner` / `build_executor`、读仿真默认关节位置）改为现在的写法；之前没有训练好的权重。
更换机器人或任务后，示例的 `policy.py`、`task.py` 需整体改写，并用本项目的内容替换本段。
