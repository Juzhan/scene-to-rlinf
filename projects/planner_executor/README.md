# planner_executor

planner 生成目标，executor 输出关节控制；`scene.yaml` 的 `control.train` 选择训练哪一个。

## 文件

| 文件 | 内容 |
| --- | --- |
| `scene.yaml` | 实体、抽象体、各组件的观察选择、控制终点（`control.output`）；用编辑器或终端修改 |
| `policy.py` | `build_planner`、`build_executor`、`decode_action`、`initial_action` |
| `task.py` | `evaluate(obs: SceneInfo)`：奖励、成功、失败与指标 |
| `observation.py` | 自动生成，不要手改：观察类型和上面函数的返回类型 |
| `train.yaml` | RLinf 训练配方 |

## 要改的地方

- 改了 `scene.yaml` 的实体、观察或控制终点后，先 `check` 刷新 `observation.py`，再按其中的类型改
  `policy.py` 与 `task.py`；`view`、`train`、`evaluate` 都会构建策略，二者须先与场景一致。
- `policy.py`、`task.py` 文件开头的说明列出共享层调用的函数、参数和返回约定。
- 网络结构、特征、缩放和偏移都写在 `policy.py`；奖励、阈值和计时写在 `task.py`。
  函数返回类型与约定见 `observation.py` 末尾，说明见工作区根 README。
- Task 需要某个连杆的位置（如夹爪末端）时，在场景中添加 `frame_transformer`，`target_body` 填连杆名。

## 运行

```bash
bash run.sh check planner_executor      # 静态检查，刷新 observation.py
bash run.sh edit planner_executor       # 浏览器编辑场景（viser）
bash run.sh view planner_executor --num-envs 1 --steps 400
bash run.sh train planner_executor
bash run.sh evaluate planner_executor runner.ckpt_path=$PWD/outputs/planner_executor/runs/<运行>/planner_executor/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt
```

默认后端 Isaac Lab 3 + Newton（`train.yaml` 的 `env.train.init_params.backend/physics`）；换 OvPhysX 在命令后加
`env.train.init_params.physics=ovphysx`，评估、预览跟随训练的选择。
evaluate 不给 `runner.ckpt_path` 时评估的是未训练的策略；每轮跑完整一个回合（`train.yaml` 的 `env.eval`）。
几次 PPO 更新只说明接口跑通，不代表学会任务。

## 当前内容

（新建项目后在这里记录本项目的选择：机器人、观察、动作含义、奖励、已验证到哪一步。）

模板创建时的示例是 Go2 行走：planner 每 0.2 s 把 `destination` 的位置和朝向换算成 4 维目标，
executor 每 0.02 s 输出 12 个关节角，`decode_action` 返回 `{'legs': action}`。
更换机器人或任务后，示例的 `policy.py`、`task.py` 需整体改写，并用本项目的内容替换本段。
