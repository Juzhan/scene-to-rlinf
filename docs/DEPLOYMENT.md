# Isaac Lab 3 部署方案

范围：只部署 Isaac Lab 3（下称 IL3，后端名 `isaaclab3`），物理引擎用 Newton（默认）或 OvPhysX，
不需要 Isaac Sim / Kit。Isaac Lab 2（`isaac`）与 mjlab 后端的代码仍在仓库里，但不在本方案内部署。

随仓库同步的项目：模板 `planner_executor`，以及 `piper_pick`、`go2_chase`、`go2_piper_wholebody`
三个案例（含已训练权重）。日常命令只有 `new`、`edit`、`check`、`view`（`record`）、`train`、`evaluate`。

## 1. 组成

### 外部库

| 库 | 版本 | 作用 |
| --- | --- | --- |
| Isaac Lab | 3.0（`release/3.0.0` @`961760b`，源码可编辑安装） | 场景、资产、manager（观察/动作/奖励/终止/事件）、环境步进 |
| Newton | 1.6.1（MuJoCo-Warp 3.12.0） | 默认物理引擎，GPU；预览窗口用 Newton GL |
| OvPhysX | 0.6.4 | 可选物理引擎（PhysX，免 Kit），`physics=ovphysx` |
| Warp | 1.17.0 | IL3/Newton 的 GPU 内核 |
| PyTorch | 2.11.0+cu130 | 模型与张量 |
| RLinf | `77ed948`（`third_party/RLinf-upstream`，不改源码） | PPO、rollout、分布式 worker、检查点 |
| Ray | 2.48.0 | RLinf 的进程调度 |
| viser | 1.1.1 | 网页场景编辑器 |

完整包列表冻结在 `shared/backends/isaaclab3/requirements.lock.txt`（`uv pip freeze` 的结果，与验证环境逐包一致）；
`requirements.txt` 记录其中由本工作区增量加入的包及原因（例如 hydra-core 换为 RLinf 固定的版本）。

### 仓库内模块

| 位置 | 作用 |
| --- | --- |
| `run.sh` / `run.py` | 统一入口：解析项目、选择后端解释器、调用 RLinf 或预览；不实现 worker 或训练循环 |
| `projects/<名称>/` | 用户项目：`scene.yaml`、`policy.py`、`task.py`、`train.yaml`，`observation.py` 自动生成 |
| `shared/assembly/` | 与后端无关的公共层：场景描述与编辑（`SceneDocument`）、观察目录与类型生成、planner/executor 调度、Python Task 运行时、抽象体 |
| `shared/backends/isaaclab3/` | IL3 绑定：资产工厂、场景编译与连接、原生控制项、读数、Task 接入、编辑器网格导出 |
| `shared/integrations/rlinf/` | RLinf 接入：对象式策略适配（采样、logprob、value）、配置校验 |
| `tools/create.py` | `new`：复制模板生成项目 |
| `tools/web_editor.py` | `edit`：网页编辑器（viser），写回同一份 `scene.yaml` |
| `tools/preview.py` | `view` / `record`：实时窗口或离屏录像，可回放策略 |
| `tools/smoke.py` | `smoke`：几步接口、复位、终止检查 |
| `tools/asset_bundle.py` | 资产包的打包、下载、校验与解压 |
| `tools/inspect_scene.py` | 内部使用：`observation_dim/action_dim: auto` 时短暂启动环境读取实际维度；不是用户命令 |

### 不随 git 同步的内容

| 内容 | 获取方式 |
| --- | --- |
| `assets/`（机器人、物体、地形、地面、天空盒，约 157 MiB 压缩包） | `python3 tools/asset_bundle.py fetch`，见第 5 节 |
| `third_party/RLinf-upstream` | `git clone` 后 checkout 固定提交 |
| Isaac Lab 3 源码 | `git clone` 后 checkout 固定提交，可放在仓库外 |
| `.venv-il3`（唯一的 Python 环境） | 按第 2 节创建 |
| `outputs/`、`.cache/` | 运行时生成（训练记录、预览、编辑器网格缓存） |

项目权重 `projects/*/weights/*.pt` 很小（各 1–3 MB），随 git 同步；来源记录在同目录的 `WEIGHTS.json`。

## 2. 安装

### 硬件与系统

- Linux x86_64，NVIDIA GPU，驱动支持 CUDA 13（已验证 RTX 5080 16 GB，驱动 580.178）。
- 主机内存：已验证机器为 62 GB；16 环境评估约 9 GB，wholebody 1024 环境训练需按场景估算。
- `uv`、`git`、Python 3.12。

### 步骤

```bash
git clone <仓库地址> rlinf-workspace && cd rlinf-workspace

# 1. 资产（下载、校验 SHA-256、解压到 assets/）
python3 tools/asset_bundle.py fetch

# 2. RLinf（固定提交，不改源码）
git clone https://github.com/RLinf/RLinf.git third_party/RLinf-upstream
git -C third_party/RLinf-upstream checkout 77ed948d68f07f55b4bc433b7643a409b2cd1adb

# 3. Isaac Lab 3 源码
IL3=$HOME/repos/IsaacLab-3.0
git clone -b release/3.0.0 https://github.com/isaac-sim/IsaacLab.git $IL3
git -C $IL3 checkout 961760bcb91e3af48e7d116292e57e8ecaefc31e

# 4. 环境：按冻结列表逐包安装（--no-deps 保证与验证环境一致），仿真、训练和网页编辑器都在这一个环境里
uv venv --python 3.12 .venv-il3
uv pip install --python .venv-il3/bin/python --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    -r shared/backends/isaaclab3/requirements.lock.txt \
    $(for p in isaaclab isaaclab_assets isaaclab_contrib isaaclab_experimental isaaclab_newton isaaclab_ov \
               isaaclab_physx isaaclab_ppisp isaaclab_rl isaaclab_tasks isaaclab_tasks_experimental \
               isaaclab_visualizers; do echo "-e $IL3/source/$p"; done)
```

冻结列表本身不能由解析器重新求解（验证环境是增量安装的，numpy 2.5.1 与 cmeel-boost 的声明范围不符，实际运行正常），
所以用 `--no-deps`。所有命令（含 `edit`）都由 `run.sh` 自动使用 `.venv-il3`，不需要激活环境；
环境放在别处时用 `RLINF_PYTHON` 指定解释器。

### 验证安装

```bash
python3 tools/asset_bundle.py verify
bash run.sh check piper_pick                                   # 静态检查，不启动仿真
bash run.sh evaluate piper_pick env.eval.total_num_envs=16     # 约 1 分钟，success_once 应为 1.0
```

以上步骤已在全新目录、全新 `.venv-il3` 上复现（2026-10-28）：4 个项目 check 通过，`new` + `smoke` 通过，
piper_pick 评估 147 回合成功率 1.0，go2_chase 评估碰到率 1.0、摔倒 0，
go2_piper_wholebody（OvPhysX）16 条回合全部跑满 1000 步、失败 0。

### 网络

- 安装阶段需要访问 GitHub、PyPI、PyTorch 源和资产包地址（Google Drive）。访问不了 Google Drive 时，在浏览器下载压缩包后用 `fetch --file <压缩包>`，校验方式相同。
- 运行时完全离线：四个项目用到的机器人、物体、地形、地面和天空盒都从本地 `assets/` 加载，代码用
  `nucleus_copy` 取本地副本，文件缺失时报错并给出 vendor 命令，不回退到联网读取。
  已在断网（`unshare -rn`）且清空临时目录的情况下验证 `record`、`evaluate`（三个案例，含 OvPhysX）、`new` + `smoke` 与编辑器网格导出（2026-10-28）。
- RLinf 训练用的 Ray 只使用本机回环地址。

## 3. 日常流程

```bash
bash run.sh new my_task                       # 从模板 planner_executor 复制（backend: isaaclab3）；也可在编辑器中“新建”
bash run.sh edit my_task                      # 网页编辑器，http://127.0.0.1:8080（--port 改端口）
bash run.sh check my_task                     # 静态检查，刷新 observation.py
bash run.sh view my_task --actions policy     # 实时窗口；默认零动作，--actions policy 运行项目策略，--checkpoint 指定训练权重
bash run.sh record my_task --actions policy --steps 300   # 离屏录像，写到 outputs/<项目>/previews/
bash run.sh train my_task                     # RLinf PPO；--board 同时开 TensorBoard
bash run.sh evaluate my_task env.eval.total_num_envs=16
```

- 物理引擎：命令后加 `env.train.init_params.physics=ovphysx`，评估、预览随之切换。
- 后端由项目 train.yaml 的 `env.train.init_params.backend` 决定；没写的旧项目（如 apple）沿用 `configs/state_ppo.yaml` 的 `isaac`（Isaac Lab 2），`check` 会打印 WARN。要在 IL3 上运行，先在其 train.yaml 写上 `backend: isaaclab3`。
- 训练产物：`outputs/<项目>/runs/<时间>/`，含有效配置与项目源码快照；用
  `evaluate <项目> --snapshot <运行目录> runner.ckpt_path=<.../full_weights.pt>` 按当时的代码回放。
- `check` 与编辑器保存时刷新 `observation.py`；改了观察或控制终点后先 `check`，再按新类型改 `policy.py` / `task.py`。

要写代码的位置（详见各文件开头的说明）：

| 文件 | 写什么 |
| --- | --- |
| `scene.yaml` | 实体、抽象体、各组件的观察选择、控制终点；用编辑器或终端修改 |
| `policy.py` | `build_planner` / `build_executor`（网络、特征、权重加载），`decode_action`（动作 → 各终点的实际单位目标），可选 `initial_action` |
| `task.py` | `evaluate(obs)` 或 `Task.evaluate(self, obs)`：返回 reward / success / failure / metrics，可选 `set`；跨步状态用 `reset(env_ids)` 或 `TaskMemory` |
| `train.yaml` | 后端与物理引擎、环境数、回合长度、PPO 参数、模型参数 |

## 4. 案例

### piper_pick：机械臂抓取

Piper X 从地面抓起方块，抬升 4 cm 稳住 0.5 s 即成功。executor 用 `pick.pt`（在 mjlab 上训练）。

```bash
bash run.sh view piper_pick --actions policy
bash run.sh evaluate piper_pick env.eval.total_num_envs=16
bash run.sh evaluate piper_pick env.eval.total_num_envs=16 env.train.init_params.physics=ovphysx
```

### go2_chase：四足追球

Go2 追一个会逃跑的虚拟球（抽象体，由 `task.py` 的 `set` 每步移动）。executor 是
[go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) 发布的平地 actor（`flat.pt`，未重新训练），
场景使用与权重一致的 supermimic 电机。另有粗糙地形配方 `--config train_terrain`。

```bash
bash run.sh view go2_chase --actions policy
bash run.sh evaluate go2_chase env.eval.total_num_envs=16
bash run.sh evaluate go2_chase --config train_terrain env.eval.total_num_envs=16
```

### go2_piper_wholebody：四足加机械臂全身控制

Go2 背 Piper X，在粗糙地块上同时跟踪机身速度命令和末端目标（复刻 LeggedManip_Lab GO2-PIPER），
`wholebody.pt` 在 mjlab 上训练。

```bash
bash run.sh view go2_piper_wholebody --actions policy
bash run.sh evaluate go2_piper_wholebody env.eval.total_num_envs=16 env.train.init_params.physics=ovphysx
```

### 新项目

`new` 生成的 Go2 走向目标点示例可以直接 `check`、`view`、`train`，再按自己的机器人和任务改写
`scene.yaml`、`policy.py`、`task.py`，并把项目 README 的“当前内容”换成本项目的说明。

## 5. 资产与权重的维护

- 资产包：`assets.lock.json`（随 git）记录版本、压缩包名、下载地址、SHA-256 与包含的资产目录。
  - 使用者：`python3 tools/asset_bundle.py fetch`（或 `--file <本地压缩包>`）；已存在的目录不覆盖，`--force` 替换。
  - 维护者：新增或更新资产后 `python3 tools/asset_bundle.py pack [目录 ...]`（默认按 lock 中的目录），
    把 `outputs/asset_bundle/rlinf-assets-<版本>.tar.gz` 上传到 Google Drive，共享设为“知道链接的任何人”，把分享链接（`https://drive.google.com/file/d/<ID>/view…`）写入 lock 的 `url` 并提交。
    `fetch` 会把分享链接换成直接下载地址（跳过大文件的病毒扫描确认页）；拿到网页而不是压缩包时报错，不会解压。
    旧版本压缩包保留，使旧提交仍能取到对应资产。
  - 新资产用 `run.py asset convert` 生成 USD 与 MJCF 并通过对齐检查，记录 `ASSET.json`（只在维护资产时需要，依赖 mjlab，不在本部署环境内）。
  - 当前包含：anymal_c、clear_sky、dex_cube、franka_panda、go2、ground_plane（IL3 默认棋盘地面）、m20、piper、piper_x、
    seattle_table、terrains（地形地块）、white_marble。只被未同步的旧实验使用的资产不打包。
- 权重：放在 `projects/<项目>/weights/`，`WEIGHTS.json` 写清来源（仓库、提交或训练运行目录、步数）、
  输入输出约定与许可证。修改输入顺序、维度或动作含义时换新文件名，不覆盖旧权重。

## 6. 当前验证状态（IL3）

| 项目 | Newton | OvPhysX | 说明 |
| --- | --- | --- | --- |
| planner_executor | 接口跑通 | — | 4 环境 × 16 步 smoke 契约通过；示例策略未训练 |
| piper_pick | 完整回合成功（150 回合 100%） | 完整回合成功（108 回合 100%） | 权重来自 mjlab；IL3 短训练只验证到权重更新 |
| go2_chase | 完整回合成功（平地 100%，粗糙地形 95.5%） | 完整回合成功（平地 100%） | 外部发布权重，未在 IL3 训练 |
| go2_piper_wholebody | 部分回合完整（63% 轨迹翻倒） | 完整回合成功（0 失败，末端误差 5.6 cm） | Newton 接触与焊接较软，机身低约 4 cm |

没有项目达到“多条件可靠成功”：都只在各自的默认初始范围内评估，未系统改变物体尺寸、摩擦、质量或地形。
各项目的详细数字见其 README，后端实现细节与差异见 [SCENE_BACKENDS.md](SCENE_BACKENDS.md)。

## 7. 设计思路

- 仿真器负责物理、控制、传感器和环境推进；RLinf 负责采样、优化、分布式和检查点。项目不写 worker、runner 或训练循环。
- 项目只描述场景、观察、动作和任务：模型通过 `obs.robot.joint_position` 这样的对象属性读取张量，
  不依赖字段字符串或拼接好的状态向量；`observation.py` 由场景生成，给出形状、单位和坐标系。
- Task 用 Python/Torch 直接写奖励、成功和失败；场景配置不承担任务语义。
- planner / executor 两层，`control.train` 选择训练哪层；控制终点只声明实体、关节和控制类型，换算在 `decode_action` 中完成。
- 后端相关代码都在 `shared/backends/<后端>/`，上层只调用各后端的同名钩子，不写 `if backend == ...`；
  某后端不支持的能力在钩子内明确报错。
- 不修改第三方源码，不用全局 monkey patch；需要的补充放在共享接入层。
- 验证分级：接口跑通、权重更新、完整回合成功、多条件可靠成功，汇报时写明到了哪一级。

## 8. TODO

- 完善 IL3 接入：Newton 上 wholebody 的接触/焊接参数，IL3 上的 PPO 训练验收，传感器与连接类型补全。
- 界面交互：编辑器的选择、聚焦、拖动体验，模板选择与项目管理。
- 动态场景扩展：Task 移动物理实体（目前只能移动抽象体），回合内增删物体。
- 虚幻引擎接入：作为新后端按钩子清单分阶段接入，先渲染与场景读取，再控制与训练。
- 资产库：资产包版本化、更多机器人与物体的 USD/MJCF 对齐。
- 测试库：把部署验证（check、smoke、评估基线）整理成可重复执行的回归测试。
- 策略库：已训练权重的登记（来源、接口、适用场景、验证结果）。
- 任务库：常用 Task 片段（到达、抓取、保持、跟踪）的复用。
