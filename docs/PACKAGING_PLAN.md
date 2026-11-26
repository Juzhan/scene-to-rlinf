# 模块解耦与按需安装：分析与计划

目标：像 `pip install pkg[extra]` 那样按用途装环境。只想编辑场景的人不装仿真器，只想跑仿真预览的人不装训练框架，
真机部署只装推理需要的部分；每个命令缺依赖时明确说出缺哪一组、怎么装。
本文写于 2026-11-26：第 1–3 节是当时的现状分析与计划；阶段 1、2、“删 IL2 目录”的两处维护工具和 3.1 的
URDF/MJCF 转换已实施，其余 3.x 未做，结果与偏差见 §6。

## 1. 现状

### 分层与规模

| 层 | 位置 | 规模 | 说明 |
| --- | --- | --- | --- |
| 前端 | `tools/editor_ui/`（app.js / app.css）、`tools/viser_client/`（打补丁后构建好的 viser 页面）、`tools/web_editor.py`、`tools/simready_library.py` | web_editor.py 7143 行，app.js 约 1600 行 | 网页编辑器；Blender 插件不入库 |
| 核心（与后端无关） | `shared/assembly/`（39 个文件，约 8.9k 行）、`shared/policy.py`、`shared/predicates.py` | | 场景描述与编辑、观察目录与类型生成、策略分层、Python Task 运行时 |
| 资产 | `shared/asset_import.py`、`shared/asset_formats.py`、`shared/simready.py`、`shared/environments.py` | asset_formats 3192 行 | 导入、格式转换、对齐检查、SimReady |
| 后端 | `shared/backends/{isaaclab3, isaac, mjlab, mujoco}` | | 仿真绑定；isaaclab3 为主，isaac（IL2）与 mjlab 冻结 |
| 训练 | `shared/integrations/rlinf/`、`run.py` 的 train / evaluate | | RLinf 适配 |
| 命令与脚本 | `run.py`、`tools/*.py`（约 40 个） | | 其中约一半是历史实验脚本：smolvla_*、pi05_libero_probe、lift_ablation、frankasim_reference、rsl_reference、go2_evaluation、stack_evaluation、import_bowl_corrections |

### 依赖与环境

- `pyproject.toml` 只声明 hydra-core、omegaconf、ruamel.yaml，没有可选依赖组。
- 实际依赖由环境决定：
  - `.venv-il3`：约 17 GB，258 个包冻结在 `shared/backends/isaaclab3/requirements.lock.txt`，仿真、训练、编辑器全在里面。
  - `.venv-editor`：192 MB，有 viser 和 trimesh，没有 torch。
  - `.venv`：带 Isaac Sim 的 conda 环境，URDF/MJCF → USD 转换要用它（现已不需要，见“3.1 实施”）。
  - 另有 `.venv-mjlab` 和若干实验环境。
- 选择解释器的入口有 5 处：
  - `run.sh`：依次找 `.venv` / `.venv-il3`；
  - 后端 `INTERPRETER` 声明；
  - `RLINF_PYTHON`；
  - `RLINF_EDITOR_PYTHON`；
  - `ISAAC_PYTHON` / `RLINF_EXPORT_PYTHON`（导入转换、编辑器网格导出）。

### 已经做对的地方（实测）

- 在没有 torch 的 `.venv-editor` 里，下面这些模块都能直接导入：
  - `shared.assembly` 的 layout、editing、catalog、object_schema、project；
  - `shared.asset_import`、`tools.web_editor`、`tools.create`。
- 重依赖（torch、isaaclab、newton、rlinf、pxr、mujoco、viser）在大部分模块里都是函数内延迟导入，基础不错。
- `check` 在这个环境里只差 hydra（读 train.yaml）一个包。

### 问题清单

| # | 问题 | 证据 | 影响 |
| --- | --- | --- | --- |
| 1 | 核心层反向依赖 IL2 后端目录 | `catalog.py`、`editing.py`、`project.py` 导入 `shared/backends/isaac/` 的 `scene_metadata`、`task_config`（选择表、校验）、`entity_observations`（信号解析）、`derived`、`task_evaluation`。这些文件本身没有重依赖，是放错位置的通用代码。`catalog.py` 里还写着 `isaaclab.envs.mdp.*` 实现名 | 去掉或冻结 IL2 后端会连核心一起坏；违背 AGENTS §1 |
| 2 | 预览依赖训练框架 | `tools/preview.py` 顶层导入 `shared.isaac_state`，后者顶层导入 rlinf | `view` / `record` 即使不回放权重也要装 RLinf |
| 3 | 通用环境边界依赖训练集成包 | `shared/backends/environment.py` 从 `shared.integrations.rlinf.terminal` 取 `snapshot`（`snapshot` 本身只用 torch / numpy） | 后端层 → 训练层的反向依赖 |
| 4 | shared/ 顶层散落旧模块 | `isaac_state.py`（rlinf）、`observations.py` / `rewards.py` / `terrains.py`（isaaclab 顶层导入）、`policies.py`（rlinf）只被 IL2 后端和旧项目使用；`policy_math.py` 被 predicates 和模板使用 | 看不出哪些是公共能力；装哪组都会碰到 |
| 5 | URDF/MJCF 导入需要 Isaac Sim | `asset_formats.to_usd` 用 Isaac Lab 的 Kit 转换器，解释器默认 `.venv`（带 Isaac Sim 的环境） | IL3 单环境部署不能导入 URDF/MJCF；文档里“依赖 mjlab”的说法也不完整 |
| 6 | 解释器发现规则分散 | 5 个变量或约定，各自报错 | 用户不知道装哪个、设哪个 |
| 7 | 编辑器服务端单文件过大 | `web_editor.py` 7143 行，混合 viser 场景、HTTP API、导入 / SimReady 任务、网格导出、经典面板 | 难拆出“无仿真也能用”的部分，难单测 |
| 8 | tools/ 混着产品命令和研究脚本 | 约 20 个历史实验脚本与 create / preview / smoke 并列 | 打包时分不清哪些属于产品 |
| 9 | 没有独立的推理运行时 | 真机部署只需要 `policy.py`、权重、观察类型、ObjectView、`decode_action`，但这些散在 `shared/assembly` 里，和编辑、场景编译放在一起 | 真机侧只能整仓带上；act_lerobot 走的是另一套（robot_toolkit） |
| 10 | 只有一份全量锁文件 | 258 包一起冻结 | 无法按组装出与验证环境版本一致的小环境 |

## 2. 目标：功能组

IL3 与 RLinf 目前都从源码安装（固定提交），不能全靠 PyPI 的 extras 表达。所以分成两部分：

- PyPI 上能装的部分写成 `pyproject` 的可选依赖组；
- 源码依赖由一个安装脚本按组拉取，例如 `tools/setup_env.py --with editor,isaaclab3,train`。

| 组 | 能做什么 | 新增依赖 | 现状 |
| --- | --- | --- | --- |
| `core` | `new`、`check`、终端编辑 scene.yaml、生成 observation.py | hydra-core、omegaconf、ruamel.yaml、numpy | 基本可行，缺 hydra 即可 |
| `editor` | 网页编辑器（无仿真时用示意代理） | viser、trimesh、pillow | 可行；需要问题 1 理顺后才稳定 |
| `import` | 导入网格 / USD / URDF / xacro / MJCF | usd-exchange、urdf-usd-converter、trimesh、mujoco、xacro | 网格、USD、URDF 可行（URDF 转换已不依赖 Isaac Sim，见 3.1 实施）；MJCF 转换在 `usd` 组 |
| `usd` | 不启动 Kit 的 MJCF → USD | mujoco-usd-converter（自带 mujoco-core 4.x、usd-exchange） | 可行（见 3.1 实施）；必须单独一个环境，与 `import` 组的 mujoco 3.12 不能同装 |
| `isaaclab3` | `view` / `record` / `smoke` / `evaluate`（未训练策略或项目自带权重），编辑器原生网格 | torch、Isaac Lab 3（源码）、newton、warp；可选 ovphysx | 可行；需要问题 2 理顺后才不必装 RLinf |
| `train` | `train`、用训练检查点 `evaluate` | RLinf（源码）、ray、tensorboard | 现状 |
| `deploy` | 在真机或其它进程里加载 policy.py 与权重推理 | torch | 缺独立运行时（问题 9） |
| `mjlab`、`isaac` | 冻结后端 | 各自 SDK | 保持现状，不扩展 |
| `blender` | Blender 插件 | bpy（Blender 自带） | 不入库，保持外置 |

依赖关系：`editor`、`import`、`isaaclab3`、`deploy` 都含 `core`；`train` 含 `isaaclab3`（或另一个仿真组）。

## 3. 分阶段计划

### 阶段 1：理顺边界（不改用户接口，优先做）

| 编号 | 改动 | 验收 | 工作量 / 风险 |
| --- | --- | --- | --- |
| 1.1 | 把 `shared/backends/isaac/` 里被核心引用的通用代码搬进 `shared/assembly`：`scene_metadata`、`task_config` 的选择表与校验、`entity_observations` 的信号解析、`derived.variable_information`、`task_evaluation.validate_sources`。IL2 后端改从核心导入；`catalog.py` 里的 `isaaclab.envs.mdp.*` 实现名改由各后端声明 | `grep` 不到上层对具体后端包的导入（只经 `backend_module`）；临时移走 `shared/backends/isaac` 后，core / editor 测试通过 | 中；搬动多、行为不变，现有测试覆盖较全 |
| 1.2 | `snapshot` 移到后端通用层；`preview.py` 不再顶层导入 `isaac_state` / rlinf，只在用 `--checkpoint` 回放训练权重时才需要 | 没装 RLinf 的 IL3 环境能 `view` / `record` / `smoke` | 小 |
| 1.3 | shared/ 顶层旧模块归位：`isaac_state` / `observations` / `rewards` / `terrains` / `policies` 移到 `shared/backends/isaac/legacy/`（旧项目路径保留转发或同步改）；`policy_math` 并入 `shared/policy.py` | shared/ 顶层只剩公共能力 | 小到中；要核对旧项目引用 |
| 1.4 | 历史实验脚本移出 `tools/`，放到 `experiments/` 或各自项目目录 | `tools/` 只剩 run.py 子命令的实现和维护工具 | 小 |

### 阶段 2：功能组与安装

| 编号 | 改动 | 验收 | 工作量 / 风险 |
| --- | --- | --- | --- |
| 2.1 | `pyproject.toml` 写可选依赖组 core / editor / import / isaaclab3 / train / deploy；`tools/setup_env.py --with <组>` 用 uv 建环境，源码依赖（IL3、RLinf）按组拉取到固定提交 | 新目录里装 `core,editor` 不到 300 MB，能 `new` / `check` / `edit`（示意代理） | 中 |
| 2.2 | 每个 run.py 命令声明需要的组；入口统一做能力探测，缺什么就报“需要 [train]：`tools/setup_env.py --with train`”；把 5 处解释器发现合并到一个注册表（组 → 解释器），后端的 `INTERPRETER` 声明保留 | 每个命令在缺组时都给出同一种提示；只剩一个环境变量（可选，覆盖解释器） | 中 |
| 2.3 | 分组锁文件 `requirements/<组>.lock.txt`，从同一个验证环境冻结，组之间版本一致 | 按组装出的环境与验证环境同名包版本一致 | 小 |
| 2.4 | 测试按组加标记，各组在只装该组的干净环境里跑 | core / editor 测试在小环境通过；部署文档写明每组验证到的命令 | 中 |

### 阶段 3：去掉剩余的重依赖与大文件

| 编号 | 改动 | 验收 | 工作量 / 风险 |
| --- | --- | --- | --- |
| 3.1 | 两步导入与不依赖 Kit 的转换，见下文“3.1 导入” | `import` 组不装 Isaac Sim 也能导入 xarm、piper 等，对齐结果与现在一致 | URDF 与 MJCF 转换已实施（见“3.1 实施”）；两步导入未做 |
| 3.2 | 拆分 `web_editor.py`：HTTP / API、viser 场景与交互、后台任务（导入、SimReady、网格导出）、经典面板（评估是否还需要）；`app.js` 按区域拆模块 | 单文件不超过约 1500 行；无仿真环境的编辑器测试覆盖 API 层 | 大；可以按区域逐步拆 |
| 3.3 | 推理运行时 `shared/runtime`（ObjectView、策略辅助、`decode_action` 校验、observation 类型），只依赖 torch | 只装 `deploy` 能加载 piper_pick 的 policy.py 与权重，对一帧观察推理，结果与仿真内一致 | 中 |
| 3.4 | 没有仿真组时，编辑器尽量用 pxr / trimesh 读出真实网格，不只是示意代理（`mesh_preview` 已有一部分） | `editor` + `import` 环境里看到的网格与原生导出接近 | 中 |

### 3.1 导入：哪些需要 Isaac Sim，怎么拆

现状：

| 导入的文件 | 怎么变成仿真用的 USD | 需要 |
| --- | --- | --- |
| OBJ、STL、PLY、glTF/GLB、DAE、OFF、3MF | `mesh_to_usd`：trimesh 读网格，pxr 直接写 USD，算好凸包碰撞、质量和惯量，几秒完成 | usd-exchange（提供 pxr）、trimesh；不需要 Isaac |
| FBX、.blend | 先由 Blender 读出网格，再同上 | 本机 Blender |
| USD / USDZ | 原样使用；必要时加一层包装（补质量、固定根部等），用 pxr | usd-exchange |
| URDF、ROS xacro | `urdf-usd-converter` 直接写 USD（不启动 Kit，见“3.1 实施”）；同时用 MuJoCo 编译出 MJCF，并做两者的对齐检查 | urdf-usd-converter、mujoco、xacro |
| MJCF | `mujoco-usd-converter` 直接写 USD（不启动 Kit，见“3.1 实施”），与源 MJCF 做对齐检查 | mujoco-usd-converter（自带 MuJoCo 4.x） |

两个转换器都没装时才回退到 Isaac Lab 的导入器（要 Isaac Sim）。部署用的 IL3 环境（Isaac Lab 3 + Newton）本身没有 Isaac Sim / Kit（`.venv-il3` 里没有 `isaacsim` 和 `omni.kit`），
所以“等需要执行时再用 Isaac 转换”在 IL3 上同样转不了：推迟只是挪了时间，关键是不依赖 Kit 的转换器。

拆成两步：

1. **登记与预览（立即，不需要仿真器）**：保存源文件；解析 URDF，或用 MuJoCo（pip 包）编译 URDF / MJCF，得到连杆、关节、
   网格和位姿。编辑器马上能显示、列出关节、拖关节预览；`check` 能生成观察类型。只需要 `editor` + `import` 组。
2. **生成仿真用的 USD（按需，按源文件哈希缓存）**：导入时如果有转换器就立即做，否则在第一次 smoke / view / train 时做。
   源文件不变就复用结果，沿用现有的转换记录机制（`mjcf_for` 已经这样做）。没有转换器时明确提示缺哪一组依赖。

转换器选 **A（不依赖 Kit 的转换器）**，其余两种方案不采用：

| 方案 | 做法 | 结论 |
| --- | --- | --- |
| **A. 不依赖 Kit 的转换器（采用）** | NVIDIA 为 Newton 开源的 `urdf-usd-converter` / `mujoco-usd-converter`，只依赖 usd-exchange | 采用。URDF 与 MJCF 都已接入并核对完驱动、mimic、碰撞、关节顺序与对齐结果（见“3.1 实施”）；MJCF 转换器单独在 `usd` 组 |
| B. 转换留作维护工具 | 继续在带 Isaac Sim 的环境里转换 | 不采用：普通用户和 IL3 部署环境都导入不了 URDF / MJCF，按需安装的目标落空 |
| C. 运行时不转 USD | Newton 用 `ModelBuilder.add_urdf` / `add_mjcf` 直接读 | 不采用：IL3 的场景、OvPhysX、编辑器都以 USD 为准，会多出一条不同的资产路径 |

Isaac Lab 的转换器现在只在两个独立转换器都没装时作为回退，导入不再需要 Isaac Sim。

要守住的约定：**关节顺序**。现在的 `joint_order` 取 Isaac 转换后的顺序，观察和动作的列序都以它为准（ASSET.json）。两步导入时，
第 1 步就要把顺序定下来写入 ASSET.json，第 2 步转换后核对；不一致就报错，不静默换序。
调研方案 A 时一并确认它的关节顺序规则。对齐检查（USD 与 MJCF）同样保留，不通过时照旧在导入提示里说明。

## 4. 建议顺序

1.1 → 1.2 → 1.3 / 1.4 → 2.1 / 2.2 → 2.3 / 2.4，然后视需要做 3.x。

- 阶段 1 只是搬家、调整导入，不改项目写法和命令，风险最低，又是后面按组安装的前提。
- 3.1 决定 `import` 组能不能脱离 Isaac Sim，建议先单独调研。

## 5. 不在本计划内

- 后端实现本身的功能补全（传感器、连接类型）。
- 多条件评估、训练算法扩展（SAC / GRPO 等）。
- mjlab 与 IL2 后端的新功能（按约定冻结）。

## 6. 实施结果（阶段 1、2 已完成；3.1 的转换部分已完成）

阶段 3 是"视需要"的部分：3.1 的 URDF 与 MJCF 转换都做了（见“3.1 实施”），两步导入未做；
`web_editor.py` 未拆分，推理运行时仍在 `shared/assembly` 与 `tools` 中。

### 阶段 1

| 编号 | 结果 |
| --- | --- |
| 1.1 | 完成。`scene_metadata`（键/作用域/prim 路径）移到 `shared/assembly/scene_metadata.py`，`FACTORY_SCOPES` 移到 `shared/assembly/assets.py`；观察组选择与原生术语校验移到 `shared/assembly/native_terms.py`；`validate_sources` / `variable_information` 移到 `shared/assembly/legacy/{sources,derived}.py`。`catalog.py` 不再写 `isaaclab.envs.mdp.*`：新增可选后端钩子 `native_terms.py`（isaac 声明实现路径，mjlab/isaaclab3 用自身模块路径）。`grep` 上层（run.py、tools、shared/assembly）已无 `shared.backends.isaac` 导入。当时 IL3 仍复用 IL2 的资产定义（USD 工厂、`robots/`、接触材质、初始状态），移走 IL2 目录会连 IL3 一起坏；已在下文“1.1 补完”解决。core 已无任何后端导入，用导入拦截实测（`rlinf`、`ray`、`isaaclab`、`isaacsim` 全屏蔽时 `tools.preview`、`tools.smoke`、`shared.assembly.project` 仍可导入） |
| 1.2 | 完成。`snapshot`、`TerminalObservationMixin` 移到 `shared/backends/terminal.py`（原路径保留转发）；环境构造改为 `shared/assembly/project.py` 的登记表，`Project.register()` 在没装 RLinf 时只登记环境与观察类型；`tools/preview.py` 顶层不再导入 `isaac_state`/RLinf，`--actions hold/zero/random` 与 `record` 不需要训练框架，`--actions policy` / `--checkpoint` 明确要求 `train` 组；`tools/smoke.py` 对 Python Task 项目直接调用项目模型（`Project.model_builder`），不需要 RLinf |
| 1.3 | 完成。`isaac_state`、`observations`、`rewards`、`terrains`、`policies` 移入 `shared/backends/isaac/legacy/`（顶层保留同名转发，旧项目与冻结快照仍可导入）；`policy_math` 并入 `shared/policy.py`，项目的 import 已同步（`shared/policy_math.py` 仍转发） |
| 1.4 | 完成。13 个历史实验脚本移到 `experiments/`，内部 `-m` 路径、项目测试与 README 链接同步。例外：`go2_evaluation` 留在 `tools/`，因为它实现 `run.py assess`（`projects/go2_stairs/README.md` 有记载） |

### 阶段 2

| 编号 | 结果 |
| --- | --- |
| 2.1 | 完成。`pyproject.toml` 增加 `editor`、`import`、`isaaclab3`、`train`、`deploy` 可选依赖组（`core` 是基础依赖）；`tools/setup_env.py --with <组>` 创建环境、按组安装，源码组（Isaac Lab 3、RLinf、mjlab）与 `--isaaclab-source` 见文件头。离线环境下脚本写得对但没跑过真实安装（本机无网络） |
| 2.2 | 完成。`shared/assembly/runtimes.py` 是唯一的组注册表：命令 → 组、组 → 必装模块、组 → 工作区解释器。`run.py` 入口（`__main__`）在缺组时 `execv` 到满足该组的解释器，否则只输出一行安装提示；解释器覆盖只剩 `RLINF_PYTHON`（`RLINF_EDITOR_PYTHON`、`RLINF_EXPORT_PYTHON`、`ISAAC_PYTHON` 已移除，后端 `INTERPRETER` 声明保留）。`run.sh` 改为 `python3 run.py` |
| 2.3 | 完成。`requirements/<组>.lock.txt` 由 `tools/setup_env.py --lock --from .venv-il3 --fallback .venv` 从同一验证环境冻结（只保留一个版本，`--extra-index-url` 写进锁文件）。mjlab 冻结在 `shared/backends/mjlab/requirements.txt`，不另生成锁文件 |
| 2.4 | 完成。每个测试模块末尾一行 `pytestmark = pytest.mark.group_<组>`；`WORKSPACE_TEST_GROUP=core` 时 `tests/conftest.py` 只收集该组（否则小环境连不上 torch 的模块都会在收集期报错）。`tests/test_packaging_groups.py` 校验注册表、extras、锁文件与标记一致 |

### 1.1 补完：IL3 不再依赖 IL2 目录（2026-11-26）

Isaac Lab 2 不再维护，所以没有另建“Isaac Lab 公共层”，而是让 IL3 自成一体：

- `isaaclab3/` 自己维护原先从 `shared/backends/isaac/` 借用的部分：piper、piper_x、m20 的配置（`isaaclab3/robots/`），
  clear_sky、terrain（含 `TERRAIN_VARIANTS`）、seattle_table、dex_cube、usd_robot、franka_panda 的 USD 路径，
  `contact_materials.py`、`initial_state.py`，抽象体标记的形状。内容原样搬过来，行为不变。IL2 目录原样保留自己那份。
- mjlab 的两处数值校验改从 IL3 导入（不涉及仿真 SDK）。
- 默认后端从 `isaac` 改为 `isaaclab3`（`shared/assembly/assets.py` 与 `configs/state_ppo.yaml`）；资产目录、生成类型这类
  “所有后端的合集”只遍历目录还在的后端（`installed_backends()`），`runtimes.declared_python` 在后端目录不在时退回 `.venv`。
  没写 backend 的旧项目因此改在 IL3 上运行；为 IL2 写的旧项目要继续在 IL2 上跑，须在 train.yaml 写明 `backend: isaac`。

验收（用导入拦截模拟 `shared/backends/isaac` 已删除，拦截抛 `ModuleNotFoundError`，与真删一致）：

- 6 个 IL3 项目 `check` 全部通过；piper_pick、go2_piper_wholebody、go2_chase、zz_dual_arm_microwave 的 `smoke` 通过。
- 不开拦截时全量测试与改前一致（763 通过，同样 18 个外部环境失败）。
- 开拦截时，失败的只剩专门测 IL2 实现的测试。它们随 IL2 目录一起删除：
  - 整个文件：test_abstract_entities、test_assembly、test_assembly_history、test_assembly_reset、test_assembly_task_config、
    test_authored_targets、test_control_chain、test_isaac_state、test_named_controls、test_native_consumers、
    test_scene_asset_placement、test_scene_compiler、test_shared_variables、test_assembly_project、test_assembly_inventory、
    test_franka_isaac；
  - 单个用例：test_task_evaluation 的旧公式引擎、test_information_types 的 IL2 命令元数据、test_projects 里引用旧项目
    （cartpole 等直接导入 IL2）的用例。
- 原先借 IL2 测试 IL3 内容的测试改为测 IL3：机器人配置库、抽象体标记、地形变体表、Piper 参数检查、Python Task 运行时，
  以及 `dual_backend_scene` 的跨后端对比（改为 IL3 对 mjlab）。

### 删 IL2 目录的两个维护工具（2026-11-26 处理）

| 工具 | 原依赖 | 现在 |
| --- | --- | --- |
| 地形块生成（`shared/terrain_assets.py`） | 在 Isaac Sim 环境里跑 `shared/backends/isaac/terrain.py` | 已改到 IL3：新增 `shared/backends/isaaclab3/terrain.py`，用 IL3 自己的 `TERRAIN_VARIANTS`/`terrain_variant`。Isaac Lab 3 的地形生成器是纯 Python（numpy），在 `.venv-il3` 里直接运行，不启动 Kit 或仿真；IL2 的 `terrain.py` 已删除。验收：`stairs`、`rough`、`pit`、`stepping_stones` 用已生成地块记录的参数重跑，USD 与 MJCF 逐字节相同；`repeated_boxes` 的 USD/MJCF 相同，只有凸包碰撞 OBJ 的顶点顺序不同（同一点集、同体积，是本文件 `build()` 的序列化差异，与生成器无关） |
| URDF/MJCF → USD（`shared/asset_formats.to_usd`） | 在 Isaac Sim 解释器里启动无界面 Kit，用 `isaaclab.sim.converters` | 目录层面已无依赖：屏蔽 `shared.backends.isaac` 后用一个 2 连杆 URDF 实测转换成功（IL2 包未被导入）。URDF 改走 `urdf-usd-converter`（`import` 组）、MJCF 改走 `mujoco-usd-converter`（`usd` 组），都不启动 Kit（见“3.1 实施”）；Isaac 导入器只在两个包都没装时作为回退，缺解释器时 `to_usd` 直接给出安装建议，而不是 `FileNotFoundError` |

因此现在删除 `shared/backends/isaac/` 不会影响 `import` 组、地形生成、IL3 的 check/smoke，也不会影响测试里
不专门测 IL2 的部分。本轮补的验证（都在删除动作之前做）：

```bash
# 用导入拦截模拟 IL2 目录已删除（sitecustomize 对 shared.backends.isaac 抛 ModuleNotFoundError）
PYTHONPATH=/tmp/blocker ./run.sh check go2_goal           # planner_executor、piper_pick 同样 PASS
PYTHONPATH=/tmp/blocker ./run.sh smoke piper_pick         # 4 envs × 16 步，contract_passed，clean_exit
PYTHONPATH=/tmp/blocker .venv/bin/python -c "<to_usd on a 2-link URDF>"    # 写出 tiny.usd，未导入 IL2 包
PYTHONPATH=/tmp/blocker .venv-il3/bin/python -c "<terrain_assets.generate('flat')>"   # 生成并写出地形
```

用已提交地块记录的参数重跑 `stairs`、`rough`、`pit`、`stepping_stones`，USD 与 MJCF 的 sha256 全部相同。

`assembly_lift`（本地未跟踪的旧项目）的 train.yaml 现在写明 `backend: isaac`：它用 IL2 字段
（`simulation.device`），而 `tests/test_projects.py` 的 native 配置用例会跑它的 `check`；`test_headless_editor_round_trip`
在复制这份项目后只在配方没有 backend 时才补这一行，两种状态下都能跑。其余没写 backend 的旧配方按新默认留在 IL3 上，
需要 IL2 语义时自己补这一行。编辑器的“导入 URDF”快捷入口也按后端判断：默认后端没有 `urdf_robot` 时
（IL3/mjlab），它直接提示先跑 `run.py asset import <文件> --kind robot --project <项目>` 转成 USD，
而不是登记一个保存时才报错的条目。

### 3.1 调研结果：不依赖 Kit 的 URDF/MJCF → USD

结论：方案 A 有现成的 PyPI 包，两侧都已实施（见“3.1 实施”）。以下是调研时的记录。

- NVIDIA 的独立转换器已发布在 PyPI：`urdf-usd-converter`（0.4.0，依赖 newton-usd-schemas、numpy-stl、pycollada、
  tinyobjloader、usd-exchange）与 `mujoco-usd-converter`（0.10.0，另依赖 `mujoco-core>=4.0.1,<4.1`）。
  Isaac Lab 3 的 `UrdfConverter`/`MjcfConverter` 调用 `isaacsim.asset.importer.urdf|mjcf`（不在 Kit 里时由 wheel
  `isaacsim-asset-isolated` 提供），据调研其底层就是这两个包（本机没有该 wheel，未能核对）；直接用这两个包，
  不必装 Isaac Sim 的 wheel。
- `.venv-il3` 已有 usd-exchange 3.0.1、newton-usd-schemas 0.5.0、pycollada，URDF 一侧只缺 numpy-stl、tinyobjloader。
  MJCF 一侧要的 `mujoco-core` 4.x 与环境里的 `mujoco` 3.12（对齐检查、mjlab 用）是否能共存需要先确认。
  `usd-exchange` 本身只是 USD 编写 API，`newton` 只把 URDF/MJCF 读成物理模型、不写 USD，都不能单独充当转换器。
- 两侧都按上面的下一步实施了，逐项核对结果见下面的“3.1 实施”。MJCF 一侧的依赖结论：`mujoco-usd-converter`
  要求 `mujoco-core` 4.x，它提供的 `mujoco` 模块与 `import` 组的 `mujoco` 3.12 冲突（3.12 里也没有转换器用到的
  `MjsJoint.springyield`，实测直接报错），而 Newton 在 `isaaclab3` 组钉住 `mujoco~=3.12.0`，所以转换器单独一个 `usd` 组、
  自己的 `.venv-usd` 环境。
- 计划里的“两步导入”（先登记与预览、再按源文件哈希缓存转换）尚未做：编辑器预览已经能在没有
  Isaac 导入器时用 MuJoCo 编译出的 MJCF 显示，缺的是 `import` 组在无转换器时也能写出 ASSET.json。

### 3.1 实施：URDF/MJCF 转换都不再启动 Kit

`import` 组装 `urdf-usd-converter`（0.4.0），`usd` 组装 `mujoco-usd-converter`（0.10.0）。`shared.asset_formats.to_usd`
按源格式各找一个装了对应包的解释器（先看 `RLINF_PYTHON`、再看工作区各环境），用子进程转换（转换器的进度行只在失败时打印），
不启动 Kit；都找不到时仍走原来的 Isaac 导入器。两个转换器都是 NVIDIA 为 Newton 发布的独立包，输出都是单一自包含文件。

因为转换器依赖 `usd-exchange`，而它与 `usd-core` 都提供 `pxr`、不能装在同一环境，`import` 组把 `usd-core` 换成了
`usd-exchange`（3.0.1）。仓库用到的 pxr 模块（Usd / UsdGeom / UsdPhysics / UsdShade / Sdf / Gf / UsdUtils）在 usd-exchange 里都有；
`requirements/import.lock.txt` 已按 `.venv-il3` + `.venv` 重新冻结。

与 Isaac 导入器（`UrdfConverter`，`fix_base=False`）逐项比对，用的是 `piper_x.urdf`（10 连杆 / 8 可动关节）和带 `<mimic>`
的夹爪 URDF（7 连杆 / 6 可动关节，另有只由固定关节连接的 `link_tcp`）：

| 项目 | 结果 |
| --- | --- |
| 连杆名、位姿、质量、质心、惯量、碰撞体 | 一致（piper_x 的 10 个连杆逐项相同；4 个连杆的四元数符号相反，旋转等价） |
| 可动关节及其类型/轴/限位 | 一致 |
| 关节顺序 `joint_order` | 一致（BFS、同层按名字排序；夹爪的固定关节 `joint_tcp` 两者都不计入） |
| 对齐检查 | 两者都 `aligned: True`（MJCF 由 MuJoCo 从同一 URDF 编译，与 USD 无关） |
| 关节驱动 | 导入器写 625/0，转换器不写任何驱动。URDF 导入本来就把每个关节当作无驱动、由实体参数 `actuators` 给默认 PD（100/10），所以导入后的实体的增益不变 |
| mimic | 转换器不写 USD 侧的耦合；管线一直从 URDF 读 `<mimic>`（`asset_import.mimic_joints` → 实体 `couplings`），不受影响 |
| 固定关节 | 导入器把每个连杆都写成刚体（含 `link_tcp`）；转换器把只由固定关节连接的叶子连杆留成 Xform 帧，链中间的固定关节仍写 `PhysicsFixedJoint`。`usd_body_paths` 因此也收这类帧，挂在工具/TCP 帧上的传感器与 tframe 照样解析。这类连杆不在关节体的 body 列表里，所以 `asset convert` 从 URDF 算出它们相对所挂刚体的位姿，写进 ASSET.json 的 `frames`（`urdf_frames`），`robot_frames` 把它当作资产默认坐标系：`frame_pose('link_tcp')`、`differential_ik` 与 `obs.<机器人>.<连杆>` 照常可用。实测 piper_x + d435 的 URDF：记录的位姿与 USD 中 Xform 一致，IL3 smoke 里 `d435_camera_link` 随手臂运动、与 TCP 传感器的距离恒为 0.12785 m（URDF 计算值 0.127845 m） |
| 根连杆 | 转换器总是把根连杆焊到 world（`Physics/root_joint`）。`fix_base=False`（默认）时把这个焊点 `SetActive(False)`，由实体的 `fix_root_link` 决定是否锚定：usd-exchange 的 pxr 没有 `Sdf.Layer.RemovePrim` / `Prim.MoveTo` / `ApplyEdits`，所以是停用而不是删除 |
| 层级 | 转换器保留自己的嵌套（`Geometry/<连杆>/<子连杆>`），导入器是平的；IL3 建场景时用 `asset_body_paths` 把 `<实体>/<身体名>` 换成文件里的实际子路径 |
| 输出文件 | 转换器按源里的名字命名（URDF 的 `<robot name>`，piper 是 `agx_arm.usdc`），管线改名为 `<name>.usd`，记录写在 ASSET.json 的 `steps` 里：`tool`、`version`、`fix_base`、`link_hierarchy: nested` |

MJCF 一侧（`go2.xml` 与 piper_x 的转换 MJCF，对 Isaac 导入器的 `MjcfConverter`）：

| 项目 | 结果 |
| --- | --- |
| 刚体、可动关节、`joint_order` | 一致（go2 19 刚体 / 12 关节，顺序与已提交的 `assets/go2/ASSET.json` 相同；piper_x 13 / 8） |
| 对齐检查 | go2 两者都 `aligned: True`；piper_x 两者都只因同一个无质量 `Camera` 本体按 PhysX 默认 1 kg 计入而 `aligned: False`（同一处差异，与转换器无关） |
| 无关节本体（MuJoCo 里焊在父本体上） | 转换器仍写成刚体并加 `PhysicsFixedJoint`，与导入器相同；`usd_body_paths` 收得到 |
| 关节驱动 | 两者都不写驱动（MJCF 没有驱动信息时）→ `undriven_joints` 相同，实体仍取默认 PD |
| 根部固定 | 转换器也把根本体焊到 world（`Geometry/<根>/PhysicsFixedJoint`）；`fix_base=False` 时同样 `SetActive(False)`，`--fix-base` 时保留 |
| 层级 | 与 URDF 一样是 `Geometry/<本体>/<子本体>`，由 `asset_body_paths` 解析 |

对环境的划分：`usd` 组只装转换器（`mujoco-usd-converter` 带 `mujoco-core` 4.x、`usd-exchange`），放在自己的 `.venv-usd`；
`import` 组继续用 `mujoco` 3.12 做编译与对齐检查。锁文件按组分开冻结（`requirements/usd.lock.txt` 从 `.venv-usd` + `.venv-il3`），
共享包（numpy、newton-usd-schemas）在两个环境里版本一致，`tests/test_packaging_groups.py` 会核对。

验证：

- `run.py asset convert` 转 `piper_x.urdf` 与夹爪 URDF 各约 1.7 秒、转 `go2.xml` 约 1.1 秒（原来启动 Kit 约 1 分钟），
  都 `aligned: True`；`run.py asset check` 对一个转换结果正常输出关节顺序与对齐结果；转换失败时给出转换器自己的报错。
  `--fix-base` 时焊点保留、不带时停用，两种都在 `.usd` 上核对过。
- 转换出的 USD 当 `usd_object` 在 IL3 上 `smoke`：`fix_root_link: false`（自由根）与 `true`（锚定）都 `contract_passed`，
  挂在 `base_link` 的接触传感器和高度扫描器路径正确；自由根那个 16 步里复位 64 次（手臂落地触发任务失败），
  锚定的 0 次，说明焊点的停用确实让根部自由。
- 新增 `tests/test_urdf_conversion.py`（`group_import`）与 `tests/test_mjcf_conversion.py`（`group_usd`），没装转换器就跳过：
  嵌套身体路径、`joint_order`、无驱动关节、自由/固定根的焊点、与另一格式的对齐。

### 与计划的差异

- `check` 实际需要 `core` + 项目后端组：装配项目的 `check` 要编译原生控制项（`install_terms`），
  故 `run.py check` 在解析后端后切换到该后端的解释器。小环境能跑的是 `new`、`catalog`、场景文件终端编辑、
  `observation.py` 生成，以及 `tests -m group_core`（core 组另有 79 个测试）。
- 编辑器组的解释器按 `runtimes.py` 的顺序选取（`.venv`、`.venv-il3`、`.venv-editor`、`.venv-mjlab` 中第一个装了 viser 的）。
- 锁文件从验证环境一起冻结：`import` 组以 `.venv-il3` 为主、`.venv` 补缺（`xacro` 等只在后者），`usd` 组以
  `.venv-usd` 为主、`.venv-il3` 补缺（core 的几个包）。因而脚本支持 `--fallback`，共享包在各组里仍只有一份版本。

### 如何验证

```bash
tools/setup_env.py --check --python .venv-il3/bin/python --with isaaclab3,train   # 组是否齐全
.venv-il3/bin/python -m pytest tests -q                                          # 全量（与改前同样的 18 个外部环境失败，768 通过）
WORKSPACE_TEST_GROUP=core <只装了 core+editor 的解释器> -m pytest tests           # core 组（在无 torch 环境实测通过）
.venv-il3/bin/python -m pytest tests/test_urdf_conversion.py -q                   # URDF 转 USD 的契约（没装转换器则跳过）
.venv-usd/bin/python  -m pytest tests/test_mjcf_conversion.py -q                  # MJCF 转 USD 的契约（同上）
run.py asset convert <某个 .urdf> /tmp/check && run.py asset check /tmp/check      # 转换与对齐检查
```
