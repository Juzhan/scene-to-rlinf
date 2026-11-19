# UE 室外场景接入（第一阶段：CARLA Town10HD → USD → IL3）

目标：把 Unreal Engine 的室外场景当作工作区的静态场景环境（`usd_object`，`scope: global`），用 IL3（Newton / OvPhysX）
训练。本文记录已下载的内容、实际跑通的管线及其耗时和内存、哪些东西丢了，以及后续建议。
标注说明：**[实测]** 是本机实际运行验证过的；**[文档]** 来自源码、README 或发布页，未在本机验证。

## 1. 下载与存放位置

大体积 UE 数据都放在 `/home/jz/data/`，不放在工作区里（VS Code 会扫描工作区）。

| 内容 | 位置 | 大小 | 说明 |
|---|---|---|---|
| UE 5.7.4 Linux 二进制版 | `~/Downloads/Linux_Unreal_Engine_5.7.4.zip` → `/home/jz/UnrealEngine-5.7.4` | zip 32.1 GB；解压 56 GB | 解压时跳过了 `*.debug`（10.3 GB，只用于调试符号）。自带 USDImporter/USDExporter 插件和 Python。[实测] |
| CARLA UE5 场景资产（`ue5-dev`） | `/home/jz/data/ue_sources/carla_ue5_content` | 41.7 GB（43 991 个 LFS 文件），磁盘占用约 40 GB | `https://bitbucket.org/carla-simulator/carla-content.git`，提交 `639d6ef`（2026-07-08）；公开仓库，不需要登录；**CC-BY 4.0**。用 UE 5.5 保存（包头中 UE5 对象版本为 1013），UE 5.7 能直接读取。未下载 `HDMaps/*.pcd`（1.6 GB 点云）。工作树文件和 `.git/lfs/objects` 是硬链接（省了一份 41 GB）：**不要在 UE 里保存这些资产**，否则会改写 LFS 对象。[实测] |
| Town10HD 裁剪导出 | `/home/jz/data/ue_sources/exports/town10hd_crop` | 263 MB | UE 导出的 USD，加上碰撞叠加层；`UE_EXPORT.json` 记录来源、参数和许可。[实测] |
| 包装后的环境 | `assets/carla_town10hd/usd/town10hd_crop.usda`（及 `.visuals.usdc`） | 160 MB | 由 `run.sh asset environment ... --expand-instancers --add-colliders` 直接包装原始导出生成；引用 `/home/jz/data/...` 中的导出。[实测] |
| UE 工程、临时导出、Zen DDC | `/home/jz/data/ue_work/`（`CarlaExport/`、`out/`、`zen/`） | 约 17 GB | 只含内容的 UE 工程，用符号链接把 CARLA 内容挂成 `/Game/Carla`，排除了 71 个 `SK_*.uasset`（见 §3）。原来的 `~/.config/Epic/UnrealEngine/Common/Zen` 已移到这里，用符号链接指回原处。 |

脚本和小日志放在工作区 `tmp/ue/scripts/`、`tmp/ue/logs/`：

- `export_level_usd.py`：在编辑器里运行的导出脚本，读取环境变量 `UE_MAP`、`UE_OUT`、`UE_CROP`、`UE_SOURCE`、`UE_LOD(_MIN)`、`UE_PAYLOAD`、`UE_BAKE`；
- `add_colliders.py`：早期的临时碰撞叠加层，已由包装器的 `--expand-instancers --add-colliders` 取代，保留作对照；
- `stage_stats.py`：统计 prim 和三角形数量；
- `carla_drop.py`：IL3 烟测；
- `run_capped.sh`：内存看门狗。

其他可获得的来源（未下载）：

- CARLA UE4 内容包：`https://carla-assets.s3.us-east-005.backblazeb2.com/20250912_2171890.tar.gz`，21.6 GB，HEAD 返回 200，不需要登录。资产是 UE 4.26 格式，未试过在 5.7 中加载。[实测下载链接；兼容性未测]
- AirSim v1.8.1 发布页：有 AirSimNH、LandscapeMountains、Africa 等打包好的环境。它们是 cooked 二进制，拿不到几何，只适用于 §5 的渲染路线。[文档]
- Epic 官方示例（City Sample、Electric Dreams、Valley of the Ancient 等）和 Fab/Megascans：都要 Epic 账号登录和 Fab 授权，本次没有使用。[文档]
- URLab（UnrealRoboticsLab）：一个 UE 5.7 插件，在 UE 内运行 MuJoCo；Apache-2.0；Linux 可用；仓库不附带场景。[文档：README]

## 2. 几何路线：UE 关卡 → USD → 场景包装 → IL3

### 2.1 命令

每次运行 UE 都必须经过看门狗，并降低优先级（`nice`），同一时间只运行一个 UE 进程：

```bash
UE_MAP=/Game/Carla/Maps/Town10HD_Opt UE_CROP=-60,-95,50,15 UE_SOURCE=0 UE_PAYLOAD=usdc \
UE_OUT=/home/jz/data/ue_work/out/<名>/<名>.usda RUN_CAPPED_GB=24 nice -n 10 \
  bash tmp/ue/scripts/run_capped.sh timeout 3600 /home/jz/UnrealEngine-5.7.4/Engine/Binaries/Linux/UnrealEditor-Cmd \
  /home/jz/data/ue_work/CarlaExport/CarlaExport.uproject -ExecutePythonScript=$PWD/tmp/ue/scripts/export_level_usd.py \
  -unattended -nullrhi -nosplash -nosound -nop4 -ddc=NoShared -stdout
BARE=$(.venv-il3/bin/python -c "from shared.asset_formats import BARE_NAMES; print(BARE_NAMES + '|quercus|palm|pine')")
bash run.sh asset environment <导出>.usda assets/carla_<名> --name <名> --expand-instancers --add-colliders --bare "$BARE" \
  --simplify-colliders 0.05 --simplify-visuals 10000000
```

导出时编辑器无界面运行（`UnrealEditor-Cmd` + `-ExecutePythonScript`）：脚本用 `EditorLoadingAndSavingUtils.load_map` 打开关卡，
再直接调用 UE 自带的 `usd_unreal.level_exporter.export`。[实测]

UE 5.7 的问题和对应绕法 [实测]：

- `AssetExportTask` 加 `LevelExporterUSD` 并传入自定义 options 时，C++ 把当前任务记在传入的 options 实例上，Python 端却读取 CDO，结果报 `AttributeError: 'NoneType' ... filename`。绕法：直接调用 `level_exporter.export(context)`。
- 根层不能写成 `.usdc`：UE 自带的 USD 25.08 无法打包 assetInfo 字典里的 `SdfAssetPath`。改为根层用 `.usda`，网格写入 `.usdc` payload（`UE_PAYLOAD=usdc`）。
- 退出时报 `free(): invalid pointer` 并段错误（退出码 139），这发生在导出完成之后，输出完整。
- 退出时编辑器重启的 CEF/EGL 子进程会刷大量 `Could not open the default X display` 日志，不影响结果。

### 2.2 各步骤耗时与内存（本机：24 核，62 GB 内存，RTX 5080 16 GB；内存为 `/usr/bin/time -v` 的最大常驻内存）

| 步骤 | 耗时 | 峰值内存 | 结果 |
|---|---|---|---|
| 模板关卡导出（`-nullrhi`） | 导出 0.4 s | — | 网格和灯光齐全；材质只有 UE 引用 [实测] |
| 模板关卡烘焙材质（`-RenderOffscreen`） | 导出 17 s | — | 得到 UsdPreviewSurface 和 PNG 贴图 [实测] |
| Town10HD 首次加载（含骨骼网格） | 1 min 后崩溃 | 15.1 GB | Nanite 构建器断言 `ClusterDAG.cpp:274`，出现在载具骨骼网格上 [实测] |
| Town10HD 整图，源数据（`UE_SOURCE=1`） | 导出 342 s，总计 6 min 37 s | **35.3 GB** | 8.3 GB 文本 usda；超过现行 24 GB 上限，不再使用 [实测] |
| Town10HD 整图，渲染数据（`UE_SOURCE=0`，usdc payload） | 导出 12.5 s，总计 46 s | 16.9 GB | 616 MB [实测] |
| Town10HD 裁剪 110 m × 110 m（经看门狗，冷 DDC，机器负载高） | 导出 142 s，总计 6 min 30 s | 16.4 GB | 261 MB；热缓存时导出只要 6 s [实测] |
| 裁剪区域烘焙材质（`-RenderOffscreen`） | 导出 201 s，总计 4 min 6 s | 14.4 GB | 137 个 UsdPreviewSurface，404 张 PNG，274 MB [实测] |
| Mine_01（含 Landscape），`-nullrhi` | 约 2 min 后崩溃 | — | 关卡导出器总会烘焙 Landscape 材质，没有渲染目标时触发 `Canvas.cpp:880` 断言 [实测] |
| Mine_01，`-RenderOffscreen` | — | **≥48 GB，被 OOM 杀掉** | 这次把整个会话一起带崩了；之后加了 24 GB 看门狗 [实测] |
| `run.sh asset environment --expand-instancers --add-colliders`（裁剪，原始导出直接包装） | 76–102 s（早先经临时脚本两步：95–112 s，机器高负载时 862 s） | 0.9 GB | 见 2.3 [实测] |
| `environments.spawn_points` | 6.6 s | — | 8 个出生点 [实测] |
| IL3 Newton 烟测（1 个环境） | 建环境 19–21 s，100 步 1.9–2.9 s；高负载且源文件在 `/home/jz/data` 冷缓存时，建环境 378 s、100 步 16.8 s | 5.2 GB | 见 2.4 [实测] |
| IL3 OvPhysX 烟测（1 个环境） | 建环境 28 s，100 步 2.3 s | 6.0 GB | 见 2.4 [实测] |

### 2.3 导出了什么、丢了什么

Town10HD_Opt 整图共有 2611 个 actor [实测]：

- StaticMeshActor 1541 个；
- DecalActor 422 个；
- 蓝图 actor：路沿/围栏样条 147 个，路灯等若干；
- Emitter 40 个；
- 停放车辆（`*_Parked` 蓝图，内含静态网格）若干。

整图导出结果 [实测]：

- 4494 个 Mesh 和 1039 个 PointInstancer（52 867 个实例）；
- 普通网格 71 M 三角形，实例化网格合计 340 M 三角形；
- 包围盒约 30 km × 50 km，原因是含一块海滩平面和海面。城区本身约 400 m × 440 m，地面在 z≈0。

- **单位、坐标轴**：导出为 Z-up、`metersPerUnit=0.01`。包装层会自动加 0.01 缩放，不需要手动改。UE 是左手系，导出时 Y 轴取反：在 UE 坐标中写的裁剪框，到了 USD 里 y 要变号。这一点与观测到的碰撞范围一致。[实测]
- **Nanite**：
  - `UE_SOURCE=1` 导出 Nanite 高精度源网格：单个墙体约 460 MB 文本，整图峰值内存 35 GB。
  - `UE_SOURCE=0` 导出渲染数据，轻得多，但网格依然很密：一棵树 64 万三角形，商店墙体 18 万。
  - 这些网格大多只有一个 LOD，所以请求 LOD3 几乎没有减少三角形（65 M → 61 M）。
  - 最终靠包装器的 `--simplify-visuals` 降面。按 2 M 预算，每个网格上限只有 650 个左右三角形，楼面在编辑器里皱成一团；现用 10 M 预算（单网格上限 11261，298 个网格被聚类），楼体轮廓正常，大墙面仍有聚类留下的三角斑块。无相机的 Newton 烟测里两种预算的建环境时间相同（约 19 s）、显存峰值 3.3 GB（含空闲 1.7 GB）；带相机或 view/record 时的显存未测。[实测]
- **Nanite 骨骼网格**：UE 5.7 在加载时重建载具骨骼网格的 Nanite 数据，构建器断言崩溃。现在的做法是挂载内容时排除全部 71 个 `SK_*`，损失的是会动的载具和行人。停放车辆是静态网格，保留了。[实测]
- **HISM/ISM 与植被（PointInstancer）**：
  - 建筑的模块化墙体和植被都导出成 PointInstancer。Newton 的 USD 导入不读 PointInstancer（`newton/_src/utils/import_usd.py` 中没有相关处理），所以这些部件既不显示也没有碰撞。[源码]
  - 包装器 `--expand-instancers` 把每个实例展开为“Xform + 引用源文件中的原型”，再停用原 instancer：展开 118 个 instancer 的 2975 个实例（早先临时脚本展开 3037 个，差别是它没有跳过 `invisibleIds` 中的实例，也展开了树木原型）。
  - 原型全部匹配 `--bare` 的 instancer（草、花等）保持为 PointInstancer（Newton 中不可见）。
  - 整图另有 1657 个实例索引指向缺失的原型。[实测]
- **碰撞**：UE 的碰撞在 BodySetup 里，USD 导出器不写任何 UsdPhysics。导出端源码中只有导入时才读 `CollisionAPI`。[源码]
  - 包装器 `--add-colliders` 给非植被网格加三角网格碰撞：3172 个网格，34.5 M 三角形；草、贴花、天空不加。默认的 `BARE_NAMES` 不含树种名，CARLA 的树（`SM_QuercusRubra…`、`Palm`、`Pine`）要在 `--bare` 后追加，否则会多出约 900 万三角形的树叶碰撞。结果与早先临时脚本逐项一致。
  - 包装器 `--simplify-colliders 0.05` 把它们压到 263 个物体碰撞体，共 4.6 M 三角形。[实测]
- **Landscape（地形）**：Town10HD 的地面是静态网格，没有 Landscape。Mine_01 的 Landscape 导出失败（见 2.2），原因是导出器总要烘焙 Landscape 材质：不开 RHI 会崩溃，开了内存超过 48 GB。[实测]
- **材质**：
  - 不烘焙时，每个网格只带 `unreal` 渲染上下文的材质引用（`/Game/...` 路径）。Newton/IL3 里看到的是默认灰色，USD 工具会对这些路径给出无法解析的警告，但不影响加载。
  - 烘焙（`UE_BAKE=1`，必须用 `-RenderOffscreen`，并编译着色器）得到 UsdPreviewSurface 和 128² PNG（默认尺寸）。
  - 裁剪区域烘焙要多 3 分钟左右；尚未接到包装层上验证其显示效果。[实测]
- **其他**：贴花（大部分路面标线）、粒子、蓝图逻辑（红绿灯）、天空和天气不导出或不可用。灯光导出为 UsdLux。[实测：类别统计；可见效果未检查]

### 2.4 IL3 烟测 [实测]

`projects/zz_carla_probe`（`projects/zz_*` 已被 gitignore）的设置：

- 场景：Franka、苹果、会游走的运动学箱子，加上 `town` 实体（`usd_object`，`scope: global`，位姿为 `[46.936, 14.653, -0.06]`，即把出生点 `(-46.9, -14.7, 0.06)` 放到环境原点）；
- 刻意不加地面平面。

`tmp/ue/scripts/carla_drop.py` 用 1 个环境、零动作运行 100 步，苹果从 0.3 m 处落下：

- Newton：最终 z = 0.004–0.06（临时脚本包装的版本，共 3 次）；包装器直接包装后 z = 0.002（2 M 与 10 M 外观预算各一次），建环境 18–19 s，100 步 1.8 s；
- OvPhysX：最终 z = 0.009（两个版本相同），建环境 21 s，100 步 2.3 s。

苹果没有穿透，说明城区路面的碰撞体在两种物理后端里都生效了。z 的抖动和水平位移来自机械臂的零动作姿态和 5 cm 的碰撞网格量化，没有单独分析。

这只证明了接口跑通：静态加载和地面碰撞没问题。没有渲染相机画面，没有检查外观，也没有做多环境或训练。

## 3. 渲染路线：UE 负责渲染，Newton 负责物理（位姿同步）

做法是 Newton/IL3 推进物理，每步把刚体位姿发给 UE，UE 渲染相机画面后回传。URLab 走的是反方向，在 UE 内跑 MuJoCo，用 ZMQ 与 Python 通信；AirSim 和 CARLA 则是 UE 自带物理，通过 RPC 交互。[文档]

- **代价**：
  - 每个渲染进程要占一个完整的 UE 运行时：加载 Town10HD 至少 15 GB 内存，外加显存。
  - UE 没有 IL3 那种按环境批量的 tiled camera，一个进程只能服务少数几个视角，所以 PPO 常用的上千环境做不到。
  - 每步要来回同步，帧率通常只有几十 FPS。
  - 需要 C++ 工程或插件（URLab 要求 C++ 工程，只测试过 5.7），还要写 USD prim 与 UE actor 的映射，以及传感器时序和种子的复现。
- **何时值得做**：只有需要照片级相机数据时才值得，例如视觉策略的离线数据生成、少量环境下的视觉评估、sim-to-real 的外观对照。以状态或低维观察为主的训练用不上这条路线。

## 4. 建议与后续工作

**建议**：训练用几何路线，按区域裁剪或按子关卡导出，导出渲染数据（`UE_SOURCE=0`），再加碰撞、包装成 `usd_object`。
渲染路线留到确实需要照片级图像时再做，届时优先评估 URLab 或直接使用 CARLA 的传感器。整图和 Landscape 导出受内存限制（≥35 GB / ≥48 GB），不在本机的 24 GB 上限内。

后续工作（粗估工作量；涉及 `shared/` 的改动需要用户确认）：

1. ~~把“加碰撞”和“展开 PointInstancer”收进包装器~~：已完成（`--expand-instancers --add-colliders --bare`，见 `docs/SCENE_BACKENDS.md`）。UE 导出暂未登记到 `shared/environments.py` 目录：目录条目会自动下载源文件，而 UE 导出需要本机 UE 工程，不能自动下载。
2. 碰撞质量：在 UE Python 中读取 BodySetup 的简单碰撞（box/convex）一并导出，代替整网格三角碰撞。约 1–2 天。
3. 视觉：在 UE 中对高面数网格生成 LOD 或调低 Nanite fallback 后再导出，代替包装器的粗暴聚类；验证烘焙材质在 IL3 相机中的显示效果。约 1–2 天。
4. Landscape：用 Python 直接导出高度图或指定 LOD 的网格，跳过材质烘焙，控制在 24 GB 内。约 1–2 天。
5. 骨骼网格 Nanite 崩溃：在不保存原内容的前提下关闭这些资产的 Nanite，或改用 UE4 版内容。如果不需要动态载具，可以不做。
6. 其他城镇：Town01/02/03/07 的 `_Opt` 关卡可以按同样流程导出；Town11–15 是 World Partition，需要先按区域加载 cell。

状态：接口跑通（加载与碰撞）已验证；权重更新、完整回合、视觉外观都还没有涉及。
