# BEHAVIOR-1K / OmniGibson 调研：资产与工具能否用于本工作区

调研日期 2026-11-18。上游仓库 `StanfordVL/BEHAVIOR-1K` 提交 `d4ebfe9`（2026-11-18），OmniGibson 3.9.3，
资产数据集 `behavior-1k-assets` 3.9.0，机器人资产 3.8.2。

标注说明：**[实测]** 表示在本机打开文件、读取远端 zip 索引或计算得到；**[代码]** 表示读上游源码得到；
**[文档]** 表示读上游文档或 HuggingFace 数据集卡片得到，未独立验证。

## 结论

1. **场景和物体资产（behavior-1k-assets）现在仍然加密，许可证只允许在 OmniGibson 内使用，所以不能直接接入本工作区。**
   全部 9589 个物体/粒子 USD 都是 `*.encrypted.usd`（Fernet 加密）[实测]。解密密钥只在用户同意
   《BEHAVIOR Data Bundle EULA》后自动下载。条款写明：仅限非商用学术研究；"只能在 OmniGibson 内使用"；
   禁止从 OmniGibson 中提取数据或逆向；禁止再分发密钥或数据的任何部分 [代码]。
   因此，即使用户本人同意了 EULA，把解密后的 USD 放进 Isaac Lab 3（Newton/OvPhysX）或网页编辑器
   也违反第 3、4 条。这不是格式转换能解决的技术问题。原因是物体主要购自 TurboSquid
   （文档列出 842 个购买链接）[文档]，上游自己也不能分发原始文件。
   本次没有同意 EULA，没有下载密钥，也没有解密任何文件。
2. **可以合法使用的部分**：
   - 代码（MIT）：OmniGibson、BDDL3、knowledgebase、asset_pipeline、joylo。
   - 机器人资产（MIT 标注，未加密）：R1 / R1Pro / Fetch / Tiago / Stretch / Franka（含灵巧手变体）/ A1 / UR5e 等。
   - 任务实例 JSON（2025 版 MIT）：只有布局和语义，没有几何。
   - 遥操作示范（MIT，LeRobot 格式，2–3.4 TB）。
3. **建议**：不要把 BEHAVIOR-1K 当作场景或物体来源。场景和物体继续使用 GRScenes、Isaac 环境和 SimReady。
   值得借鉴的是任务语义层：BDDL 活动定义、谓词和物体知识库。如需移动双臂平台，可以把 R1Pro 机器人
   作为新资产引入。工作量估计见文末。

## 1. BEHAVIOR-1K 现在提供什么

### 场景 [实测：zip 索引]

- 资产 zip 中有 51 个场景目录。文档写"50 个"，差异来自 `gates_bedroom` 等附加场景。分为两组：
  - iGibson 时代沿用的住宅（Beechwood、Benevolence、Ihlen、Merom、Pomaria、Rs、Wainscott），
    其中 `_int` 为室内，`_garden` 为带庭院版本；
  - 新建的大型场景：`house_single_floor`、`house_double_floor_lower/upper`、杂货店 ×4、餐厅 ×6、
    办公室 ×5、学校 ×5、酒店 ×3、大厅 ×4。
- **场景不是一个整体 USD**，`scenes/` 下全部内容只有 0.04 GB，包括：
  - `json/<场景>_best.json`：物体清单与状态；
  - `layout/`：楼层可通行图、语义图和实例分割图（PNG）。
- 墙、地板、天花板、窗户、门也都是普通物体模型：walls 1416 个、floors 386 个、ceilings 360 个模型。
- [实测，任务模板 JSON] 以 `house_single_floor` 的 picking_up_toys 模板为例：
  - 共 601 个 `DatasetObject`，涉及 81 个类别，其中 147 段墙、56 个筒灯、30 扇窗、28 棵树；
  - 536 个 `fixed_base: true`，52 个带关节状态（柜门、抽屉等）；
  - 每个物体写明 `category / model / scale / in_rooms`，`state.registry` 中有根位姿和关节位置；
  - 坐标范围约 x −25…28 m、y −16…43 m（含庭院），z −1…6.8 m；
  - 很多物体使用非等比缩放，例如柜子 `scale [0.85, 0.42, 0.90]`。
- OmniGibson 加载场景时，按 JSON 逐个实例化物体 USD。

### 物体 [实测：zip 索引]

- 1829 个类别、8635 个物体模型，另有 954 个粒子系统模板 USD（`systems/`，共 996 个系统定义）。
- 体积：zip 31.46 GB，解压后 34.7 GB。
  - 贴图约 28.4 GB（PNG 26.75 GB + EXR 1.65 GB），未加密，但仍受 EULA 约束；
  - USD 共 6.3 GB，全部加密；中位数 169 KB，最大 60 MB（某天花板模型）；
  - 每个模型的目录结构为 `usd/<model>.encrypted.usd`、`material/*.png`（diffuse、normal、roughness、
    metalness、reflection、refraction、ior）和 `misc/metadata.json`。
- 物体能力来自 BDDL3 公开数据 [代码：`bddl3/bddl/generated_data`，MIT]。
  - `object_inventory.json` 按模型列出 meta link，各类型出现的模型数：collision 9589、fillable 895、
    togglebutton 661、joint 541、com 246、subpart 213、connectedpart 132、openfillable 118、
    fluidsource 75、particleapplier 74、fluidsink 69、attachment 67、heatsource 54、slicer 29、
    extrapart 24、particleremover 13。
  - 3484 个 WordNet synset 标注了属性，例如 rigidBody 2098、cookable 782、heatable 706、fillable 265、
    sliceable 207、toggleable 196、cloth 97、openable 29、heatSource 20。
- 包括铰接物体（柜、门、抽屉、电器）、布料（cloth/drapeable，约 100 个 synset），以及粒子物质。
  粒子分四类 [文档]：
  - GranularSystem：米粒等；
  - FluidSystem：水、油；
  - MacroPhysicalParticleSystem：药片、切丁；
  - MacroVisualParticleSystem：污渍、灰尘，只附着显示。

### 任务定义（BDDL）[代码]

- `bddl3/bddl/activity_definitions` 下有 1018 个活动，每个活动一个 `problem0.bddl`，采用 PDDL 风格的
  `:objects / :init / :goal`。例如 `picking_up_trash` 的目标是"所有易拉罐都 inside 垃圾桶"。
- 谓词共约 27 个：ontop、inside、nextto、under、touching、open、toggled_on、cooked、frozen、hot、
  on_fire、filled、contains、covered、saturated、folded、attached、draped、inroom、grasped、broken 等。
- BDDL 包只依赖 numpy/networkx/nltk，**与仿真器无关**：求值通过回调 `evaluate_fn(predicate_cls, *entity_names)`
  完成，由宿主仿真提供每个谓词的判断。
- 2025/2026 挑战各选 50 个任务，提供预采样的任务实例 JSON：
  - 2025 版在 HF `behavior-1k/2025-challenge-task-instances`，MIT；
  - 2026 版在 `zipped-datasets` 中，108 MB；
  - 任务实例只在 3 个房屋场景上采样。

### 机器人 [实测：robot-assets 索引]

- `omnigibson-robot-assets` 3.8.2：zip 641 MB（解压 2.9 GB）。HF 仓库 `behavior-1k/omnigibson-robot-assets`
  标注 MIT，未加密，可逐文件下载。
- 包含 a1、behavior_robot（VR 身体）、fetch、franka（panda / mounted / allegro / inspire / leap 手）、
  freight、husky、kinova、locobot、r1、r1pro、stretch、tiago、turtlebot、ur5e、vx300s。
  每个机器人有 URDF、USD（`.usda`）、curobo 配置和 yaml。
- 挑战赛的主力机器人是 R1Pro（Galaxea 双臂移动平台）。示范数据集 `2025-challenge-demos` 有
  10000 条 R1Pro 遥操作示范：50 个任务，30 fps，LeRobot v2.1，2.07 TB，MIT [文档：数据集卡片]。
  2026 版为 3.41 TB。

### 工具 [代码/文档]

- **OmniGibson 仿真栈**（MIT）：
  - 运行环境：Isaac Sim 6.0.1、Python 3.12（`setup.sh` 安装 `isaacsim==6.0.1.0`）；
  - 物理只有 PhysX 后端，但已抽象出 `physics_backends/`（`base.py` / `physx_backend.py`），目前没有 Newton 后端；
  - 自带 Environment、Scene、Simulator、控制器、传感器、向量环境，与 Isaac Lab 的职责重叠。
- **物体状态**：共 34 种。
  - 运动学类：OnTop、Inside、NextTo、Under、Touching、AABB、Adjacency 等，通过 AABB、射线和接触计算；
  - 语义类：Temperature、Heated、Cooked、Burnt、Frozen、Saturated、ToggledOn 等，作为每步更新的内部变量；
  - 依赖 meta link，例如 ToggledOn 需要 togglebutton，HeatSource 需要 heatsource；
  - 需开启 `gm.ENABLE_OBJECT_STATES`。
- **转换规则**（`transition_rules.py`）：切片、切丁、融化、烹饪粒子、机器配方（搅拌机）等，
  用删除物体再生成新物体或粒子的方式模拟物理引擎做不到的变化。
- **采样**：`sampling_utils` 用物理后端射线在支撑面上采样 OnTop/Inside 位置；`scripts/sampling/*` 是
  BDDL 任务实例化流程：先生成模板，再采样任务相关物体，然后复制实例、预采样机器人位姿。
  该流程需要完整仿真，并用 JoyLo 人工验证。
- **kinematic_only**：`compute_kinematic_only` 的规则是：固定基座、无活动关节、（等比缩放或无固定关节）、
  无 attachment 时，物体只作运动学体，不参与动力学，以节省开销。
- **asset_pipeline**：基于 DVC，需要 Windows、3ds Max 2022 和 V-Ray 5，把 3ds Max 原始文件转成 OmniGibson USD。
  原始 3ds Max 文件只对斯坦福人员开放。对我们只有参考价值。
- **asset_conversion_utils**：`import_og_asset_from_urdf` 把自定义 URDF/OBJ 转成 OmniGibson 格式，
  生成 meta link、碰撞凸包，并可加密。依赖 Isaac Sim 的 URDF 导入器。
- **joylo**：低成本双臂遥操作硬件（GELLO 系）与采集脚本。

## 2. 资产格式与许可

| 项 | 内容 | 依据 |
|---|---|---|
| 托管 | HuggingFace `behavior-1k/zipped-datasets`，公开、未 gated，无需登录 | [实测] HF API `gated: False` |
| 资产包 | `behavior-1k-assets-3.9.0.zip` 31,457,673,073 字节；机器人 `omnigibson-robot-assets-3.8.2.zip` 641,004,353 字节 | [实测] |
| 加密 | 资产包中所有 USD 都是 `.encrypted.usd`，没有明文 `.usd`。抽查 `objects/apple/dfgurb/usd/dfgurb.encrypted.usd` 的开头为 `gAAAAABq…`（Fernet token） | [实测] 只在内存中读了开头约 100 字节，未保存 |
| 解密方式 | OmniGibson 加载时用 `omnigibson.key` 把文件解密到临时文件再打开。密钥 URL 在源码中做了混淆，同意 EULA 后才下载 | [代码] `asset_utils.download_key / decrypt_file`，`usd_object._prepare_to_load` |
| 许可 | BEHAVIOR Data Bundle EULA（2022-12-08 修订）：①回答 Yes 后才可使用；②仅限非商用学术研究；③禁止提取或逆向；④只能在 OmniGibson 内使用；⑤禁止再分发密钥或数据 | [代码] `print_user_agreement` |
| 代码许可 | 仓库 MIT（OmniGibson、BDDL3 各自也是 MIT） | [实测] LICENSE |
| 机器人资产 | HF 仓库标注 MIT，未加密；URDF 和网格来自各厂商，商用前要逐个核对原厂许可 | [实测] README frontmatter |
| 物体单位、朝向、材质 | 无法实测（加密）。从转换代码看：Z 向上、米制；材质是自带 MDL `omnigibson_vray_mtl.mdl`（`OmniGibsonVRayMtl`，从 V-Ray 转来），不是 UsdPreviewSurface | [代码] `asset_conversion_utils._force_asset_pipeline_materials` |
| Isaac Sim 版本 | 当前 OmniGibson 3.9.3 面向 Isaac Sim 6.0.1（`omnigibson_6_0.kit`） | [代码] `setup.sh` |

明文 pxr 能否打开：

- 加密物体 USD：不能。文件是 base64 文本，不是 USD [实测]。
- 机器人 USD：能，见第 4 节 [实测]。

## 3. 对本工作区的可行性

### 场景和物体：法律上不可行，技术上即使解密也要较多改造

- 许可：见第 2 节。本工作区的 IL3/Newton/OvPhysX、网页编辑器网格导出和 `asset_formats` 包装层都在
  OmniGibson 之外，均不在许可范围内。商用场景更不允许。
- 下面的技术障碍是假设许可允许时的情况，供与其他数据源对比，结论来自代码阅读：
  - **没有场景 USD**：需要写一个 JSON 到 USD 的组装器，引用约 600 个物体文件，并应用位姿、非等比缩放和关节初值。
    `environments.py` / `composition.py` 目前假设场景是一个 USD 文件树。
  - **非等比缩放的铰接物体**：OmniGibson 用 `kinematic_only` 与自定义处理规避，Newton 和 PhysX 对缩放后的关节锚点处理不同，需要逐类验证。
  - **材质**：自定义 V-Ray MDL。Newton GL 和编辑器的 `asset_formats.base_color` 不认识它的输入名，要么新增解析，
    要么直接读 `material/*diffuse.png`。
  - **meta link**：`meta__togglebutton_*`、`heatsource`、`fillable` 体积、`attachment` 等都是普通 Xform/Mesh 子节点，
    需要排除碰撞并隐藏，否则会变成多余的几何体。
  - **粒子和布料**：依赖 PhysX PBD 粒子和布料，Newton/OvPhysX 路径没有对应接入。
  - **物体状态和转换规则**：依赖 OmniGibson 的仿真循环、PhysX 接触与射线视图，无法直接复用。
- 相比之下，GRScenes（CC BY-NC-SA，明文 USD）与 SimReady（CC-BY-4.0）已经通过本工作区的
  包装、简化和组成流程。

### 值得复用的部分

| 部分 | 价值 | 怎么用 | 依据 |
|---|---|---|---|
| BDDL 活动定义与谓词 | 1018 个日常任务的结构化目标，与仿真无关 | 在项目 `task.py` 里由 `SceneInfo` 计算 ontop、inside、open、toggled_on 等，交给 BDDL 的 `evaluate_fn` 求值。按 AGENTS.md，阈值写在 Python 里，不新建 YAML 公式 | [代码] |
| 物体知识库（synset 到属性、meta link 清单） | 告诉我们哪些类别需要哪些能力（可开、可切、热源、可装液体） | 给 SimReady/GRScenes 物体做语义标注时参考 | [代码] |
| 状态判定算法 | OnTop/Inside/NextTo 的 AABB 加射线规则、Temperature 的升降温模型 | 作为参考实现，在 shared 或项目 Task 中用 Torch 批量重写，不引入 OmniGibson 运行时 | [代码] |
| 任务实例 JSON（MIT） | 房间级布局、物体类别和关节初值 | 只能作为语义布局参考，几何需换成自己的资产 | [实测] |
| 机器人资产（MIT） | R1Pro、Tiago、Fetch、Stretch 等，含 USD 与 URDF | 用 `run.py asset vendor <HF URL> --base https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets/resolve/<commit>` 引入，再做 IL3 配置 | [实测] |
| 示范数据（MIT） | 1 万条 R1Pro 双臂移动操作示范 | 在线回放和评估依赖加密场景，不能在本工作区复现；离线 SFT 预训练可以考虑 | [文档] |
| OmniGibson 仿真循环、控制器、环境 | 与 Isaac Lab 职责重叠 | 不采用 | — |
| asset_pipeline | 依赖 3ds Max + V-Ray + 私有原始文件 | 不采用 | [代码] |

## 4. 下载与 pxr 检查（实测）

下载内容（均无需登录或同意条款）：

| 路径 | 内容 | 大小 | 许可 |
|---|---|---|---|
| `assets/behavior1k_sources/omnigibson-robot-assets-3.8.2/` | `README.md`、`VERSION`、`models/r1pro/`（USD、URDF、网格、curobo、yaml） | 178 MB | MIT（HF 标注） |
| `assets/behavior1k_sources/2025-challenge-task-instances/` | `house_single_floor_task_picking_up_toys_0_0_template.json` 和 README | 0.95 MB | MIT |

- 来源：前者从 `zipped-datasets/omnigibson-robot-assets-3.8.2.zip` 用 HTTP range 只取出所需文件；
  后者来自 HF `behavior-1k/2025-challenge-task-instances` main。
- 没有下载 `behavior-1k-assets`（加密、受 EULA 约束），只读取了它的 zip 目录索引。
- 临时文件放在 `tmp/b1k/`（可删）：
  - 上游仓库浅克隆约 1 GB；
  - zip 索引 JSON；
  - 脚本 `scripts/zipindex.py`、`zipextract.py`、`httpfile.py`、`inspect_usd.py`、`inspect_r1*.py`。
- 没有安装任何包。

`.venv-il3/bin/python`（pxr 0.26.8）只读打开 `r1pro.usda`，运行时加了 24 GB 地址空间上限：

- `metersPerUnit 1.0`、`upAxis Z`、defaultPrim `/r1pro`；
- 382 个 prim，202 个 Mesh，共 54 万三角形；
- 44 个刚体，21 个转动关节、7 个移动关节、15 个固定关节，一个 articulation root（`/r1pro/base_footprint_x`）；
- 碰撞：164 个 `MeshCollisionAPI`，其中 161 个 convexHull、3 个 boundingSphere；
- 材质：34 个 OmniPBR.mdl（Kit 内置），贴图为 `./materials/*.jpg` 相对路径，未引用外部层；
- 只有少量 `physxArticulation` / `physxJoint` 自定义属性；
- 层外有 `/physicsScene`（PhysicsScene）。它不在 defaultPrim 下，按 defaultPrim 引用时不会带入；
- 底盘不走轮子物理：轮子和舵机都是固定关节，底盘由 6 个虚拟关节（`base_footprint_x/y/z/rx/ry/rz`）
  实现全向移动，与本工作区 Go2 那类接触驱动底盘不同；
- 关节驱动为 force 型，stiffness/damping 为 0（OmniGibson 在控制器中设增益），maxForce 为 55–100；
- 有 3 个相机：左右腕部 realsense 和头部 zed；`left/right_eef_link` 是末端参考系。

没有在 Isaac Lab 3 / Newton / OvPhysX 中实际加载 R1Pro，因为按要求本次不启动仿真。
所以上面只是结构检查，不表示能在 IL3 中运行。

## 5. 建议与工作量

1. **不引入 BEHAVIOR-1K 场景和物体。** 许可明确只能在 OmniGibson 内用，并禁止提取。
   如果确实需要这些场景，唯一合规的做法是直接在 OmniGibson 里跑（单独的 Isaac Sim 6.0.1 环境，非商用），
   这和本工作区"IL3 + RLinf 统一入口"的路线是两套系统，不建议。
2. **借用 BDDL 做任务语义（推荐，约 3–5 天）**：
   - 写一个最小的谓词实现集合（ontop、inside、nextto、open、toggled_on、touching、inroom），基于 SceneInfo 和抽象体批量计算；
   - 选 5–10 个只涉及刚体和铰接的 BDDL 活动，用 SimReady/GRScenes 物体实例化，接到项目 `task.py`；
   - 粒子、布料、烹饪、切片类活动先不做；
   - 可选做法：只拷贝解析器，或作为外部依赖安装到单独环境。BDDL3 是 MIT，可以 vendor。
3. **R1Pro 机器人（可选，约 2–4 天到 smoke）**：
   - 用 `asset vendor` 从 HF 仓库按提交引入；
   - 写 IL3 关节、执行器配置（增益要自己定），把虚拟底盘关节映射成 Isaac Lab 的关节位置/速度控制；
   - 先验证在 Newton 和 OvPhysX 上能加载、站稳、手臂可控；
   - 不要把这一步和"能复现 BEHAVIOR 挑战"混为一谈。
4. **示范数据**：只有在要做 R1Pro 双臂移动操作 SFT 时才考虑。数据量为 TB 级，而且观测来自 OmniGibson 渲染的场景，
   与本工作区仿真存在视觉和动力学差距。
