历史实现位于 shared/assembly/legacy/；新项目请使用 task.py 和生成的 SceneInfo。

旧表达式引擎参考
================

当前项目入口
------------

本文件仅记录仍供历史实验使用的表达式引擎，不是新项目的设计规范。
planner_executor、assembly_lift、assembly_go2、assembly_go2_navigation 已改用项目
``task.py``，不接受 scene.yaml 中的 task / variables，也不再使用下面的配置语言。

新项目定义 ``evaluate(obs: SceneInfo)`` 或 ``Task.evaluate(obs: SceneInfo)``，
直接用 Python / Torch 计算并返回 reward、success、failure、metrics；可选 set 在本步移动抽象体
（``{名: [B,3] 位置或 [B,7] xyz+wxyz}``，复位之后、观察之前生效）。
``observation.py`` 从场景生成完整 SceneInfo 及策略所选 Obs 类型；Task 按需读取基础信息，
与策略选择分开。历史可用 TaskMemory；环境推进、奖励积分、终止、复位仍交给原生 manager。
具体接口与示例见 ``projects/planner_executor/README.md`` 和 ``task.py``。

以下内容用于理解旧表达式引擎测试与历史配置；不要据此生成新项目。

统一变量与 Torch 公式
---------------------

顶层 variables 是唯一变量定义区，统一数值、向量、布尔和历史变量。
task 直接配置 success、failure、reward、metrics，不再接受 evaluation / variables / values / conditions 子区。
变量、奖励、指标名字由项目定义。变量名字须是有效的 Python 标识符（支持中文），
不能以 _ 开头，不能与 torch、历史辅助函数或 Python 关键字重名。

.. code-block:: yaml

   variables:
     position:
       expr: robot.position
       history: 100
     distance:
       expr: torch.linalg.vector_norm(robot.position - object.position, dim=-1)
       history: 100
     reached:
       expr: distance < 0.03
       history: 25
     recent_mean:
       expr: history_mean(distance)
     recent_path:
       expr: >
         torch.linalg.vector_norm(
           torch.diff(position.history, dim=1), dim=-1
         ).sum(dim=1)
   task:
     success: reached
     reward: "-distance"
     metrics:
       distance_m: distance
       recent_mean_m: recent_mean
       recent_path_m: recent_path

变量用 expr 直接引用完整场景属性、其他变量或计算公式。新模板无需 op / signal / ref。
history 是所有 variables 声明的可选属性，包括常量、布尔、Torch 派生量和历史派生量；
不需要另建 history 配置区。变量也可简写为 ``distance: "...公式..."``。
计算按依赖顺序执行，允许前向引用，拒绝循环（包括自身的 .history，因为历史包含当前值）。
每一步每个变量只计算一次、每个源只读一次，多个指标或奖励引用不会重复写历史。

success / failure 接受布尔表达式；两者同时满足时失败优先。
reward 直接写总奖励表达式，例如 ``-1.0 * reach_distance - 4.0 * target_distance``。
不需要为了加权求和强制拆成多个 input / weight 项。其结果必须是每环境一个数值或布尔值，
布尔按 0/1 计分；Isaac 原生奖励管理器随后乘控制步长 dt。
改变系数就是改变奖励尺度，-0.1/-0.4 与原 -1/-4 比例相同但总量小十倍。
单条公式的任何依赖无效时，本步总奖励为 0。需要单独记录某个贡献时可声明变量并加入 metrics。
配置化 Task 只写单条 reward。task.rewards 属于原生 manager 扩展，不是另一套变量或公式分项入口。
metrics 只记录结果与有效性，不加入策略输入，不自行改变奖励。
success / failure / metrics 也直接写字符串公式或变量名。

历史与有效性
------------

* x：当前值，形状 [环境数, 数据维度...]。
* x.history：最近 history 个控制步的快照，形状 [环境数, 时间, 数据维度...]。
  包含当前样本，时间从旧到新；窗口未满时左侧填零，不把填零视为有效数据。
* x.history_valid：每个时间槽的有效标记，形状 [环境数, 时间]。
* x.history_time：回合内的采样时间（秒），形状 [环境数, 时间]，须配合有效标记使用。
* history_mean(x) / history_sum(x)：忽略无效样本，对时间轴求均值/和。
* history_all(x) / history_any(x)：忽略无效样本，对布尔历史按时间求全满足/任一满足。
  没有有效样本时以上辅助函数的结果无效。

直接使用 .history / .history_time 的公式默认需要所引用的整个窗口有效；
窗口未满、含缺失或非有限样本时，结果标为无效，奖励贡献为 0，成功/失败不触发。
同一公式显式读取 x.history_valid 时，表示用户自行处理掩码，此时只要求该历史至少有一个
有效样本，公式结果还必须有限。即使用户只取最后一个历史样本，未显式使用掩码时仍检查完整窗口。
对当前变量的依赖也会传播其有效性；torch.where 的无效分支不采用自动短路有效性推导。

注意 history_all 是“已有有效样本全满足”，不表示完整窗口已满或连续保持足够时长。
要连续保持 0.5 秒，可使用 ``hold(reached, seconds=0.5)``；要全部 25 个时间槽有效且满足，使用
``torch.all(reached.history & reached.history_valid, dim=1)``。

示例 recent_path 是最近 100 个位置样本之间的折线路程，最多 99 段，包含竖直位移；
不是本回合总路程。整回合累计仍可使用 accumulate / integral / running_mean 等已有
增量规则，避免为了一个累计值保留完整轨迹。历史不会跳过无效样本后把两段轨迹强行连接。
历史按环境分别重置，终止样本先参与评价，随后由原生重置事件清空该环境。
默认不保存历史；仅为声明 history 的变量分配有界缓冲。Task 历史默认不记录梯度。
variables 供 Task 使用，不作为模型观察。模型的基础输入读取不经过 Task 变量缓存。
默认按环境决策步采样，dt 为所训练角色的 period；重复读取不推进采样。Task 的即时计算可配置 sampling: physics，只引用基础信息及同为 physics 的纯变量；
不能配置 history 或时序操作。物理步内重复评价复用缓存，且不会推进决策历史。
需要记录实时计算的决策采样历史时，另声明 decision 变量引用它并设置 history。
单个环境复位清理自己的历史，存活环境在同一步不会再次累计。
此机制没有替换 VLA 项目的专用历史预处理。

表达式支持范围
--------------

使用 Python 算术、单次比较、& / | / ~ 逻辑、切片、函数调用语法。
不支持 Python and/or、链式比较、语句、循环、推导式、lambda 或导入。
使用 torch.where 表达条件选择。长公式可用 YAML 的 > 换行，或拆成多个命名变量。

目前放行的真实 Torch 函数：

* abs, square, sqrt, exp, log, tanh, sin, cos, atan2, clamp, minimum, maximum, where
* sum, mean, amin, amax, all, any, diff, stack, cat
* logical_and, logical_or, logical_not, isfinite
* linalg.vector_norm, linalg.norm

支持 Tensor 的 sum/mean/amin/amax/all/any/abs/square/sqrt/clamp/float/bool/unsqueeze/squeeze 方法。
另外提供以下领域函数，复用已有计算实现：

* inverse_rotate(vector, quaternion)：将向量从四元数的参考坐标系变换到实体坐标系，wxyz/xyzw 按信息声明解释。
* inside_region(position, region)：判断根位置是否在区域内；矩形判断 XY 投影，盒形判断三维。
  区域来源已随 ``regions:`` 段移除，仅保留给直接提供 10 维区域张量的旧测试。
* hold(condition, seconds=0.5)：连续满足指定秒数，条件不满足或无效时清零。
* running_mean(value)：整个回合的有效样本平均，不保存全轨迹。

这些不是 Torch 原生函数，但不再需要另一套 op / signal / ref 参数写法。
每个 hold / running_mean 调用位置有独立状态，按环境重置；嵌套调用同样支持。
数学由 Torch 实现，本层只解析表达式、发现依赖、读取变量并调用上述函数；没有 eval()。
不开放原地修改、out 参数、设备切换或任意属性/函数调用；函数白名单不是完整 Torch API。
归约要明确 dim，第一维始终是环境轴；禁止对它归约、拼接或插入维度。
切片使用 ``x[:, ...]`` 或 ``x[..., ...]`` 保留环境轴，不能用 x[0] 取某个环境。
两个批量张量参与广播时须保持相同秩，例如 [N,3] 乘 [N] 应显式把后者
unsqueeze(-1) 成 [N,1]，避免 N 恰好等于特征数时默默混用环境数据。

复杂模型/项目算法继续通过 TERMS.signals 注册只读信号，再由 source 引用；
expr 不支持任意导入 Python 函数。Blender 任务页支持添加、删除变量，编辑 expr、history、
成功、失败和总奖励公式。完整参数区以 JSON 编辑 information、metrics 和结构化表达式，
通过共享编辑接口校验后写回同一份 YAML；校验失败不写文件。原生任务预设仍保留，
这里编辑的是顶层 variables 和 task，不会给单条总奖励公式捏造分项权重。

类型检查边界
------------

source 和直接别名保留已知信息元数据；已有 op 规则保留原有类型、单位和坐标检查。
任意 Torch 公式不假装能自动推导物理单位；check 的 untyped_variables 列出未知变量。
复杂结果可以补充 information，例如：

.. code-block:: yaml

   distance:
     expr: torch.linalg.vector_norm(robot.position - object.position, dim=-1)
     information: {kind: number, shape: [], unit: m}

shape 不含环境轴。此声明是作者提供的语义，不是对任意 Torch 计算的物理正确性证明。
运行时检查环境维、声明的形状、布尔条件、每环境标量奖励与数值有效性。

兼容的结构化表达式
------------------

仍支持 {signal: 路径}、{ref: 变量}、{value: 常量} 和 {op: 操作, ...}。
下面的规则表作为几何、累计、时序等便捷操作以及已有实验的兼容入口保留。
新数学公式优先调用 Torch，不为它已提供的每个函数再添加一种 op。

Isaac 内置信息源
---------------

直接复用观察目录的实体属性路径（如 ``object.position``、``robot.wrist_camera.rgb``），
不要求勾选为 observation。嵌套实体使用点分隔名字，底层 scene key 由接入层转换。

* position：根部位置，米，减去 env_origins 的环境坐标。
* orientation：根部朝向，wxyz 四元数。
* linear_velocity / angular_velocity：世界轴线速度、角速度。
* joint_position / joint_velocity / joint_effort：关节状态。
* ray_hit_positions / contact_force / rgb：相应传感器原始数据，实体必须具备对应能力。
* simulation.gravity：环境重力加速度向量，不隐式投影到机身坐标。
* frame_positions：坐标测量器各目标坐标系的位置，形状 [frame, xyz]，环境坐标；单个末端的选择在变量公式中写明。
* ``object.target.position`` 等：targets 显式设置的期望属性。当前支持固定的三维位置、线速度、角速度；
  均为环境坐标。声明目标必须对应场景中存在的实体。
  点、方向、矩形、长方体等空间量是 ``entities`` 中 ``kind: abstract`` 的抽象体，字段为
  ``<name>.position`` / ``<name>.orientation``（另有 ``size`` 或 ``direction``），可选作模型输入。
  没有 goal 类型：随机目标就是带 ``reset: {region: world/<区域>, yaw_range, interval}`` 的抽象体，
  含义由 task.py / 模型决定。不再有 ``regions:`` 段和 visualization.goals。不包含规划或避障。
  回合内运动的目标由新模板 task.py 返回 ``set`` 移动，未跟随且非区域的抽象体另提供 ``<name>.linear_velocity``。
  目标与机器人的相对位置、距离、朝向误差通过顶层 variables 声明，不自动注册派生字段。
* 基础信息提供实体 linear_velocity/angular_velocity/orientation、关节当前/默认值、simulation.gravity，
  ray_hit_positions[ray,xyz]、frame_positions[frame,xyz]、contact_force[body,xyz]、contact_force_samples[sample,body,xyz]。
  Task 所需派生量在 variables 中计算；网络所需机身速度、重力投影、关节偏移、扫描高度在 policy.py 中计算。
* 新控制链使用 planner.output / planner.previous_output 或 executor.output / executor.previous_output，明确被评价的角色；component.input 为最近一次实际输入。旧项目的 policy.action / policy.previous_action 保留兼容。
* 新项目外部目标使用抽象体并在 task.py 中计算；组件观察选择基础信息，模型在 Python 中加工。
* 旧表达式引擎的 ``regions.<name>`` 区域来源已随 ``regions:`` 段移除；需要区域判断的新项目在 task.py
  中读取抽象体的 position/orientation/size 计算。

旧实验的 entity__ / region__ / command__ 和显式 preset 仍可读取，用于保留旧实验语义；新模板不使用。

信息目录不意味着传感器自动创建或启用。RGB 必须存在并启用相机。当前 check 检查表达式、
引用、实体存在性；能力、实际数据维度和命令是否可读取还需要 smoke。
已知类型的单位、坐标系、形状会在配置检查时校验，转换必须显式配置。
未知类型会在 untyped_signals 中列出，不假装通过完整语义检查；混合关节单位等尚未声明的
元数据也不会被猜成 rad 或 m。详细分类与推导见 INFORMATION_TYPES.rst。

底层几何与时序操作表（49 项）
-----------------

下面是计算引擎的结构化操作，不是 49 个可以直接调用的 Torch 函数。
常规配置优先用上面的 expr；需要专用几何操作时可在顶层变量声明中使用结构化表达式。
a、b、input、lower、upper、position、quaternion 均为表达式；inputs 是表达式列表。
向量归约仅作用在最后一维，不混合不同环境。

计算与几何（19 项）
~~~~~~~~~~~~~~~~~~

* distance(a, b)：欧氏距离，要求有特征维度。
* subtract(a, b)、add(a, b)、multiply(a, b)、divide(a, b)：四则运算。
* dot(a, b)：向量点积。
* angle(a, b)：两向量夹角，弧度；零向量使结果无效。
* abs(input)：逐元素绝对值。
* norm(input)：向量 L2 范数。
* sum(input)、mean(input)、min(input)、max(input)：最后一维归约。
* select(input, indices)：选择最后一维；整数得到单分量，列表保留向量维度。
* rotate(input, quaternion, inverse=false)：wxyz 四元数旋转三维向量，可逆向旋转。
* transform_point(input, quaternion, position)：旋转三维点后加平移。
* field(input, name)：按名称取组合信息的字段。例如位姿的 position / orientation，
  速度命令的 linear_velocity / angular_velocity；输出保留字段自己的单位与坐标系。
* rotation_distance(a, b)：旋转的最短角误差，弧度；支持 wxyz / xyzw，不把 q 与 -q 判成不同朝向。
* inside_region(input, region)：点进入场景中有朝向的区域。box 检查三维范围；rectangle
  仅检查区域局部 XY 投影，不限制离平面的高度，不表示完整物体被包含。

比较与有效性（10 项）
~~~~~~~~~~~~~~~~~~~~

* less / less_equal / greater / greater_equal(a, b)：小于、小于等于、大于、大于等于。
* equal / not_equal(a, b)：精确相等或不等，浮点目标优先使用 near。
* near(a, b, tolerance)：逐分量绝对误差 <= 容差。三维球形距离使用 distance 再比较，
  不要把逐分量 near 误当作欧氏距离判定。
* between(input, lower, upper)：闭区间，逐分量判断。
* in_set(input, choices)：数值、布尔或已声明类别属于指定常量集合。
* inside_box(input, lower, upper)：所有坐标均位于闭区间；轴对齐盒中的点判定，
  不自动推断物体尺寸，也不代表完整物体被容纳。

逻辑与集合（8 项）
~~~~~~~~~~~~~~~~~

* valid(input)：得到每个环境的数据有效性；自身的结果有效。
* not(input)：布尔否定，原数据无效时仍无效，不能用否定把无效变成成功。
* all(inputs)、any(inputs)：全部或任一标量条件成立。
* at_least(inputs, count)：至少 count 项成立。
* all_elements(input)、any_elements(input)：布尔向量中全部或任一元素成立。
* count(input)：统计布尔向量成立数量。

时间与历史（12 项）
~~~~~~~~~~~~~~~~~~

* hold(input, seconds)：连续保持；不满足或无效会清零计时。每次有效真样本计入一个 dt。
* ever(input)：本回合曾经成立；支持布尔向量，可累计访问点或扫描网格的并集。
* sequence(inputs)：依次完成并锁存进度；每步最多推进一个阶段。不是边沿触发语义，
  某条件早已成立并保持到其阶段时也可通过。需要“此时新发生”时配合 rising / changed。
* within(input, seconds)：回合开始后指定秒数内曾经成立，成功后锁存；不是从任意事件触发计时。
* changed(input, tolerance=0)：与上一个有效且连续的样本相比，任一分量变化超过容差。
* rising(input)：布尔由假变真；首样本建立基线，不产生边沿。
* delta(input)：当前值减上一步值，首样本或无效间断后的首样本无效；可配负权重表示距离减小。
* accumulate(input)：累计有效输入；输入为新增覆盖量时可以累计，重复覆盖不自动去重。
* integral(input)：累计有效输入 × dt。
* running_mean(input)、running_min(input)、running_max(input)：本回合有效样本均值、最小值、最大值。

判定、无效数据与奖励
--------------------

* success 与 failure 都必须是每个环境一个布尔值；同时成立时 failure 优先。
  与预设原生终止项并存，其他原生终止仍有效。不能把任意原生终止自动解释为成功。
* all / any / at_least 使用保守有效性：任何依赖无效时，该组合结果无效。
  ever / within / sequence 保存已有有效事件，不因后续暂时无效丢失历史。
* NaN、Inf 和显式 Reading.valid=false 都产生无效结果；无效条件不触发成功/失败，
  无效奖励贡献为零，指标保留有效性标记。程序错误仍报错，不吞掉异常假装模型判断失败。
* reward 为每环境标量公式，系数直接写入公式；布尔转为 0/1。
  条件奖励默认每个满足步均贡献；首次达成可用 ``changed(ever(condition))``。
  首样本就满足的情况不会触发 changed，可改成成功终止奖励或自定义评价。
* Isaac 原生 RewardManager 还会将奖励乘 step_dt。这里不抵消这个行为；若需要严格的一次性
  固定分数，应按控制频率设置权重。奖励权重与 dt 的解释必须一致。
* 超时继续使用现有 train.yaml 的 max_episode_steps，不新增第二个回合时钟配置。
* 同一步的共享引用与信号只计算/读取一次，所有配置节点都会更新，不做逻辑短路，
  避免时间条件漏记。未引用的场景信息不会读取。
* 并行环境分别维护状态，原生重置只清理对应槽位。重置不需要用户编写额外配置。

覆盖率组合示例
--------------

下面假定外部组件提供 ``scan/visible_cells``，形状为 [环境数, 网格数] 的布尔张量；
它负责视野、遮挡、清晰度等判定。各网格面积相等，总共 100 格。

.. code-block:: yaml

   variables:
     scanned: {op: ever, input: {signal: scan/visible_cells}}
     count: {op: count, input: {ref: scanned}}
     coverage: {op: divide, a: {ref: count}, b: {value: 100}}
   task:
     success: coverage >= 0.9
     metrics:
       coverage: coverage

重复看到同一个网格不会重复计数。面积不等时可乘每格面积后求和。
本版提供累计与组合，不包含扫描可见性、接触支撑语义、路径规划或视觉整洁模型算法。

项目自定义 / 外部模型
--------------------

通用 TaskEvaluator 接受已有 Signals 实例，任何后端都可以注册信息并调用；
当前自动接入的是 Isaac assembly，MuJoCo 项目尚未自动替换原有 task.py；
MuJoCo 的信息目录已经补充类型、形状与四元数编码描述。

Isaac 可选的 terms.py 用 ``TERMS['signals']`` 注册工厂，工厂只在启动后绑定 env，返回 Signal：

.. code-block:: python

   from shared.assembly.signals import Signal
   from shared.assembly.legacy.task import Reading
   from shared.assembly.information import Information

   def inspection_signal(env):
       # 项目先准备 evaluator；避免模块导入时启动网络服务或下载模型。
       def read():
           scores, available = env.inspection_evaluator.read_latest()
           return Reading(scores, available)
       return Signal(read, category="evaluation", description="桌面整洁评分",
                     info=Information("number", (), "float", "1"), source="model")

   # 静态 check 可读取这个属性，无需启动模型工厂。
   inspection_signal.information = Information("number", (), "float", "1")

   TERMS = {"signals": {"inspection/tidiness": inspection_signal}}

随后在顶层变量用 ``source: inspection/tidiness`` 引用。工厂不得依赖 env.inspection_evaluator 自动存在，
示例中的模型准备由项目自己的已有扩展负责，也可在工厂闭包中创建并持有只读评价器。
本版不提供外部服务调度、异步队列、超时重试或模型选择；昂贵模型的调用频率及结果新鲜度
由评价组件负责，不能把缓存的旧评分当成新一帧评价来重复计时。

信号的数值张量须包含首维环境轴，例如标量 [N]、位置 [N,3]、布尔集合 [N,K]。
单环境同样保留首维。纯标量信号会广播到所有环境；YAML 常量数组不含环境轴。
Reading.valid 为布尔标量或 [N]，当前按整个环境读数标记有效，尚不支持逐像素/逐元素有效性。

运行与边界
----------

.. code-block:: bash

   bash run.sh check assembly_lift
   bash run.sh check assembly_go2
   bash run.sh smoke assembly_lift env.train.total_num_envs=2 env.train.max_episode_steps=8
   bash run.sh smoke assembly_go2 env.train.total_num_envs=2 env.train.max_episode_steps=8
   .venv/bin/python -m pytest tests/test_task_evaluation.py tests/test_assembly_task_config.py -q

Isaac 编译为原生 task_success / task_failure 终止项、task_<name> 奖励项以及内部 reset 事件。
逐环境指标在 info['task_metrics'][name] 中包含 value 和 valid；
info['metrics'] 的 success/failure 供 RLinf 统计配置化 Task 的结果。
命名不能与这些保留原生项冲突。预设原生奖励仍保留，新增评价奖励是叠加，若需要替换应显式
禁用相应预设奖励。仅由其他原生终止项触发的失败未自动分类到此 Task 的 failure 指标；
需要统计该语义时应显式配置 failure，Lift 的落桌失败已这样配置。

这次先完成配置和执行接入。Blender 尚无本规则编辑面板，未实现自动单位换算、按实体标签批量
生成条件、滑动窗口统计、事件触发后限时或异步模型评分。集合规则目前
作用在提供的布尔向量上，不会自动遍历全部实体。相关需求可添加小型评价组件，不新增训练框架。

规则验证不代表策略学会任务。本轮没有启动长训练，也没有报告新的策略成功率。

观察与显示配置
--------------

planner / executor 的 observations 均为有序名称列表，不使用 source 字典混排。
只选择基础信息、目标或组件输出，不选择 Task variables。模型预处理放 policy.py，
不再支持 input_processing 配置，当前模板不施加输入噪声。PPO 保存原始观察，在采样与更新时
采用同一模型预处理；Task 独立读取基础信息计算评价。
抽象体显示使用 visualization.abstract 的名称列表，默认空，与采样/控制无关。
