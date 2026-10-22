当前项目读取 Information 元数据并生成对象类型；数学计算写在 policy.py/task.py。
下文关于 YAML variables、history、op 和公式推导的段落属于历史表达式引擎，
仅由 shared/assembly/legacy 使用，不是当前配置要求。

场景信息分类与 Task 类型推导
===========================

信息含义和数据组织分开
----------------------

Information 是元数据，不改变原生仿真存储，不要求用户每项填写，也不读取图像或启动模型。

2026-10-12：Task 已统一到 variables。source 和直接别名保留类型；任意 Torch expr 的
单位、坐标和形状不会被自动猜测，未知项在 check.untyped_variables 报告。
变量可用 information 显式声明结果语义；运行时检查其形状和布尔类型。
每个变量可启用 history，历史维度是 [环境, 时间, 数据...]，带独立有效标记和时间戳；
history 不是一种新的物理信息类型。已有 op 的语义检查继续保留。
后端、资产或自定义评价器声明，Observation 和 Task 共用 Signals 目录。

当前语义类别：

* number 数值：质量、温度、评分、覆盖率、单关节角度。标量 shape=()，集合 shape=(K,)。
* boolean 真假：开关、接触、检查通过；也可为布尔集合或网格。
* category 类别：运行模式、阶段。labels 列举名称，支持字符串值和从 0 开始的整数编码。
* position 位置：空间中的点，有单位和参考坐标系，不等同于位移向量。
* vector 空间向量：速度、力、方向、位移，有分量、单位与坐标系。
* rotation 旋转：当前实现四元数，encoding 必须声明 wxyz 或 xyzw。
* pose 位姿：位置和朝向组成的结构，使用 field 分开计算，不对七个分量直接算距离。
* region 区域：当前有朝向的矩形、盒形区域。其他几何算法可另接组件，暂不提供网格表面相交算法。
* image 图像：RGB、深度等，通过 encoding、dtype、axes 描述，不当作普通数值向量归约。
* point_cloud 点云：已提供分类和描述入口，没有内置点云匹配/重建算法。
* grid 空间网格：已提供分类入口，没有通用地图规划算法。布尔覆盖矩阵可声明 boolean + 空间轴，
  以使用 ever / count；grid 原始地图必须先由组件提取可计算的内容。
* record 组合信息：例如速度命令中的平面线速度和转向角速度；每个字段独立带单位。
* unknown 未声明：兼容已有自定义信息，check 明确列出，不根据长度猜语义。

位置/速度都是三个数，但不能随便相加；四元数有四个数，不能因此用普通欧氏距离评价朝向。
新项目通过生成的 SceneInfo 类型读取基础信息，计算写在 Python Task 中。
历史 YAML 表达式的运算检查保存在 legacy/task_types.py，不是新项目配置接口。

描述字段
--------

* kind：上述含义；不是实体类别。机器人/相机/物体分组继续用 Signal.category 与 entity。
* shape：单个环境内的数据形状，例如 ()、(3,)、(12,)、(480,640,3)。None 维度表示启动后确定。
* dtype：float / int / bool / uint8 等。
* unit：m、s、rad、m/s、N、kg 等。四元数编码不再放在单位字段里。
* frame：信息在哪个坐标系表达。
* target_frame：旋转描述的对象坐标系。四元数把 target_frame 下的向量旋转到 frame。
* encoding：wxyz、xyzw、RGB、区域打包格式等编码约定。
* axes：各维的含义，例如 joint、height/width/channel；环境批次维不在其中。
* labels：枚举类别名称，或关节集合的元素名称。
* fields：组合信息各字段的索引和独立类型。由实现者提供，普通任务使用 field 名称即可。
* Signal.entity / source：所属实体和来源，例如 simulation、sensor、model、authored。

时间历史可用 time 轴表达；本轮没有新增通用时序存储、异步数据时间戳或过期策略。
有效性仍由 Reading.valid 显式提供，不能靠元数据把旧模型评分自动判断为新鲜观测。

Signals.describe() 展示 information、entity、source，并附上候选操作列表。
候选列表用于界面提示，具体多个输入是否兼容仍由 Task 检查，例如距离需要匹配坐标系与维数。

已接入的位置
------------

* Isaac entity 信息目录：位置、朝向、速度、关节集合、扫描高度、接触力和图像。
* Isaac 显式 targets：固定三维期望位置、线速度、角速度，继承对应属性的类型与环境坐标。
  原生命令描述只用于旧实验兼容，新模板不从原生任务注入目标。
* 抽象体（``kind: abstract``）：position（m，环境坐标）、orientation（wxyz，环境坐标）、矩形/长方体的 size
  （全长，m）、方向的 direction（本地 +X 的单位向量，环境坐标）。静态、随机复位、跟随实体根部的读数共用同一字段。
* Isaac 独立场景预览：根位姿、速度、质量、关节状态、帧、相机与时钟的主要动态信息。
  任意参数字典、元数据或项目扩展未全部强制转换，允许 unknown。
* MuJoCo/robosuite 信息目录：物体位姿速度、机器人 TCP/关节、相机。保留原有数组布局，
  单环境 MuJoCo 还没有自动接入 batched TaskEvaluator，不能把类型描述当成任务已迁移。

Isaac 的根位置减去 env_origins；根速度虽然读取 world 轴，因副本坐标仅有平移，
可同样标记 environment 轴。不要把它与机器人自身旋转坐标混用。

Task 如何使用类型
----------------

配置为顶层 ``variables`` 和 ``task``，没有增加用户必填的 schema 配置。
check 在不启动仿真的情况下给出 sources、information 和 untyped_signals。
运行时绑定实际元数据再校验，并检查实际读数形状和布尔 dtype。

推导示例：

.. code-block:: text

   方块位置：position[3], m, environment
   目标位置：position[3], m, environment
          │ distance
          ▼
   目标距离：number[], m
          │ less_equal(0.03)
          ▼
   已到达：boolean[]
          │ hold(0.5 s)
          ▼
   稳定到达：boolean[] → success

* position - position → vector，保留长度单位与坐标系。
* vector 的 norm → number，保留单位；scalar 本身不能再执行向量归约。
* 两个 position 相加、位置减去不同坐标系位置、米与秒比较会报错。
* rotation 不允许普通 distance / norm；改用 rotation_distance，输出弧度。
* pose / record 不允许任意切片破坏结构；field 或完全匹配已声明字段的 select 才能保留语义。
* 旋转操作检查输入/输出坐标系，按声明处理 wxyz / xyzw，不改原始信号数据。
* 布尔集合需 all_elements / any_elements 归约后，才能作为单一成功条件。
* unknown 保留运行兼容，但必须明确标记缺失描述；已知类别不允许退化成普通数字绕过检查。

常量的数字本身没有物理单位，在比较/加减中按所选输入的上下文解释：距离阈值 0.03 表示 0.03 m。
乘除中的无单位字面数作为无量纲系数。两个声明过单位的信号必须兼容。
乘除、点积、积分有简单符号单位推导，例如 (m/s) × s → m。不会自动把 cm 换算成 m，
也不是完整的 SI 单位等价求解器；涉及换算请由信息提供方显式转换并声明输出单位。
未知单位保持未知，尤其不把任意关节集合默认标成 rad。

三种配置示例
------------

比较实际位置与显式目标（assembly_lift 已使用）：

.. code-block:: yaml

   target_distance:
     op: distance
     a: {signal: object.position}
     b: {signal: object.target.position}

期望位置在 targets.object.position 设定，和实际位置一样带有 position 类型、米单位和环境坐标。
assembly_go2 同样比较 robot.linear_velocity 与 robot.target.linear_velocity，不再切片原生命令。

旧表达式中的 ``region: {signal: regions.<name>}`` 已随 ``regions:`` 段移除。新项目在 task.py 中读取
抽象体（如 ``obs.spawn_area.position/orientation/size``）自行判断：先把点转到区域局部坐标，矩形比较
XY，长方体比较三维。这只判断实体根位置，不代表整个物体几何都在区域里，也不判断支撑或接触。

设备类别（须由项目注册该 Signal）：

.. code-block:: python

   info = Information("category", (), "int", labels=("stopped", "running", "error"))
   signal = Signal(read_mode, info=info, entity="machine", source="sensor")

.. code-block:: yaml

   running:
     op: equal
     a: {signal: machine/mode}
     b: {value: running}

输入可返回 ["running", "error"] 或 [1, 2]，两种方式含义相同；枚举拼写错误会在静态元数据完整时报告。
自定义 Isaac 工厂可设置 factory.information = info，让 check 无需执行工厂便可获知类型。

验证与边界
----------

完整运算表与时序、奖励语义继续见 TASK_RULES.rst（现有 49 项）。
本轮没有添加 Blender 的可视化任务编辑器；目录已经提供类型及候选操作，可供后续界面消费。
没有自动生成点云、视觉整洁、场景遮挡等算法。没有重新包装仿真推进或训练循环。
