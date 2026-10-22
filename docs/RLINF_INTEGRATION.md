# RLinf 接入边界与历史补丁

2026-10-19：统一入口默认使用 `third_party/RLinf-upstream`，基于固定提交
`77ed948d68f07f55b4bc433b7643a409b2cd1adb`。此目录没有本地源码修改。
本机用 `git -C third_party/RLinf worktree add --detach ../RLinf-upstream HEAD` 创建；
换机可在该目录独立 clone 官方仓库并 checkout 上述提交。运行所需的 Python、Torch、Ray、Isaac 依赖仍需安装。

## 新模板如何使用原生接口

- 环境通过原有 `REGISTER_ISAACLAB_ENVS` 注册，模型通过 `register_model` 注册。
- `shared/integrations/rlinf/observations.py` 将命名 Tensor 字典放入原生 `states` 字段；图像、射线、关节保持形状和 dtype，没有统一展平。
- `task_descriptions` 是原生传输元数据，也让 RLinf 能推断批次数；不进入用户的对象观察。
- `ObjectPolicy` 解包 `states` 并构造 `Obs`。训练轨迹的 `forward_inputs` 仍按原生批处理要求使用平铺命名 Tensor 字典。
- `state_ppo.yaml` 的原生 `sampling_params` 区分训练与评估；共享模型读取 `do_sample`，直接调用也支持显式 `mode`。不修改 rollout worker。
- 默认模板使用原生学习率配置，移除 `adaptive_kl_target`，不另外实现 actor/优化循环。旧实验的自适应学习率结果不可当作新配方结果。
- Isaac 终态快照与自动复位接入仍在共享 adapter 中，复用原生循环；不要求项目维护 env/worker。
- 检查点保留数量使用已有的共享 Runner 子类；启动入口选择该子类，未修改第三方类或训练循环。

## 保留的历史源码改动

旧目录 `third_party/RLinf` 没有被重置。下列变动是历史实验的实际依赖，不是新模板要求；
运行历史配方需要先阅读其 README，并明确设置 `RLINF_ROOT="$PWD/third_party/RLinf"`。
不能把它们简单变成全局 monkey patch 来宣称不依赖补丁。

| 修改位置 | 用途 | 新路径或处理方式 |
| --- | --- | --- |
| data/schema/embodied_types.py | 自定义观察透传、批次推断 | 新模板改用原生 states 字典，无需补丁 |
| workers/rollout/hf/huggingface_worker.py | 自定义模型接收 train/eval mode | 新模板使用原生 do_sample 参数 |
| envs/sim/isaaclab/isaaclab_env.py | 保留原始 infos | assembly adapter 已自行接入 step/metrics，无需补丁 |
| envs/__init__.py | 新增 gymnasium 注册入口 | gridworld 仍依赖历史副本；尚未迁移，不冒充 Isaac 后端 |
| algorithms/utils.py、workers/actor/embodied_fsdp_actor_worker.py | 自适应 Gaussian KL 学习率 | 新模板移除该扩展；旧配方在干净依赖上明确报错 |
| runners/async_embodied_runner.py | 按 replay 样本预算停止 | 旧异步 SAC 实验专用，干净依赖不支持该参数 |
| workers/actor/async_fsdp_sac_policy_worker.py | 每轮等待新样本配额 | 同上，干净依赖不支持该参数 |
| workers/actor/fsdp_sac_policy_worker.py | 恢复 critic 更新计数 | 历史 SAC 续训路径，干净入口拒绝该续训配方，尚未迁移验收 |
| envs/sim/frankasim/frankasim_env.py | 渲染尺寸与图像缩放 | 旧 MuJoCo 图像配方依赖，干净入口明确拒绝这些参数；需后续环境子类适配 |
| tests/unit_tests/test_worker.py | 历史 worker 扩展测试 | 留在历史副本，不当作干净依赖的验收 |
| toolkits/standalone_eval_scripts/openpi/* | 专用评估工具与说明 | 与维护中 assembly 模板无关 |

当前只保证维护模板走干净依赖。SFT、MuJoCo/VLA、历史 SAC 的能力继续以项目说明为准；
本轮没有宣称整个工作区的所有历史实验都已摆脱补丁。

## 维护约束

扩展优先顺序：原生配置与注册入口 → 共享环境/模型 adapter → 原生支持的子类或钩子。
若必须替换完整 worker/训练循环，先说明缺少的上游能力，不复制训练框架，也不静默忽略参数。
旧实验权重、快照、源码与训练记录保持原路径；升级依赖必须重新核对接口与运行结果。

## 验证

`tests/test_stock_rlinf.py` 在新进程中确认实际导入干净目录，验证原生 EnvOutput、PolicyInput
路由、图像 dtype/维度、轨迹批处理、确定性评估与一次本地梯度更新。
`tests/test_assembly_native_rlinf.py` 用 CPU 传输替身验证完整原生 adapter 的部分复位和终态保存。
这两者都不是 Isaac 物理测试，也不是 RLinf 分布式 PPO 训练成功。

本轮结果：215 项核心回归通过，60 项控制/编辑入口回归通过（两组有重叠，不相加）；
四个维护模板的 check 均通过。独立新进程测试还调用了原生 MultiStepRolloutWorker.predict，
确认原生采样参数实际到达对象策略。测试记录存于 outputs/framework_cleanup_20261019/。
检查时 GPU 仅余约 0.75 GB，本轮未启动 Isaac 物理测试或分布式训练；当前版本仍待这些实机资源验证。

## mjlab 原生 PPO 验收

mjlab 环境通过现有环境注册入口构建，底层使用 mjlab 原生 manager 循环。已完成 GPU 部分复位与最终观察测试、原生 RLinf rollout、两次 PPO 更新、检查点保存和加载后的 4 回合评估。记录见 [后端说明](SCENE_BACKENDS.md)。原生评估合并器会把 states 字典变成生产者列表，共享 observations.fields 将其按批次合并；训练的递归字典路径保持不变。未修改上游代码，也未安装全局 monkey patch。

该验收不包含策略学会任务、相机输入、策略渲染或历史 MuJoCo/VLA 项目迁移。
