# Generated from scene.yaml; edit the scene and run check to refresh.
from typing import TypedDict

from torch import Tensor
from shared.assembly.objects import ObjectView
from shared.assembly.robot import RobotView

class SceneInfo_arena(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 2], float, m, arena, 尺寸（全长）


class SceneInfo_contact_forces(ObjectView):
    contact_force: Tensor  # ['B', None, 3], float, N, environment, 各刚体接触力
    contact_force_samples: Tensor  # ['B', None, None, 3], float, N, environment, 接触传感器采样窗口


class SceneInfo_destination(ObjectView):
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 线速度（Task set 位移 / 决策步长）
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置


class SceneInfo_executor_input_executor(ObjectView):
    output: Tensor  # ['B', None], float, 本步输出（动作）


class SceneInfo_executor_input_planner(ObjectView):
    goal: Tensor  # ['B', 3], float, 目标位置（m，环境坐标）
    heading: Tensor  # ['B', 1], float, 目标朝向（rad，绕 z）


class SceneInfo_executor_input_robot(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_executor_input(ObjectView):
    executor: SceneInfo_executor_input_executor
    planner: SceneInfo_executor_input_planner
    robot: SceneInfo_executor_input_robot


class SceneInfo_executor(ObjectView):
    input: SceneInfo_executor_input
    output: Tensor  # ['B', None], float, 本步输出（动作）
    previous_output: Tensor  # ['B', None], float, 上一步输出（动作）


class SceneInfo_height_scanner(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    ray_hit_positions: Tensor  # ['B', None, 3], float, m, environment, 射线命中位置


class SceneInfo_planner_input_destination(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置


class SceneInfo_planner_input(ObjectView):
    destination: SceneInfo_planner_input_destination


class SceneInfo_planner(ObjectView):
    goal: Tensor  # ['B', 3], float, 目标位置（m，环境坐标）
    heading: Tensor  # ['B', 1], float, 目标朝向（rad，绕 z）
    input: SceneInfo_planner_input


class SceneInfo_robot(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：无
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    bounds: Tensor  # ['B', 2, 3], float, m, robot/root, 包围盒（本体坐标的最小角, 最大角）
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    default_joint_velocity: Tensor  # ['B', None], float, 默认关节速度
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_limits: Tensor  # ['B', None, 2], float, 关节范围（下限, 上限）
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_simulation(ObjectView):
    gravity: Tensor  # ['B', 3], float, m/s^2, environment, 环境重力加速度
    step_dt: Tensor  # ['B', 1], s
    time: Tensor  # ['B', 1], s


class SceneInfo(ObjectView):
    arena: SceneInfo_arena
    contact_forces: SceneInfo_contact_forces
    destination: SceneInfo_destination
    executor: SceneInfo_executor
    height_scanner: SceneInfo_height_scanner
    planner: SceneInfo_planner
    robot: SceneInfo_robot
    simulation: SceneInfo_simulation


class PlannerObs_destination(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置


class PlannerObs(ObjectView):
    destination: PlannerObs_destination


class ExecutorObs_executor(ObjectView):
    output: Tensor  # ['B', None], float, 本步输出（动作）


class ExecutorObs_planner(ObjectView):
    goal: Tensor  # ['B', 3], float, 目标位置（m，环境坐标）
    heading: Tensor  # ['B', 1], float, 目标朝向（rad，绕 z）


class ExecutorObs_robot(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：无
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class ExecutorObs(ObjectView):
    executor: ExecutorObs_executor
    planner: ExecutorObs_planner
    robot: ExecutorObs_robot


Obs = ExecutorObs
SCENE_FIELDS = ('robot.position', 'robot.orientation', 'robot.linear_velocity', 'robot.angular_velocity', 'robot.joint_position', 'robot.joint_velocity', 'robot.joint_effort', 'robot.default_joint_position', 'robot.default_joint_velocity', 'robot.joint_limits', 'robot.bounds', 'height_scanner.position', 'height_scanner.ray_hit_positions', 'contact_forces.contact_force_samples', 'contact_forces.contact_force', 'simulation.gravity', 'arena.position', 'arena.orientation', 'arena.size', 'destination.position', 'destination.orientation', 'destination.linear_velocity', 'planner.goal', 'planner.heading', 'executor.output', 'executor.previous_output', 'simulation.step_dt', 'simulation.time', 'planner.input.destination.position', 'planner.input.destination.orientation', 'executor.input.robot.linear_velocity', 'executor.input.robot.angular_velocity', 'executor.input.robot.orientation', 'executor.input.robot.position', 'executor.input.robot.joint_position', 'executor.input.robot.joint_velocity', 'executor.input.planner.goal', 'executor.input.planner.heading', 'executor.input.executor.output')


# planner 的 forward 返回字典（goal、heading），下一层读 obs.planner.<名字>
Action = Tensor  # 被训练层 executor 的 forward 返回：动作 ['B', 动作个数]；个数就是模型最后一层的大小，构造参数写 actions 时得到受控关节总数（默认 decode 的情形）
# initial_action(obs) 返回同形状的 Action（复位后第一步之前的动作，缺省为 0）


class Targets(TypedDict):  # decode_action(action, obs: ExecutorObs) 返回；实际单位，每台受控机器人一项
    robot: Tensor  # ['B', 关节数], joint_position, robot 的全部关节，顺序同 robot.joint_position


class TaskResult(TypedDict, total=False):  # evaluate(obs: SceneInfo) 返回；缺省 reward=0, success/failure=False
    # 每决策步评价一次：reward 乘 step_dt 计入回报；success 或 failure 为真即结束该环境的回合。
    # 计时等跨步状态：写 Task 类，在 __init__ 中建 shared.assembly.task_memory.TaskMemory（hold 计时、mean 平均），按环境随复位清零。
    reward: Tensor  # ['B']
    success: Tensor  # ['B'], bool
    failure: Tensor  # ['B'], bool
    metrics: dict[str, Tensor]  # 名称 -> ['B']；回合最后一步的值记入训练/评估日志 env/task/名称
    # set：本步把抽象体移到新位置（任务事件），在复位之后、观察之前生效；刚复位的环境保持采样位姿。
    # 读数 <名>.linear_velocity = 位移 / step_dt。可设置（未跟随、非 reset 区域）：destination
    set: dict[str, Tensor]  # 抽象体名 -> ['B', 3] 位置或 ['B', 7] xyz + wxyz，环境坐标
