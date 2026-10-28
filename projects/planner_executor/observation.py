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
    output: Tensor  # ['B', 12], float, 本步输出（解码前动作）


class SceneInfo_executor_input_planner(ObjectView):
    output: Tensor  # ['B', 4], float, 本步输出（解码前动作）


class SceneInfo_executor_input_robot(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    default_joint_velocity: Tensor  # ['B', None], float, 默认关节速度
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_executor_input_simulation(ObjectView):
    gravity: Tensor  # ['B', 3], float, m/s^2, environment, 环境重力加速度


class SceneInfo_executor_input(ObjectView):
    executor: SceneInfo_executor_input_executor
    planner: SceneInfo_executor_input_planner
    robot: SceneInfo_executor_input_robot
    simulation: SceneInfo_executor_input_simulation


class SceneInfo_executor(ObjectView):
    input: SceneInfo_executor_input
    output: Tensor  # ['B', 12], float, 本步输出（解码前动作）
    previous_output: Tensor  # ['B', 12], float, 上一步输出（解码前动作）


class SceneInfo_height_scanner(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    ray_hit_positions: Tensor  # ['B', None, 3], float, m, environment, 射线命中位置


class SceneInfo_planner_input_destination(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置


class SceneInfo_planner_input(ObjectView):
    destination: SceneInfo_planner_input_destination


class SceneInfo_planner(ObjectView):
    input: SceneInfo_planner_input
    output: Tensor  # ['B', 4], float, 本步输出（解码前动作）
    previous_output: Tensor  # ['B', 4], float, 上一步输出（解码前动作）


class SceneInfo_robot(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：无
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    default_joint_velocity: Tensor  # ['B', None], float, 默认关节速度
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
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
    output: Tensor  # ['B', 12], float, 本步输出（解码前动作）


class ExecutorObs_planner(ObjectView):
    output: Tensor  # ['B', 4], float, 本步输出（解码前动作）


class ExecutorObs_robot(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：无
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    default_joint_velocity: Tensor  # ['B', None], float, 默认关节速度
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class ExecutorObs_simulation(ObjectView):
    gravity: Tensor  # ['B', 3], float, m/s^2, environment, 环境重力加速度


class ExecutorObs(ObjectView):
    executor: ExecutorObs_executor
    planner: ExecutorObs_planner
    robot: ExecutorObs_robot
    simulation: ExecutorObs_simulation


Obs = ExecutorObs
SCENE_FIELDS = ('robot.position', 'robot.orientation', 'robot.linear_velocity', 'robot.angular_velocity', 'robot.joint_position', 'robot.joint_velocity', 'robot.joint_effort', 'robot.default_joint_position', 'robot.default_joint_velocity', 'height_scanner.position', 'height_scanner.ray_hit_positions', 'contact_forces.contact_force_samples', 'contact_forces.contact_force', 'simulation.gravity', 'arena.position', 'arena.orientation', 'arena.size', 'destination.position', 'destination.orientation', 'destination.linear_velocity', 'planner.output', 'planner.previous_output', 'executor.output', 'executor.previous_output', 'simulation.step_dt', 'simulation.time', 'planner.input.destination.position', 'planner.input.destination.orientation', 'executor.input.robot.linear_velocity', 'executor.input.robot.angular_velocity', 'executor.input.robot.orientation', 'executor.input.simulation.gravity', 'executor.input.robot.joint_position', 'executor.input.robot.default_joint_position', 'executor.input.robot.joint_velocity', 'executor.input.robot.default_joint_velocity', 'executor.input.planner.output', 'executor.input.robot.position', 'executor.input.executor.output')


PlannerOutput = Tensor  # ['B', 4], 固定组件 forward 返回
ExecutorOutput = Tensor  # ['B', 12], 训练组件 forward 返回 (ExecutorOutput 均值, Tensor ['B', 1] 价值)，模型另需参数 actor_logstd；build 的 config 即 train.yaml 的 actor.model（action_dim、hidden_dims 等）；解码前动作，initial 返回同形状


class ExecutorTargets(TypedDict):  # decode(action, obs: ExecutorObs) 返回；实际单位，键与 control.output 一致
    legs: Tensor  # ['B', 12], joint_position, robot: FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint


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
