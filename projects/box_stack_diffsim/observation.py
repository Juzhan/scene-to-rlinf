# Generated from scene.yaml; edit the scene and run check to refresh.
from typing import TypedDict

from torch import Tensor
from shared.assembly.objects import ObjectView
from shared.assembly.robot import RobotView

class SceneInfo_cube_lower_target(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 固定期望值（targets）


class SceneInfo_cube_lower(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    target: SceneInfo_cube_lower_target


class SceneInfo_cube_upper_target(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 固定期望值（targets）


class SceneInfo_cube_upper(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    target: SceneInfo_cube_upper_target


class SceneInfo_ee_frame(ObjectView):
    frame_positions: Tensor  # ['B', None, 3], float, m, environment, 测量坐标系位置


class SceneInfo_executor_input_cube_lower(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_executor_input_cube_upper(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_executor_input_ee_frame(ObjectView):
    frame_positions: Tensor  # ['B', None, 3], float, m, environment, 测量坐标系位置


class SceneInfo_executor_input_planner(ObjectView):
    output: Tensor  # ['B', 3], float, 本步输出（解码前动作）


class SceneInfo_executor_input_robot(ObjectView):
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）


class SceneInfo_executor_input(ObjectView):
    cube_lower: SceneInfo_executor_input_cube_lower
    cube_upper: SceneInfo_executor_input_cube_upper
    ee_frame: SceneInfo_executor_input_ee_frame
    planner: SceneInfo_executor_input_planner
    robot: SceneInfo_executor_input_robot


class SceneInfo_executor(ObjectView):
    input: SceneInfo_executor_input
    output: Tensor  # ['B', 8], float, 本步输出（解码前动作）
    previous_output: Tensor  # ['B', 8], float, 上一步输出（解码前动作）


class SceneInfo_planner_input_cube_upper_target(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 固定期望值（targets）


class SceneInfo_planner_input_cube_upper(ObjectView):
    target: SceneInfo_planner_input_cube_upper_target


class SceneInfo_planner_input(ObjectView):
    cube_upper: SceneInfo_planner_input_cube_upper


class SceneInfo_planner(ObjectView):
    input: SceneInfo_planner_input
    output: Tensor  # ['B', 3], float, 本步输出（解码前动作）
    previous_output: Tensor  # ['B', 3], float, 上一步输出（解码前动作）


class SceneInfo_robot_tcp(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 坐标系朝向（tcp）
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class SceneInfo_robot_wrist_camera(ObjectView):
    rgb: Tensor  # ['B', 120, 160, 3], uint8, RGB, RGB 图像（独立图像通道）


class SceneInfo_robot(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    default_joint_velocity: Tensor  # ['B', None], float, 默认关节速度
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    tcp: SceneInfo_robot_tcp
    wrist_camera: SceneInfo_robot_wrist_camera


class SceneInfo_simulation(ObjectView):
    gravity: Tensor  # ['B', 3], float, m/s^2, environment, 环境重力加速度
    step_dt: Tensor  # ['B', 1], s
    time: Tensor  # ['B', 1], s


class SceneInfo(ObjectView):
    cube_lower: SceneInfo_cube_lower
    cube_upper: SceneInfo_cube_upper
    ee_frame: SceneInfo_ee_frame
    executor: SceneInfo_executor
    planner: SceneInfo_planner
    robot: SceneInfo_robot
    simulation: SceneInfo_simulation


class PlannerObs_cube_upper_target(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 固定期望值（targets）


class PlannerObs_cube_upper(ObjectView):
    target: PlannerObs_cube_upper_target


class PlannerObs(ObjectView):
    cube_upper: PlannerObs_cube_upper


class ExecutorObs_cube_lower(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class ExecutorObs_cube_upper(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class ExecutorObs_ee_frame(ObjectView):
    frame_positions: Tensor  # ['B', None, 3], float, m, environment, 测量坐标系位置


class ExecutorObs_planner(ObjectView):
    output: Tensor  # ['B', 3], float, 本步输出（解码前动作）


class ExecutorObs_robot(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）


class ExecutorObs(ObjectView):
    cube_lower: ExecutorObs_cube_lower
    cube_upper: ExecutorObs_cube_upper
    ee_frame: ExecutorObs_ee_frame
    planner: ExecutorObs_planner
    robot: ExecutorObs_robot


Obs = ExecutorObs
SCENE_FIELDS = ('robot.position', 'robot.orientation', 'robot.linear_velocity', 'robot.angular_velocity', 'robot.joint_position', 'robot.joint_velocity', 'robot.joint_effort', 'robot.default_joint_position', 'robot.default_joint_velocity', 'robot.wrist_camera.rgb', 'cube_lower.position', 'cube_lower.orientation', 'cube_lower.linear_velocity', 'cube_lower.angular_velocity', 'cube_upper.position', 'cube_upper.orientation', 'cube_upper.linear_velocity', 'cube_upper.angular_velocity', 'ee_frame.frame_positions', 'robot.tcp.position', 'robot.tcp.orientation', 'simulation.gravity', 'cube_lower.target.position', 'cube_upper.target.position', 'planner.output', 'planner.previous_output', 'executor.output', 'executor.previous_output', 'simulation.step_dt', 'simulation.time', 'planner.input.cube_upper.target.position', 'executor.input.robot.joint_position', 'executor.input.robot.joint_velocity', 'executor.input.cube_upper.position', 'executor.input.cube_lower.position', 'executor.input.planner.output', 'executor.input.ee_frame.frame_positions', 'executor.input.robot.default_joint_position')


PlannerOutput = Tensor  # ['B', 3], 固定组件 forward 返回
ExecutorOutput = Tensor  # ['B', 8], 训练组件 forward 返回 (ExecutorOutput 均值, Tensor ['B', 1] 价值)，模型另需参数 actor_logstd；build 的 config 即 train.yaml 的 actor.model（action_dim、hidden_dims 等）；解码前动作，initial 返回同形状


class ExecutorTargets(TypedDict):  # decode(action, obs: ExecutorObs) 返回；实际单位，键与 control.output 一致
    arm: Tensor  # ['B', 7], joint_position, robot: panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5, panda_joint6, panda_joint7
    gripper: Tensor  # ['B', 2], joint_position, robot: panda_finger_joint1, panda_finger_joint2


class TaskResult(TypedDict, total=False):  # evaluate(obs: SceneInfo) 返回；缺省 reward=0, success/failure=False
    # 每决策步评价一次：reward 乘 step_dt 计入回报；success 或 failure 为真即结束该环境的回合。
    # 计时等跨步状态：写 Task 类，在 __init__ 中建 shared.assembly.task_memory.TaskMemory（hold 计时、mean 平均），按环境随复位清零。
    reward: Tensor  # ['B']
    success: Tensor  # ['B'], bool
    failure: Tensor  # ['B'], bool
    metrics: dict[str, Tensor]  # 名称 -> ['B']；回合最后一步的值记入训练/评估日志 env/task/名称
    # set：本步把抽象体移到新位置（任务事件），在复位之后、观察之前生效；刚复位的环境保持采样位姿。
    # 读数 <名>.linear_velocity = 位移 / step_dt。可设置（未跟随、非 reset 区域）：无
    set: dict[str, Tensor]  # 抽象体名 -> ['B', 3] 位置或 ['B', 7] xyz + wxyz，环境坐标
