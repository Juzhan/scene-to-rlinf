# Generated from scene.yaml; edit the scene and run check to refresh.
from typing import TypedDict

from torch import Tensor
from shared.assembly.objects import ObjectView
from shared.assembly.robot import RobotView

class SceneInfo_base_contact(ObjectView):
    contact_force: Tensor  # ['B', None, 3], float, N, environment, 各刚体接触力
    contact_force_samples: Tensor  # ['B', None, None, 3], float, N, environment, 接触传感器采样窗口


class SceneInfo_camera(ObjectView):
    rgb: Tensor  # ['B', 120, 160, 3], uint8, RGB, RGB 图像（独立图像通道）


class SceneInfo_dex_cube(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_ee_workspace(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 3], float, m, ee_workspace, 尺寸（全长）


class SceneInfo_executor_input_executor(ObjectView):
    output: Tensor  # ['B', 18], float, 本步输出（解码前动作）


class SceneInfo_executor_input_feet_contact(ObjectView):
    contact_force: Tensor  # ['B', None, 3], float, N, environment, 各刚体接触力


class SceneInfo_executor_input_piperx_tcp(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class SceneInfo_executor_input_piperx(ObjectView):
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    tcp: SceneInfo_executor_input_piperx_tcp


class SceneInfo_executor_input_planner(ObjectView):
    output: Tensor  # ['B', 6], float, 本步输出（解码前动作）


class SceneInfo_executor_input_robot(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向


class SceneInfo_executor_input_simulation(ObjectView):
    gravity: Tensor  # ['B', 3], float, m/s^2, environment, 环境重力加速度


class SceneInfo_executor_input(ObjectView):
    executor: SceneInfo_executor_input_executor
    feet_contact: SceneInfo_executor_input_feet_contact
    piperx: SceneInfo_executor_input_piperx
    planner: SceneInfo_executor_input_planner
    robot: SceneInfo_executor_input_robot
    simulation: SceneInfo_executor_input_simulation


class SceneInfo_executor(ObjectView):
    input: SceneInfo_executor_input
    output: Tensor  # ['B', 18], float, 本步输出（解码前动作）
    previous_output: Tensor  # ['B', 18], float, 上一步输出（解码前动作）


class SceneInfo_feet_contact(ObjectView):
    contact_force: Tensor  # ['B', None, 3], float, N, environment, 各刚体接触力
    contact_force_samples: Tensor  # ['B', None, None, 3], float, N, environment, 接触传感器采样窗口


class SceneInfo_piperx_tcp(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 坐标系朝向（tcp）
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class SceneInfo_piperx(RobotView):
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
    tcp: SceneInfo_piperx_tcp


class SceneInfo_planner_input_ee_workspace(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 3], float, m, ee_workspace, 尺寸（全长）


class SceneInfo_planner_input_piperx(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_planner_input(ObjectView):
    ee_workspace: SceneInfo_planner_input_ee_workspace
    piperx: SceneInfo_planner_input_piperx


class SceneInfo_planner(ObjectView):
    input: SceneInfo_planner_input
    output: Tensor  # ['B', 6], float, 本步输出（解码前动作）
    previous_output: Tensor  # ['B', 6], float, 上一步输出（解码前动作）


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
    base_contact: SceneInfo_base_contact
    camera: SceneInfo_camera
    dex_cube: SceneInfo_dex_cube
    ee_workspace: SceneInfo_ee_workspace
    executor: SceneInfo_executor
    feet_contact: SceneInfo_feet_contact
    piperx: SceneInfo_piperx
    planner: SceneInfo_planner
    robot: SceneInfo_robot
    simulation: SceneInfo_simulation


class PlannerObs_ee_workspace(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 3], float, m, ee_workspace, 尺寸（全长）


class PlannerObs_piperx(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class PlannerObs(ObjectView):
    ee_workspace: PlannerObs_ee_workspace
    piperx: PlannerObs_piperx


class ExecutorObs_executor(ObjectView):
    output: Tensor  # ['B', 18], float, 本步输出（解码前动作）


class ExecutorObs_feet_contact(ObjectView):
    contact_force: Tensor  # ['B', None, 3], float, N, environment, 各刚体接触力


class ExecutorObs_piperx_tcp(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class ExecutorObs_piperx(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    tcp: ExecutorObs_piperx_tcp


class ExecutorObs_planner(ObjectView):
    output: Tensor  # ['B', 6], float, 本步输出（解码前动作）


class ExecutorObs_robot(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：无
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向


class ExecutorObs_simulation(ObjectView):
    gravity: Tensor  # ['B', 3], float, m/s^2, environment, 环境重力加速度


class ExecutorObs(ObjectView):
    executor: ExecutorObs_executor
    feet_contact: ExecutorObs_feet_contact
    piperx: ExecutorObs_piperx
    planner: ExecutorObs_planner
    robot: ExecutorObs_robot
    simulation: ExecutorObs_simulation


Obs = ExecutorObs
SCENE_FIELDS = ('robot.position', 'robot.orientation', 'robot.linear_velocity', 'robot.angular_velocity', 'robot.joint_position', 'robot.joint_velocity', 'robot.joint_effort', 'robot.default_joint_position', 'robot.default_joint_velocity', 'piperx.position', 'piperx.orientation', 'piperx.linear_velocity', 'piperx.angular_velocity', 'piperx.joint_position', 'piperx.joint_velocity', 'piperx.joint_effort', 'piperx.default_joint_position', 'piperx.default_joint_velocity', 'base_contact.contact_force_samples', 'base_contact.contact_force', 'feet_contact.contact_force_samples', 'feet_contact.contact_force', 'dex_cube.position', 'dex_cube.orientation', 'dex_cube.linear_velocity', 'dex_cube.angular_velocity', 'camera.rgb', 'piperx.tcp.position', 'piperx.tcp.orientation', 'simulation.gravity', 'ee_workspace.position', 'ee_workspace.orientation', 'ee_workspace.size', 'planner.output', 'planner.previous_output', 'executor.output', 'executor.previous_output', 'simulation.step_dt', 'simulation.time', 'planner.input.piperx.position', 'planner.input.ee_workspace.position', 'planner.input.ee_workspace.size', 'planner.input.ee_workspace.orientation', 'executor.input.robot.linear_velocity', 'executor.input.robot.angular_velocity', 'executor.input.robot.orientation', 'executor.input.simulation.gravity', 'executor.input.robot.joint_position', 'executor.input.robot.default_joint_position', 'executor.input.robot.joint_velocity', 'executor.input.planner.output', 'executor.input.executor.output', 'executor.input.piperx.orientation', 'executor.input.feet_contact.contact_force', 'executor.input.piperx.position', 'executor.input.piperx.joint_position', 'executor.input.piperx.joint_velocity', 'executor.input.piperx.tcp.position', 'executor.input.piperx.default_joint_position')


PlannerOutput = Tensor  # ['B', 6], 固定组件 forward 返回
ExecutorOutput = Tensor  # ['B', 18], 训练组件 forward 返回 (ExecutorOutput 均值, Tensor ['B', 1] 价值)，模型另需参数 actor_logstd；build 的 config 即 train.yaml 的 actor.model（action_dim、hidden_dims 等）；解码前动作，initial 返回同形状


class ExecutorTargets(TypedDict):  # decode(action, obs: ExecutorObs) 返回；实际单位，键与 control.output 一致
    legs: Tensor  # ['B', 12], joint_position, robot: FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint
    arm: Tensor  # ['B', 6], joint_position, piperx: joint1, joint2, joint3, joint4, joint5, joint6


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
