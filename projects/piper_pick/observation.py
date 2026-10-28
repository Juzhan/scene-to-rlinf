# Generated from scene.yaml; edit the scene and run check to refresh.
from typing import TypedDict

from torch import Tensor
from shared.assembly.objects import ObjectView
from shared.assembly.robot import RobotView

class SceneInfo_cube(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_cube_area(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 2], float, m, cube_area, 尺寸（全长）


class SceneInfo_executor_input_cube(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_executor_input_piper(ObjectView):
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）


class SceneInfo_executor_input_tcp_frame(ObjectView):
    frame_positions: Tensor  # ['B', None, 3], float, m, environment, 测量坐标系位置


class SceneInfo_executor_input(ObjectView):
    cube: SceneInfo_executor_input_cube
    piper: SceneInfo_executor_input_piper
    tcp_frame: SceneInfo_executor_input_tcp_frame


class SceneInfo_executor(ObjectView):
    input: SceneInfo_executor_input
    output: Tensor  # ['B', 7], float, 本步输出（解码前动作）
    previous_output: Tensor  # ['B', 7], float, 上一步输出（解码前动作）


class SceneInfo_piper_tcp(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 坐标系朝向（tcp）
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class SceneInfo_piper(RobotView):
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
    tcp: SceneInfo_piper_tcp


class SceneInfo_planner_input_cube(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_planner_input_piper_tcp(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class SceneInfo_planner_input_piper(ObjectView):
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    tcp: SceneInfo_planner_input_piper_tcp


class SceneInfo_planner_input(ObjectView):
    cube: SceneInfo_planner_input_cube
    piper: SceneInfo_planner_input_piper


class SceneInfo_planner(ObjectView):
    input: SceneInfo_planner_input
    output: Tensor  # ['B', 3], float, 本步输出（解码前动作）
    previous_output: Tensor  # ['B', 3], float, 上一步输出（解码前动作）


class SceneInfo_simulation(ObjectView):
    gravity: Tensor  # ['B', 3], float, m/s^2, environment, 环境重力加速度
    step_dt: Tensor  # ['B', 1], s
    time: Tensor  # ['B', 1], s


class SceneInfo_tcp_frame(ObjectView):
    frame_positions: Tensor  # ['B', None, 3], float, m, environment, 测量坐标系位置


class SceneInfo_wrist_camera(ObjectView):
    rgb: Tensor  # ['B', 120, 160, 3], uint8, RGB, RGB 图像（独立图像通道）


class SceneInfo(ObjectView):
    cube: SceneInfo_cube
    cube_area: SceneInfo_cube_area
    executor: SceneInfo_executor
    piper: SceneInfo_piper
    planner: SceneInfo_planner
    simulation: SceneInfo_simulation
    tcp_frame: SceneInfo_tcp_frame
    wrist_camera: SceneInfo_wrist_camera


class PlannerObs_cube(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class PlannerObs_piper_tcp(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class PlannerObs_piper(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    tcp: PlannerObs_piper_tcp


class PlannerObs(ObjectView):
    cube: PlannerObs_cube
    piper: PlannerObs_piper


class ExecutorObs_cube(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class ExecutorObs_piper(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）


class ExecutorObs_tcp_frame(ObjectView):
    frame_positions: Tensor  # ['B', None, 3], float, m, environment, 测量坐标系位置


class ExecutorObs(ObjectView):
    cube: ExecutorObs_cube
    piper: ExecutorObs_piper
    tcp_frame: ExecutorObs_tcp_frame


Obs = ExecutorObs
SCENE_FIELDS = ('piper.position', 'piper.orientation', 'piper.linear_velocity', 'piper.angular_velocity', 'piper.joint_position', 'piper.joint_velocity', 'piper.joint_effort', 'piper.default_joint_position', 'piper.default_joint_velocity', 'cube.position', 'cube.orientation', 'cube.linear_velocity', 'cube.angular_velocity', 'wrist_camera.rgb', 'tcp_frame.frame_positions', 'piper.tcp.position', 'piper.tcp.orientation', 'simulation.gravity', 'cube_area.position', 'cube_area.orientation', 'cube_area.size', 'planner.output', 'planner.previous_output', 'executor.output', 'executor.previous_output', 'simulation.step_dt', 'simulation.time', 'planner.input.cube.position', 'planner.input.cube.orientation', 'planner.input.piper.joint_position', 'planner.input.piper.joint_velocity', 'planner.input.piper.joint_effort', 'planner.input.piper.tcp.position', 'executor.input.piper.joint_position', 'executor.input.piper.joint_velocity', 'executor.input.piper.joint_effort', 'executor.input.cube.position', 'executor.input.tcp_frame.frame_positions')


PlannerOutput = Tensor  # ['B', 3], 固定组件 forward 返回
ExecutorOutput = Tensor  # ['B', 7], 训练组件 forward 返回 (ExecutorOutput 均值, Tensor ['B', 1] 价值)，模型另需参数 actor_logstd；build 的 config 即 train.yaml 的 actor.model（action_dim、hidden_dims 等）；解码前动作，initial 返回同形状


class ExecutorTargets(TypedDict):  # decode(action, obs: ExecutorObs) 返回；实际单位，键与 control.output 一致
    arm: Tensor  # ['B', 8], joint_position, piper: joint1, joint2, joint3, joint4, joint5, joint6, gripper_joint1, gripper_joint2


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
