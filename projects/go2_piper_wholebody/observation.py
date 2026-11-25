# Generated from scene.yaml; edit the scene and run check to refresh.
from typing import TypedDict

from torch import Tensor
from shared.assembly.objects import ObjectView
from shared.assembly.robot import RobotView

class SceneInfo_base_contact(ObjectView):
    contact_force: Tensor  # ['B', None, 3], float, N, environment, 各刚体接触力
    contact_force_samples: Tensor  # ['B', None, None, 3], float, N, environment, 接触传感器采样窗口


class SceneInfo_body_input_body(ObjectView):
    output: Tensor  # ['B', None], float, 本步输出（动作）


class SceneInfo_body_input_command(ObjectView):
    ee_target: Tensor  # ['B', 3], float, 末端目标（m），Piper 基座坐标系
    velocity: Tensor  # ['B', 3], float, 机身速度指令 vx、vy（m/s）、wz（rad/s），机身坐标系


class SceneInfo_body_input_feet_contact(ObjectView):
    contact_force: Tensor  # ['B', None, 3], float, N, environment, 各刚体接触力


class SceneInfo_body_input_piperx_tcp(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class SceneInfo_body_input_piperx(ObjectView):
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    tcp: SceneInfo_body_input_piperx_tcp


class SceneInfo_body_input_robot(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向


class SceneInfo_body_input(ObjectView):
    body: SceneInfo_body_input_body
    command: SceneInfo_body_input_command
    feet_contact: SceneInfo_body_input_feet_contact
    piperx: SceneInfo_body_input_piperx
    robot: SceneInfo_body_input_robot


class SceneInfo_body(ObjectView):
    input: SceneInfo_body_input
    output: Tensor  # ['B', None], float, 本步输出（动作）
    previous_output: Tensor  # ['B', None], float, 上一步输出（动作）


class SceneInfo_camera(ObjectView):
    rgb: Tensor  # ['B', 120, 160, 3], uint8, RGB, RGB 图像（独立图像通道）


class SceneInfo_command_input_ee_workspace(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 3], float, m, ee_workspace, 尺寸（全长）


class SceneInfo_command_input_piperx(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_command_input(ObjectView):
    ee_workspace: SceneInfo_command_input_ee_workspace
    piperx: SceneInfo_command_input_piperx


class SceneInfo_command(ObjectView):
    ee_target: Tensor  # ['B', 3], float, 末端目标（m），Piper 基座坐标系
    input: SceneInfo_command_input
    velocity: Tensor  # ['B', 3], float, 机身速度指令 vx、vy（m/s）、wz（rad/s），机身坐标系


class SceneInfo_dex_cube(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    bounds: Tensor  # ['B', 2, 3], float, m, dex_cube/root, 包围盒（本体坐标的最小角, 最大角）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_ee_workspace(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 3], float, m, ee_workspace, 尺寸（全长）


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
    bounds: Tensor  # ['B', 2, 3], float, m, piperx/root, 包围盒（本体坐标的最小角, 最大角）
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    default_joint_velocity: Tensor  # ['B', None], float, 默认关节速度
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_limits: Tensor  # ['B', None, 2], float, 关节范围（下限, 上限）
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    tcp: SceneInfo_piperx_tcp


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


class SceneInfo_terrain_rough_40m(ObjectView):
    bounds: Tensor  # ['B', 2, 3], float, m, terrain_rough_40m/root, 包围盒（本体坐标的最小角, 最大角）
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo(ObjectView):
    base_contact: SceneInfo_base_contact
    body: SceneInfo_body
    camera: SceneInfo_camera
    command: SceneInfo_command
    dex_cube: SceneInfo_dex_cube
    ee_workspace: SceneInfo_ee_workspace
    feet_contact: SceneInfo_feet_contact
    piperx: SceneInfo_piperx
    robot: SceneInfo_robot
    simulation: SceneInfo_simulation
    terrain_rough_40m: SceneInfo_terrain_rough_40m


class CommandObs_ee_workspace(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 3], float, m, ee_workspace, 尺寸（全长）


class CommandObs_piperx(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class CommandObs(ObjectView):
    ee_workspace: CommandObs_ee_workspace
    piperx: CommandObs_piperx


class BodyObs_body(ObjectView):
    output: Tensor  # ['B', None], float, 本步输出（动作）


class BodyObs_command(ObjectView):
    ee_target: Tensor  # ['B', 3], float, 末端目标（m），Piper 基座坐标系
    velocity: Tensor  # ['B', 3], float, 机身速度指令 vx、vy（m/s）、wz（rad/s），机身坐标系


class BodyObs_feet_contact(ObjectView):
    contact_force: Tensor  # ['B', None, 3], float, N, environment, 各刚体接触力


class BodyObs_piperx_tcp(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class BodyObs_piperx(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    tcp: BodyObs_piperx_tcp


class BodyObs_robot(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：无
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向


class BodyObs(ObjectView):
    body: BodyObs_body
    command: BodyObs_command
    feet_contact: BodyObs_feet_contact
    piperx: BodyObs_piperx
    robot: BodyObs_robot


Obs = BodyObs
SCENE_FIELDS = ('robot.position', 'robot.orientation', 'robot.linear_velocity', 'robot.angular_velocity', 'robot.joint_position', 'robot.joint_velocity', 'robot.joint_effort', 'robot.default_joint_position', 'robot.default_joint_velocity', 'robot.joint_limits', 'robot.bounds', 'piperx.position', 'piperx.orientation', 'piperx.linear_velocity', 'piperx.angular_velocity', 'piperx.joint_position', 'piperx.joint_velocity', 'piperx.joint_effort', 'piperx.default_joint_position', 'piperx.default_joint_velocity', 'piperx.joint_limits', 'piperx.bounds', 'base_contact.contact_force_samples', 'base_contact.contact_force', 'feet_contact.contact_force_samples', 'feet_contact.contact_force', 'dex_cube.position', 'dex_cube.orientation', 'dex_cube.linear_velocity', 'dex_cube.angular_velocity', 'dex_cube.bounds', 'camera.rgb', 'terrain_rough_40m.position', 'terrain_rough_40m.orientation', 'terrain_rough_40m.bounds', 'piperx.tcp.position', 'piperx.tcp.orientation', 'simulation.gravity', 'ee_workspace.position', 'ee_workspace.orientation', 'ee_workspace.size', 'command.velocity', 'command.ee_target', 'body.output', 'body.previous_output', 'simulation.step_dt', 'simulation.time', 'command.input.piperx.position', 'command.input.ee_workspace.position', 'command.input.ee_workspace.size', 'command.input.ee_workspace.orientation', 'body.input.robot.linear_velocity', 'body.input.robot.angular_velocity', 'body.input.robot.orientation', 'body.input.robot.joint_position', 'body.input.robot.joint_velocity', 'body.input.command.velocity', 'body.input.command.ee_target', 'body.input.body.output', 'body.input.piperx.orientation', 'body.input.feet_contact.contact_force', 'body.input.piperx.position', 'body.input.piperx.joint_position', 'body.input.piperx.joint_velocity', 'body.input.piperx.tcp.position')


# command 的 forward 返回字典（velocity、ee_target），下一层读 obs.command.<名字>
Action = Tensor  # 被训练层 body 的 forward 返回：动作 ['B', 动作个数]；个数就是模型最后一层的大小，构造参数写 actions 时得到受控关节总数（默认 decode 的情形）
# initial_action(obs) 返回同形状的 Action（复位后第一步之前的动作，缺省为 0）


class Targets(TypedDict):  # decode_action(action, obs: BodyObs) 返回；实际单位，每台受控机器人一项
    robot: Tensor  # ['B', 关节数], joint_position, robot 的全部关节，顺序同 robot.joint_position
    piperx: Tensor  # ['B', 关节数], joint_position, piperx 的全部关节，顺序同 piperx.joint_position


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
