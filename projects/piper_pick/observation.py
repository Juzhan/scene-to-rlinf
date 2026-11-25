# Generated from scene.yaml; edit the scene and run check to refresh.
from typing import TypedDict

from torch import Tensor
from shared.assembly.objects import ObjectView
from shared.assembly.robot import RobotView

class SceneInfo_cube(ObjectView):
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    bounds: Tensor  # ['B', 2, 3], float, m, cube/root, 包围盒（本体坐标的最小角, 最大角）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_cube_area(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 朝向
    position: Tensor  # ['B', 3], float, m, environment, 位置
    size: Tensor  # ['B', 2], float, m, cube_area, 尺寸（全长）


class SceneInfo_piper_tcp(ObjectView):
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 坐标系朝向（tcp）
    position: Tensor  # ['B', 3], float, m, environment, 坐标系位置（tcp）


class SceneInfo_piper(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    angular_velocity: Tensor  # ['B', 3], float, rad/s, environment, 根部角速度
    bounds: Tensor  # ['B', 2, 3], float, m, piper/root, 包围盒（本体坐标的最小角, 最大角）
    default_joint_position: Tensor  # ['B', None], float, 默认关节位置
    default_joint_velocity: Tensor  # ['B', None], float, 默认关节速度
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_limits: Tensor  # ['B', None, 2], float, 关节范围（下限, 上限）
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）
    linear_velocity: Tensor  # ['B', 3], float, m/s, environment, 根部线速度
    orientation: Tensor  # ['B', 4], float, 1, environment, wxyz, 根部朝向
    position: Tensor  # ['B', 3], float, m, environment, 根部位置
    tcp: SceneInfo_piper_tcp


class SceneInfo_policy_input_cube(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class SceneInfo_policy_input_piper(ObjectView):
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）


class SceneInfo_policy_input_tcp_frame(ObjectView):
    frame_positions: Tensor  # ['B', None, 3], float, m, environment, 测量坐标系位置


class SceneInfo_policy_input(ObjectView):
    cube: SceneInfo_policy_input_cube
    piper: SceneInfo_policy_input_piper
    tcp_frame: SceneInfo_policy_input_tcp_frame


class SceneInfo_policy(ObjectView):
    input: SceneInfo_policy_input
    output: Tensor  # ['B', None], float, 本步输出（动作）
    previous_output: Tensor  # ['B', None], float, 上一步输出（动作）


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
    piper: SceneInfo_piper
    policy: SceneInfo_policy
    simulation: SceneInfo_simulation
    tcp_frame: SceneInfo_tcp_frame
    wrist_camera: SceneInfo_wrist_camera


class PolicyObs_cube(ObjectView):
    position: Tensor  # ['B', 3], float, m, environment, 根部位置


class PolicyObs_piper(RobotView):
    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；
    # frame 取本体 body 名或已命名坐标系：tcp
    joint_effort: Tensor  # ['B', None], float, 关节实际力 / 力矩
    joint_position: Tensor  # ['B', None], float, 关节位置（绝对值）
    joint_velocity: Tensor  # ['B', None], float, 关节速度（绝对值）


class PolicyObs_tcp_frame(ObjectView):
    frame_positions: Tensor  # ['B', None, 3], float, m, environment, 测量坐标系位置


class PolicyObs(ObjectView):
    cube: PolicyObs_cube
    piper: PolicyObs_piper
    tcp_frame: PolicyObs_tcp_frame


Obs = PolicyObs
SCENE_FIELDS = ('piper.position', 'piper.orientation', 'piper.linear_velocity', 'piper.angular_velocity', 'piper.joint_position', 'piper.joint_velocity', 'piper.joint_effort', 'piper.default_joint_position', 'piper.default_joint_velocity', 'piper.joint_limits', 'piper.bounds', 'cube.position', 'cube.orientation', 'cube.linear_velocity', 'cube.angular_velocity', 'cube.bounds', 'wrist_camera.rgb', 'tcp_frame.frame_positions', 'piper.tcp.position', 'piper.tcp.orientation', 'simulation.gravity', 'cube_area.position', 'cube_area.orientation', 'cube_area.size', 'policy.output', 'policy.previous_output', 'simulation.step_dt', 'simulation.time', 'policy.input.piper.joint_position', 'policy.input.piper.joint_velocity', 'policy.input.piper.joint_effort', 'policy.input.cube.position', 'policy.input.tcp_frame.frame_positions')


Action = Tensor  # 被训练层 policy 的 forward 返回：动作 ['B', 动作个数]；个数就是模型最后一层的大小，构造参数写 actions 时得到受控关节总数（默认 decode 的情形）
# initial_action(obs) 返回同形状的 Action（复位后第一步之前的动作，缺省为 0）


class Targets(TypedDict):  # decode_action(action, obs: PolicyObs) 返回；实际单位，每台受控机器人一项
    piper: Tensor  # ['B', 关节数], joint_position, piper 的全部关节，顺序同 piper.joint_position


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
