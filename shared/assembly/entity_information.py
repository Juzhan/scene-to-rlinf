"""Public entity information names and types; no simulator SDK dependencies."""
import re
from shared.assembly.information import Information, Field, pose_information

# signal: label, native function, capability, units, dimension
SIGNALS = {
    'position': ('根部位置', 'root_pos_w', 'root_state', 'm · 环境坐标', 3),
    'orientation': ('根部朝向', 'root_quat_w', 'root_state', 'wxyz 四元数', 4),
    'linear_velocity': ('根部线速度', 'root_lin_vel_w', 'root_state', 'm/s · 世界轴', 3),
    'angular_velocity': ('根部角速度', 'root_ang_vel_w', 'root_state', 'rad/s · 世界轴', 3),
    'joint_position': ('关节位置（绝对值）', 'joint_pos', 'joint_state', 'rad / m', None),
    'joint_velocity': ('关节速度（绝对值）', 'joint_vel', 'joint_state', 'rad/s / m/s', None),
    'joint_effort': ('关节实际力 / 力矩', 'joint_effort', 'joint_state', 'N / N·m', None),
    'default_joint_position': ('默认关节位置', 'default_joint_position', 'joint_state', 'rad / m', None),
    'default_joint_velocity': ('默认关节速度', 'default_joint_velocity', 'joint_state', 'rad/s / m/s', None),
    'ray_hit_positions': ('射线命中位置', 'ray_hit_positions', 'height_scan', 'm', None),
    'frame_positions': ('测量坐标系位置', 'frame_positions', 'frame_transform', 'm', None),
    'contact_force_samples': ('接触传感器采样窗口', 'contact_force_samples', 'contact_forces', 'N', None),
    'contact_force': ('各刚体接触力', 'contact_force', 'contact_forces', 'N · 世界轴', None),
    'rgb': ('RGB 图像（独立图像通道）', 'image', 'image', 'uint8 · H×W×3', None),
    'distance_to_image_plane': ('深度（到成像平面距离）', 'image', 'image', 'm · H×W×1', None),
    'distance_to_camera': ('深度（到相机光心距离）', 'image', 'image', 'm · H×W×1', None),
    'normals': ('表面法向', 'image', 'image', '单位向量 · 世界轴 · H×W×3', None),
}


def signal_information(name):
    if name == "simulation.gravity":
        return Information("vector", (3,), "float", "m/s^2", "environment")
    key, signal = parse_signal(name)
    # Cloned environment frames differ by translation, not orientation. Root
    # velocity world axes are consequently also environment-frame axes.
    if signal == "position": return Information("position", (3,), "float", "m", "environment")
    if signal in ("ray_hit_positions", "frame_positions"):
        return Information("number", (None, 3), "float", "m", "environment", axes=("ray" if signal == "ray_hit_positions" else "frame", "xyz"))
    if signal == "contact_force_samples":
        return Information("number", (None, None, 3), "float", "N", "environment", axes=("sample", "body", "xyz"))
    if signal == "orientation": return Information("rotation", (4,), "float", "1", "environment", key + "/root", "wxyz")
    if signal in ("linear_velocity", "angular_velocity"):
        return Information("vector", (3,), "float", "m/s" if signal == "linear_velocity" else "rad/s", "environment", axes=("xyz",))
    if signal in ("joint_position", "joint_velocity", "joint_effort", "default_joint_position", "default_joint_velocity"):
        # Revolute/prismatic collections can have mixed units. Do not label
        # arbitrary articulated assets as all-radian or all-torque.
        return Information("number", (None,), "float", axes=("joint",))
    if signal == "contact_force":
        return Information("number", (None, 3), "float", "N", "environment", axes=("body", "xyz"))
    axes = ("height", "width", "channel")
    if signal.startswith("distance_to_"):
        return Information("image", (None, None, 1), "float", "m", encoding="depth", axes=axes)
    if signal == "normals":
        return Information("image", (None, None, 3), "float", "1", "environment", encoding="normals", axes=axes)
    return Information("image", (None, None, 3), "uint8", encoding="RGB", axes=axes)


def command_information(config):
    """Metadata from native command configuration, not project command names."""
    cls = type(config).__name__ if not isinstance(config, dict) else config.get("type", "")
    get = (lambda key: config.get(key)) if isinstance(config, dict) else (lambda key: getattr(config, key, None))
    asset = get("asset_name")
    frame = asset + "/root" if asset else None
    if cls in ("UniformPoseCommandCfg", "UniformPose2dCommandCfg"):
        if cls == "UniformPoseCommandCfg": return pose_information(frame)
    if cls in ("UniformVelocityCommandCfg", "NormalVelocityCommandCfg"):
        return Information("record", (3,), "float", fields={
            "linear_velocity": Field((0, 1), Information("vector", (2,), "float", "m/s", frame)),
            "angular_velocity": Field((2,), Information("number", (), "float", "rad/s", frame)),
        })
    return Information()


def signal_profile(name):
    parsed = parse_signal(name)
    return 'camera__' + parsed[0] if parsed and SIGNALS[parsed[1]][2] == 'image' else 'states'



def check_image_sources(sources, outputs):
    """Selected images must be rendered by their camera; ``outputs`` maps entity key to its data types."""
    for source in sources:
        parsed = parse_signal(source)
        if parsed and SIGNALS[parsed[1]][2] == 'image' and parsed[0] in outputs and parsed[1] not in outputs[parsed[0]]:
            raise ValueError(f'{source!r} is not rendered: camera outputs {list(outputs[parsed[0]])}; '
                             f'add {parsed[1]!r} to its data_types or deselect the observation')


def signal_id(key, signal):
    return key.replace('__', '.') + '.' + signal


def parse_signal(name):
    if '.' in name and not name.startswith('entity__'):
        key, signal = name.rsplit('.', 1)
        if '.target.' in name:
            return None
        key = key.replace('.', '__')
        if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key) and signal in SIGNALS:
            return key, signal
        return None
    if not name.startswith('entity__') or '__' not in name[8:]:
        return None
    key, signal = name[8:].rsplit('__', 1)
    if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key) and signal in SIGNALS:
        return key, signal
    return None


