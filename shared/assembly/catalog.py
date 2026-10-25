"""Scene-aware authoring descriptions shared by CLI, Blender and agents.

Static inspection never constructs SDK assets or imports Isaac. Entries are
configuration candidates, not a guarantee that a native articulation supports
every control mode. Exact joint names, limits and tensor shapes are resolved
by the native environment when it is inspected. Project ``ASSET_METADATA`` and
``TERM_METADATA`` optionally add explanations without duplicating SDK behavior.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from importlib import import_module
import inspect
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from .layout import load_scene_config, world_spec_from_config
from shared.backends.isaac.assets import FACTORIES
from .assets import asset_backends, image_outputs
from shared.backends.isaac.scene import scene_metadata
from shared.backends.isaac.task_config import _ACTION_CLASSES, _OBSERVATION_NAMES, _TASK_NAMES, _selection


_ASSETS = {
    "piper": ("Piper 六轴机械臂", "本地 Piper USD，六轴臂与双指夹爪；保留已有驱动与预备姿态。", ["articulation", "root_state", "joint_state"]),
    "piper_x": ("Piper X 机械臂", "本地 Piper X，含原模型夹爪与 D435 外观；图像采集需另配传感器。", ["articulation", "root_state", "joint_state"]),
    "m20": ("M20 轮足机器人", "本地 M20，12 个腿关节与 4 个轮关节；轮子沿用速度驱动参数。", ["articulation", "root_state", "joint_state"]),
    "go2": ("Go2 四足机器人", "原生 Unitree Go2 关节与驱动配置。", ["articulation", "root_state", "joint_state"]),
    "anymal_c": ("ANYmal-C 四足机器人", "原生 ANYbotics ANYmal-C，12 个腿关节与 ANYdrive 执行器配置。", ["articulation", "root_state", "joint_state"]),
    "franka_panda": ("Franka Panda 机械臂", "原生 Panda 机械臂和双指夹爪。", ["articulation", "root_state", "joint_state"]),
    "dex_cube": ("可抓取方块", "原生 DexCube 刚体；尺寸由资产与 scale 决定。", ["rigid_body", "root_state"]),
    "ground_plane": ("平面地面", "各并行环境共享的物理地面。", ["geometry"]),
    "box": ("长方体", "完整边长（米）；质量为空时是静态设施。", ["geometry"]),
    "ball": ("球", "自由刚体球；半径、质量、颜色与摩擦可调。", ["rigid_body", "root_state"]),
    "urdf_robot": ("导入的 URDF 机器人", "工作区内的 URDF，首次使用时转换为 USD（缓存于 .cache/urdf_usd）；所有关节共用一组隐式 PD 增益，请按机器人调整。", ["articulation", "root_state", "joint_state"]),
    "usd_robot": ("导入的 USD 机器人", "工作区内带关节的 USD；未填增益时沿用 USD 中的关节驱动。", ["articulation", "root_state", "joint_state"]),
    "usd_object": ("导入的 USD 物体", "质量为空时是静态设施；填写质量时 USD 内需已有刚体与碰撞属性。`run.py asset terrain` 生成的双格式地形地块也用它。", ["geometry"]),
    "seattle_table": ("工作台", "SeattleLab 工作台资产。", ["geometry"]),
    "terrain": ("程序化地形", "仅 Isaac，只为已有项目保留，不在资产架提供：复用 Isaac Lab 地形生成器按难度课程排布子地形。新场景用 `run.py asset terrain` 生成双后端地块，再以 usd_object 引用。", ["terrain", "geometry"]),
    "height_scanner": ("地形高度扫描器", "从机器人附近向地形投射射线。", ["height_scan"]),
    "contact_sensor": ("接触力传感器", "测量所选刚体接触力，可记录足端离地时间。", ["contact_forces"]),
    "camera": ("相机", "针孔相机，可输出 RGB、深度与法向，各输出同一分辨率；可安装在机器人或场景中，观察图像时自动启用相机。", ["image"]),
    "frame_transformer": ("坐标测量器", "读取同一资产上目标刚体坐标系的位姿。", ["frame_transform"]),
    "dome_light": ("环境灯", "统一环境照明。", ["lighting"]),
    "clear_sky": ("天空照明", "使用原生晴空环境贴图。", ["lighting"]),
}
# Still loadable for existing scenes, but not offered when adding entities.
_NOT_ON_SHELF = {"terrain"}
_OBSERVATIONS = {
    "base_lin_vel": ("机身线速度", "机器人自身坐标系中的三轴线速度。", "base_lin_vel", "m/s", "robot_root", ["robot"], 3),
    "base_ang_vel": ("机身角速度", "机器人自身坐标系中的三轴角速度。", "base_ang_vel", "rad/s", "robot_root", ["robot"], 3),
    "projected_gravity": ("机身中的重力方向", "世界重力的单位方向在机身坐标系中的投影，可反映倾斜。", "projected_gravity", "1", "robot_root", ["robot"], 3),
    "velocity_commands": ("期望行走速度", "任务采样的前后、左右速度和转向角速度；不是实测速度。", "generated_commands", "m/s,m/s,rad/s", "robot_root", ["command:base_velocity"], 3),
    "joint_pos": ("关节相对默认位置", "关节位置减去资产默认关节位置；旋转关节用弧度，移动关节用米。", "joint_pos_rel", "rad or m", "joint", ["robot"], None),
    "joint_vel": ("关节相对默认速度", "关节速度减去资产默认关节速度。", "joint_vel_rel", "rad/s or m/s", "joint", ["robot"], None),
    "actions": ("最近一次策略动作", "动作管理器保存的原始策略输入；不是关节实测状态，也不是电机执行后的力矩。", "last_action", "policy action units", "action", ["actions"], None),
    "height_scan": ("周围地形高度", "扫描器高度减去射线命中高度及偏移量，再按观察配置裁剪/加噪。", "height_scan", "m", "world_vertical", ["height_scanner"], None),
    "object_position": ("物体相对机器人位置", "目标物体根位置在机器人根坐标系中的三维坐标。", "object_position_in_robot_root_frame", "m", "robot_root", ["robot", "object"], 3),
    "target_object_position": ("期望物体位姿", "物体目标命令的完整七维位姿：xyz 加 wxyz 四元数，名称虽含 position 但不只有位置。", "generated_commands", "m,quaternion", "robot_root", ["command:object_pose"], 7),
}
_REWARDS = {
    "track_lin_vel_xy_exp": ("跟随水平速度", "速度误差越小奖励越高。", "track_lin_vel_xy_exp", "exp(-sum((command_xy - base_velocity_xy)^2) / std^2)", ["robot", "command:base_velocity"]),
    "track_ang_vel_z_exp": ("跟随转向速度", "绕竖直轴的角速度误差越小奖励越高。", "track_ang_vel_z_exp", "exp(-(command_yaw_rate - base_yaw_rate)^2 / std^2)", ["robot", "command:base_velocity"]),
    "lin_vel_z_l2": ("减少上下跳动", "机身竖直速度平方；通常配负权重。", "lin_vel_z_l2", "base_velocity_z^2", ["robot"]),
    "ang_vel_xy_l2": ("减少机身摇晃", "机身横滚、俯仰角速度平方和；通常配负权重。", "ang_vel_xy_l2", "sum(base_angular_velocity_xy^2)", ["robot"]),
    "dof_torques_l2": ("减少关节力矩", "选中关节实际施加的力矩平方和；不是能量或功率。", "joint_torques_l2", "sum(applied_torque[selected_joints]^2)", ["robot"]),
    "dof_acc_l2": ("减少关节加速度", "选中关节加速度平方和；通常配负权重。", "joint_acc_l2", "sum(joint_acceleration[selected_joints]^2)", ["robot"]),
    "action_rate_l2": ("平滑相邻动作", "相邻两次原始策略动作差值的平方和；通常配负权重。", "action_rate_l2", "sum((action - previous_action)^2)", ["actions"]),
    "feet_air_time": ("足端离地时间", "首次触地时按离地时长与阈值差计算，静止命令时抑制该项。", "feet_air_time", "sum((last_air_time - threshold) * first_contact) * moving_command", ["contact_forces", "command:base_velocity"]),
    "undesired_contacts": ("减少不期望接触", "接触力超过阈值的选中刚体数量；通常配负权重。", "undesired_contacts", "count(selected_body_contact_force > threshold)", ["contact_forces"]),
    "flat_orientation_l2": ("保持机身水平", "重力投影的水平分量平方和；通常配负权重。", "flat_orientation_l2", "sum(projected_gravity_xy^2)", ["robot"]),
    "dof_pos_limits": ("避免关节越界", "超出软关节位置范围的距离之和；通常配负权重。", "joint_pos_limits", "sum(distance_outside_soft_joint_limits)", ["robot"]),
    "reaching_object": ("末端靠近物体", "测量末端坐标系到物体的距离。", "object_ee_distance", "1 - tanh(distance(end_effector, object) / std)", ["object", "ee_frame"]),
    "lifting_object": ("物体抬高", "物体世界坐标高度超过阈值时为 1，否则为 0；不是完整抓取成功判定。", "object_is_lifted", "object_world_z > minimal_height", ["object"]),
    "object_goal_tracking": ("物体接近目标", "物体抬高后，按其到目标位置的距离给奖励。", "object_goal_distance", "lifted * (1 - tanh(distance(object, goal) / std))", ["robot", "object", "command:object_pose"]),
    "object_goal_tracking_fine_grained": ("物体精细对准目标", "与目标距离奖励相同，但原生预设使用更小的距离尺度。", "object_goal_distance", "lifted * (1 - tanh(distance(object, goal) / std))", ["robot", "object", "command:object_pose"]),
    "action_rate": ("平滑相邻动作", "相邻两次原始策略动作差值的平方和；通常配负权重。", "action_rate_l2", "sum((action - previous_action)^2)", ["actions"]),
    "joint_vel": ("减少关节速度", "选中关节速度平方和；通常配负权重。", "joint_vel_l2", "sum(joint_velocity[selected_joints]^2)", ["robot"]),
}
_ACTION_LABELS = {
    "joint_position": ("关节目标位置", "策略输入经缩放和偏移后成为关节位置目标，具体跟踪由原生驱动器负责。"),
    "relative_joint_position": ("关节位置增量", "每次输入相对当前关节位置的增量，非相对上一动作。"),
    "joint_velocity": ("关节目标速度", "策略输入经缩放和偏移后成为关节速度目标；需匹配驱动器设置。"),
    "joint_effort": ("关节力/力矩", "策略输入经缩放和偏移后成为关节力或力矩目标；需匹配驱动器设置。"),
    "binary_joint_position": ("开合两态关节", "单个输入通过原生阈值逻辑选择一组打开/关闭关节位置，常用于夹爪。"),
}


def _parameter(label, kind, **options):
    return {"label": label, "type": kind, **options}


CATEGORIES = {
    "robot": "机器人", "object": "物体 / 设备", "terrain": "地面 / 地形",
    "sensor": "传感器", "light": "照明", "group": "组合",
}


def entity_category(capabilities, kind="entity", factory=None):
    """Presentation categories describe capabilities, not mandatory YAML roles."""
    if "articulation" in capabilities or kind == "robot":
        return "robot"
    if kind == "sensor" or set(capabilities) & {"image", "height_scan", "contact_forces", "frame_transform"}:
        return "sensor"
    if "terrain" in capabilities or factory == "ground_plane":
        return "terrain"
    if "lighting" in capabilities:
        return "light"
    return "object" if factory else "group"


def asset_capabilities(factory, capabilities, parameters=None):
    """Parameter-dependent capabilities a factory declares only by its arguments.

    ``box`` is a static facility without a mass and a rigid root body with one, which the
    backends already read as ``position``/``linear_velocity``; the metadata must follow or
    such a body cannot be selected as an observation.
    """
    parameters = parameters or {}
    if factory == "box" and parameters.get("mass") is not None and "rigid_body" not in capabilities:
        return [*capabilities, "rigid_body", "root_state"]
    return capabilities


# Display names for annotated factory parameters; the YAML keys stay the factory argument names.
_PARAMETER_LABELS = {
    "camera": {"width": "宽度（像素）", "height": "高度（像素）", "data_types": "输出图像",
               "focal_length": "焦距（mm，水平孔径 20.955 mm）", "clipping_range": "近 / 远裁剪面（m）",
               "depth_clipping": "超出远裁剪面的深度"},
    "box": {"size": "边长 xyz（m）", "mass": "质量（kg，空为静态）", "color": "颜色 RGB（0–1）", "friction": "摩擦系数"},
    "ball": {"radius": "半径（m）", "mass": "质量（kg）", "color": "颜色 RGB（0–1）", "friction": "摩擦系数"},
}
_OPTION_LABELS = {
    "rgb": "RGB 彩色", "distance_to_image_plane": "深度 · 到成像平面", "distance_to_camera": "深度 · 到光心",
    "normals": "表面法向", "max": "取远裁剪面距离", "zero": "置 0", "none": "保留 inf",
}


def _parameter_type(annotation, default):
    """Editor field type from the annotation (else the default); unknown structures stay JSON values."""
    options = {}
    if get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation):
        remaining = [arg for arg in get_args(annotation) if arg is not type(None)]
        options["nullable"] = True
        annotation = remaining[0] if len(remaining) == 1 else inspect.Parameter.empty
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        return {"type": "choice", "options": list(args), **options}
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis and get_origin(args[0]) is Literal:
        return {"type": "choices", "options": list(get_args(args[0])), **options}
    if origin is tuple and args and all(arg in (int, float) for arg in args):
        return {"type": "vector", "length": len(args), **options}
    if annotation is inspect.Parameter.empty and default is not inspect.Parameter.empty and default is not None:
        annotation = type(default)
        if annotation is tuple and default and all(type(v) in (int, float) for v in default):
            return {"type": "vector", "length": len(default), **options}
    kinds = {bool: "boolean", int: "integer", float: "number", str: "text"}
    return {"type": kinds.get(annotation, "value"), **options}


def _factory_parameters(factory, name=None):
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return {}
    try:
        hints = get_type_hints(factory)
    except Exception:  # noqa: BLE001 - unresolved annotations only lose the editor field type
        hints = {}
    labels, result = _PARAMETER_LABELS.get(name, {}), {}
    for key, parameter in parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        field = _parameter_type(hints.get(key, inspect.Parameter.empty), parameter.default)
        entry = _parameter(labels.get(key, key), field.pop("type"), required=parameter.default is parameter.empty, **field)
        if "options" in entry:
            entry["option_labels"] = {option: _OPTION_LABELS.get(option, option) for option in entry["options"]}
        if parameter.default is not parameter.empty and isinstance(parameter.default, (str, int, float, bool, tuple, list, dict, type(None))):
            entry["default"] = deepcopy(parameter.default)
        result[key] = entry
    return result


def _parameter_values(entity, assets):
    """Authored parameters over factory defaults."""
    factory = assets.get(entity["factory"])
    defaults = {} if factory is None else {
        name: entry.get("default") for name, entry in _factory_parameters(factory).items()}
    return {**defaults, **entity["parameters"]}


def _image_size(info, entity, assets):
    """Image height/width come from the camera's width/height parameters (authored or factory default)."""
    if info.kind != "image" or entity["factory"] not in assets:
        return info
    values = _parameter_values(entity, assets)
    height, width = values.get("height"), values.get("width")
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        return info
    return replace(info, shape=(height, width, *info.shape[2:]))


def _project_package(directory):
    parts, current = [], Path(directory).resolve()
    while (current / "__init__.py").exists():
        if not current.name.isidentifier():
            return None
        parts.append(current.name)
        current = current.parent
    package = '.'.join(reversed(parts))
    if not package:
        return None
    from importlib.util import find_spec
    try:
        return package if find_spec(package) is not None else None
    except (ImportError, ModuleNotFoundError, ValueError):
        return None


def describe_scene(source, *, directory=None, factories=None, term_registry=None,
                   asset_metadata=None, term_metadata=None):
    """Return a JSON-safe candidate catalog, retaining stable YAML identifiers.

``source`` is a scene path or parsed mapping. In-memory scenes must provide
their source ``directory`` so assembly references resolve correctly. Lazy
project registries and metadata are discovered for ordinary Python packages.
"""
    source_path = None
    if isinstance(source, Mapping):
        if directory is None:
            raise ValueError("An in-memory scene catalog requires directory")
        settings = deepcopy(dict(source))
    else:
        source = Path(source).resolve()
        source_path = source
        directory, settings = source.parent, load_scene_config(source)
    package = _project_package(directory)
    assets, terms = dict(FACTORIES), {}
    asset_docs, term_docs = {}, {}
    if package:
        from .project import project_registry
        assets, terms, _ = project_registry(package)
        from importlib.util import find_spec
        for suffix, target, attribute in (("assets", asset_docs, "ASSET_METADATA"), ("terms", term_docs, "TERM_METADATA")):
            name = package + "." + suffix
            if find_spec(name) is not None:
                target.update(getattr(import_module(name), attribute, {}))
    assets.update(factories or {})
    for category, entries in (term_registry or {}).items():
        terms[category] = {**terms.get(category, {}), **entries}
    asset_docs.update(asset_metadata or {})
    for category, entries in (term_metadata or {}).items():
        term_docs[category] = {**term_docs.get(category, {}), **entries}
    world = world_spec_from_config(settings, directory=directory, cameras=True, factories=assets)
    metadata, index = scene_metadata(world), world.index()
    entities = []
    for path, (node, pose) in index.items():
        if not node.factory:
            continue
        builtin = _ASSETS.get(node.factory, (node.factory, "项目自定义资产。", []))
        docs = asset_docs.get(node.factory, {})
        capabilities = asset_capabilities(node.factory, list(docs.get("capabilities", builtin[2])), node.parameters)
        category = entity_category(capabilities, node.kind, node.factory)
        entities.append({"path": path, "native_key": metadata.keys[path], "factory": node.factory,
                         "supported_backends": asset_backends(node.factory),
                         "category": category, "category_label": CATEGORIES[category],
                         "kind": node.kind, "label": node.name, "capabilities": capabilities,
                         "local_pose": asdict(node.pose), "layout_pose": asdict(pose),
                         "parameters": deepcopy(node.parameters), "scope": metadata.scopes[path],
                         "pose_authored": node.pose_authored})
    by_key = {entity["native_key"]: entity for entity in entities}
    robots = [entity for entity in entities if "articulation" in entity["capabilities"] or entity["kind"] == "robot"]
    task, preset = settings.get("task", {}), settings.get("task", {}).get("preset")
    command_names = set(_TASK_NAMES.get(preset, {}).get("commands", ()))
    if task.get("commands", {}) is None:
        command_names.clear()
    else:
        for name, value in task.get("commands", {}).items():
            (command_names.discard if value is None else command_names.add)(name)

    def availability(requires):
        missing = [name for name in requires if
                   (name == "actions" and not settings.get("actions")) or
                   (name.startswith("command:") and name.split(":", 1)[1] not in command_names) or
                   (name != "actions" and not name.startswith("command:") and name not in by_key)]
        return {"available": not missing, "availability": "missing_dependency" if missing else "candidate",
                "missing": missing, "runtime_required": True}

    selected = {}
    for group, value in settings.get("observations", {}).items():
        names, _, _ = _selection(value, f"observations.{group}")
        for name in names:
            selected.setdefault(name, []).append(group)
    # 编辑中允许暂时为空（例如刚删掉唯一被观察的实体）；check 与启动时再要求至少一项。
    for role, component in settings.get('control', {}).get('components', {}).items():
        for source in component.get('observations') or []:
            selected.setdefault(source, []).append(role)
    observations = []
    for name in _OBSERVATION_NAMES.get(preset, ()):
        label, description, function, units, frame, requires, dimension = _OBSERVATIONS[name]
        implementation = "isaaclab.envs.mdp." + function
        if name == "object_position":
            implementation = "isaaclab_tasks.manager_based.manipulation.lift.mdp." + function
        observations.append({"id": name, "label": label, "description": description, "implementation": implementation,
                             "units": units, "frame": frame, "dimension": dimension, "requires": requires,
                             "parameters": {"params": _parameter("原生项参数与实体绑定", "mapping"),
                                            "scale": _parameter("输入缩放", "number"),
                                            "clip": _parameter("裁剪范围", "range")},
                             "selected_in": selected.get(name, []), **availability(requires)})
    actions = []
    for name, class_name in _ACTION_CLASSES.items():
        label, description = _ACTION_LABELS[name]
        parameters = {
            "asset_name": _parameter("机器人", "entity", required=True, choices=[e["native_key"] for e in robots]),
            "joint_names": _parameter("关节选择", "string_list", default=[".*"], runtime_required=True),
        }
        if name == "binary_joint_position":
            parameters.update({
                "open_command_expr": _parameter("打开时各关节目标位置", "mapping", required=True),
                "close_command_expr": _parameter("关闭时各关节目标位置", "mapping", required=True)})
        else:
            parameters["scale"] = _parameter("动作缩放", "number_or_mapping", default=1.)
            if name in ("joint_position", "joint_velocity"):
                parameters["use_default_offset"] = _parameter("以默认关节状态为零点", "boolean", default=True)
            elif name == "relative_joint_position":
                parameters["use_zero_offset"] = _parameter("使用零偏移", "boolean", default=True)
            else:
                parameters["offset"] = _parameter("目标偏移", "number_or_mapping", default=0.)
        configured = [key for key, value in settings.get("actions", {}).items() if value.get("type") == name]
        actions.append({"id": name, "label": label, "description": description, "requires": ["articulation"],
                        "implementation": "isaaclab.envs.mdp." + class_name, "parameters": parameters,
                        "available": bool(robots), "availability": "candidate" if robots else "missing_dependency",
                        "runtime_required": True, "configured": configured,
                        "targets": [{key: entity[key] for key in ("path", "native_key", "factory")} for entity in robots]})
    rewards = []
    updates = task.get("rewards", {})
    for name in sorted(_TASK_NAMES.get(preset, {}).get("rewards", ())):
        label, description, function, formula, requires = _REWARDS[name]
        implementation = "isaaclab.envs.mdp." + function
        if function.startswith("object_"):
            implementation = "isaaclab_tasks.manager_based.manipulation.lift.mdp." + function
        elif function == "feet_air_time":
            implementation = "isaaclab_tasks.manager_based.locomotion.velocity.mdp." + function
        current = updates.get(name, {}) if updates is not None else None
        rewards.append({"id": name, "label": label, "description": description, "formula": formula,
                        "implementation": implementation, "requires": requires,
                        "parameters": {"weight": _parameter("奖励权重", "number"),
                                       "params": _parameter("原生项参数与实体绑定", "mapping")},
                        "enabled": current is not None, "overrides": deepcopy(current),
                        "contribution": "raw_term * weight * environment_step_seconds",
                        **availability(requires)})
    collections = {"observations": observations, "actions": actions, "rewards": rewards}
    for category, collection in collections.items():
        known = {entry["id"] for entry in collection}
        for name, entry in terms.get(category, {}).items():
            docs = term_docs.get(category, {}).get(name, {})
            description = inspect.getdoc(entry) or "项目自定义组件；参数与能力由项目声明。"
            item = {"id": name, "label": name, "description": description,
                    "implementation": f"{getattr(entry, '__module__', 'project')}.{getattr(entry, '__qualname__', name)}",
                    "parameters": {}, "requires": [], "available": True, "availability": "custom_unverified",
                    "runtime_required": True, **deepcopy(docs)}
            if name in known:
                collection[:] = [item if old["id"] == name else old for old in collection]
            else:
                collection.append(item)
        for item in collection:
            item.update(deepcopy(term_docs.get(category, {}).get(item["id"], {})))
    asset_entries = []
    for name, factory in assets.items():
        label, description, capabilities = _ASSETS.get(name, (name, inspect.getdoc(factory) or "项目自定义资产。", []))
        capabilities = asset_capabilities(name, capabilities)
        asset_entries.append({"id": name, "label": label, "description": description, "capabilities": capabilities,
                              "supported_backends": asset_backends(name),
                              "implementation": f"{getattr(factory, '__module__', 'project')}.{getattr(factory, '__qualname__', name)}",
                              "parameters": _factory_parameters(factory, name), "available": True,
                              "on_shelf": name not in _NOT_ON_SHELF,
                              **deepcopy(asset_docs.get(name, {}))})
        entry = asset_entries[-1]
        entry["category"] = entity_category(entry["capabilities"], factory=name)
        entry["category_label"] = CATEGORIES[entry["category"]]

    def term_sources(item, override):
        requires = list(item.get("requires", []))
        # Resolve explicit native bindings instead of attributing every joint
        # observation to the preset's default robot. Keep cross-entity terms.
        params = override.get("params", {})
        parsed = parse_signal(item['id'])
        aliases = {"asset_cfg": parsed[0] if parsed else "robot", "robot_cfg": "robot", "object_cfg": "object",
                   "ee_frame_cfg": "ee_frame", "sensor_cfg": next(
                       (key for key in requires if key in by_key and by_key[key]["category"] == "sensor"),
                       "height_scanner" if item["id"] == "height_scan" else "contact_forces")}
        for field, default in aliases.items():
            binding = params.get(field)
            if isinstance(binding, Mapping) and "name" in binding:
                requires = [binding["name"] if key == default else key for key in requires]
                if binding["name"] not in requires:
                    requires.append(binding["name"])
        if "command_name" in params:
            requires = ["command:" + params["command_name"] if key.startswith("command:") else key for key in requires]
        sources = [{"path": by_key[key]["path"], "label": by_key[key]["label"],
                    "category": by_key[key]["category"]} for key in dict.fromkeys(requires) if key in by_key]
        group = (sources[0]["path"] if len(sources) == 1 else "relations" if sources else
                 "commands" if any(key.startswith("command:") for key in requires) else
                 "actions" if "actions" in requires else "custom")
        label = (sources[0]["label"] if len(sources) == 1 else
                 {"relations": "实体间关系", "commands": "任务指令", "actions": "策略动作历史", "custom": "自定义 / 未声明来源"}[group])
        return {"sources": sources, "group": group, "group_label": label, **availability(requires)}

    # Preset terms keep their behavior/order. Entity candidates start unchecked.
    from shared.assembly.entity_information import SIGNALS, signal_id, signal_profile, parse_signal, signal_information
    for entity in entities:
        images = image_outputs(assets.get(entity["factory"]), entity["parameters"])
        for signal, (label, function, capability, units, dimension) in SIGNALS.items():
            if capability not in entity["capabilities"] and not (signal == "position" and "height_scan" in entity["capabilities"]):
                continue
            if capability == "image" and signal not in images:
                continue  # Only outputs the camera renders (its data_types parameter).
            name = signal_id(entity["native_key"], signal)
            if preset:
                name = f'entity__{entity["native_key"]}__{signal}'
            observations.append({"id": name, "label": label, "description": label + "，来源：" + entity["path"],
                                 "implementation": ("shared.backends.isaac.entity_observations.primitive" if signal in {"position", "default_joint_position", "default_joint_velocity", "ray_hit_positions", "frame_positions", "contact_force", "contact_force_samples"} else "isaaclab.envs.mdp." + function),
                                 "requires": [entity["native_key"]], "units": units, "dimension": dimension,
                                 "information": _image_size(signal_information(name), entity, assets).describe(),
                                 "profile": signal_profile(name),
                                 "selected_in": selected.get(name, []), **availability([entity["native_key"]])})
    if not preset:
        # Named robot frames (asset defaults such as tcp, plus entity frames), computed from body poses.
        from .information import Information
        from .robot import frame_observations, robot_frames
        frames = robot_frames(world)
        for name, (robot, frame, field) in frame_observations(frames).items():
            body, (offset, _) = frames[robot][frame]
            info = (Information("position", (3,), "float", "m", "environment") if field == "position" else
                    Information("rotation", (4,), "float", "1", "environment", robot + "/" + frame, "wxyz"))
            label = {"position": "坐标系位置", "orientation": "坐标系朝向"}[field] + f"（{frame}）"
            observations.append({"id": name, "label": label,
                                 "description": f"{label}：{body or '根部'} 偏移 {list(offset)} m，环境坐标；由本体位姿计算，无需传感器。",
                                 "requires": [robot], "units": "m · 环境坐标" if field == "position" else "wxyz 四元数 · 环境坐标",
                                 "dimension": info.shape[0], "information": info.describe(), "profile": "states",
                                 "selected_in": selected.get(name, []), **availability([robot])})
    gravity_info = signal_information('simulation.gravity')
    observations.append({'id': 'simulation.gravity', 'label': '环境重力加速度', 'description': '仿真配置中的重力向量',
                         'requires': [], 'information': gravity_info.describe(), 'units': gravity_info.unit,
                         'dimension': 3, 'profile': 'states', 'parameters': {},
                         'selected_in': selected.get('simulation.gravity', []), **availability([])})
    from shared.assembly.targets import target_catalog
    from shared.assembly.abstract import LABELS as ABSTRACT_LABELS, abstract_entities, entity_key
    abstracts = abstract_entities(settings)
    for name, (value, info) in target_catalog(settings).items():
        entry = {"id": name, "requires": [], "units": info.unit, "dimension": info.shape[0],
                 "information": info.describe(), "profile": "states", "parameters": {},
                 "selected_in": selected.get(name, [])}
        abstract_name, _, field = name.partition('.')
        if abstract_name in abstracts:
            node = abstracts[abstract_name]
            labels = {'position': ('位置', 'm · 环境坐标'), 'orientation': ('朝向', 'wxyz 四元数 · 环境坐标'),
                      'size': ('尺寸（全长）', 'm'), 'direction': ('方向（本地 +X）', '单位向量 · 环境坐标')}
            state = ('随机放置' if node.reset is not None else '跟随 ' + node.mount.entity
                     if node.mount is not None else '固定')
            entry.update(label=labels[field][0], units=labels[field][1],
                         description=f'抽象{ABSTRACT_LABELS[node.shape]}（{state}），环境坐标；无模型、无物理。',
                         requires=[] if node.mount is None else [entity_key(settings, node.mount.entity)],
                         entity=settings.get('name', 'world') + '/' + abstract_name)
            if not isinstance(value, dict):
                entry['value'] = value
        else:
            entity = name.split('.target.')[0].replace('.', '__')
            entry.update(label='固定期望值（targets）', description='显式设定的固定目标，环境坐标。',
                         requires=[entity], value=value)
        observations.append({**entry, **availability(entry['requires'])})
    from .information import Information
    computed = {}
    if settings.get('variables'):
        from shared.backends.isaac.derived import variable_information
        computed = {name: variable_information(settings, name, terms.get("signals")) for name in settings['variables']}
    from .control import declared_dimension
    for role in settings.get('control', {}).get('components', {}):
        for field in ('input', 'output', 'previous_output'):
            dimension = declared_dimension(settings['control'], role) if field != 'input' else None
            computed[role + '.' + field] = Information('number', (dimension,), 'float')
    interfaces = {'input': '最近基础输入（对象结构）', 'output': '本步输出（解码前动作）',
                  'previous_output': '上一步输出（解码前动作）'}
    for name, info in computed.items():
        observations.append({'id': name, 'label': interfaces.get(name.rsplit('.', 1)[-1], name) if '.' in name else name,
                             'description': '组件接口' if '.' in name else '用户定义的计算信息',
                             'requires': [], 'information': info.describe(), 'units': info.unit,
                             'dimension': info.shape[0] if info.shape else 1, 'profile': 'states', 'parameters': {},
                             'selected_in': selected.get(name, []), 'observation_selectable': name not in settings.get('variables', {}), **availability([])})
    for item in observations:
        item["source_groups"] = {"default": term_sources(item, {})}
        for profile, selection in settings.get("observations", {}).items():
            _, _, overrides = _selection(selection, f"observations.{profile}")
            item["source_groups"][profile] = term_sources(item, overrides.get(item["id"], {}))
    for item in rewards:
        item["source_group"] = term_sources(item, item.get("overrides") or {})
    resolved_interface = None
    if source_path is not None:
        from .interfaces import load_interface
        resolved_interface = load_interface(source_path)
    if resolved_interface:
        measured = resolved_interface.get("articulations", {})
        for action in actions:
            for target in action.get("targets", []):
                if target["native_key"] in measured:
                    target["joint_names"] = list(measured[target["native_key"]].get("joint_names", []))
                    target["joint_names_measured"] = True
    return {"schema_version": 1, "preset": preset, "assets": asset_entries, "entities": entities,
            "reward_formula": task.get('reward'),
            "resolved_interface": resolved_interface, "control": settings.get("control"),
            **collections,
            "limitations": ["目录只列当前已实现组件；关节名称、维度、驱动能力需运行检查。",
                            "观察选择顺序就是策略输入顺序。",
                            "抽象矩形/长方体只定义几何范围；被 reset.region 引用后才参与复位，不自动避碰或投影到地面。"]}
