"""Selection and validation of native MDP terms named by scene.yaml.

Python Task projects select observations through ``policy.layers[].inputs``; the
``observations:``/``actions:``/``task:`` sections are read only for projects kept
on Isaac Lab's native terms. Their standard preset names are listed here as data,
so ``check`` and the editor validate a project without importing a simulator.
Native implementation paths (``isaaclab.envs.mdp.*``) belong to the backend: ask
``implementation_names(backend)`` instead of writing them here.
"""
from collections.abc import Mapping
from importlib import import_module

# Observation group options; the rest of a selection names native terms.
GROUP_OPTIONS = {
    "enable_corruption", "concatenate_terms", "concatenate_dim", "history_length", "flatten_history_dim",
}
GROUP_CLASSES = {
    "observations": "ObservationsCfg",
    "commands": "CommandsCfg", "rewards": "RewardsCfg",
    "terminations": "TerminationsCfg", "events": "EventCfg",
    "curriculum": "CurriculumCfg",
}
TASK_GROUPS = tuple(name for name in GROUP_CLASSES if name != "observations")
# Standard presets that existing projects name in ``task.preset``.
PRESET_NAMES = ("velocity", "lift")
PRESET_OBSERVATION_NAMES = {
    "velocity": ("base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                 "joint_pos", "joint_vel", "actions", "height_scan"),
    "lift": ("joint_pos", "joint_vel", "object_position", "target_object_position", "actions"),
}
PRESET_TASK_NAMES = {
    "lift": {
        "commands": {"object_pose"},
        "rewards": {"reaching_object", "lifting_object", "object_goal_tracking",
                    "object_goal_tracking_fine_grained", "action_rate", "joint_vel"},
        "terminations": {"time_out", "object_dropping"},
        "events": {"reset_all", "reset_object_position"},
        "curriculum": {"action_rate", "joint_vel"},
    },
    "velocity": {
        "commands": {"base_velocity"},
        "rewards": {"track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2",
                    "dof_torques_l2", "dof_acc_l2", "action_rate_l2", "feet_air_time", "undesired_contacts",
                    "flat_orientation_l2", "dof_pos_limits"},
        "terminations": {"time_out", "base_contact"},
        "events": {"physics_material", "add_base_mass", "base_com", "base_external_force_torque",
                   "reset_base", "reset_robot_joints", "push_robot"},
        "curriculum": {"terrain_levels"},
    },
}
ACTION_CLASSES = {
    "joint_position": "JointPositionActionCfg",
    "relative_joint_position": "RelativeJointPositionActionCfg",
    "joint_velocity": "JointVelocityActionCfg",
    "joint_effort": "JointEffortActionCfg",
    "binary_joint_position": "BinaryJointPositionActionCfg",
}


class _DeclaredNames:
    """Fallback when a backend declares no native implementation names (its own modules bind the terms)."""

    def __init__(self, backend):
        self.backend = backend

    def observation(self, function):
        return self.backend + "." + function

    def action(self, class_name):
        return self.backend + "." + class_name

    def reward(self, function):
        return self.backend + "." + function

    def signal(self, signal, function):
        return self.backend + "." + function


def implementation_names(backend):
    """The backend's native implementation paths, e.g. ``isaaclab.envs.mdp.joint_pos_rel``.

    A backend that binds these terms in its own code (mjlab) declares none and the
    names stay its own module paths; missing module is not an error here.
    """
    try:
        module = import_module(f"shared.backends.{backend}.native_terms")
    except ModuleNotFoundError:
        return _DeclaredNames(backend)
    return module


def selection(value, label):
    """A group selection: a term list, or ``{terms, overrides, <options>}``."""
    if isinstance(value, (list, tuple)):
        names, options, overrides = value, {}, {}
    else:
        value = _mapping(value, label)
        unknown = value.keys() - (GROUP_OPTIONS | {"terms", "overrides"})
        if unknown:
            raise ValueError(f"{label}: unknown options {sorted(unknown)}")
        names = value.get("terms")
        options = {key: value[key] for key in GROUP_OPTIONS if key in value}
        overrides = _mapping(value.get("overrides", {}), f"{label}.overrides")
    if not isinstance(names, (list, tuple)) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{label}.terms must be a nonempty list of observation names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label}: repeated observation terms")
    if overrides.keys() - set(names):
        raise ValueError(f"{label}: overrides must name selected observation terms")
    for name, override in overrides.items():
        _mapping(override, f"{label}.overrides.{name}")
    for option in ("enable_corruption", "concatenate_terms", "flatten_history_dim"):
        if option in options and not isinstance(options[option], bool):
            raise ValueError(f"{label}.{option} must be a boolean")
    for option in ("history_length", "concatenate_dim"):
        if option in options and options[option] is not None and (
            isinstance(options[option], bool) or not isinstance(options[option], int)
        ):
            raise ValueError(f"{label}.{option} must be an integer")
    if options.get("history_length") is not None and options["history_length"] < 0:
        raise ValueError(f"{label}.history_length must be nonnegative")
    return list(names), options, overrides


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def validate_task_settings(settings, *, term_registry=None, task_presets=None, require_training=True):
    """Check names/shape without importing the simulator or invoking factories.

    Custom presets are opaque Python extensions: their term names are validated
    when their native configs are built. Standard presets are checked here.
    """
    registry = term_registry or {}
    _mapping(registry, "TERMS")
    categories = set(GROUP_CLASSES) | {"actions", "signals", "object_api"}
    if registry.keys() - categories:
        raise ValueError(f"TERMS: unknown categories {sorted(registry.keys() - categories)}")
    for category, entries in registry.items():
        _mapping(entries, f"TERMS.{category}")
    presets = task_presets or {}
    _mapping(presets, "TASKS")
    task = _mapping(settings.get("task", {}), "task")
    if 'object_api' in registry:
        if task or settings.get('variables'):
            raise ValueError('Python Task projects define evaluation in task.py; remove task/variables from scene.yaml')
    else:
        from .legacy.task import evaluation_config, TASK_FIELDS
        evaluation = evaluation_config(settings)
        unknown = task.keys() - (set(TASK_GROUPS) | {"preset"} | TASK_FIELDS)
        if unknown:
            raise ValueError(f"task: unknown fields {sorted(unknown)}")
        from .legacy.sources import validate_sources
        validate_sources(evaluation, registry.get("signals"), settings)
    preset = task.get("preset")
    if preset is not None and preset not in set(PRESET_NAMES) | set(presets):
        raise ValueError(f"Unknown task preset: {preset}")
    observations = _mapping(settings.get("observations", {}), "observations")
    if not observations and require_training:
        raise ValueError("observations must select at least one group")
    from .entity_information import parse_signal
    from .targets import target_catalog
    observation_names = set(PRESET_OBSERVATION_NAMES.get(preset, ())) | set(registry.get("observations", {})) \
        | set(target_catalog(settings)) | {"simulation.gravity"}
    selected = {}
    for name, entry in observations.items():
        if not isinstance(name, str) or not name or name.startswith("_"):
            raise ValueError("observation group names must be nonempty public names")
        names, _, _ = selection(entry, f"observations.{name}")
        unknown = {term for term in names if term not in observation_names and not parse_signal(term)}
        if preset not in presets and unknown:
            raise ValueError(f"observations.{name}: unknown terms {sorted(unknown)}")
        selected[name] = names
    actions = _mapping(settings.get("actions", {}), "actions")
    if not actions and require_training:
        raise ValueError("actions must contain at least one native action term")
    available_actions = set(ACTION_CLASSES) | set(registry.get("actions", {}))
    for name, value in actions.items():
        value = _mapping(value, f"actions.{name}")
        if value.get("type") not in available_actions:
            raise ValueError(f"actions.{name}: unknown action type {value.get('type')!r}")
    for category in TASK_GROUPS:
        entries = task.get(category, {})
        if entries is None:
            continue
        for name, value in _mapping(entries, f"task.{category}").items():
            if preset not in presets and name not in PRESET_TASK_NAMES.get(preset, {}).get(category, ()):
                if not isinstance(value, Mapping) or "type" not in value:
                    raise ValueError(f"task.{category}.{name}: unknown term; choose a registered type for a new term")
            if value is None:
                continue
            value = _mapping(value, f"task.{category}.{name}")
            if "type" in value and value["type"] not in registry.get(category, {}):
                raise ValueError(f"task.{category}.{name}: unknown term type {value['type']!r}")
    return {"preset": preset, "observations": selected, "actions": list(actions)}
