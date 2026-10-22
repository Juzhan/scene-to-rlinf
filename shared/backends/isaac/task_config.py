"""Select and configure Isaac Lab's native MDP terms.

This module contains configuration translation only. Isaac Lab owns action
processing, rewards, commands, observations, reset events and simulation steps.
Project ``terms.py`` may expose ``TERMS`` (category -> name -> native config or
lazy config factory) and ``TASKS`` (preset name -> lazy native group factory).
"""

from collections.abc import Mapping
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from .entity_observations import parse_signal, build_signal
from .targets import target_catalog, build_target


_GROUP_CLASSES = {
    "observations": "ObservationsCfg",
    "commands": "CommandsCfg", "rewards": "RewardsCfg",
    "terminations": "TerminationsCfg", "events": "EventCfg",
    "curriculum": "CurriculumCfg",
}
_TASK_GROUPS = tuple(name for name in _GROUP_CLASSES if name != "observations")
_PRESET_MODULES = {
    "velocity": "isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg",
    "lift": "isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg",
}
_OBSERVATION_NAMES = {
    "velocity": ("base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                 "joint_pos", "joint_vel", "actions", "height_scan"),
    "lift": ("joint_pos", "joint_vel", "object_position", "target_object_position", "actions"),
}
_TASK_NAMES = {
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
_ACTION_CLASSES = {
    "joint_position": "JointPositionActionCfg",
    "relative_joint_position": "RelativeJointPositionActionCfg",
    "joint_velocity": "JointVelocityActionCfg",
    "joint_effort": "JointEffortActionCfg",
    "binary_joint_position": "BinaryJointPositionActionCfg",
}
_GROUP_OPTIONS = {
    "enable_corruption", "concatenate_terms", "concatenate_dim", "history_length", "flatten_history_dim",
}


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _selection(value, label):
    if isinstance(value, (list, tuple)):
        names, options, overrides = value, {}, {}
    else:
        value = _mapping(value, label)
        unknown = value.keys() - (_GROUP_OPTIONS | {"terms", "overrides"})
        if unknown:
            raise ValueError(f"{label}: unknown options {sorted(unknown)}")
        names = value.get("terms")
        options = {key: value[key] for key in _GROUP_OPTIONS if key in value}
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


def validate_task_settings(settings, *, term_registry=None, task_presets=None, require_training=True):
    """Check names/shape without importing the simulator or invoking factories.

    Custom presets are opaque Python extensions: their term names are validated
    when their native configs are built. Standard presets are checked here.
    """
    registry = term_registry or {}
    _mapping(registry, "TERMS")
    categories = set(_GROUP_CLASSES) | {"actions", "signals", "object_api"}
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
        from shared.assembly.legacy.task import evaluation_config, TASK_FIELDS
        evaluation = evaluation_config(settings)
        unknown = task.keys() - (set(_TASK_GROUPS) | {"preset"} | TASK_FIELDS)
        if unknown:
            raise ValueError(f"task: unknown fields {sorted(unknown)}")
        from .task_evaluation import validate_sources
        validate_sources(evaluation, registry.get("signals"), settings)
    preset = task.get("preset")
    if preset is not None and preset not in set(_PRESET_MODULES) | set(presets):
        raise ValueError(f"Unknown task preset: {preset}")
    observations = _mapping(settings.get("observations", {}), "observations")
    if not observations and require_training:
        raise ValueError("observations must select at least one group")
    observation_names = set(_OBSERVATION_NAMES.get(preset, ())) | set(registry.get("observations", {})) | set(target_catalog(settings)) | {"simulation.gravity"}
    selected = {}
    for name, selection in observations.items():
        if not isinstance(name, str) or not name or name.startswith("_"):
            raise ValueError("observation group names must be nonempty public names")
        names, _, _ = _selection(selection, f"observations.{name}")
        unknown = {term for term in names if term not in observation_names and not parse_signal(term)}
        if preset not in presets and unknown:
            raise ValueError(f"observations.{name}: unknown terms {sorted(unknown)}")
        selected[name] = names
    actions = _mapping(settings.get("actions", {}), "actions")
    if not actions and require_training:
        raise ValueError("actions must contain at least one native action term")
    available_actions = set(_ACTION_CLASSES) | set(registry.get("actions", {}))
    for name, value in actions.items():
        value = _mapping(value, f"actions.{name}")
        if value.get("type") not in available_actions:
            raise ValueError(f"actions.{name}: unknown action type {value.get('type')!r}")
    for category in _TASK_GROUPS:
        entries = task.get(category, {})
        if entries is None:
            continue
        for name, value in _mapping(entries, f"task.{category}").items():
            if preset not in presets and name not in _TASK_NAMES.get(preset, {}).get(category, ()):
                if not isinstance(value, Mapping) or "type" not in value:
                    raise ValueError(f"task.{category}.{name}: unknown term; choose a registered type for a new term")
            if value is None:
                continue
            value = _mapping(value, f"task.{category}.{name}")
            if "type" in value and value["type"] not in registry.get(category, {}):
                raise ValueError(f"task.{category}.{name}: unknown term type {value['type']!r}")
    return {"preset": preset, "observations": selected, "actions": list(actions)}


def _members(value):
    return dict(value) if isinstance(value, Mapping) else dict(vars(value))


def _patch(config, changes, label):
    """Update native config fields, preserving nested config types and tuples."""
    for key, value in changes.items():
        if isinstance(config, dict):
            current = config.get(key)
        elif not hasattr(config, key):
            raise ValueError(f"{label}: unknown native field {key}")
        else:
            current = getattr(config, key)
        if isinstance(value, Mapping) and current is not None:
            _patch(current, value, f"{label}.{key}")
            continue
        value = tuple(value) if isinstance(current, tuple) and isinstance(value, list) else deepcopy(value)
        if isinstance(config, dict):
            config[key] = value
        else:
            setattr(config, key, value)
    return config


def _native_preset(name, task_presets):
    if name is None:
        return {}
    if name in task_presets:
        preset = task_presets[name]
        preset = preset() if callable(preset) else preset
    else:
        preset = import_module(_PRESET_MODULES[name])
    if isinstance(preset, Mapping):
        unknown = preset.keys() - _GROUP_CLASSES.keys()
        if unknown:
            raise ValueError(f"Task preset {name}: unknown groups {sorted(unknown)}")
        return deepcopy(dict(preset))
    return {name: getattr(preset, class_name)() for name, class_name in _GROUP_CLASSES.items()}


def _prototype(entry):
    return deepcopy(entry() if callable(entry) else entry)


def _flatten_spatial_signal(env, path):
    """Flatten only the vector policy boundary, preserving Task's native layout."""
    from .entity_observations import primitive
    value = primitive(env, path)
    return value.reshape(value.shape[0], -1)


def build_task_config(settings, *, term_registry=None, task_presets=None):
    """Return native manager configs; call only after starting Isaac Sim.

    Observation list order is the policy input order. Presets supply an
    observation catalog and task terms, never a hidden observation vector.
    Actions are explicit native terms concatenated in YAML insertion order.
    """
    validate_task_settings(settings, term_registry=term_registry, task_presets=task_presets)
    from isaaclab.managers import ObservationGroupCfg

    registry = term_registry or {}
    task = settings.get("task", {})
    native = _native_preset(task.get("preset"), task_presets or {})
    native_observations = _members(native.pop("observations", {}))
    source_group = native_observations.get("policy")
    catalog = {} if source_group is None else {
        name: value for name, value in _members(source_group).items() if name not in _GROUP_OPTIONS and value is not None
    }
    catalog.update(registry.get("observations", {}))
    catalog.update({name: build_target(value) for name, (value, _) in target_catalog(settings).items()})
    observations = {}
    if 'object_api' in registry:
        for selection in settings['observations'].values():
            if not isinstance(selection, list):
                raise ValueError('Object observations are plain field lists; preprocessing belongs to the model')
            names, _, _ = _selection(selection, 'observations')
            for name in names:
                if name not in catalog:
                    catalog[name] = build_signal(name)
                group = ObservationGroupCfg(enable_corruption=False, concatenate_terms=True)
                setattr(group, 'value', _prototype(catalog[name]))
                observations[name] = group
    else:
        for group_name, selection in settings["observations"].items():
            names, options, overrides = _selection(selection, f"observations.{group_name}")
            defaults = {} if source_group is None else {
                name: getattr(source_group, name) for name in _GROUP_OPTIONS if hasattr(source_group, name)
            }
            group = ObservationGroupCfg(**{**defaults, **options})
            for name in names:
                if name not in catalog and (name == "simulation.gravity" or parse_signal(name)):
                    catalog[name] = build_signal(name)
                if name not in catalog:
                    raise ValueError(f"observations.{group_name}: unknown term {name!r}")
                term = _prototype(catalog[name])
                _patch(term, overrides.get(name, {}), f"observations.{group_name}.{name}")
                if parse_signal(name) and parse_signal(name)[1] in {'ray_hit_positions', 'frame_positions', 'contact_force', 'contact_force_samples'}:
                    term.func = _flatten_spatial_signal
                setattr(group, name, term)
            observations[group_name] = group
    actions = {}
    for name, action in settings["actions"].items():
        parameters = dict(action)
        type_name = parameters.pop("type")
        if type_name in registry.get("actions", {}):
            entry = registry["actions"][type_name]
            actions[name] = deepcopy(entry(**parameters)) if callable(entry) else _patch(
                deepcopy(entry), parameters, f"actions.{name}"
            )
        else:
            mdp = import_module("isaaclab.envs.mdp")
            actions[name] = getattr(mdp, _ACTION_CLASSES[type_name])(**parameters)
    # CommandManager and ActionManager advertise dict support but write
    # ``cfg.debug_vis`` during construction in the installed SDK. Give these
    # two managers an attribute configuration, as the native task presets do.
    output = {"observations": observations, "actions": SimpleNamespace(**actions)}
    for category in _TASK_GROUPS:
        entries = {} if native.get(category) is None else deepcopy(_members(native.get(category, {})))
        updates = task.get(category, {})
        if updates is None:
            output[category] = SimpleNamespace() if category == "commands" else {}
            continue
        for name, update in updates.items():
            if update is None:
                entries[name] = None
                continue
            update = dict(update)
            type_name = update.pop("type", None)
            if type_name is not None:
                entries[name] = _prototype(registry[category][type_name])
            if name not in entries or entries[name] is None:
                raise ValueError(f"task.{category}.{name}: choose a registered type for a new term")
            _patch(entries[name], update, f"task.{category}.{name}")
        output[category] = SimpleNamespace(**entries) if category == "commands" else entries
    from .abstract_terms import attach_abstract
    attach_abstract(output, settings)
    if 'object_api' not in registry and (settings.get("variables") or task.keys() & {"reward", "success", "failure", "metrics"}):
        from shared.assembly.legacy.task import evaluation_config
        from .task_evaluation import attach_evaluation
        attach_evaluation(output, evaluation_config(settings), registry.get("signals"), settings)
    if 'object_api' in registry:
        from .python_task import attach_task
        attach_task(output, registry['object_api'], settings)
    return output
