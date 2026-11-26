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
from shared.assembly.native_terms import (ACTION_CLASSES as _ACTION_CLASSES,  # noqa: F401 - used here
                                          GROUP_CLASSES as _GROUP_CLASSES, GROUP_OPTIONS as _GROUP_OPTIONS,
                                          TASK_GROUPS as _TASK_GROUPS, selection as _selection,
                                          validate_task_settings)  # noqa: F401 - re-exported for projects
from .entity_observations import parse_signal, build_signal
from .targets import target_catalog, build_target


_PRESET_MODULES = {
    "velocity": "isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg",
    "lift": "isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg",
}


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


def build_task_config(settings, *, term_registry=None, task_presets=None, visualize=False):
    """Return native manager configs; call only after starting Isaac Sim.

    Observation list order is the policy input order. Presets supply an
    observation catalog and task terms, never a hidden observation vector.
    Actions are explicit native terms concatenated in YAML insertion order.
    ``visualize`` adds markers for ``visualization.abstract``; only the preview window/video sets it.
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
    attach_abstract(output, settings, visualize=visualize)
    if 'object_api' not in registry and (settings.get("variables") or task.keys() & {"reward", "success", "failure", "metrics"}):
        from shared.assembly.legacy.task import evaluation_config
        from .task_evaluation import attach_evaluation
        attach_evaluation(output, evaluation_config(settings), registry.get("signals"), settings)
    if 'object_api' in registry:
        from .python_task import attach_task
        attach_task(output, registry['object_api'], settings)
    return output
