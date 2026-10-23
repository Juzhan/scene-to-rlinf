"""Validate the common project contract against the Isaac scene, without launching Isaac."""
from shared.assembly.entity_information import parse_signal
from shared.assembly.targets import target_catalog

# Project terms.py TERMS/TASKS are native Isaac manager terms, and an assets.py
# FACTORIES table without a backend key predates mjlab; both belong to Isaac.
PROJECT_TERMS = True
UNKEYED_FACTORIES = True


def validate_scene(description, settings):
    """Scene-only check; the Isaac scene compiler accepts every shared scene field."""


def validate_project(description, settings, terms, *, tasks=None, factories=None,
                     require_training=True, evaluation=None):
    from shared.assembly.assets import image_outputs
    from shared.assembly.entity_information import check_image_sources
    from shared.assembly.project import _selected_sources
    from .contact_materials import validate_materials
    from .scene import scene_metadata
    from .task_config import _selection, validate_task_settings
    metadata = scene_metadata(description)
    validate_materials(settings.get("world", {}).get("contact_materials", {}), metadata.keys.values())
    from shared.assembly.robot import frame_observations
    # Named robot frames are read through the robot interface, not as scene entities.
    known = set(metadata.keys.values()) | {robot + '__' + frame for robot, frame, _ in
                                           frame_observations(settings.get('_robot_frames') or {}).values()}
    for signal_path in target_catalog(settings):
        if '.target.' not in signal_path:
            # Abstract entities are validated together with the scene tree.
            continue
        if signal_path.split('.target.')[0].replace('.', '__') not in known:
            raise ValueError(f'Target {signal_path!r} references an unknown scene entity')
    for group, selection in settings.get('observations', {}).items():
        for signal_path in _selection(selection, f'observations.{group}')[0]:
            parsed = parse_signal(signal_path)
            if signal_path not in target_catalog(settings) and parsed and parsed[0] not in known:
                raise ValueError(f'Observation {signal_path!r} references an unknown scene entity')
    if 'control' in settings:
        from shared.assembly.control import component_observations
        for name, component in settings['control']['components'].items():
            for entry in component_observations(component['observations']):
                parsed = parse_signal(entry['source'])
                if entry["source"] not in target_catalog(settings) and parsed and parsed[0] not in known:
                    raise ValueError(f"Component {name!r} references unknown entity in {entry['source']!r}")
    check_image_sources(_selected_sources(settings), {
        metadata.keys[path]: image_outputs(factories[node.factory], node.parameters)
        for path, (node, _) in description.index().items() if node.factory and path in metadata.keys})
    validate_task_settings(settings, term_registry=terms, task_presets=tasks or {},
                           require_training=require_training)
    if evaluation is not None:
        from .task_evaluation import validate_sources
        custom_signals = terms.get("signals", {})
        for signal in validate_sources(evaluation, custom_signals, settings)["signals"]:
            parsed = parse_signal(signal)
            if signal not in custom_signals and signal not in target_catalog(settings) and parsed and parsed[0] not in known:
                raise ValueError(f"Task signal {signal!r} references an unknown scene entity")
