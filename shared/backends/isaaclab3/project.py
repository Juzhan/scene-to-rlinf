"""Validate the common project contract against implemented Isaac Lab 3 capabilities."""
from shared.assembly.entity_information import parse_signal
from shared.assembly.targets import target_catalog
from .scene import validate

# Native Isaac Lab 2 project terms and unkeyed assets.py FACTORIES are not portable here;
# project assets reach this backend through BACKEND_FACTORIES['isaaclab3'].
PROJECT_TERMS = False
UNKEYED_FACTORIES = False
validate_scene = validate


def validate_project(description, settings, terms, *, tasks=None, factories=None,
                     require_training=True, evaluation=None):
    from shared.assembly.robot import frame_observations
    from .readers import SUPPORTED
    layout = {k: v for k, v in settings.items() if k in {
        'schema_version', 'name', 'entities', 'connections', 'simulation', 'world', 'authoring', 'visualization'}}
    layout['world'] = {k: v for k, v in layout.get('world', {}).items() if k != 'decimation'}
    validate(description, layout)
    if 'object_api' not in terms:
        raise ValueError('isaaclab3 projects use the shared Python task.py contract; native task presets are not portable')
    if require_training and 'control' not in settings:
        raise ValueError('Declare the shared planner/executor control configuration')
    keys = {path.removeprefix(description.root.name + '/').replace('/', '__')
            for path, (node, _) in description.index().items() if node.factory}
    targets = target_catalog(settings)
    frames = frame_observations(settings.get('_robot_frames') or {})
    for source in targets:
        if '.target.' in source and source.split('.target.')[0].replace('.', '__') not in keys:
            raise ValueError(f'Target references unknown entity: {source}')
    for role, component in settings.get('control', {}).get('components', {}).items():
        for source in component['observations']:
            if source in targets or source in frames or source == 'simulation.gravity' or source in {
                    role + '.output', role + '.previous_output', 'planner.output'}:
                continue
            parsed = parse_signal(source)
            if parsed is None or parsed[0] not in keys:
                raise ValueError(f'Unknown entity observation: {source}')
            if parsed[1] not in SUPPORTED:
                raise NotImplementedError(f'isaaclab3 information reader not implemented: {source}')
    if factories is not None:
        from shared.assembly.assets import image_outputs
        from shared.assembly.entity_information import check_image_sources
        from shared.assembly.project import _selected_sources
        check_image_sources(_selected_sources(settings), {
            path.removeprefix(description.root.name + '/').replace('/', '__'): image_outputs(factories[node.factory], node.parameters)
            for path, (node, _) in description.index().items() if node.factory})
    for name, binding in settings.get('control', {}).get('output', {}).items():
        if binding['asset_name'] not in keys:
            raise ValueError(f'Control {name} references unknown entity: {binding["asset_name"]}')
