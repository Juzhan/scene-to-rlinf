"""Validate the common project contract against implemented mjlab capabilities."""
from shared.assembly.entity_information import parse_signal
from shared.assembly.targets import target_catalog
from .scene import validate

# Native Isaac project terms and unkeyed assets.py FACTORIES are not portable here;
# project assets reach mjlab through BACKEND_FACTORIES['mjlab'].
PROJECT_TERMS = False
UNKEYED_FACTORIES = False
validate_scene = validate


def validate_project(description, settings, terms, *, tasks=None, factories=None,
                     require_training=True, evaluation=None):
    from .readers import SUPPORTED
    # The scene compiler receives scene data only; policy/task remain shared.
    scene_fields = {'schema_version', 'name', 'entities', 'connections', 'simulation', 'world', 'authoring', 'visualization'}
    layout = {k: v for k, v in settings.items() if k in scene_fields}
    layout['world'] = {k: v for k, v in layout.get('world', {}).items() if k != 'decimation'}
    validate(description, layout)
    if 'object_api' not in terms:
        raise ValueError('mjlab projects use the shared Python task.py contract; native Isaac task presets are not portable')
    if require_training and 'control' not in settings:
        raise ValueError('Declare the shared planner/executor control configuration')
    keys = {path.removeprefix(description.root.name + '/').replace('/', '__')
            for path, (node, _) in description.index().items() if node.factory}
    targets = target_catalog(settings)
    for source in targets:
        if '.target.' in source and source.split('.target.')[0].replace('.', '__') not in keys:
            raise ValueError(f'Target references unknown entity: {source}')
    for role, component in settings.get('control', {}).get('components', {}).items():
        for source in component['observations']:
            if source in targets or source == 'simulation.gravity' or source in {
                    role + '.output', role + '.previous_output', 'planner.output'}:
                continue
            parsed = parse_signal(source)
            if parsed is None or parsed[0] not in keys:
                raise ValueError(f'Unknown entity observation: {source}')
            if parsed[1] not in SUPPORTED:
                raise NotImplementedError(f'mjlab information reader not implemented: {source}')
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


def inspect_scene(description, settings, factories, report, *, view=False):
    """Compile the scene alone with CPU MuJoCo; ``view`` opens the native viewer. No policy runs."""
    from .scene import compile_scene
    scene, model, data, keys = compile_scene(description, settings, factories)
    report.update(simulator_started=True, runtime='mjlab scene compiled with CPU MuJoCo',
                  model={'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
                         'bodies': model.nbody, 'geometries': model.ngeom})
    for entry in report['entities']:
        if keys[entry['path']] in scene.sensors:
            entry.update(sensor=type(scene.sensors[keys[entry['path']]]).__name__)
            continue
        entity = scene.entities[keys[entry['path']]]
        body = model.body(entity.root_body.name)
        entry.update(actual_position=data.xpos[body.id].tolist(),
                     actual_orientation=data.xquat[body.id].tolist(),
                     joints=list(entity.joint_names))
    if view:
        import mujoco.viewer
        # Native viewer owns interaction. No policy or training is running.
        mujoco.viewer.launch(model=model, data=data)
