"""Backend-neutral scene loading and a static report, independent of training."""
from dataclasses import asdict
from pathlib import Path
from .assets import asset_backends, backend_module, check_parameters, factories_for
from .layout import load_scene_config, load_world_spec


def prepare(path, backend):
    path = Path(path).resolve()
    settings = load_scene_config(path)
    factories = factories_for(backend)
    from .catalog import _project_package
    from .project import project_registry
    package = _project_package(path.parent)
    if package:
        factories, _, _ = project_registry(package, backend=backend)
    description = load_world_spec(path, cameras=True, factories=factories)
    index = description.index()
    for name, (entity, _) in index.items():
        if not entity.factory:
            continue
        try:
            check_parameters(factories[entity.factory], entity.parameters)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{backend} entity {name}: {error}') from error
    backend_module(backend, 'project').validate_scene(description, settings)
    report = {'backend': backend, 'scene': str(path), 'scope': 'scene construction',
              'entities': [{'path': name, 'asset': node.factory,
                            'supported_backends': sorted(set(asset_backends(node.factory)) | {backend}),
                            'layout_pose': asdict(pose), 'initial_state': node.initial_state}
                           for name, (node, pose) in index.items() if node.factory],
              'abstract': [{'path': name, 'shape': node.shape, 'size': list(node.size), 'pose': asdict(node.pose)}
                           for name, (node, _) in index.items() if node.kind == 'abstract'],
              'simulator_started': False, 'rlinf_training_connected': False,
              'limitations': ['This scene-only command does not configure policy, task or training.',
                              'Matching asset names does not guarantee matching dynamics.']}
    return settings, description, factories, report
