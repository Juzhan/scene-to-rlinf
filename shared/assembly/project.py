"""One local project entry for checking, previewing and training a configured scene.

This module only reads configuration. Isaac and RLinf imports are deferred until
launch/registration; project asset and term factories must defer SDK imports too.
"""

from dataclasses import asdict, dataclass
from importlib import import_module, util
import os
from pathlib import Path
import sys

from .assets import DEFAULT_BACKEND, backend_module, selected_backend


def require_online_backend(parameters):
    selected_backend(parameters)


def backend_interpreter(backend):
    """Interpreter able to import the selected simulator.

    A backend package may declare ``SDK`` (module it needs) and ``INTERPRETER`` (workspace
    path of a separate environment). That interpreter is chosen automatically unless the
    user named one with ``RLINF_PYTHON``.
    """
    package = import_module(f'shared.backends.{selected_backend({"backend": backend})}')
    sdk = getattr(package, 'SDK', None)
    if sdk is None or util.find_spec(sdk) is not None:
        return sys.executable
    interpreter = getattr(package, 'INTERPRETER', None)
    candidate = Path(__file__).resolve().parents[2] / interpreter if interpreter else None
    if os.environ.get('RLINF_PYTHON') or candidate is None or not candidate.is_file():
        raise RuntimeError(f'{sys.executable} has no {sdk}; set RLINF_PYTHON to an interpreter with {sdk}'
                           + (f' (e.g. {interpreter})' if interpreter else ''))
    return str(candidate)


def follow_train_backend(cfg, overrides, previous=None):
    """A backend chosen for training on the command line also moves evaluation.

    Recipes resolve ``env.eval`` from ``env.train`` before command-line overrides apply, so
    without this a train override would silently leave evaluation on the old simulator.
    An explicit eval backend, or an eval backend that already differed (``previous``), is kept.
    """
    keys = {item.lstrip('+').split('=', 1)[0] for item in overrides}
    if 'env.train.init_params.backend' not in keys or 'env.eval.init_params.backend' in keys:
        return cfg
    if previous is not None and selected_backend(cfg.env.eval.init_params) != previous:
        return cfg
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.env.eval.init_params.backend = cfg.env.train.init_params.backend
    return cfg


@dataclass
class PreparedScene:
    path: Path
    settings: dict
    application: dict
    factories: dict
    terms: dict
    tasks: dict
    description: object
    backend: str = DEFAULT_BACKEND


def project_registry(package=None, *, factories=None, terms=None, tasks=None, backend=DEFAULT_BACKEND):
    """Discover optional project assets.py and terms.py; never invoke factories."""
    from .assets import factories_for

    assets, registered_terms, registered_tasks = factories_for(backend), {}, {}
    support = backend_module(backend, 'project')
    if package:
        for suffix in ("assets", "terms"):
            name = package + "." + suffix
            if util.find_spec(name) is None:
                continue
            module = import_module(name)
            if suffix == "assets":
                if support.UNKEYED_FACTORIES:
                    assets.update(getattr(module, "FACTORIES", {}))
                assets.update(getattr(module, 'BACKEND_FACTORIES', {}).get(backend, {}))
            else:
                if not support.PROJECT_TERMS:
                    continue  # Native simulator terms are not portable across backends.
                registered_terms.update(getattr(module, "TERMS", {}))
                registered_tasks.update(getattr(module, "TASKS", {}))
    assets.update(factories or {})
    for category, entries in (terms or {}).items():
        registered_terms[category] = {**registered_terms.get(category, {}), **entries}
    registered_tasks.update(tasks or {})
    for label, registry in (("Asset", assets), ("Task", registered_tasks)):
        if not isinstance(registry, dict) or any(
            not isinstance(name, str) or not name or not callable(factory)
            for name, factory in registry.items()
        ):
            raise ValueError(f"{label} registry must map nonempty names to callable factories")
    return assets, registered_terms, registered_tasks


def _selected_sources(settings):
    """Observation sources chosen in observation groups and control components."""
    from shared.backends.isaac.task_config import _selection
    sources = [name for group, selection in settings.get('observations', {}).items()
               for name in _selection(selection, f'observations.{group}')[0]]
    if 'control' in settings:
        from .control import component_observations
        sources += [entry['source'] for component in settings['control']['components'].values()
                    for entry in component_observations(component['observations'])]
    return sources


def _check_factory(name, parameters, factories, location):
    if name not in factories:
        raise ValueError(f"{location}: unknown asset factory {name!r}")
    if not isinstance(parameters, dict):
        raise ValueError(f"{location}: parameters must be a mapping")
    from .assets import check_parameters
    try:
        check_parameters(factories[name], parameters)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location}: factory {name!r}: {error}") from error


def prepare_geometry(scene_config, *, project_package=None):
    """Scene tree and asset factories only, for visual export of a scene still being edited.

    Control, observations and Task are not compiled: an unfinished control section
    must not block seeing the scene.
    """
    from .layout import load_scene_config, load_world_spec
    path = Path(scene_config).expanduser().resolve()
    assets, _, _ = project_registry(project_package)
    settings = load_scene_config(path)
    application = {"headless": True, "enable_cameras": False, **settings.get("application", {})}
    application.update(headless=True, enable_cameras=False)
    description = load_world_spec(path, cameras=False, factories=assets)
    for entity_path, (node, _) in description.index().items():
        if node.factory:
            _check_factory(node.factory, node.parameters, assets, entity_path)
    return PreparedScene(path, settings, application, assets, {}, {}, description)


def refresh_observation_types(scene, package, **options):
    """Regenerate a live project's observation.py; a frozen run snapshot keeps the file it recorded.

    The generator's comments follow the shared code, so rewriting a snapshot would break its hash."""
    if package and package.startswith('_experiment_'):
        return
    from .object_schema import generate
    generate(scene, **options)


def prepare_scene(parameters, *, scene_config, project_package=None, factories=None,
                  terms=None, tasks=None, headless=None, enable_cameras=None, require_training=True):
    """Same static preparation for check, preview and every RLinf worker."""
    from .layout import load_scene_config, load_world_spec
    backend = selected_backend(parameters)
    path = Path(parameters.get("scene_config", scene_config)).expanduser().resolve()
    assets, registered_terms, registered_tasks = project_registry(
        project_package, factories=factories, terms=terms, tasks=tasks, backend=backend)
    try:
        settings = load_scene_config(path)
        if (Path(scene_config).parent / 'task.py').is_file() and project_package:
            if settings.get('variables') or settings.get('task'):
                raise ValueError('Python Task projects define evaluation in task.py; remove task/variables from scene.yaml')
            # Project-local assets must be discoverable while the typed schema
            # is regenerated, exactly as they are for the environment itself.
            refresh_observation_types(path, project_package, factories=assets, terms=registered_terms)
            registered_terms['object_api'] = {'module': project_package}
            task_module = import_module(project_package + '.task')
            if not callable(getattr(getattr(task_module, 'Task', task_module), 'evaluate', None)):
                raise ValueError('task.py must define evaluate(obs) or Task.evaluate(self, obs)')
        # Model registration belongs to the project, not an alternate scene file.
        if path != Path(scene_config).expanduser().resolve():
            original = load_scene_config(scene_config).get('control', {})
            alternate = settings.get('control', {})
            def model_boundary(control):
                role = control.get('train')
                component = control.get('components', {}).get(role, {})
                return role, component.get('model'), component.get('parameters', {})
            if model_boundary(original) != model_boundary(alternate):
                raise ValueError('Change the trained component in project scene.yaml; scene_config overrides may not change model registration')
        evaluation = None
        if 'object_api' not in registered_terms:
            from .legacy.task import evaluation_config
            evaluation = evaluation_config(settings)
        from .control import compile_control
        from .robot import robot_frames
        settings['_robot_frames'] = robot_frames(load_world_spec(path, cameras=False, factories=assets))
        settings, registered_terms = compile_control(settings, registered_terms, project_package=project_package, backend=backend)
        application = {"headless": True, "enable_cameras": False, **settings.get("application", {})}
        if headless is not None:
            application["headless"] = headless
        if enable_cameras is not None:
            application["enable_cameras"] = enable_cameras
        description = load_world_spec(path, cameras=application["enable_cameras"], factories=assets)
        for entity_path, (node, _) in description.index().items():
            if node.factory:
                _check_factory(node.factory, node.parameters, assets, entity_path)
        backend_module(backend, 'project').validate_project(
            description, settings, registered_terms, tasks=registered_tasks, factories=assets,
            require_training=require_training, evaluation=evaluation)
        return PreparedScene(path, settings, application, assets, registered_terms, registered_tasks, description, backend)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def check_project(cfg, *, scene_config, project_package=None, **registries):
    """Static CLI report. Physical behavior and tensor sizes require a smoke run."""
    environment = cfg.env.eval if cfg.runner.only_eval else cfg.env.train
    prepared = prepare_scene(environment.init_params, scene_config=scene_config,
                             project_package=project_package, **registries)
    if prepared.settings.get("control"):
        from .control import require_listed_joints
        require_listed_joints(prepared.settings["control"]["output"])
    from shared.backends.isaac.scene import scene_metadata
    index = prepared.description.index()
    metadata = scene_metadata(prepared.description)
    joints = prepared.description.ordered_connections()
    layout = []
    for path, (node, _) in index.items():
        mount = prepared.description.resolve_mount(path, node.mount) if node.mount else None
        layout.append({"path": path, "native_key": metadata.keys.get(path), "kind": node.kind, "factory": node.factory,
                       "local_pose": asdict(node.pose), "parameters": node.parameters,
                       "mount": asdict(mount) if mount else None,
                       "scope": node.scope, "meshes": list(node.meshes), "prim_name": node.prim_name})
    task_evaluation = None
    if 'object_api' not in prepared.terms:
        from .legacy.task import evaluation_config
        from shared.backends.isaac.task_evaluation import validate_sources
        task_evaluation = validate_sources(evaluation_config(prepared.settings),
                                           prepared.terms.get("signals"), prepared.settings)
    return {"scene_config": str(prepared.path), "entities": list(index),
            "task_evaluation": task_evaluation,
            "control": prepared.settings.get("control"),
            "connections": len(joints), "observations": prepared.settings["observations"],
            "actions": list(prepared.settings["actions"]), "task": "task.py" if "object_api" in prepared.terms else prepared.settings.get("task", {}).get("preset", "custom"),
            "capabilities": {"backend": prepared.backend, "training": "RLinf PPO", "task": "Python" if "object_api" in prepared.terms else "legacy native terms",
                             "commands": ["check", "inspect", "view", "record", "smoke", "train", "evaluate"],
                             "validation": "static; runtime capabilities require smoke/evaluation"},
            "layout": layout, "physical_connections": [asdict(joint) for joint in joints],
            "layout_note": "Poses are authored relative to logical parents; sensor mounts use body-local poses. "
                           "Fixed attachments use parent_pose/child_pose. These are not measured world poses.",
            "enable_cameras": prepared.application["enable_cameras"],
            "checks": "asset names/arguments, composition references, task and observation selection",
            "limitations": "Native SDK configuration, tensor dimensions and physical behavior require smoke/view"}


class Project:
    """The entire project entry: scene.yaml, optional assets/terms, and policy.py.

    Projects bind ``register = project.register`` and ``check = project.check``;
    no project environment class or simulator loop is needed.
    """

    def __init__(self, source, package, *, task_id, model_type):
        self.scene_config = Path(source).resolve().parent / "scene.yaml"
        self.package, self.task_id, self.model_type = package, task_id, model_type

    def check(self, cfg):
        return check_project(cfg, scene_config=self.scene_config, project_package=self.package)

    def make_environment(self, parameters, num_envs, seed, horizon, **options):
        make_environment = backend_module(selected_backend(parameters), 'environment').make_environment
        return make_environment(parameters, num_envs, seed, horizon, scene_config=self.scene_config,
                                project_package=self.package, **options)

    def register(self):
        from rlinf.models import register_model
        from shared.integrations.rlinf.isaac import register_isaac_assembly_env
        object_api = (self.scene_config.parent / 'task.py').is_file()
        if object_api:
            factories, terms, _ = project_registry(self.package)
            refresh_observation_types(self.scene_config, self.package, factories=factories, terms=terms)
        policy = import_module(self.package + ".policy")
        from .layout import load_scene_config
        settings = load_scene_config(self.scene_config)
        if 'control' in settings:
            from .control import component_config, load_factory
            control = component_config(settings['control'], self.package)
            selected = control['components'][control['train']]
            factory = load_factory(selected['model'])
            def build_selected(config, torch_dtype):
                return factory(config=config, torch_dtype=torch_dtype, **selected.get('parameters', {}))
            build = build_selected
        else:
            build = policy.build_model
        if object_api:
            project_build = build
            def build_object(config, torch_dtype):
                from shared.integrations.rlinf.object_policy import ObjectPolicy
                schema = import_module(self.package + '.observation')
                model = project_build(config, torch_dtype)
                import torch
                if config.num_action_chunks != 1:
                    raise ValueError('The PPO object adapter currently supports one action chunk')
                shapes = config.get('observation_shapes')
                if not shapes:
                    raise ValueError('Inspect the scene before constructing the object policy')
                # Materialize project LazyLinear layers before FSDP/optimizers.
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    dtypes = config.get('observation_dtypes', {})
                    samples = {}
                    for key, shape in shapes.items():
                        dtype = getattr(torch, dtypes.get(key) or 'float32')
                        samples[key] = torch.zeros((1, *shape), dtype=torch_dtype if dtype.is_floating_point else dtype)
                    model(schema.Obs(samples))
                model.train(was_training)
                result = ObjectPolicy(model, schema.Obs, config.get('collect_gaussian_stats', False))
                checkpoint = config.get('initial_checkpoint')
                if checkpoint:
                    import torch
                    result.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
                return result
            build = build_object
        register_model(self.model_type, build, category="embodied", force=True)
        register_isaac_assembly_env(self.task_id, factory=self.make_environment)
