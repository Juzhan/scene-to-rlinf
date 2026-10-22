"""Native term selection and source-level preset checks, without launching Kit."""

import ast
from copy import deepcopy
from dataclasses import MISSING
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType, SimpleNamespace
import math
import sys

import pytest
import yaml

from shared.backends.isaac.task_config import build_task_config, validate_task_settings


ROOT = Path(__file__).resolve().parents[1]


def project(name):
    return yaml.safe_load((ROOT / 'projects' / name / 'scene.yaml').read_text())


def native_lift():
    """Isolate native term compilation from the separate control-chain compiler."""
    settings = project('assembly_lift')
    control = settings.pop('control')
    settings['actions'] = control['output']
    settings['observations'] = {'states': ['robot.joint_position', 'robot.joint_velocity',
        'object.position', 'object.target.position', 'ee_frame.frame_positions']}
    return settings


def test_project_task_selection_is_static_and_matches_policy_dimensions():
    lift = validate_task_settings(native_lift())
    from shared.assembly.control import compile_control
    compiled, registry = compile_control(project('assembly_go2'), {}, project_package='projects.assembly_go2')
    go2 = validate_task_settings(compiled, term_registry=registry)
    assert lift['observations']['states'] == [
        'robot.joint_position', 'robot.joint_velocity', 'object.position', 'object.target.position', 'ee_frame.frame_positions']
    # Authored attachments may add observations after the core robot inputs.
    assert go2['observations']['policy'][:9] == [
        'robot.linear_velocity', 'robot.angular_velocity', 'robot.orientation', 'simulation.gravity', 'planner.output',
        'robot.joint_position', 'robot.joint_velocity', 'height_scanner.position', 'height_scanner.ray_hit_positions']
    assert lift['actions'] == ['arm', 'gripper']
    assert go2['actions'] == ['control_chain']
    for name in ('assembly_lift', 'assembly_go2'):
        model = yaml.safe_load((ROOT / 'projects' / name / 'train.yaml').read_text())['actor']['model']
        assert model['obs_dim'] is None and model['action_dim'] == 'auto'


@pytest.mark.parametrize('mutate, message', [
    (lambda s: s['observations'].update(states=['joint_poz']), 'unknown terms'),
    (lambda s: s['observations'].update(states=['joint_pos', 'joint_pos']), 'repeated'),
    (lambda s: s['actions']['arm'].update(type='missing'), 'unknown action type'),
    (lambda s: s.setdefault('task', {}).update(rewards={'nonexistent': {'weight': 1}}), 'unknown term'),
    (lambda s: s['observations'].update(states={'terms':['joint_pos'], 'overrides':{'joint_vel':{}}}), 'selected'),
])
def test_static_typos_fail_before_simulator_start(mutate, message):
    settings = native_lift()
    mutate(settings)
    with pytest.raises(ValueError, match=message):
        validate_task_settings(settings)


def test_static_custom_extensions_are_not_invoked():
    def unopened():
        raise AssertionError('check must not start user factories')
    settings = {'observations': {'states':['custom']}, 'actions': {'move': {'type':'custom'}},
                'task': {'preset':'custom'}}
    report = validate_task_settings(settings, term_registry={'actions':{'custom':unopened}},
                                    task_presets={'custom':unopened})
    assert report['observations'] == {'states':['custom']}


def _package_source(package):
    spec = find_spec(package)
    if spec is None or spec.origin is None:
        pytest.skip(f'{package} source is not installed')
    root = Path(spec.origin).parent
    candidates = [root / 'source' / package / package, root]
    for candidate in candidates:
        if (candidate / ('utils' if package == 'isaaclab' else 'manager_based')).is_dir():
            return candidate
    pytest.skip(f'{package} source layout unavailable')


def _classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


@pytest.fixture
def sdk(monkeypatch):
    # Execute the installed configclass decorator and config declarations. No
    # simulator classes, managers, observation/reward functions are executed.
    source = _package_source('isaaclab')
    tasks = _package_source('isaaclab_tasks')
    config_module = ModuleType('_assembly_test_native_config')
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    config_tree = ast.parse((source / 'utils/configclass.py').read_text())
    config_tree.body = [node for node in config_tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    config_module.__dict__.update(class_to_dict=lambda value: vars(value), update_class_from_dict=lambda *a, **k: None)
    exec(compile(config_tree, str(source / 'utils/configclass.py'), 'exec'), config_module.__dict__)
    configclass = config_module.configclass
    terms = ModuleType('_assembly_test_native_terms')
    monkeypatch.setitem(sys.modules, terms.__name__, terms)
    terms.__dict__.update(configclass=configclass, MISSING=MISSING)
    _classes(source / 'managers/manager_term_cfg.py', {
        'ManagerTermBaseCfg', 'ObservationTermCfg', 'ObservationGroupCfg', 'RewardTermCfg',
        'TerminationTermCfg', 'EventTermCfg', 'CurriculumTermCfg'}, terms.__dict__)

    class Config(SimpleNamespace):
        def __init__(self, name=None, **kwargs):
            if name is not None: kwargs['name'] = name
            super().__init__(**kwargs)
    Config.Ranges = Config

    class Mdp:
        def __getattr__(self, name):
            if name.endswith('Cfg'): return Config
            def function(*args, **kwargs):
                raise AssertionError(f'Config compilation should not call MDP function {name}')
            function.__name__ = name
            return function

    presets = {}
    for preset, relative in {
        'lift':'manipulation/lift/lift_env_cfg.py',
        'velocity':'locomotion/velocity/velocity_env_cfg.py',
    }.items():
        module = ModuleType(f'_assembly_test_{preset}')
        monkeypatch.setitem(sys.modules, module.__name__, module)
        module.__dict__.update(configclass=configclass, MISSING=MISSING, math=math, mdp=Mdp(),
            SceneEntityCfg=Config, Unoise=Config,
            ObsGroup=terms.ObservationGroupCfg, ObsTerm=terms.ObservationTermCfg,
            RewTerm=terms.RewardTermCfg, DoneTerm=terms.TerminationTermCfg,
            EventTerm=terms.EventTermCfg, CurrTerm=terms.CurriculumTermCfg)
        _classes(tasks / 'manager_based' / relative,
                 {'CommandsCfg','ObservationsCfg','EventCfg','RewardsCfg','TerminationsCfg','CurriculumCfg'}, module.__dict__)
        presets[preset] = module
    managers = ModuleType('isaaclab.managers')
    managers.ObservationGroupCfg = terms.ObservationGroupCfg
    managers.ObservationTermCfg = terms.ObservationTermCfg
    managers.SceneEntityCfg = Config
    managers.EventTermCfg = terms.EventTermCfg
    managers.RewardTermCfg = terms.RewardTermCfg
    managers.TerminationTermCfg = terms.TerminationTermCfg
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', managers)
    mdp = ModuleType('isaaclab.envs.mdp')
    mdp.__getattr__ = Mdp().__getattr__
    envs = ModuleType('isaaclab.envs')
    envs.mdp = mdp
    monkeypatch.setitem(sys.modules, 'isaaclab.envs', envs)
    for name in ('JointPositionActionCfg','BinaryJointPositionActionCfg'):
        setattr(mdp, name, Config)
    monkeypatch.setitem(sys.modules, 'isaaclab.envs.mdp', mdp)
    return SimpleNamespace(presets=presets, terms=terms, Config=Config)


def test_lift_compiles_actual_sdk_terms_in_selected_order(sdk):
    settings = native_lift()
    snapshot = deepcopy(settings)
    groups = build_task_config(settings, task_presets=sdk.presets, term_registry={'object_api': {'module': 'projects.assembly_lift'}})
    obs = groups['observations']['object.position']
    assert isinstance(obs, sdk.terms.ObservationGroupCfg)
    names = [name for name, term in vars(obs).items() if isinstance(term, sdk.terms.ObservationTermCfg)]
    assert names == ['value']
    assert list(groups['observations']) == settings['observations']['states']
    assert groups['observations']['object.target.position'].value.params['value'] == [.5, 0, .3]
    assert vars(groups['commands']) == {}
    assert set(groups['rewards']) == {'project_reward'}
    assert set(groups['events']) == {'project_task_reset'}
    entities = {item['name']: item for item in settings['entities']}
    assert entities['spawn_area']['kind'] == 'abstract' and entities['spawn_area']['size'] == [.2, .5]
    assert entities['object']['reset']['region'] == 'world/spawn_area'
    assert set(groups['terminations']) == {'project_success', 'project_failure'}
    assert groups['curriculum'] == {}
    assert groups['actions'].arm.asset_name == 'robot'
    assert settings == snapshot


def test_go2_declares_physical_control_and_project_task():
    from shared.assembly.control import compile_control
    settings = project('assembly_go2')
    compiled, registry = compile_control(settings, {}, project_package='projects.assembly_go2')
    assert compiled['actions'] == {'control_chain': {'type': 'control_chain'}}
    go2 = [f'{leg}_{part}_joint' for part in ('hip', 'thigh', 'calf') for leg in ('FL', 'FR', 'RL', 'RR')]
    # Listed in the native articulation order, so the frozen walking weights keep their action order.
    assert settings['control']['output']['legs'] == {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': go2}
    assert settings['control']['train'] == 'executor'
    assert set(settings['control']['components']) == {'planner', 'executor'}
    assert 'task' not in settings and 'variables' not in settings
    assert (ROOT / 'projects/assembly_go2/task.py').is_file()


def test_python_task_adds_native_reward_and_reset_without_changing_information(sdk):
    settings = native_lift()
    groups = build_task_config(settings, term_registry={'object_api': {'module': 'projects.assembly_lift'}})
    reward = groups['rewards']['project_reward']
    assert reward.weight == 1.0
    assert reward.params['field'] == 'reward'
    assert groups['terminations']['project_success'].time_out is False
    assert groups['events']['project_task_reset'].mode == 'reset'
    assert 'target_distance' not in groups['observations']


def test_selection_custom_terms_and_prototypes_are_independent(sdk):
    prototype = sdk.terms.ObservationTermCfg(func=lambda env: None, params={'gain':1.0})
    reward = sdk.terms.RewardTermCfg(func=lambda env: None, weight=1.0, params={'gain':2.0})
    settings = {'observations':{'states':{'terms':['custom'], 'overrides':{'custom':{'scale':0.5}},
                                          'enable_corruption':False}},
                'actions':{'move':{'type':'local', 'scale':0.1}},
                'task':{'rewards':{'bonus':{'type':'local', 'weight':3.0}}}}
    registry = {'observations':{'custom':prototype}, 'actions':{'local':lambda **kw: sdk.Config(**kw)},
                'rewards':{'local':lambda: reward}}
    groups = build_task_config(settings, term_registry=registry)
    assert groups['observations']['states'].custom.scale == 0.5
    assert prototype.scale is None
    assert groups['rewards']['bonus'].weight == 3.0
    assert reward.weight == 1.0
    groups['observations']['states'].custom.params['gain'] = 5.0
    assert prototype.params == {'gain':1.0}
    assert groups['actions'].move.scale == 0.1


def _sdk_manager_constructor(name):
    """Run the installed constructors, with term instantiation isolated from Kit."""
    import torch
    source = _package_source('isaaclab')
    filename = 'command_manager.py' if name == 'CommandManager' else 'action_manager.py'
    path = source / 'managers' / filename
    tree = ast.parse(path.read_text())
    declaration = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    methods = {'__init__', 'total_action_dim', 'action_term_dim'}
    declaration.body = [node for node in declaration.body
                        if isinstance(node, ast.FunctionDef) and node.name in methods]

    class ManagerBase:
        def __init__(self, cfg, env):
            self.cfg = deepcopy(cfg)
            self.num_envs, self.device = env.num_envs, env.device
            terms = self.cfg.items() if isinstance(self.cfg, dict) else vars(self.cfg).items()
            self._terms = {
                key: SimpleNamespace(cfg=SimpleNamespace(debug_vis=getattr(term, 'debug_vis', False)), action_dim=1)
                for key, term in terms if term is not None
            }

    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              declaration], type_ignores=[])
    namespace = {'ManagerBase':ManagerBase, 'torch':torch}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('project_name', ['assembly_lift'])
@pytest.mark.parametrize('category, manager_name', [('commands','CommandManager'), ('actions','ActionManager')])
def test_compiled_configs_work_in_real_sdk_manager_constructors(sdk, project_name, category, manager_name):
    settings = native_lift()
    groups = build_task_config(settings, task_presets=sdk.presets, term_registry={'object_api': {'module': 'projects.assembly_lift'}})
    manager_type = _sdk_manager_constructor(manager_name)
    environment = SimpleNamespace(num_envs=2, device='cpu')
    manager = manager_type(groups[category], environment)
    assert isinstance(manager.cfg.debug_vis, bool)
    assert set(manager._terms) == set(vars(groups[category]))
    assert 'debug_vis' not in vars(groups[category])
    if category == 'actions':
        assert manager._action.shape == (2, len(settings['actions']))
    # This is the real constructor's regression: a plain dict gets past the
    # manager base but fails when the SDK assigns its debug-visibility flag.
    if vars(groups[category]):
        with pytest.raises(AttributeError, match='debug_vis'):
            manager_type(vars(groups[category]), environment)
