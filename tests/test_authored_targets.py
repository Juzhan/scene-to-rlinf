from shared.assembly.legacy.task import evaluation_config
"""Scene-owned targets have no dependency on native task presets or commands."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from shared.assembly.project import prepare_scene
from shared.backends.isaac.entity_observations import parse_signal
from shared.backends.isaac.targets import target_catalog, read_target
from shared.backends.isaac.task_evaluation import evaluate_term

ROOT = Path(__file__).resolve().parents[1]


def test_same_target_is_read_by_observation_and_task_for_each_environment():
    settings = {'targets': {'robot': {'linear_velocity': [.5, 0, 0]}}}
    targets = target_catalog(settings)
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0, extras={})
    path = 'robot.target.linear_velocity'
    config = {'metrics': {'target': {'signal': path}}}
    evaluate_term(env, config, 'success', targets=targets)
    observation = read_target(env, targets[path][0])
    torch.testing.assert_close(env.extras['task_metrics']['target']['value'], observation)
    assert observation.shape == (2, 3)
    assert targets[path][1].frame == 'environment'


@pytest.mark.parametrize('value', [[1, 2], [1, 2, float('nan')], [True, 0, 0], 'sample'])
def test_targets_reject_ambiguous_or_invalid_values(value):
    with pytest.raises(ValueError, match='three finite numbers'):
        target_catalog({'targets': {'robot': {'position': value}}})


def test_target_must_belong_to_scene_entity(tmp_path):
    settings = {'schema_version': 1, 'name': 'world', 'entities': [{'name': 'robot', 'factory': 'go2'}],
                'targets': {'missing_robot': {'position': [1, 0, 0]}},
                'observations': {'states': ['robot.position']},
                'actions': {'move': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['.*']}}}
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match='unknown scene entity'):
        prepare_scene({}, scene_config=path)


def test_nested_property_and_target_are_distinct():
    assert parse_signal('robot.arm.joint_position') == ('robot__arm', 'joint_position')
    assert parse_signal('robot.target.position') is None


@pytest.mark.parametrize('name', ['assembly_lift', 'assembly_go2'])
def test_new_templates_declare_only_evaluation_and_explicit_targets(name):
    settings = yaml.safe_load((ROOT / 'projects' / name / 'scene.yaml').read_text())
    assert 'task' not in settings and 'variables' not in settings
    assert target_catalog(settings)
    from shared.backends.isaac.task_evaluation import validate_sources
    report = validate_sources(evaluation_config(settings), settings=settings)
    assert not any(path.startswith(('command__', 'entity__')) for path in report['signals'])
    assert not report['untyped_signals']


NAVIGATION = {'name': 'world', 'entities': [
    {'name': 'arena', 'kind': 'abstract', 'shape': 'rectangle', 'size': [6, 6], 'pose': {'position': [0, 0, .4]}},
    {'name': 'destination', 'kind': 'abstract', 'shape': 'point',
     'reset': {'region': 'world/arena', 'yaw_range': [-3.14, 3.14], 'interval': 8}}]}


def test_dynamic_navigation_target_changes_are_visible_to_task_and_observation():
    targets = target_catalog(NAVIGATION)
    command = SimpleNamespace(command=torch.tensor([[1., 2., 0., 1., 0., 0., 0.], [3., 4., 0., 0., 0., 0., 1.]]))
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.2, common_step_counter=1, extras={},
                          command_manager=SimpleNamespace(get_term=lambda name: command))
    path = 'destination.position'
    config = {'metrics': {'goal': {'expr': path}}}
    evaluate_term(env, config, 'success', targets=targets)
    torch.testing.assert_close(env.extras['task_metrics']['goal']['value'], command.command[:, :3])
    command.command = command.command * 2  # replacement, not merely in-place mutation
    env.common_step_counter += 1
    evaluate_term(env, config, 'success', targets=targets)
    torch.testing.assert_close(env.extras['task_metrics']['goal']['value'], read_target(env, targets[path][0]))
    torch.testing.assert_close(read_target(env, targets['destination.orientation'][0]), command.command[:, 3:])


def test_navigation_formula_preserves_original_reward_and_contact_failure():
    from projects.assembly_go2_navigation.task import evaluate
    from projects.assembly_go2_navigation.observation import SceneInfo
    data = {
        'destination.position': torch.tensor([[.3, .4, 0.], [3., 4., 0.]]),
        'destination.orientation': yaw_quaternion(torch.tensor([.2, -.3])),
        'contact_forces.contact_force_samples': torch.tensor([[[[0., 0., 0.]]], [[[2., 0., 0.]]]]),
        'robot.position': torch.zeros(2, 3),
        'robot.orientation': torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        'planner.output': torch.tensor([[.1, .2, .3], [.4, .5, .6]]),
        'planner.previous_output': torch.zeros(2, 3),
    }
    data.update(navigation_primitives(2))
    result = evaluate(SceneInfo(data))
    distance = torch.tensor([.5, 5.])
    expected = .5 * (1 - torch.tanh(distance / 2)) + .5 * (1 - torch.tanh(distance / .2))
    expected -= .2 * torch.tensor([.2, .3]) + 400 * torch.tensor([0., 1.])
    expected -= .01 * data['planner.output'].square().sum(dim=1)
    torch.testing.assert_close(result['reward'], expected)
    assert result['failure'].tolist() == [False, True]
    assert 'success' not in result  # reaching remains a metric, not a new early termination


def test_navigation_project_has_no_custom_terms_or_native_task_configuration():
    from shared.backends.isaac.task_evaluation import validate_sources
    root = ROOT / 'projects/assembly_go2_navigation'
    settings = yaml.safe_load((root / 'scene.yaml').read_text())
    assert not (root / 'terms.py').exists()
    assert 'task' not in settings and 'variables' not in settings
    evaluation = evaluation_config(settings)
    report = validate_sources(evaluation, settings=settings)
    assert not any(path.startswith('command__') for path in report['signals'])
    assert not report['untyped_signals']


def test_abstract_entities_publish_geometry_and_validate_visualization():
    settings = {**NAVIGATION, 'visualization': {'abstract': ['destination']}}
    catalog = target_catalog(settings)
    assert set(catalog) == {'arena.position', 'arena.orientation', 'arena.size',
                            'destination.position', 'destination.orientation'}
    assert catalog['arena.size'][0] == [6., 6.]
    with pytest.raises(ValueError, match='abstract entity name list'):
        target_catalog({**NAVIGATION, 'visualization': {'goals': ['destination']}})
    with pytest.raises(ValueError, match='existing abstract'):
        target_catalog({**NAVIGATION, 'visualization': {'abstract': ['missing']}})


def test_explicit_relative_geometry_preserves_heading_frame_and_angle_wrap():
    import math
    from shared.policy_math import relative_goal
    yaw = torch.tensor([math.pi / 2, math.pi - .1, 0., 0.])
    q = torch.stack((torch.cos(yaw / 2), torch.zeros(4), torch.zeros(4), torch.sin(yaw / 2)), dim=1)
    relative, error = relative_goal(torch.zeros(4, 3), q,
        torch.tensor([[1., 0., 0.]]).expand(4, -1),
        torch.tensor([0., -math.pi + .1, math.pi, -math.pi]))
    torch.testing.assert_close(relative[0], torch.tensor([0., -1., 0.]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(error[:, 0], torch.tensor([-math.pi / 2, .2, math.pi, -math.pi]), atol=1e-6, rtol=1e-6)


def test_abstract_command_resamples_only_selected_rows_inside_its_area(monkeypatch):
    import sys
    from shared.backends.isaac.abstract_terms import _config_type
    class Term:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self.num_envs, self.device = env.num_envs, env.device
    class Cfg:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', SimpleNamespace(CommandTerm=Term, CommandTermCfg=Cfg))
    monkeypatch.setitem(sys.modules, 'isaaclab.utils', SimpleNamespace(configclass=lambda cls: cls))
    _config_type.cache_clear()
    try:
        area = {'shape': 'rectangle', 'size': [2., 4.], 'pose': {'position': [0., 0., .4], 'orientation': [1., 0., 0., 0.]}}
        entity = {'shape': 'point', 'size': [], 'mount': None,
                  'pose': {'position': [2., 3., .4], 'orientation': [1., 0., 0., 0.]},
                  'reset': {'region': 'world/arena', 'yaw_range': [.5, .5], 'inset': 0.}}
        cfg = _config_type()(entity=entity, areas={'world/arena': area})
        term = cfg.class_type(cfg, SimpleNamespace(num_envs=64, device='cpu'))
        torch.testing.assert_close(term.command[0], torch.tensor([2., 3., .4, 1., 0., 0., 0.]))
        term._resample_command(list(range(1, 64)))
        torch.testing.assert_close(term.command[0], torch.tensor([2., 3., .4, 1., 0., 0., 0.]))
        torch.testing.assert_close(term.command[1:, 2], torch.full((63,), .4))
        assert term.command[1:, 0].abs().max() <= 1. and term.command[1:, 1].abs().max() <= 2.
        assert term.command[1:, :2].std(0).min() > .2
        torch.testing.assert_close(term.command[1:, 6], torch.full((63,), math.sin(.25)))
    finally:
        _config_type.cache_clear()


def yaw_quaternion(angle):
    zero = torch.zeros_like(angle)
    return torch.stack((torch.cos(angle / 2), zero, zero, torch.sin(angle / 2)), dim=-1)


def navigation_primitives(n):
    return {'robot.linear_velocity': torch.zeros(n,3), 'robot.angular_velocity': torch.zeros(n,3),
            'simulation.gravity': torch.tensor([[0.,0.,-9.81]]).expand(n,-1),
            'robot.joint_position': torch.zeros(n,12), 'robot.joint_velocity': torch.zeros(n,12),
            'robot.default_joint_position': torch.zeros(n,12), 'robot.default_joint_velocity': torch.zeros(n,12),
            'height_scanner.position': torch.zeros(n,3), 'height_scanner.ray_hit_positions': torch.zeros(n,187,3)}
