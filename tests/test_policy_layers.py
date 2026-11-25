"""scene.yaml policy layers: reading them, what policy.py says a layer returns, editing them, and running them."""
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from shared.assembly.layers import control_from_policy, layer_tokens, output_structure, type_name
from shared.assembly.objects import ObjectView
from shared.assembly.policy_runtime import FeedbackChain
from shared.policy import flat, mlp

POLICY_SOURCE = '''
from torch import Tensor, nn


class Upper(nn.Module):
    """Where to go"""
    def forward(self, obs) -> dict:
        return {'goal': obs.goal,          # [B, 3] m, environment frame
                'picture': obs.camera,     # [B, 64, 64, 3] a camera view
                'flag': obs.flag}


def build_upper() -> Upper:
    return Upper()


class Lower(nn.Module):
    def forward(self, obs) -> Tensor:
        ...
'''


def two_layers(**changes):
    policy = {'train': 'lower', 'decode': '.policy:decode_action', 'layers': {
        'upper': {'model': '.policy:Upper', 'period': .2, 'inputs': ['destination.position']},
        'lower': {'model': '.policy:Lower', 'period': .02, 'inputs': ['robot.joint_position', 'upper', 'lower'],
                  'history': {'robot.joint_position': 3, 'lower': 2}}}}
    policy.update(changes)
    return policy


def test_one_layer_needs_no_train_and_reads_its_own_previous_action():
    control = control_from_policy({'decode': '.policy:decode_action', 'layers': {
        'policy': {'model': '.policy:Policy', 'period': .02, 'inputs': ['robot.joint_position', 'policy']}}})
    assert control['train'] == control['driver'] == 'policy' and control['layers'] is True
    assert control['components']['policy']['observations'] == ['robot.joint_position', 'policy.output']
    assert control['components']['policy']['decode'] == '.policy:decode_action'


def test_two_layers_read_the_parts_policy_py_declares(tmp_path):
    (tmp_path / 'policy.py').write_text(POLICY_SOURCE)
    control = control_from_policy(two_layers(), tmp_path)
    lower = control['components']['lower']
    assert lower['observations'] == ['robot.joint_position', 'upper.goal', 'upper.picture', 'upper.flag', 'lower.output']
    assert lower['history'] == {'robot.joint_position': 3, 'lower.output': 2}
    assert 'decode' in lower and 'decode' not in control['components']['upper']
    found = control['structure']['upper']
    assert found['shapes'] == {'goal': [3], 'picture': [64, 64, 3]} and 'flag' not in found['shapes']
    assert found['notes']['goal'] == 'm, environment frame'
    assert control['structure']['lower'] == {'tensor': True}
    # A factory that names the class it returns is followed to it.
    policy = two_layers()
    policy['layers']['upper']['model'] = '.policy:build_upper'
    assert output_structure(policy, tmp_path)['upper']['parts'] == ['goal', 'picture', 'flag']
    # The editor writes those parts back as the layer's name.
    assert layer_tokens(lower['observations'], policy['layers']) == ['robot.joint_position', 'upper', 'lower']


@pytest.mark.parametrize('change, message', [
    ({'train': None}, 'policy.train'),
    ({'decode': None}, 'policy.decode'),
    ({'layers': {}}, 'one layer'),
    ({'extra': 1}, 'unknown fields'),
])
def test_wrong_policies_say_what_is_missing(change, message):
    policy = {key: value for key, value in two_layers(**change).items() if value is not None}
    with pytest.raises(ValueError, match=message):
        control_from_policy(policy)


def test_layer_type_names_are_python_class_names():
    assert type_name('planner') == 'Planner' and type_name('low_level') == 'LowLevel'


def test_policy_helpers():
    assert flat(torch.zeros(2, 3), torch.zeros(2, 4, 5)).shape == (2, 23)
    assert mlp(8, 3)(torch.zeros(2, 5)).shape == (2, 3)


class Input:
    """A layer's inputs: the scene value ``feedback`` and the layer outputs it links to."""
    def __init__(self, config, env, links, settings):
        self.env, self.links = env, links
        self.sources = [source for source in config['observations']]
        self.last = {source: self._value(source).clone() for source in self.sources}

    def _value(self, source):
        return self.links[source]() if source in self.links else self.env.feedback

    def __call__(self, rows=None):
        for source in self.sources:
            self.last[source] = self._value(source).clone()
        return ObjectView(self.last)

    def observe(self, source):
        return self._value(source)

    def mark(self, ids=None): pass
    def state(self): return {k: v.clone() for k, v in self.last.items()}
    def restore(self, state): self.last.update(state)
    def reset(self, ids): pass


class Upper(nn.Module):
    def forward(self, obs):
        return {'goal': obs.robot.feedback[:, :2] * 10, 'scale': torch.ones_like(obs.robot.feedback[:, :1])}


class Lower(nn.Module):
    """Reads its own previous action; how many it returns is found by trial."""
    def __init__(self):
        super().__init__()
        self.net = mlp(4, 5)

    def forward(self, obs):
        return self.net(flat(obs.robot.feedback, obs.upper.goal, obs.upper.scale, obs.lower.output))


def chain(monkeypatch=None):
    models = {'Upper': Upper, 'Lower': Lower, 'decode': lambda action, obs: action}
    control = {'train': 'lower', 'driver': 'lower', 'layers': True, 'components': {
        'upper': {'model': 'Upper', 'period': .2, 'dimension': None, 'observations': ['robot.feedback']},
        'lower': {'model': 'Lower', 'period': .02, 'dimension': None, 'decode': 'decode',
                  'observations': ['robot.feedback', 'upper.goal', 'upper.scale', 'lower.output']}}}
    env = SimpleNamespace(num_envs=2, device='cpu', common_step_counter=0, feedback=torch.ones(2, 3))
    return FeedbackChain(control, .005, env, 7, {}, input_factory=Input, model_factory=models.get), env


def test_structured_results_are_measured_and_the_trial_leaves_random_draws_alone():
    torch.manual_seed(3)
    expected = torch.rand(3)
    torch.manual_seed(3)
    built, env = chain()
    assert torch.equal(torch.rand(3), expected)  # trial steps restore the random state
    assert built.dimension == 5 and built.outputs['lower'].shape == (2, 5)
    assert set(built.outputs['upper']) == {'goal', 'scale'} and built.outputs['upper']['goal'].shape == (2, 2)
    env.feedback.fill_(2.)
    built.observation('robot.feedback')  # the fixed upper layer refreshes before the trained one observes
    torch.testing.assert_close(built.read('upper', 'goal'), torch.full((2, 2), 20.))
    with pytest.raises(ValueError, match='no part'):
        built.read('upper', 'output')
    state = built.snapshot_state()
    env.feedback.fill_(5.)
    env.common_step_counter = 40
    built.observation('robot.feedback')
    built.restore_state(state)
    torch.testing.assert_close(built.read('upper', 'goal'), torch.full((2, 2), 20.))
    built.reset(torch.tensor([0]))
    assert not built.outputs['upper']['goal'][0].any() and built.outputs['upper']['goal'][1].any()


def test_the_layer_driving_the_robots_returns_a_tensor():
    models = {'Upper': Upper, 'decode': lambda action, obs: action}
    control = {'train': 'upper', 'driver': 'upper', 'layers': True, 'components': {
        'upper': {'model': 'Upper', 'period': .02, 'dimension': None, 'observations': ['robot.feedback'], 'decode': 'decode'}}}
    env = SimpleNamespace(num_envs=2, device='cpu', common_step_counter=0, feedback=torch.ones(2, 3))
    with pytest.raises(TypeError, match='one Tensor'):
        FeedbackChain(control, .005, env, 7, {}, input_factory=Input, model_factory=models.get)


def test_split_and_merge_edit_only_the_policy(tmp_path):
    from shared.assembly.editing import SceneDocument
    (tmp_path / 'scene.yaml').write_text('''schema_version: 1
name: world
entities:
  - name: robot
    factory: go2
policy:
  layers:
    policy:
      model: .policy:Policy
      period: 0.02
      inputs:
        - robot.joint_position
        - policy
  decode: .policy:decode_action
''')
    document = SceneDocument(tmp_path / 'scene.yaml')
    with pytest.raises(ValueError, match='重名'):
        document.add_layer('robot', model='.policy:Plan', period=.2, inputs=['robot.position'])
    document.add_layer('plan', model='.policy:Plan', period=.2, inputs=['robot.position'])
    policy = document.data['policy']
    assert list(policy['layers']) == ['plan', 'policy'] and policy['train'] == 'policy'
    assert list(policy['layers']['policy']['inputs']) == ['robot.joint_position', 'policy', 'plan']
    document.set_observations('policy', ['robot.joint_position', 'plan.goal', 'plan.heading'])
    assert list(policy['layers']['policy']['inputs']) == ['robot.joint_position', 'plan']
    document.remove_layer('plan')
    assert list(policy['layers']) == ['policy'] and 'train' not in policy
    assert list(policy['layers']['policy']['inputs']) == ['robot.joint_position']


def test_the_adapter_adds_action_noise_and_a_value_network():
    pytest.importorskip('rlinf')
    from shared.integrations.rlinf.object_policy import ObjectPolicy

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = mlp(16, 4)

        def forward(self, obs):
            return self.net(flat(obs.robot.joint_position))
    adapter = ObjectPolicy(Policy(), ObjectView, action_dim=4, hidden=(8,), log_std=-1.)
    distribution, value = adapter._evaluate({'robot.joint_position': torch.zeros(3, 6)})
    assert distribution.mean.shape == (3, 4) and value.shape == (3, 1)
    torch.testing.assert_close(distribution.stddev, torch.full((3, 4), torch.e ** -1))
    assert any(name.startswith('value_head.') for name, _ in adapter.named_parameters())  # RLinf's value_lr
    with pytest.raises(ValueError, match='images'):
        adapter._evaluate({'robot.joint_position': torch.zeros(3, 6), 'camera.rgb': torch.zeros(3, 4, 4, 3, dtype=torch.uint8)})


def test_blank_template_is_a_scene_to_fill_in(tmp_path):
    import yaml
    from tools.create import MAINTAINED_TEMPLATES, create_project
    assert MAINTAINED_TEMPLATES[0] == 'blank'
    target, files = create_project('empty_start', 'blank', destination=tmp_path)
    scene = yaml.safe_load((target / 'scene.yaml').read_text())
    assert [entity['factory'] for entity in scene['entities']] == ['ground_plane', 'clear_sky']
    assert scene['policy']['layers']['policy']['inputs'] == []
    assert 'NotImplementedError' in (target / 'policy.py').read_text()
    assert (target / 'observation.py').is_file() and 'empty_start' in (target / 'README.md').read_text()
    assert control_from_policy(scene['policy'])['components']['policy']['observations'] == []  # filled in the editor


def test_models_asking_for_config_get_the_recipe_model_section():
    seen = {}

    class Sized(nn.Module):
        def __init__(self, config):
            super().__init__()
            seen['hidden'] = list(config.hidden_dims)
            self.net = mlp(*config.hidden_dims, 3)

        def forward(self, obs):
            return self.net(flat(obs.robot.feedback, obs.sized.output))
    control = {'train': 'sized', 'driver': 'sized', 'layers': True, 'components': {
        'sized': {'model': 'Sized', 'period': .02, 'dimension': None, 'decode': 'decode',
                  'observations': ['robot.feedback', 'sized.output']}}}
    env = SimpleNamespace(num_envs=2, device='cpu', common_step_counter=0, feedback=torch.ones(2, 3))
    built = FeedbackChain(control, .005, env, 3, {'_model_config': {'hidden_dims': [8, 4], 'own_setting': 1}},
                          input_factory=Input, model_factory={'Sized': Sized, 'decode': lambda a, o: a}.get)
    assert built.dimension == 3 and seen['hidden'] == [8, 4]
