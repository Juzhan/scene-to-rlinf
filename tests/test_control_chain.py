from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from shared.assembly.control import compile_control
from shared.backends.isaac.control import FeedbackChain

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def local_rlinf(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'third_party/RLinf'))


def settings(project='assembly_go2_navigation'):
    return yaml.safe_load((ROOT / 'projects' / project / 'scene.yaml').read_text())


def test_boundary_compilation_is_static_and_does_not_mutate_source():
    original = settings()
    before = deepcopy(original)
    result, registry = compile_control(original, {}, project_package='projects.assembly_go2_navigation')
    assert original == before
    assert result['world']['decimation'] == 40
    assert list(result['actions']) == ['control_chain']
    assert result['observations'] == {'policy': original['control']['components']['planner']['observations']}
    assert callable(registry['actions']['control_chain'])
    assert 'task' not in result


@pytest.mark.parametrize('change', ['frequency', 'selected', 'conflict', 'dimension', 'future_input'])
def test_invalid_boundaries_fail_before_simulation(change):
    cfg = settings()
    if change == 'frequency': cfg['control']['components']['executor']['period'] = .03
    if change == 'selected': cfg['control']['train'] = 'unknown'
    if change == 'conflict': cfg['world']['decimation'] = 4
    if change == 'dimension': cfg['control']['components']['planner']['dimension'] = 0
    if change == 'future_input': cfg['control']['components']['planner']['observations'] = ['executor.output']
    with pytest.raises(ValueError):
        compile_control(cfg, {}, project_package='projects.assembly_go2_navigation')


def fake_chain(monkeypatch, selected):
    import shared.backends.isaac.control as module
    calls = []
    class Input:
        def __init__(self, cfg, env, links, settings):
            self.name = cfg['label']
            self.env, self.links = env, links
            self.dimension = 3
            self.last = {'feedback': torch.zeros(2, 3)}
        def __call__(self, rows=None):
            calls.append(self.name)
            value = self.env.feedback if self.name == 'planner' else self.links['planner.output']() + self.env.feedback
            self.last['feedback'].copy_(value)
            return self.last['feedback']
        def read(self, source):
            return self()
        observe = read
        def mark(self, ids=None): pass
        def state(self): return {k: v.clone() for k, v in self.last.items()}
        def restore(self, state):
            for k, v in state.items(): self.last[k].copy_(v)
        def reset(self, ids):
            self.last['feedback'][slice(None) if ids is None else ids] = 0
    monkeypatch.setattr(module, 'ComponentInput', Input)
    monkeypatch.setattr(module, 'load_factory', lambda _: lambda **kw: torch.nn.Identity())
    control = {'train': selected, 'components': {
        'planner': {'label': 'planner', 'dimension': 3, 'period': .2, 'model': 'test'},
        'executor': {'label': 'executor', 'dimension': 'auto', 'period': .02, 'model': 'test', 'decode': 'test'}}}
    env = SimpleNamespace(num_envs=2, device='cpu', common_step_counter=0, feedback=torch.zeros(2, 3))
    return FeedbackChain(control, .005, env, 3, {}), env, calls


def test_planner_training_uses_fresh_downstream_feedback_and_holds_targets(monkeypatch):
    chain, env, calls = fake_chain(monkeypatch, 'planner')
    chain.process(torch.ones(2, 3))
    torch.testing.assert_close(chain.advance(), torch.ones(2, 3))
    env.feedback.fill_(2.)
    for _ in range(3):
        torch.testing.assert_close(chain.advance(), torch.ones(2, 3))
    torch.testing.assert_close(chain.advance(), torch.full((2, 3), 3.))
    for _ in range(35): chain.advance()
    assert calls.count('executor') == 10
    assert all(not p.requires_grad for model in chain.models.values() for p in model.parameters())
    old = chain.outputs['executor'][1].clone()
    chain.reset(torch.tensor([0]))
    assert chain.tick == 40
    assert not chain.outputs['executor'][0].any()
    torch.testing.assert_close(chain.outputs['executor'][1], old)


def test_executor_training_prepares_planner_at_its_frequency_and_resets_per_env(monkeypatch):
    chain, env, calls = fake_chain(monkeypatch, 'executor')
    env.feedback.fill_(1.)
    torch.testing.assert_close(chain.observation('feedback'), torch.full((2, 3), 2.))
    env.feedback.fill_(2.)
    env.common_step_counter = 1
    torch.testing.assert_close(chain.observation('feedback'), torch.full((2, 3), 3.))
    assert calls.count('planner') == 1
    chain.reset(torch.tensor([0]))
    chain.observation('feedback')
    torch.testing.assert_close(chain.outputs['planner'], torch.tensor([[2., 2., 2.], [1., 1., 1.]]))
    env.common_step_counter = 10
    state = chain.snapshot_state()
    chain.observation('feedback')  # terminal observation must not consume live component memory
    chain.restore_state(state)
    torch.testing.assert_close(chain.planner_due, state[-1])
    torch.testing.assert_close(chain.outputs['planner'], state[0]['planner'])
    chain.process(torch.randn(2, 3))
    torch.testing.assert_close(chain.advance(), chain.outputs['executor'])


def test_project_models_and_configuration_have_no_simulator_glue():
    for name in ('assembly_go2_navigation', 'planner_executor'):
        project = ROOT / 'projects' / name
        assert not (project / 'controllers.py').exists()
        assert not (project / 'terms.py').exists()
        assert not any('isaaclab' in p.read_text() for p in project.glob('*.py'))
        from shared.assembly.layout import load_scene_config
        cfg = load_scene_config(project / 'scene.yaml')  # a policy: is read as the internal control mapping
        for component in cfg['control']['components'].values():
            assert component['model'].startswith('.policy:')


def test_executor_auto_dimension_is_resolved_from_native_control(monkeypatch):
    chain, env, _ = fake_chain(monkeypatch, 'executor')
    assert chain.dimension == 3
    with pytest.raises(ValueError): chain.process(torch.zeros(2, 4))


def test_generated_project_uses_new_roles_without_legacy_files(tmp_path):
    from tools.create import create_project
    target, files = create_project('example', destination=tmp_path)
    cfg = yaml.safe_load((target / 'scene.yaml').read_text())
    # The default template is one policy layer; its size comes from the model, not scene.yaml.
    assert list(cfg['policy']['layers']) == ['policy'] and 'control' not in cfg
    assert cfg['policy']['layers']['policy']['model'] == '.policy:Policy'
    assert 'dimension' not in cfg['policy']['layers']['policy']
    assert set(files) == {'README.md', '__init__.py', 'policy.py', 'scene.yaml', 'train.yaml', 'task.py', 'observation.py'}
    assert 'env.py' not in files
    assert 'example__go2_goal_policy' in (target / 'train.yaml').read_text()
    assert not (target / 'controllers.py').exists()


def test_computed_information_and_component_interfaces_are_discoverable():
    from shared.assembly.catalog import describe_scene
    # Public catalog is shared with the editor; names are not hard-coded in UI.
    catalog = describe_scene(ROOT / 'projects/planner_executor/scene.yaml')
    entries = {item['id']: item for item in catalog['observations']}
    # planner returns a dict: its entries are the planner's outputs; the trained executor's output is its action.
    assert {'planner.goal', 'planner.heading', 'executor.output', 'destination.position', 'destination.orientation'} <= entries.keys()
    assert 'distance' not in entries
    assert entries['destination.position']['available']


def test_component_observation_edit_only_selects_basic_information():
    from shared.assembly.editing import SceneDocument
    document = SceneDocument(ROOT / 'projects/assembly_go2_navigation/scene.yaml')
    document.set_observations('executor', ['planner.output', 'robot.joint_position'])
    component = document.data['control']['components']['executor']
    assert component['observations'] == ['planner.output', 'robot.joint_position']
    assert 'input_processing' not in component
    assert 'observations' not in document.data


@pytest.mark.parametrize('old_input', ['processing', 'variable'])
def test_model_processing_is_not_a_scene_configuration(old_input):
    cfg = settings()
    component = cfg['control']['components']['planner']
    if old_input == 'processing':
        component['input_processing'] = {}
    else:
        component['observations'] = ['goal_heading_error']
    with pytest.raises(ValueError):
        compile_control(cfg, {}, project_package='projects.assembly_go2_navigation')


def test_physical_output_reset_only_restores_selected_environments(monkeypatch):
    chain, _, _ = fake_chain(monkeypatch, 'executor')
    chain.initial_output = torch.full_like(chain.outputs['executor'], .7)
    chain.outputs['executor'].fill_(2.)
    chain.previous['executor'].fill_(3.)
    chain.reset(torch.tensor([0]))
    torch.testing.assert_close(chain.outputs['executor'][0], chain.initial_output[0])
    torch.testing.assert_close(chain.previous['executor'][0], chain.initial_output[0])
    assert (chain.outputs['executor'][1] == 2).all()
    assert (chain.previous['executor'][1] == 3).all()


def test_control_endpoint_rejects_hidden_scaling():
    from shared.assembly.control import component_config
    from shared.assembly.layout import load_scene_config
    cfg = load_scene_config(ROOT / 'projects/planner_executor/scene.yaml')['control']
    cfg['output'] = {'legs': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['FL_hip_joint'],
                              'scale': .25}}
    with pytest.raises(ValueError, match='scaling in policy'):
        component_config(cfg, 'projects.planner_executor')


def test_component_without_observations_is_named_at_check():
    from shared.assembly.control import component_config
    from shared.assembly.layout import load_scene_config
    cfg = load_scene_config(ROOT / 'projects/planner_executor/scene.yaml')['control']
    cfg['components']['planner']['observations'] = []
    with pytest.raises(ValueError, match='planner 尚未选择输入'):
        component_config(cfg, 'projects.planner_executor')


def test_without_control_output_every_robot_is_driven_by_name():
    """go2_piper_wholebody: the dog and the arm are robots; the cube and the terrain stay passive."""
    from shared.assembly.control import automatic_outputs, effective_outputs
    from shared.assembly.layout import load_world_spec
    scene = ROOT / 'projects/go2_piper_wholebody/scene.yaml'
    world = load_world_spec(scene)
    from shared.assembly.layout import load_scene_config
    scene_settings = load_scene_config(scene)  # its policy: read as the internal control mapping
    outputs = effective_outputs(scene_settings['control'], world)
    assert outputs == {name: {'type': 'joint_position', 'asset_name': name, 'joint_names': ['.*']}
                       for name in ('robot', 'piperx')}
    assert list(automatic_outputs(world, {'piperx': 'none', 'robot': 'joint_velocity'})) == ['robot']
    assert automatic_outputs(world, {'robot': 'joint_effort'})['robot']['type'] == 'joint_effort'
    with pytest.raises(ValueError, match='not robots: dex_cube'):
        automatic_outputs(world, {'dex_cube': 'none'})
    with pytest.raises(ValueError, match='maps robot names'):
        automatic_outputs(world, {'robot': 'joint_speed'})
    with pytest.raises(ValueError, match='modes applies'):
        effective_outputs({'output': outputs, 'modes': {'robot': 'none'}}, world)
    result, _ = compile_control(scene_settings, {}, project_package='projects.go2_piper_wholebody',
                                backend='isaaclab3', world=world)
    assert result['control']['output'] == outputs and 'modes' not in result['control']
