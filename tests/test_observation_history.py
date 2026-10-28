"""Shared observation windows: per-env reset, padding, order, offline == online, interface id."""
from copy import deepcopy
from importlib import import_module, util
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid

import pytest
import torch
import yaml

from shared.assembly.history import HistoryWindow, history_windows
from shared.assembly.policy_runtime import ComponentInput, FeedbackChain

ROOT = Path(__file__).resolve().parents[1]


def frame(*values):
    return torch.tensor(values, dtype=torch.float32).view(-1, 1)


def test_first_frame_pads_and_frames_are_oldest_to_newest():
    window = HistoryWindow(torch.zeros(1, 1), 3)
    window.push(frame(1.))
    assert window.buffer.view(-1).tolist() == [1., 1., 1.]
    assert window.valid.tolist() == [[False, False, True]]
    window.push(frame(2.))
    window.push(frame(3.))
    window.push(frame(4.))
    assert window.buffer.view(-1).tolist() == [2., 3., 4.]
    assert window.valid.all()


def test_reset_is_per_environment_and_never_crosses_episodes():
    window = HistoryWindow(torch.zeros(3, 2), 3)
    for value in (1., 2., 3.):
        window.push(torch.full((3, 2), value))
    window.reset(torch.tensor([1]))
    assert not window.buffer[1].any() and window.count.tolist() == [3, 0, 3]
    window.push(torch.full((3, 2), 9.))
    assert window.buffer[1].eq(9.).all()                      # new episode: only its own frame
    assert window.buffer[0, :, 0].tolist() == [2., 3., 9.]    # others keep their history
    assert window.valid.tolist() == [[True] * 3, [False, False, True], [True] * 3]


def test_masked_push_advances_only_selected_rows_for_any_dtype():
    window = HistoryWindow(torch.zeros(2, 4, 4, 3, dtype=torch.uint8), 2)
    image = torch.arange(2 * 4 * 4 * 3, dtype=torch.uint8).view(2, 4, 4, 3)
    window.push(image, torch.tensor([True, False]))
    assert window.buffer.shape == (2, 2, 4, 4, 3) and window.buffer.dtype == torch.uint8
    assert torch.equal(window.buffer[0, 1], image[0]) and not window.buffer[1].any()
    assert window.count.tolist() == [1, 0]


def test_matches_native_isaac_circular_buffer():
    path = Path('/home/jz/repos/RoboDojo/third_party/IsaacLab/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py')
    if not path.exists():
        pytest.skip('Isaac Lab source not available')
    spec = util.spec_from_file_location('_isaac_circular_buffer', path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.CircularBuffer(4, 3, 'cpu')
    ours = HistoryWindow(torch.zeros(3, 2), 4)
    generator = torch.Generator().manual_seed(0)
    for step in range(12):
        if step in (5, 9):
            ids = [step % 3]
            native.reset(ids)
            ours.reset(torch.tensor(ids))
        value = torch.randn(3, 2, generator=generator)
        native.append(value)
        ours.push(value)
        torch.testing.assert_close(ours.buffer, native.buffer)


def test_offline_windows_restart_at_episode_and_missing_steps():
    frames = torch.arange(7.).view(-1, 1)
    windows, valid = history_windows(frames, 3, episode=[0, 0, 0, 1, 1, 1, 1], step=[0, 1, 2, 0, 1, 3, 4])
    assert windows[2].view(-1).tolist() == [0., 1., 2.]
    assert windows[3].view(-1).tolist() == [3., 3., 3.]      # new episode
    assert windows[5].view(-1).tolist() == [5., 5., 5.]      # step 2 missing: no stitching
    assert windows[6].view(-1).tolist() == [5., 5., 6.]
    assert valid[6].tolist() == [False, True, True]
    with pytest.raises(ValueError):
        history_windows(frames, 3, step=[0, 1])


# ---- runtime wiring through the shared component input / feedback chain ----

class Recorder(torch.nn.Module):
    def __init__(self, dimension, seen):
        super().__init__()
        self.dimension, self.seen = dimension, seen

    def forward(self, obs):
        self.seen.append(obs.feedback.clone())
        return torch.zeros(obs.feedback.shape[0], self.dimension)


def make_chain(selected, history, num_envs=2):
    env = SimpleNamespace(num_envs=num_envs, device='cpu', common_step_counter=0, feedback=torch.zeros(num_envs, 1))
    seen = {'planner': [], 'executor': []}
    control = {'train': selected, 'components': {
        'planner': {'dimension': 3, 'period': .2, 'model': 'planner', 'observations': ['feedback'],
                    'history': {'feedback': history}},
        'executor': {'dimension': 'auto', 'period': .02, 'model': 'executor', 'decode': 'decode',
                     'observations': ['feedback', 'planner.output'], 'history': {'feedback': history}}}}
    def factory(name):
        if name == 'decode':
            return lambda action, obs: {}
        return lambda **_: Recorder(3, seen[name])
    def inputs(config, env, links, settings):
        return ComponentInput(config, env, links, settings, bind_reader=lambda path: lambda: env.feedback)
    chain = FeedbackChain(control, .005, env, 3, {}, input_factory=inputs, model_factory=factory)
    return chain, env, seen


def env_step(chain, env, *, reset_ids=None, terminal=False):
    """Isaac/mjlab order: process -> decimation advances -> (terminal snapshot) -> reset -> compute."""
    chain.process(torch.zeros_like(chain.outputs[chain.selected]))
    for _ in range(chain.periods[chain.selected]):
        chain.advance()
    env.common_step_counter += 1
    env.feedback += 1
    terminal_obs = None
    if terminal:
        state = chain.snapshot_state()
        terminal_obs = chain.observation('feedback').clone()
        chain.restore_state(state)
    if reset_ids is not None:
        chain.reset(reset_ids)
        env.feedback[reset_ids] = -10 * (env.common_step_counter)
    first = chain.observation('feedback').clone()
    again = chain.observation('feedback').clone()           # repeated compute in one step
    torch.testing.assert_close(first, again)
    return first, terminal_obs


def test_trained_window_advances_once_per_decision_and_resets_per_env():
    chain, env, _ = make_chain('executor', 3)
    chain.reset()
    assert chain.observation('feedback')[:, :, 0].tolist() == [[0.] * 3] * 2
    env_step(chain, env)
    obs, _ = env_step(chain, env, reset_ids=torch.tensor([1]))
    assert obs[0, :, 0].tolist() == [0., 1., 2.]
    assert obs[1, :, 0].tolist() == [-20.] * 3                # fresh episode, no leak from env 0 or old episode
    obs, _ = env_step(chain, env)
    assert obs[:, :, 0].tolist() == [[1., 2., 3.], [-20., -20., -19.]]
    # Generated input links (executor.input.feedback) see the same stored window.
    assert chain.read('executor', 'input')['feedback'] is chain.readers['executor'].last['feedback']


def test_terminal_snapshot_shows_terminal_frame_without_advancing_live_window():
    chain, env, _ = make_chain('executor', 3)
    chain.reset()
    chain.observation('feedback')
    env_step(chain, env)
    before = chain.readers['executor'].windows['feedback'].count.clone()
    obs, terminal = env_step(chain, env, reset_ids=torch.tensor([0]), terminal=True)
    assert terminal[:, :, 0].tolist() == [[0., 1., 2.], [0., 1., 2.]]
    assert obs[1, :, 0].tolist() == [0., 1., 2.]              # live window: one push, not two
    assert before.tolist() == [2, 2]


def test_explicit_partial_reset_compute_only_pushes_reset_rows():
    chain, env, _ = make_chain('executor', 3)
    chain.reset()
    chain.observation('feedback')
    env_step(chain, env)
    chain.reset(torch.tensor([1]))                            # e.g. RLinf explicit reset of env 1 only
    env.feedback[1] = 7.
    obs = chain.observation('feedback')
    assert obs[:, :, 0].tolist() == [[0., 0., 1.], [7., 7., 7.]]


def test_fixed_components_advance_windows_at_their_own_period():
    chain, env, seen = make_chain('executor', 2)
    chain.reset()
    for _ in range(25):
        chain.observation('feedback')
        env_step(chain, env)
    # planner period / executor period = 10 decisions; frames are 10 executor steps apart.
    assert len(seen['planner']) == 3
    assert seen['planner'][1][0, :, 0].tolist() == [0., 10.]
    assert seen['planner'][2][0, :, 0].tolist() == [10., 20.]

    chain, env, seen = make_chain('planner', 2)
    chain.reset()
    chain.observation('feedback')
    for _ in range(2):
        env_step(chain, env)
    # executor runs every 4 physics steps: 10 pushes per 40-step planner decision.
    assert len(seen['executor']) == 20
    assert seen['executor'][10][0, :, 0].tolist() == [0., 1.] and seen['executor'][-1][0, :, 0].tolist() == [1., 1.]


def test_offline_windows_equal_online_windows():
    chain, env, _ = make_chain('executor', 4, num_envs=3)
    chain.reset()
    online = {0: [], 1: [], 2: []}
    records = {0: [], 1: [], 2: []}
    episode = [0, 0, 0]
    obs = chain.observation('feedback').clone()
    resets = {3: [0], 5: [1, 2], 9: [0]}
    for step in range(14):
        for env_id in range(3):
            online[env_id].append(obs[env_id])
            records[env_id].append((episode[env_id], env.feedback[env_id].clone()))
        ids = resets.get(step)
        if ids:
            for env_id in ids:
                episode[env_id] += 1
        obs, _ = env_step(chain, env, reset_ids=None if ids is None else torch.tensor(ids))
    for env_id in range(3):
        episodes = [e for e, _ in records[env_id]]
        frames = torch.stack([f for _, f in records[env_id]])
        windows, valid = history_windows(frames, 4, episode=episodes)
        torch.testing.assert_close(windows, torch.stack(online[env_id]))
        window = chain.readers['executor'].windows['feedback']
        assert valid[-1].sum() == min(window.count[env_id], 4)


def test_history_is_validated_at_check():
    from shared.assembly.control import component_config
    base = yaml.safe_load((ROOT / 'projects/planner_executor/scene.yaml').read_text())['control']
    good = deepcopy(base)
    good['components']['executor']['history'] = {'robot.joint_position': 3}
    assert component_config(good, 'projects.planner_executor')['components']['executor']['history'] == {'robot.joint_position': 3}
    for value in ([3], {'robot.not_selected': 3}, {'robot.joint_position': 1}, {'robot.joint_position': True},
                  {'robot.joint_position': 2.0}, {'robot.joint_position': '3'}):
        bad = deepcopy(base)
        bad['components']['executor']['history'] = value
        with pytest.raises(ValueError, match='history'):
            component_config(bad, 'projects.planner_executor')


def test_generated_types_show_the_time_axis():
    from shared.assembly.object_schema import generate
    scene = ROOT / 'projects/planner_executor/scene.yaml'
    settings = yaml.safe_load(scene.read_text())
    plain = generate(scene, settings=deepcopy(settings), write=False)
    settings['control']['components']['executor']['history'] = {'robot.joint_position': 3}
    text = generate(scene, settings=settings, write=False)
    executor = text[text.index('class ExecutorObs_robot('):]
    line = next(l for l in executor.splitlines() if l.strip().startswith('joint_position:'))
    assert "['B', 3, " in line and '旧→新' in line
    planner = text[text.index('class SceneInfo_robot('):]
    raw = next(l for l in planner.splitlines() if l.strip().startswith('joint_position:'))
    assert "['B', 3" not in raw and '旧→新' not in raw        # SceneInfo keeps the raw reading
    assert '旧→新' not in plain


def test_deselecting_an_observation_drops_its_window():
    from shared.assembly.editing import SceneDocument
    document = SceneDocument(ROOT / 'projects/planner_executor/scene.yaml')
    component = document.data['control']['components']['executor']
    component['history'] = {'robot.joint_position': 3, 'robot.joint_velocity': 2}
    document.set_observations('executor', ['robot.joint_position', 'planner.output'])
    assert component['history'] == {'robot.joint_position': 3}
    document.set_observations('executor', ['planner.output'])
    assert 'history' not in component


def test_removing_an_entity_drops_the_windows_of_its_observations():
    from shared.assembly.editing import SceneDocument
    document = SceneDocument(ROOT / 'projects/planner_executor/scene.yaml')
    planner = document.data['control']['components']['planner']
    planner['observations'].append('simulation.gravity')
    planner['history'] = {'destination.position': 3, 'simulation.gravity': 2}
    document.remove_entities(['world/destination'])
    assert planner['history'] == {'simulation.gravity': 2}


@pytest.fixture
def signature_project(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    name = 'history_probe_' + uuid.uuid4().hex[:8]
    root = tmp_path / name
    (root / 'assemblies').mkdir(parents=True)
    (root / '__init__.py').write_text(
        "from shared.assembly.project import Project\n"
        "project = Project(__file__, __name__, task_id='history-test', model_type='history-policy')\n"
        "def register():\n    pass\n")
    (root / 'policy.py').write_text('SCALE = 1.0\n')
    (root / 'assemblies/robot.yaml').write_text('schema_version: 1\nroot:\n  name: robot\n  kind: robot\n  factory: go2\n')
    scene = {'schema_version': 1, 'name': 'world',
             'entities': [{'name': 'robot', 'assembly': 'assemblies/robot.yaml', 'pose': {'position': [0, 0, .4]}}],
             'control': {'train': 'executor', 'components': {
                 'planner': {'model': '.policy:build_planner', 'dimension': 3, 'period': .2, 'observations': ['robot.position']},
                 'executor': {'model': '.policy:build_executor', 'dimension': 'auto', 'period': .02,
                              'observations': ['robot.joint_position'], 'decode': '.policy:decode_action',
                              'history': {'robot.joint_position': 3}}},
                 'output': {'legs': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['FL_hip_joint']}}}}
    OmegaConf.save(OmegaConf.create(scene), root / 'scene.yaml')
    monkeypatch.syspath_prepend(str(tmp_path))
    module = import_module(name)
    cfg = OmegaConf.create({
        'runner': {'only_eval': False, 'resume_dir': None, 'logger': {'log_path': str(tmp_path / 'outputs'), 'experiment_name': 'probe'}},
        'env': {'train': {'init_params': {'id': 'history-test'}}, 'eval': '${env.train}'},
        'actor': {'model': {'model_type': 'history-policy'}}})
    yield root, cfg, module
    for loaded in list(sys.modules):
        if loaded == name or loaded.startswith(name + '.'):
            sys.modules.pop(loaded, None)


def test_changed_history_length_changes_the_interface_identifier(signature_project):
    from omegaconf import OmegaConf
    from shared.experiments import project_signature
    root, cfg, module = signature_project
    before = project_signature(cfg, module)['interface_sha256']
    scene = OmegaConf.load(root / 'scene.yaml')
    scene.control.components.executor.history['robot.joint_position'] = 4
    OmegaConf.save(scene, root / 'scene.yaml')
    longer = project_signature(cfg, module)['interface_sha256']
    del scene.control.components.executor['history']
    OmegaConf.save(scene, root / 'scene.yaml')
    without = project_signature(cfg, module)['interface_sha256']
    assert len({before, longer, without}) == 3
