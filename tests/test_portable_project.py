from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from shared.assembly.project import prepare_scene, require_online_backend

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'projects/dual_backend_scene/scene.yaml'
PACKAGE = 'projects.dual_backend_scene'


def test_same_project_definition_and_model_for_both_backends():
    prepared = [prepare_scene({'backend': b}, scene_config=SCENE, project_package=PACKAGE)
                for b in ('isaac', 'mjlab')]
    for section in ('control', 'observations', 'actions'):
        assert prepared[0].settings[section] == prepared[1].settings[section]
    assert prepared[0].terms['object_api'] == prepared[1].terms['object_api']
    assert prepared[0].backend == 'isaac' and prepared[1].backend == 'mjlab'
    require_online_backend({'backend': 'mjlab'})
    with pytest.raises(ValueError, match='Unknown'):
        require_online_backend({'backend': 'missing'})


def test_reader_reference_frames_and_lazy_task_share_contract():
    from shared.backends.mjlab.readers import bind_reader
    from shared.backends.mjlab.python_task import Runtime
    data = SimpleNamespace(root_link_pos_w=torch.tensor([[9., 0., .42], [19., 0., .30]]),
        root_com_lin_vel_w=torch.ones(2,3), root_link_lin_vel_w=torch.full((2,3), 99.),
        joint_pos=torch.zeros(2,12))
    class Scene(dict):
        env_origins = torch.tensor([[10.,0,0],[20.,0,0]])
    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=data)), num_envs=2,
        device='cpu', common_step_counter=0, extras={})
    prepared = prepare_scene({'backend':'mjlab'},scene_config=SCENE,project_package=PACKAGE)
    position = bind_reader(env, prepared.settings, 'robot.position')()
    torch.testing.assert_close(position[:,0], torch.tensor([-1.,-1.]))
    assert bind_reader(env, prepared.settings, 'robot.joint_position')() is data.joint_pos
    assert bind_reader(env, prepared.settings, 'robot.linear_velocity')() is data.root_com_lin_vel_w
    task = Runtime(env, {'module':PACKAGE}, prepared.settings)
    result = task.compute()
    torch.testing.assert_close(result['reward'], torch.tensor([-.07,-.05]))
    assert task.compute() is result
    assert set(task.signals.describe()) == {'robot.position', 'robot.target.position'}


def test_mjlab_native_terms_build_without_simulation():
    pytest.importorskip('mjlab')
    from shared.backends.mjlab.python_task import attach_task
    prepared = prepare_scene({'backend':'mjlab'},scene_config=SCENE,project_package=PACKAGE)
    action = prepared.terms['actions']['control_chain']()
    assert action.entity_name == 'robot'
    assert action.control['train'] == 'executor'
    observation = prepared.terms['observations']['robot.joint_position']()
    assert observation.params == {'source':'robot.joint_position'}
    output = {'rewards':{}, 'terminations':{}, 'events':{}}
    attach_task(output, {'module':PACKAGE}, prepared.settings)
    assert output['rewards']['project_reward'].weight == 1.
    assert output['events']['project_task_reset'].mode == 'reset'
    assert not output['terminations']['project_success'].time_out


def test_native_control_rejects_wrong_actuator_mode_and_executes_shared_chain():
    pytest.importorskip('mjlab')
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.actuator import XmlMotorActuatorCfg
    from shared.backends.mjlab.assets import go2
    from shared.backends.mjlab.control import Endpoint
    cfg = go2()
    cfg.articulation = EntityArticulationInfoCfg(actuators=(XmlMotorActuatorCfg(target_names_expr=('.*',)),))
    robot = cfg.build()
    env = SimpleNamespace(scene={'robot':robot},device='cpu')
    binding = {'asset_name':'robot','joint_names':['.*_joint'],'type':'joint_effort'}
    endpoint = Endpoint(binding,env)
    assert endpoint.action_dim == 12
    with pytest.raises(ValueError, match='native actuators'):
        Endpoint({**binding,'type':'joint_position'},env)
    writes = {}
    proxy = SimpleNamespace(actuators=robot.actuators, find_joints=robot.find_joints,
        data=SimpleNamespace(joint_pos=torch.zeros(2,12), joint_vel=torch.zeros(2,12),
                             root_link_quat_w=torch.tensor([[1.,0,0,0]]).expand(2,-1)),
        set_joint_effort_target=lambda value, joint_ids: writes.update(value=value.clone(), ids=joint_ids))
    env = SimpleNamespace(scene={'robot':proxy}, device='cpu', num_envs=2, common_step_counter=0)
    prepared = prepare_scene({'backend':'mjlab'},scene_config=SCENE,project_package=PACKAGE)
    action = prepared.terms['actions']['control_chain']().build(env)
    assert action.action_dim == 12
    action.chain.observation('robot.joint_position')
    action.process_actions(torch.ones(2,12))
    action.apply_actions()
    torch.testing.assert_close(writes['value'], 10.*torch.tanh(torch.ones(2,12)))
    action.reset(torch.tensor([0]))
    assert not action.raw_action[0].any()
    assert action.raw_action[1].all()


def test_command_line_train_backend_also_moves_evaluation():
    from omegaconf import OmegaConf
    from shared.assembly.project import follow_train_backend
    recipe = lambda: OmegaConf.create({'env': {'train': {'init_params': {'backend': 'mjlab'}},
                                               'eval': {'init_params': {'backend': 'isaac'}}}})
    assert follow_train_backend(recipe(), ['env.train.init_params.backend=mjlab']).env.eval.init_params.backend == 'mjlab'
    explicit = ['env.train.init_params.backend=mjlab', 'env.eval.init_params.backend=isaac']
    assert follow_train_backend(recipe(), explicit).env.eval.init_params.backend == 'isaac'
    assert follow_train_backend(recipe(), []).env.eval.init_params.backend == 'isaac'
    # An evaluation backend that already differed from training stays where it was.
    separate = follow_train_backend(recipe(), ['env.train.init_params.backend=mjlab'], previous='mjlab')
    assert separate.env.eval.init_params.backend == 'isaac'


def test_mjlab_backend_selects_its_own_interpreter(monkeypatch, tmp_path):
    import sys
    import shared.assembly.project as project
    import pytest
    assert project.backend_interpreter('isaac') == sys.executable
    monkeypatch.setattr(project.util, 'find_spec', lambda name: None)
    monkeypatch.delenv('RLINF_PYTHON', raising=False)
    expected = project.Path(project.__file__).resolve().parents[2] / '.venv-mjlab/bin/python'
    if expected.is_file():
        assert project.backend_interpreter('mjlab') == str(expected)
    # An interpreter the user named explicitly is never silently replaced.
    monkeypatch.setenv('RLINF_PYTHON', sys.executable)
    with pytest.raises(RuntimeError, match='no mjlab'):
        project.backend_interpreter('mjlab')
