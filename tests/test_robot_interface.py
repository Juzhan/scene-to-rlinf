"""Shared robot interface: frame offsets, Jacobian, damped IK step and where methods are available."""
import math
from types import SimpleNamespace
import pytest
import torch
from shared.assembly.objects import ObjectView
from shared.assembly.robot import Kinematics, RobotView, frame_observations, orientation_error, provider, robot_frames


def yaw(angle):
    return torch.stack((torch.cos(angle / 2), 0 * angle, 0 * angle, torch.sin(angle / 2)), -1)


class PlanarArm(Kinematics):
    """Two unit links turning about +Z, plus a passive finger joint that moves no frame."""
    def __init__(self, q, frames):
        super().__init__('arm', frames)
        self.q = q

    def joint_names(self): return ('joint1', 'joint2', 'finger')
    def joint_position(self): return self.q
    def body_names(self): return ('base', 'link1', 'link2')

    def body_pose(self, body):
        q1, q2 = self.q[:, 0], self.q[:, 1]
        zero = torch.zeros_like(q1)
        if body == 'base':
            return torch.zeros(len(q1), 3), yaw(zero)
        if body == 'link1':
            return torch.zeros(len(q1), 3), yaw(q1)
        return torch.stack((torch.cos(q1), torch.sin(q1), zero), -1), yaw(q1 + q2)

    def body_jacobian(self, body):
        position, _ = self.body_pose(body)
        value = torch.zeros(len(self.q), 6, 3)
        moving = {'base': 0, 'link1': 1, 'link2': 2}[body]
        for joint in range(moving):
            axis_point = torch.zeros(3) if joint == 0 else self.body_pose('link2')[0]
            value[:, :3, joint] = torch.linalg.cross(torch.tensor([0., 0., 1.]).expand_as(position), position - axis_point)
            value[:, 5, joint] = 1
        return value


FRAMES = {'tool': ('link2', ((1., 0., 0.), (1., 0., 0., 0.)))}


def test_offset_frame_jacobian_matches_finite_differences():
    q = torch.tensor([[.3, .7, .01], [-1.1, .4, .02]])
    arm = PlanarArm(q, FRAMES)
    position, orientation = arm.frame_pose('tool')
    expected = torch.stack((torch.cos(q[:, 0]) + torch.cos(q[:, 0] + q[:, 1]),
                            torch.sin(q[:, 0]) + torch.sin(q[:, 0] + q[:, 1]), torch.zeros(2)), -1)
    torch.testing.assert_close(position, expected)
    torch.testing.assert_close(orientation, yaw(q[:, 0] + q[:, 1]))
    jacobian = arm.jacobian('tool')
    for joint in range(3):
        shifted = q.clone()
        shifted[:, joint] += 1e-3
        moved = PlanarArm(shifted, FRAMES).frame_pose('tool')[0]
        torch.testing.assert_close(jacobian[:, :3, joint], (moved - position) / 1e-3, atol=3e-3, rtol=0)
    assert torch.all(jacobian[:, :, 2] == 0)


def test_differential_ik_converges_and_keeps_off_chain_joints():
    q = torch.tensor([[.3, .7, .01], [-1.1, .4, .02]])
    target = torch.tensor([[1.2, .9, 0.], [.5, -1.4, 0.]])
    for _ in range(80):
        q = PlanarArm(q, FRAMES).differential_ik('tool', target, damping=.01, max_step=.1)
    torch.testing.assert_close(PlanarArm(q, FRAMES).frame_pose('tool')[0], target, atol=1e-4, rtol=0)
    torch.testing.assert_close(q[:, 2], torch.tensor([.01, .02]))
    step = PlanarArm(torch.zeros(2, 3), FRAMES).differential_ik('tool', target, max_step=.05)
    assert step.abs().max() <= .05 + 1e-6
    # Orientation and position together: reach a pose that fixes both joint angles.
    goal = torch.tensor([[.4, -.6, 0.]])
    pose = PlanarArm(torch.tensor([[.4, -.6, 0.]]), FRAMES).frame_pose('tool')
    q = torch.zeros(1, 3)
    for _ in range(80):
        q = PlanarArm(q, FRAMES).differential_ik('tool', *pose, damping=.01, max_step=.1)
    torch.testing.assert_close(q, goal, atol=1e-4, rtol=0)


def test_orientation_error_is_rotation_vector():
    error = orientation_error(yaw(torch.tensor([.5, -3.])), yaw(torch.tensor([.2, 3.])))
    torch.testing.assert_close(error, torch.tensor([[0, 0, .3], [0, 0, 2 * math.pi - 6]]), atol=1e-5, rtol=0)


def test_robot_methods_only_where_a_backend_provides_them():
    class Obs_arm(RobotView):
        joint_position: torch.Tensor
    class Obs(ObjectView):
        arm: Obs_arm
    q = torch.tensor([[.3, .7, .01]])
    trained = Obs({'arm.joint_position': q})
    assert trained.arm.joint_position is q
    with pytest.raises(RuntimeError, match='trained component'):
        trained.arm.frame_pose('tool')
    fixed = Obs({'arm.joint_position': q}, robots={'arm': PlanarArm(q, FRAMES)}.__getitem__)
    assert fixed.arm.joint_names() == ('joint1', 'joint2', 'finger')
    assert fixed.arm.differential_ik('tool', fixed.arm.frame_pose('tool')[0]).shape == q.shape
    with pytest.raises(ValueError, match='unknown frame'):
        fixed.arm.jacobian('camera')


def test_scene_frames_from_composed_scene_and_cached_provider(tmp_path):
    from shared.assembly.layout import world_spec_from_config
    (tmp_path / 'arm.yaml').write_text(
        'schema_version: 1\nroot:\n  name: arm\n  kind: robot\n  factory: franka_panda\n'
        '  frames:\n    - name: tcp\n      body: panda_hand\n      pose: {position: [0, 0, 0.1]}\n')
    settings = {'name': 'world', 'entities': [
        {'name': 'piper', 'kind': 'robot', 'factory': 'piper_x',
         'frames': [{'name': 'palm', 'body': 'link6', 'pose': {'position': [0, 0, .1]}}]},
        {'name': 'tcp_frame', 'kind': 'sensor', 'factory': 'frame_transformer',
         'parameters': {'target_body': 'gripper_base', 'target_position': [0, 0, .13]},
         'mount': {'entity': 'world/piper', 'body': 'gripper_base'}},
        {'name': 'panda', 'assembly': 'arm.yaml'},
        {'name': 'cube', 'factory': 'dex_cube'}]}
    frames = robot_frames(world_spec_from_config(settings, directory=tmp_path))
    # Asset defaults (tcp) plus entity frames; an entity frame replaces a default of the same name.
    # A frame_transformer sensor stays a sensor reading, not a robot frame.
    assert frames == {'piper': {'tcp': ('gripper_base', ((0., 0., .13), (1., 0., 0., 0.))),
                                'palm': ('link6', ((0, 0, .1), (1., 0., 0., 0.)))},
                      'panda': {'tcp': ('panda_hand', ((0, 0, .1), (1., 0., 0., 0.)))}}
    sources = frame_observations(frames)
    assert sources['piper.tcp.position'] == ('piper', 'tcp', 'position')
    assert set(sources) == {f'{robot}.{frame}.{field}' for robot, frame in
                            [('piper', 'tcp'), ('piper', 'palm'), ('panda', 'tcp')] for field in ('position', 'orientation')}
    for name in ('jacobian', 'joint_position'):
        clash = {'name': 'world', 'entities': [{'name': 'piper', 'kind': 'robot', 'factory': 'piper_x', 'frames': [{'name': name}]}]}
        with pytest.raises(ValueError, match='clashes'):
            robot_frames(world_spec_from_config(clash, directory=tmp_path))
    env = SimpleNamespace()
    made = []
    robots = provider(env, {'_robot_frames': frames}, lambda env, name, frames: made.append(name) or object())
    assert robots('piper') is robots('piper') and made == ['piper']
    with pytest.raises(ValueError, match='not a robot'):
        robots('cube')
    with pytest.raises(ValueError, match='prepare_scene'):
        provider(env, settings, object)('piper')


def test_fixed_component_gets_generated_view_and_robot_methods():
    import sys
    from shared.assembly.policy_runtime import ComponentInput
    class ExecutorObs_arm(RobotView):
        joint_position: torch.Tensor
    class ExecutorObs(ObjectView):
        arm: ExecutorObs_arm
    module = SimpleNamespace(ExecutorObs=ExecutorObs)
    sys.modules['robot_test_observation'] = module
    try:
        q = torch.tensor([[.3, .7, .01]])
        robots = {'arm': PlanarArm(q, FRAMES)}.__getitem__
        reader = ComponentInput({'observations': ['arm.joint_position'], '_view': 'robot_test_observation:ExecutorObs'},
                                None, {}, {}, bind_reader=lambda path: lambda: q, robots=robots)
        obs = reader()
        assert type(obs) is ExecutorObs and isinstance(obs.arm, ExecutorObs_arm)
        assert obs.arm.frame_pose('tool')[0].shape == (1, 3)
    finally:
        del sys.modules['robot_test_observation']


def test_generated_robot_nodes_inherit_the_interface(tmp_path):
    from tools.create import create_project
    target, _ = create_project('robot_example', destination=tmp_path)
    tree = {}
    exec(compile((target / 'observation.py').read_text(), 'observation.py', 'exec'), tree)
    assert issubclass(tree['ExecutorObs_robot'], RobotView)
    assert issubclass(tree['SceneInfo_robot'], RobotView)
    assert not issubclass(tree['SceneInfo_executor_input_robot'], RobotView)


def test_named_frame_is_an_observation_read_through_the_robot_interface():
    from shared.assembly.readers import bind_common
    q = torch.tensor([[.3, .7, .01], [0., .2, .02]])
    arm = PlanarArm(q, FRAMES)
    settings = {'_robot_frames': {'arm': FRAMES}}
    env = SimpleNamespace(device='cpu', num_envs=2)
    def physical(path):
        raise AssertionError(path)
    position = bind_common(env, settings, 'arm.tool.position', bind_physical=physical, bind_sampled_target=None,
                           robots={'arm': arm}.__getitem__)
    orientation = bind_common(env, settings, 'arm.tool.orientation', bind_physical=physical, bind_sampled_target=None,
                              robots={'arm': arm}.__getitem__)
    expected = arm.frame_pose('tool')
    assert torch.equal(position(), expected[0]) and torch.equal(orientation(), expected[1])
