"""Named controls preserve training-action semantics and batch/device boundaries."""
from types import SimpleNamespace
import pytest
import torch
from shared.assembly.objects import ObjectView
from shared.backends.isaac.control import validate_targets
from projects.assembly_lift.policy import decode_action


def test_lift_keeps_eight_actions_and_binary_gripper_semantics():
    action = torch.zeros(3, 8)
    action[:, 7] = torch.tensor([-1., 0., 1.])
    action[:, :7] = .4
    default = torch.arange(9).float().expand(3, -1)
    targets = decode_action(action, ObjectView({'robot.default_joint_position': default}))
    assert set(targets) == {'arm', 'gripper'}
    torch.testing.assert_close(targets['arm'], default[:, :7] + .2)
    torch.testing.assert_close(targets['gripper'], torch.tensor([[0., 0.], [.04, .04], [.04, .04]]))
    validate_targets(targets, {'arm': SimpleNamespace(action_dim=7), 'gripper': SimpleNamespace(action_dim=2)}, 3, action.device)


@pytest.mark.parametrize('bad', ['missing', 'extra', 'shape', 'integer', 'nan'])
def test_named_target_contract_rejects_bad_results(bad):
    targets = {'arm': torch.zeros(2, 7), 'gripper': torch.zeros(2, 2)}
    if bad == 'missing': targets.pop('arm')
    if bad == 'extra': targets['typo'] = torch.zeros(2, 1)
    if bad == 'shape': targets['arm'] = torch.zeros(2, 8)
    if bad == 'integer': targets['arm'] = torch.zeros(2, 7, dtype=torch.int64)
    if bad == 'nan': targets['arm'][0, 0] = float('nan')
    with pytest.raises((ValueError, RuntimeError)):
        validate_targets(targets, {'arm': SimpleNamespace(action_dim=7), 'gripper': SimpleNamespace(action_dim=2)}, 2, torch.device('cpu'))


def test_simple_executor_named_binding_does_not_copy():
    from projects.assembly_go2.policy import decode_action
    action = torch.randn(2, 12)
    assert decode_action(action, None)['legs'] is action
