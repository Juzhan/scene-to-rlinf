"""One scope, one history and one sampling clock across policy and Task."""
from types import SimpleNamespace
import pytest
import torch
from shared.assembly.legacy.task import TaskEvaluator, evaluation_config
from shared.assembly.signals import Signal, Signals
from shared.backends.isaac.derived import read_variable
from shared.backends.isaac.task_evaluation import evaluate_term, reset_evaluation


def test_consumers_share_runtime_and_history_and_partial_reset():
    data = torch.tensor([1., 3.])
    calls = []
    settings = {
        'variables': {
            'measurement': {'source': 'measurement_source', 'history': 4},
            'average': {'expr': 'history_mean(measurement)'},
            'running': {'expr': 'running_mean(measurement)'},
        },
        'task': {'reward': '-average', 'metrics': {'average': 'average'}},
    }
    factories = {'measurement_source': lambda env: Signal(lambda: (calls.append(1), data)[1])}
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.1, common_step_counter=0, extras={})
    torch.testing.assert_close(read_variable(env, settings, 'average', factories), data)
    runtime = env._assembly_task_evaluator
    result = evaluate_term(env, evaluation_config(settings), 'reward', 'total', signal_factories=factories)
    torch.testing.assert_close(result, -data)
    assert calls == [1]
    for _ in range(10):
        read_variable(env, settings, 'average', factories)
    assert calls == [1]
    assert runtime.histories['measurement'].cursor.tolist() == [1, 1]
    data[:] = torch.tensor([3., 5.])
    env.common_step_counter = 1
    torch.testing.assert_close(read_variable(env, settings, 'average', factories), torch.tensor([2., 4.]))
    evaluate_term(env, evaluation_config(settings), 'reward', 'total', signal_factories=factories)
    reset_evaluation(env, [0])
    data[:] = torch.tensor([9., 50.])
    torch.testing.assert_close(read_variable(env, settings, 'average', factories), torch.tensor([9., 4.]))
    torch.testing.assert_close(read_variable(env, settings, 'running', factories), torch.tensor([9., 4.]))
    assert runtime.histories['measurement'].cursor.tolist() == [1, 2]
    assert not hasattr(env, '_assembly_variables')


def test_observation_first_and_task_first_have_same_sampling():
    settings = {'variables': {'value': {'expr': 'robot.position[:, 0]', 'history': 3},
                              'mean': {'expr': 'history_mean(value)'}},
                'task': {'reward': '-mean'}}
    values = torch.tensor([[1., 0., 0.], [3., 0., 0.]])
    signals = Signals(); signals.register('robot.position', Signal(lambda: values))
    engines = [TaskEvaluator(evaluation_config(settings), signals, num_envs=2) for _ in range(2)]
    for step in range(4):
        values[:, 0] += 1
        first = engines[0].read_variable('mean', dt=.1, step=step)
        a = engines[0].evaluate(dt=.1, step=step)
        b = engines[1].evaluate(dt=.1, step=step)
        second = engines[1].read_variable('mean', dt=.1, step=step)
        torch.testing.assert_close(first.value, second.value)
        torch.testing.assert_close(a.reward, b.reward)


@pytest.mark.parametrize('field', ['evaluation', 'variables', 'values', 'conditions', 'history'])
def test_old_task_scopes_are_rejected(field):
    with pytest.raises(ValueError, match='Obsolete task fields'):
        evaluation_config({'task': {field: {}}})


def test_fast_feedback_updates_without_advancing_decision_history():
    values = torch.tensor([1., 2.]); calls = []
    factories = {'feedback': lambda env: Signal(lambda: (calls.append(1), values)[1])}
    settings = {'variables': {
        'raw': {'source': 'feedback', 'sampling': 'physics'},
        'twice': {'expr': '2 * raw', 'sampling': 'physics'},
        'recorded': {'expr': 'twice', 'history': 4},
    }, 'task': {'reward': '-twice'}}
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.2, physics_dt=.005,
                          common_step_counter=0, _sim_step_counter=0, extras={})
    torch.testing.assert_close(read_variable(env, settings, 'twice', factories), values * 2)
    evaluate_term(env, evaluation_config(settings), 'reward', 'total', signal_factories=factories)
    runtime = env._assembly_task_evaluator
    assert calls == [1]
    assert runtime.histories['recorded'].cursor.tolist() == [1, 1]
    values[:] += 1; env._sim_step_counter = 4
    torch.testing.assert_close(read_variable(env, settings, 'twice', factories), values * 2)
    assert calls == [1, 1]
    assert runtime.histories['recorded'].cursor.tolist() == [1, 1]
    env.common_step_counter = 1
    torch.testing.assert_close(evaluate_term(env, evaluation_config(settings), 'reward', 'total', signal_factories=factories), -values * 2)
    assert calls == [1, 1]  # same physical sample reused by Task
    assert runtime.histories['recorded'].cursor.tolist() == [2, 2]
    reset_evaluation(env, [0]); values[0] = 10
    torch.testing.assert_close(read_variable(env, settings, 'twice', factories), torch.tensor([20., 6.]))
    assert not runtime.physics_runtime.histories


@pytest.mark.parametrize('declaration', [
    {'expr': 'robot.position', 'history': 5},
    {'expr': 'hold(robot.position[:, 0] > 0, seconds=1)'},
    {'expr': 'running_mean(robot.position[:, 0])'},
    {'expr': 'slow + 1'},
])
def test_fast_feedback_has_no_implicit_second_history_clock(declaration):
    from shared.assembly.legacy.task import validate_evaluation
    with pytest.raises(ValueError):
        validate_evaluation({'variables': {'slow': {'expr': '1.0'},
                                          'fast': {**declaration, 'sampling': 'physics'}}})


def test_derived_fields_removed_from_backend_catalog():
    from shared.backends.isaac.entity_observations import SIGNALS
    assert not {'relative_joint_position', 'relative_joint_velocity', 'body_linear_velocity',
                'body_angular_velocity', 'gravity_direction', 'height', 'peak_contact_force', 'tracked_position'} & SIGNALS.keys()
    assert {'joint_position', 'default_joint_position', 'ray_hit_positions', 'contact_force_samples'} <= SIGNALS.keys()
