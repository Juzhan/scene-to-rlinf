"""Torch expressions and per-variable histories across batched episode resets."""
import pytest
import torch

from shared.assembly.signals import Signal, Signals
from shared.assembly.legacy.task import Reading, TaskEvaluator, validate_evaluation


def make(config, data, device='cpu'):
    signals = Signals()
    for name in data:
        signals.register(name, Signal(lambda name=name: data[name]))
    return TaskEvaluator(config, signals, num_envs=2, device=device)


def test_long_formula_matches_torch_and_reads_sources_once():
    a = torch.tensor([[1., 2, 3], [4, 5, 6]])
    b = torch.zeros_like(a)
    engine = make({'variables': {
        'distance': {'expr': 'torch.linalg.vector_norm(robot.position - object.position, dim=-1)'},
        'score': {'expr': 'torch.where(distance < 4, torch.exp(-distance / 2), -distance.square())'},
    }, 'rewards': {'score': {'input': 'score', 'weight': 2}}, 'metrics': {'distance': 'distance'}},
        {'robot.position': a, 'object.position': b})
    result = engine.evaluate(dt=.1, step=0)
    distance = torch.linalg.vector_norm(a - b, dim=-1)
    torch.testing.assert_close(result.reward, 2 * torch.where(distance < 4, torch.exp(-distance / 2), -distance.square()))
    assert engine.evaluate(dt=.1, step=0) is result


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_every_variable_can_have_history_and_partial_reset_isolated(device):
    data = {'robot.value': torch.tensor([1., 10.], device=device)}
    engine = make({'variables': {
        'raw': {'source': 'robot.value', 'history': 3},
        'twice': {'expr': 'raw * 2', 'history': 2},
        'ok': {'expr': 'twice > 0', 'history': 2},
        'smoothed': {'expr': 'history_mean(twice)', 'history': 2},
    }, 'metrics': {'average': 'smoothed', 'all_ok': 'history_all(ok)'}}, data, device)
    for step in range(4):
        data['robot.value'] = torch.tensor([step+1., 10*(step+1.)], device=device)
        result = engine.evaluate(dt=.1, step=step)
    torch.testing.assert_close(result.metrics['average'].value, torch.tensor([7., 70.], device=device))
    old_values, old_valid, old_times = (x.clone() for x in engine.histories['raw'].read())
    torch.testing.assert_close(old_values, torch.tensor([[2., 3, 4], [20, 30, 40]], device=device))
    engine.reset([0])
    data['robot.value'] = torch.tensor([9., 40.], device=device)
    engine.evaluate(dt=.1, step=3)  # unchanged environment must not append a duplicate
    values, valid, times = engine.histories['raw'].read()
    assert valid[0].tolist() == [False, False, True]
    assert values[0, -1] == 9
    torch.testing.assert_close(values[1], old_values[1])
    torch.testing.assert_close(times[1], old_times[1])
    assert times[0, -1].item() == pytest.approx(.1)
    assert all(not h.values.requires_grad for h in engine.histories.values())


def test_history_validity_warmup_missing_samples_and_explicit_mask():
    data = {'robot.value': Reading(torch.tensor([1., 10.]), torch.tensor([True, False]))}
    engine = make({'variables': {'x': {'source': 'robot.value', 'history': 3}}, 'metrics': {
        'mean': 'history_mean(x)', 'raw_mean': 'torch.mean(x.history, dim=1)',
        'manual': 'torch.where(x.history_valid, x.history, 0).sum(dim=1) / x.history_valid.sum(dim=1).clamp(min=1)',
    }}, data)
    result = engine.evaluate(dt=.1, step=0)
    assert result.metrics['mean'].valid.tolist() == [True, False]
    assert not result.metrics['raw_mean'].valid.any()
    assert result.metrics['manual'].value.tolist() == [1., 0.]
    data['robot.value'] = torch.tensor([3., 30.])
    result = engine.evaluate(dt=.2, step=1)
    assert result.metrics['mean'].value.tolist() == [2., 30.]
    assert result.metrics['mean'].valid.all()
    _, _, time = engine.histories['x'].read()
    torch.testing.assert_close(time[0], torch.tensor([0., .1, .3], dtype=torch.float64))


def test_path_length_from_vector_history_and_finite_window():
    data = {'robot.position': torch.zeros(2, 3)}
    engine = make({'variables': {'p': {'source': 'robot.position', 'history': 3},
        'length': {'expr': 'torch.linalg.vector_norm(torch.diff(p.history, dim=1), dim=-1).sum(dim=1)', 'history': 2}},
        'metrics': {'length': 'length'}}, data)
    for step in range(4):
        data['robot.position'] = torch.tensor([[3.*step, 4.*step, 0.], [0., 0., 2.*step]])
        result = engine.evaluate(dt=.1, step=step)
        assert result.metrics['length'].valid.all().item() == (step >= 2)
    assert result.metrics['length'].value.tolist() == [10., 4.]


@pytest.mark.parametrize('expression', [
    '__import__("os")', 'robot.value.add_(1)', 'robot.value.cpu()', 'robot.value.__class__',
    '[x for x in robot.value]', '(lambda x: x)(robot.value)',
    'torch.sum(robot.value, out=robot.value)', '0 < robot.value < 1',
])
def test_unsupported_code_fails_before_runtime(expression):
    with pytest.raises(ValueError): validate_evaluation({'variables': {'x': {'expr': expression}}})


@pytest.mark.parametrize('expression', ['torch.mean(robot.value)', 'torch.sum(robot.value, dim=0)',
    'robot.value[0]', 'robot.value.sum(dim=-2)', 'torch.cat([robot.value, robot.value], dim=0)'])
def test_cross_environment_operations_rejected(expression):
    engine = make({'variables': {'x': {'expr': expression}}}, {'robot.value': torch.ones(2, 3)})
    with pytest.raises(ValueError, match='environment|explicit dim'): engine.evaluate(dt=.1, step=0)


@pytest.mark.parametrize('variables,match', [
    ({'x': {'expr': 'y'}, 'y': {'expr': 'x'}}, 'Cyclic'),
    ({'x': {'expr': 'x.history', 'history': 3}}, 'Cyclic'),
    ({'x': {'source': 'robot.value'}, 'y': {'expr': 'history_mean(x)'}}, 'not enabled'),
    ({'x': {'value': 1, 'history': 0}}, 'positive'),
    ({'torch': {'value': 1}}, 'Invalid variable'),
    ({'x': {'expr': 'misspelled'}}, 'Unknown task variable'),
])
def test_invalid_graph_and_declarations_fail_statically(variables, match):
    with pytest.raises(ValueError, match=match): validate_evaluation({'variables': variables})


def test_runtime_contracts_and_nonfinite_reward():
    engine = make({'variables': {'x': {'expr': 'robot.value / 0', 'history': 2}},
                   'rewards': {'x': {'input': 'x'}}}, {'robot.value': torch.ones(2)})
    assert engine.evaluate(dt=.1, step=0).reward.eq(0).all()
    assert not engine.histories['x'].read()[1].any()
    engine = make({'variables': {'x': {'expr': 'robot.value', 'information': {'kind': 'number', 'shape': [3]}}}},
                   {'robot.value': torch.ones(2, 4)})
    with pytest.raises(ValueError, match='declared shape'): engine.evaluate(dt=.1, step=0)
    engine = make({'success': 'robot.value'}, {'robot.value': torch.ones(2)})
    with pytest.raises(ValueError, match='boolean'): engine.evaluate(dt=.1, step=0)


def test_static_dependency_discovery_includes_nested_properties():
    report = validate_evaluation({'variables': {'x': {'expr': 'torch.linalg.vector_norm(robot.arm.position - object.target.position, dim=-1)'}},
                                  'metrics': {'x': 'x'}})
    assert report['signals'] == ['object.target.position', 'robot.arm.position']


def test_ellipsis_cannot_hide_indexing_environment_axis():
    engine = make({'variables': {'x': {'expr': 'robot.value[..., 0]'}}}, {'robot.value': torch.ones(2)})
    with pytest.raises(ValueError, match='environment axis'): engine.evaluate(dt=.1, step=0)


def test_ambiguous_batched_broadcast_requires_unsqueeze():
    data = {'robot.vector': torch.ones(2, 2), 'robot.scalar': torch.tensor([1., 2.])}
    engine = make({'variables': {'x': {'expr': 'robot.vector * robot.scalar'}}}, data)
    with pytest.raises(ValueError, match='broadcasting'): engine.evaluate(dt=.1, step=0)
    engine = make({'variables': {'x': {'expr': 'robot.vector * robot.scalar.unsqueeze(-1)'}}, 'metrics': {'x': 'x'}}, data)
    assert engine.evaluate(dt=.1, step=0).metrics['x'].value.tolist() == [[1., 1.], [2., 2.]]


def test_total_reward_formula_matches_named_contributions():
    data = {'robot.reach': torch.tensor([.1, .3]), 'robot.target': torch.tensor([.2, .4])}
    variables = {'reach_distance': {'expr': 'robot.reach'}, 'target_distance': {'expr': 'robot.target'}}
    formula = make({'variables': variables, 'reward': '-0.1 * reach_distance - 0.4 * target_distance'}, data)
    named = make({'variables': variables, 'rewards': {
        'reach': {'input': 'reach_distance', 'weight': -.1},
        'target': {'input': 'target_distance', 'weight': -.4}}}, data)
    result = formula.evaluate(dt=.02, step=0)
    torch.testing.assert_close(result.reward, named.evaluate(dt=.02, step=0).reward)
    assert set(result.rewards) == {'total'}


@pytest.mark.parametrize('config,match', [
    ({'reward': '1', 'rewards': {}}, 'not both'),
    ({'reward': ''}, 'nonempty'),
    ({'reward': {'expr': '1'}}, 'expression string'),
    ({'variables': {'x': {'expr': 'hold(True, seconds=-1)'}}}, 'nonnegative'),
    ({'variables': {'x': {'expr': 'hold(True, unexpected=1)'}}}, 'unknown'),
    ({'variables': {'x': {'expr': 'inside_region(robot.position)'}}}, 'requires'),
])
def test_formula_reward_and_domain_call_validation(config, match):
    with pytest.raises(ValueError, match=match): validate_evaluation(config)


def test_domain_call_state_is_per_occurrence_and_resets_by_environment():
    data = {'robot.a': torch.tensor([True, True]), 'robot.b': torch.tensor([False, True]),
            'robot.number': torch.tensor([1., 10.])}
    engine = make({'variables': {
        'both': {'expr': 'hold(robot.a, seconds=0.2) & hold(robot.b, seconds=0.2)'},
        'average': {'expr': 'running_mean(robot.number)'}},
        'success': 'both', 'reward': 'average'}, data)
    assert not engine.evaluate(dt=.1, step=0).success.any()
    data['robot.number'] = torch.tensor([3., 30.])
    result = engine.evaluate(dt=.1, step=1)
    assert result.success.tolist() == [False, True]
    assert result.reward.tolist() == [2., 20.]
    engine.reset([1])
    data['robot.b'] = torch.tensor([True, True])
    data['robot.number'] = torch.tensor([5., 50.])
    result = engine.evaluate(dt=.1, step=2)
    assert not result.success.any()
    assert result.reward.tolist() == [3., 50.]
