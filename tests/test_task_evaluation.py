from shared.assembly.legacy.task import evaluation_config
"""Behavioral tests: batched histories, invalid data, no observation leakage."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from shared.assembly.signals import Observation, Signal, Signals
from shared.assembly.legacy.task import Reading, TaskEvaluator, validate_evaluation


def evaluator(config, data=None, n=2, information=None):
    data = data or {}
    signals = Signals()
    for name in data:
        signals.register(name, Signal(lambda name=name: data[name], info=(information or {}).get(name)))
    return TaskEvaluator(config, signals, num_envs=n), signals


def value(x): return {"value": x}
def ref(x): return {"ref": x}
def signal(x): return {"signal": x}


@pytest.mark.parametrize("op,arguments,expected", [
    ("distance", {"a": value([0, 0, 0]), "b": value([3, 4, 0])}, 5),
    ("subtract", {"a": value(3), "b": value(2)}, 1),
    ("add", {"a": value(3), "b": value(2)}, 5),
    ("multiply", {"a": value(3), "b": value(2)}, 6),
    ("divide", {"a": value(3), "b": value(2)}, 1.5),
    ("dot", {"a": value([1, 2]), "b": value([3, 4])}, 11),
    ("angle", {"a": value([1, 0, 0]), "b": value([0, 1, 0])}, math.pi / 2),
    ("abs", {"input": value(-2)}, 2),
    ("norm", {"input": value([3, 4])}, 5),
    ("sum", {"input": value([3, 4])}, 7),
    ("mean", {"input": value([3, 4])}, 3.5),
    ("min", {"input": value([3, 4])}, 3),
    ("max", {"input": value([3, 4])}, 4),
    ("select", {"input": value([3, 4]), "indices": 1}, 4),
    ("less", {"a": value(2), "b": value(3)}, True),
    ("less_equal", {"a": value(3), "b": value(3)}, True),
    ("greater", {"a": value(3), "b": value(2)}, True),
    ("greater_equal", {"a": value(3), "b": value(3)}, True),
    ("equal", {"a": value(3), "b": value(3)}, True),
    ("not_equal", {"a": value(3), "b": value(2)}, True),
    ("near", {"a": value(3), "b": value(3.1), "tolerance": .2}, True),
    ("between", {"input": value(3), "lower": value(2), "upper": value(4)}, True),
    ("in_set", {"input": value(3), "choices": [2, 3]}, True),
    ("inside_box", {"input": value([1, 2]), "lower": value([0, 0]), "upper": value([3, 3])}, True),
    ("valid", {"input": value(3)}, True),
    ("not", {"input": value(False)}, True),
    ("all", {"inputs": [value(True), value(False)]}, False),
    ("any", {"inputs": [value(True), value(False)]}, True),
    ("at_least", {"inputs": [value(True), value(False)], "count": 1}, True),
    ("all_elements", {"input": value([True, False])}, False),
    ("any_elements", {"input": value([True, False])}, True),
    ("count", {"input": value([True, False, True])}, 2),
])
def test_stateless_rules(op, arguments, expected):
    engine, _ = evaluator({"metrics": {"result": {"op": op, **arguments}}})
    result = engine.evaluate(dt=.1, step=1).metrics["result"]
    torch.testing.assert_close(result.value.float(), torch.full((2,), float(expected)))
    assert result.valid.all()


def test_rotation_and_coordinate_transform():
    q = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
    engine, _ = evaluator({"values": {
        "rotated": {"op": "rotate", "input": value([1, 0, 0]), "quaternion": value(q)},
        "restored": {"op": "rotate", "input": ref("rotated"), "quaternion": value(q), "inverse": True},
        "position": {"op": "transform_point", "input": value([1, 0, 0]), "quaternion": value(q), "position": value([2, 3, 4])},
    }, "metrics": {name: ref(name) for name in ("rotated", "restored", "position")}})
    results = engine.evaluate(dt=.1, step=0).metrics
    for name, expected in [("rotated", [0, 1, 0]), ("restored", [1, 0, 0]), ("position", [2, 4, 4])]:
        torch.testing.assert_close(results[name].value, torch.tensor([expected] * 2).float(), atol=1e-6, rtol=1e-6)


def test_lazy_signal_reads_cached_for_success_reward_metrics_and_not_policy():
    calls = []
    signals = Signals()
    signals.register("truth", Signal(lambda: (calls.append(1), torch.tensor([.01, 1.]))[1]))
    signals.register("camera", Signal(lambda: torch.zeros(2, 4, 4, 3)))
    signals.register("unused", Signal(lambda: pytest.fail("Unselected signal fetched")))
    config = {"conditions": {"close": {"op": "less", "a": signal("truth"), "b": value(.1)}},
              "success": ref("close"), "rewards": {"distance": {"input": signal("truth"), "weight": -1}},
              "metrics": {"distance": signal("truth")}}
    engine = TaskEvaluator(config, signals, num_envs=2)
    result = engine.evaluate(dt=.1, step=0)
    assert engine.evaluate(dt=.1, step=0) is result
    assert calls == [1]
    assert result.success.tolist() == [True, False]
    assert set(Observation({"image": "camera"}).read(signals)) == {"image"}


def test_invalid_negation_and_failure_precedence():
    data = {"model": Reading(torch.tensor([False, True]), torch.tensor([False, True]))}
    engine, _ = evaluator({"success": {"op": "not", "input": signal("model")}}, data)
    assert not engine.evaluate(dt=.1, step=0).success.any()
    engine, _ = evaluator({"success": value(True), "failure": value(True)})
    result = engine.evaluate(dt=.1, step=0)
    assert result.failure.all() and not result.success.any()
    engine, _ = evaluator({"values": {"bad": {"op": "divide", "a": value(1), "b": value(0)}},
                          "success": {"op": "not", "input": {"op": "less", "a": ref("bad"), "b": value(1)}},
                          "rewards": {"bad": {"input": ref("bad")}}})
    result = engine.evaluate(dt=.1, step=0)
    assert not result.success.any() and result.reward.eq(0).all()


def test_hold_ever_within_partial_reset_and_invalid_break():
    data = {"ok": torch.tensor([True, False])}
    nodes = {"hold": {"op": "hold", "input": signal("ok"), "seconds": .3},
             "ever": {"op": "ever", "input": signal("ok")},
             "within": {"op": "within", "input": signal("ok"), "seconds": .2}}
    engine, _ = evaluator({"conditions": nodes, "metrics": {k: ref(k) for k in nodes}}, data)
    for i in range(3): result = engine.evaluate(dt=.1, step=i)
    assert result.metrics["hold"].value.tolist() == [True, False]
    data["ok"] = torch.tensor([True, True])
    engine.reset(torch.tensor([0]))
    result = engine.evaluate(dt=.1, step=3)
    assert result.metrics["hold"].value.tolist() == [False, False]
    assert result.metrics["within"].value.tolist() == [True, False]
    data["ok"] = Reading(torch.tensor([True, True]), torch.tensor([False, True]))
    engine.evaluate(dt=.1, step=4)
    data["ok"] = torch.tensor([True, True])
    result = engine.evaluate(dt=.1, step=5)
    assert result.metrics["hold"].value.tolist() == [False, True]
    assert result.metrics["ever"].value.all()


def test_sequence_does_not_complete_every_stage_on_one_sample():
    engine, _ = evaluator({"success": {"op": "sequence", "inputs": [value(True), value(True)]}})
    assert not engine.evaluate(dt=.1, step=0).success.any()
    assert engine.evaluate(dt=.1, step=1).success.all()
    engine.reset([1])
    assert engine.evaluate(dt=.1, step=2).success.tolist() == [True, False]


def test_change_delta_and_aggregation():
    data = {"x": torch.tensor([1., 3.]), "ok": torch.tensor([False, True])}
    operations = ["changed", "delta", "accumulate", "integral", "running_mean", "running_min", "running_max"]
    nodes = {op: {"op": op, "input": signal("x")} for op in operations}
    nodes["rising"] = {"op": "rising", "input": signal("ok")}
    engine, _ = evaluator({"values": nodes, "metrics": {k: ref(k) for k in nodes}}, data)
    result = engine.evaluate(dt=.5, step=0)
    assert not result.metrics["delta"].valid.any()
    assert not result.metrics["changed"].value.any()
    data.update(x=torch.tensor([2., 1.]), ok=torch.tensor([True, True]))
    result = engine.evaluate(dt=.5, step=1)
    expected = {"delta": [1, -2], "accumulate": [3, 4], "integral": [1.5, 2],
                "running_mean": [1.5, 2], "running_min": [1, 1], "running_max": [2, 3],
                "changed": [True, True], "rising": [True, False]}
    for name, values in expected.items():
        torch.testing.assert_close(result.metrics[name].value.float(), torch.tensor(values).float())
    engine.reset([0])
    result = engine.evaluate(dt=.5, step=2)
    assert result.metrics["accumulate"].value.tolist() == [2, 5]
    assert result.metrics["delta"].valid.tolist() == [False, True]


@pytest.mark.parametrize("config,match", [
    ({"success": {"ref": "missing"}}, "unknown task reference"),
    ({"values": {"a": ref("b"), "b": ref("a")}}, "Cyclic"),
    ({"values": {"a": value(1)}, "conditions": {"a": value(True)}}, "duplicate"),
    ({"success": {"op": "hold", "input": value(True), "seconds": -1}}, "nonnegative"),
    ({"success": {"op": "all", "inputs": []}}, "nonempty"),
    ({"success": {"op": "exec", "input": value(True)}}, "unknown operation"),
    ({"success": {"op": "not", "input": value(True), "typo": 1}}, "unknown"),
])
def test_static_errors(config, match):
    with pytest.raises(ValueError, match=match): validate_evaluation(config)


def test_real_assembly_configs_on_synthetic_scene_signals():
    from projects.assembly_lift.task import Task as LiftTask
    from projects.assembly_lift.observation import SceneInfo as LiftInfo
    from projects.assembly_go2.task import Task as Go2Task
    from projects.assembly_go2.observation import SceneInfo as Go2Info
    data = {"object.target.position": torch.tensor([[.5, 0, .3]] * 2),
            "ee_frame.frame_positions": torch.tensor([[[.5, 0, .3]]] * 2),
            "object.position": torch.tensor([[.5, 0, .3], [.5, 0, .05]]),
            "object.linear_velocity": torch.zeros(2, 3),
            "simulation.step_dt": torch.full((2, 1), .02),
            "spawn_area.position": torch.tensor([[.5, 0, .055]] * 2),
            "spawn_area.orientation": torch.tensor([[1., 0, 0, 0]] * 2),
            "spawn_area.size": torch.tensor([[.2, .5]] * 2)}
    task = LiftTask()
    for _ in range(25): result = task.evaluate(LiftInfo(data))
    assert result['success'].tolist() == [True, False]
    data = {"robot.target.linear_velocity": torch.tensor([[1., 0, 0]] * 2),
            "simulation.gravity": torch.tensor([[0., 0, -9.81]] * 2),
            "robot.orientation": torch.tensor([[1., 0, 0, 0]] * 2),
            "robot.joint_effort": torch.zeros(2, 12),
            "robot.linear_velocity": torch.tensor([[1., 0, 0], [0., 0, 0]])}
    result = Go2Task().evaluate(Go2Info(data))
    assert result['metrics']["velocity_error_m_s"].tolist() == [0, 1]
    assert 'success' not in result


def test_native_adapter_custom_model_signal_cached_and_reset():
    from shared.backends.isaac.task_evaluation import evaluate_term, reset_evaluation
    calls = []
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.1, common_step_counter=0, extras={})
    factories = {"inspection": lambda env: Signal(lambda: (calls.append(1), torch.tensor([True, False]))[1])}
    config = {"success": {"op": "hold", "input": signal("inspection"), "seconds": .2},
              "metrics": {"inspection": signal("inspection")}}
    assert not evaluate_term(env, config, "success", signal_factories=factories).any()
    evaluate_term(env, config, "failure", signal_factories=factories)
    assert calls == [1]
    env.common_step_counter += 1
    assert evaluate_term(env, config, "success", signal_factories=factories).tolist() == [True, False]
    reset_evaluation(env, [0])
    env.common_step_counter += 1
    assert not evaluate_term(env, config, "success", signal_factories=factories).any()
    assert env.extras["task_metrics"]["inspection"]["valid"].all()
    assert env.extras["metrics"]["success"].tolist() == [False, False]


def test_coverage_union_does_not_count_repeated_cells_and_resets_per_env():
    data = {"visible": torch.tensor([[True, False, False, False], [False, True, True, False]])}
    engine, _ = evaluator({"values": {
        "scanned": {"op": "ever", "input": signal("visible")},
        "count": {"op": "count", "input": ref("scanned")},
        "ratio": {"op": "divide", "a": ref("count"), "b": value(4)},
    }, "success": {"op": "greater_equal", "a": ref("ratio"), "b": value(.75)},
        "metrics": {"coverage": ref("ratio")}}, data)
    engine.evaluate(dt=.1, step=0)
    data["visible"] = torch.tensor([[True, True, False, False], [False, False, True, True]])
    result = engine.evaluate(dt=.1, step=1)
    assert result.metrics["coverage"].value.tolist() == [.5, .75]
    assert result.success.tolist() == [False, True]
    engine.reset([1])
    result = engine.evaluate(dt=.1, step=2)
    assert result.metrics["coverage"].value.tolist() == [.5, .5]
