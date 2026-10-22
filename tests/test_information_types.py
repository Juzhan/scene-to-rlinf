import math
from types import SimpleNamespace

import pytest
import torch

from shared.assembly.information import Information, pose_information
from shared.assembly.signals import Signal, Signals
from shared.assembly.legacy.task import TaskEvaluator, validate_evaluation, evaluation_config
from shared.assembly.legacy.task_types import infer_evaluation, product_unit


def src(name): return {"signal": name}
def val(value): return {"value": value}


def run(expression, sources, data):
    signals = Signals()
    for name, info in sources.items():
        signals.register(name, Signal(lambda name=name: data[name], info=info))
    evaluator = TaskEvaluator({"metrics": {"result": expression}}, signals, num_envs=2)
    return evaluator, evaluator.evaluate(dt=.1, step=1).metrics["result"]


def test_classification_does_not_read_data_or_guess_three_values_are_positions():
    signals = Signals()
    signals.register("point", Signal(lambda: pytest.fail("Metadata must be lazy"),
        info=Information("position", (3,), "float", "m", "world"), entity="box", source="simulation"))
    signals.register("legacy", Signal(lambda: [1, 2, 3]))
    descriptor = signals.describe()
    assert descriptor["point"]["information"]["kind"] == "position"
    assert "distance" in descriptor["point"]["information"]["operations"]
    assert descriptor["legacy"]["information"]["kind"] == "unknown"
    assert descriptor["point"]["entity"] == "box"
    assert "norm" not in Information("number", ()).describe()["operations"]
    assert "hold" not in Information("boolean", (3,)).describe()["operations"]


@pytest.mark.parametrize("expression,sources,message", [
    ({"op": "distance", "a": src("q"), "b": src("q")},
     {"q": Information("rotation", (4,), encoding="wxyz")}, "not an ordinary"),
    ({"op": "distance", "a": src("a"), "b": src("b")},
     {"a": Information("position", (3,), unit="m", frame="world"),
      "b": Information("position", (3,), unit="m", frame="robot")}, "frame"),
    ({"op": "less", "a": src("a"), "b": src("b")},
     {"a": Information("number", (), unit="m"), "b": Information("number", (), unit="s")}, "unit"),
    ({"op": "add", "a": src("a"), "b": src("a")},
     {"a": Information("position", (3,), unit="m")}, "two positions"),
    ({"op": "norm", "input": src("image")},
     {"image": Information("image", (64, 64, 3))}, "not an ordinary"),
    ({"op": "all", "inputs": [src("score")]},
     {"score": Information("number", ())}, "boolean"),
    ({"op": "hold", "input": src("array"), "seconds": 1},
     {"array": Information("boolean", (3,), "bool")}, "one condition"),
    ({"op": "distance", "a": val([1, 2]), "b": val([1, 2, 3])}, {}, "shapes"),
    ({"op": "field", "input": src("pose"), "name": "typo"},
     {"pose": pose_information("world")}, "unknown field"),
    ({"op": "select", "input": src("p"), "indices": 9},
     {"p": Information("position", (3,))}, "out of range"),
])
def test_invalid_computations_fail_without_loading_simulation(expression, sources, message):
    config = {"metrics": {"result": expression}}
    validate_evaluation(config)
    with pytest.raises(ValueError, match=message): infer_evaluation(config, sources)


def test_units_and_derived_output_propagate():
    config = {"values": {
        "displacement": {"op": "subtract", "a": src("a"), "b": src("b")},
        "distance": {"op": "norm", "input": {"ref": "displacement"}},
        "close": {"op": "less", "a": {"ref": "distance"}, "b": val(.03)},
        "scaled": {"op": "multiply", "a": {"ref": "distance"}, "b": val(2)},
    }}
    info = Information("position", (3,), "float", "m", "world")
    report = infer_evaluation(config, {"a": info, "b": info})
    assert report["nodes"]["displacement"].kind == "vector"
    assert report["nodes"]["distance"].unit == "m"
    assert report["nodes"]["distance"].shape == ()
    assert report["nodes"]["close"].kind == "boolean"
    assert report["nodes"]["scaled"].unit == "m"
    assert product_unit("m/s", "s") == "m"


def test_quaternion_distance_respects_encoding_and_double_cover():
    sources = {"a": Information("rotation", (4,), frame="world", encoding="wxyz"),
               "b": Information("rotation", (4,), frame="world", encoding="xyzw")}
    data = {"a": [[1., 0, 0, 0], [1., 0, 0, 0]],
            "b": [[0., 0, 0, -1.], [0, 0, math.sqrt(.5), math.sqrt(.5)]]}
    evaluator, result = run({"op": "rotation_distance", "a": src("a"), "b": src("b")}, sources, data)
    torch.testing.assert_close(result.value, torch.tensor([0., math.pi / 2]))
    assert evaluator.types["expressions"]["metrics.result"].unit == "rad"


def test_xyzw_rotation_maps_declared_frames():
    sources = {"q": Information("rotation", (4,), frame="world", target_frame="body", encoding="xyzw"),
               "v": Information("vector", (3,), unit="m/s", frame="body")}
    evaluator, result = run({"op": "rotate", "input": src("v"), "quaternion": src("q")}, sources,
        {"q": [[0, 0, math.sqrt(.5), math.sqrt(.5)]] * 2, "v": [[1., 0, 0]] * 2})
    torch.testing.assert_close(result.value, torch.tensor([[0., 1, 0]] * 2), atol=1e-6, rtol=1e-6)
    assert evaluator.types["expressions"]["metrics.result"].frame == "world"


def test_named_pose_field_has_its_own_type_and_units():
    evaluator, result = run({"op": "field", "input": src("pose"), "name": "position"},
        {"pose": pose_information("world")}, {"pose": [[1., 2, 3, 1, 0, 0, 0]] * 2})
    assert result.value.tolist() == [[1, 2, 3]] * 2
    assert evaluator.types["expressions"]["metrics.result"].kind == "position"


@pytest.mark.parametrize("kind,expected", [("box", [True, False]), ("rectangle", [True, True])])
def test_rotated_region_containment(kind, expected):
    sources = {"p": Information("position", (3,), unit="m", frame="world"),
               "r": Information("region", (10,), unit="m", frame="world", encoding=kind + "_center_wxyz_halfsize")}
    # local X axis is world Y; z outside box but ignored for planar rectangle.
    data = {"p": [[0., 1.5, 0], [0, 1.5, 3]],
            "r": [[0., 0, 0, math.sqrt(.5), 0, 0, math.sqrt(.5), 2, .5, 1]] * 2}
    _, result = run({"op": "inside_region", "input": src("p"), "region": src("r")}, sources, data)
    assert result.value.tolist() == expected


def test_categories_accept_declared_labels_and_reject_typos():
    sources = {"mode": Information("category", (), "int", labels=("stopped", "running", "error"))}
    _, result = run({"op": "equal", "a": src("mode"), "b": val("running")}, sources,
                    {"mode": ["running", "error"]})
    assert result.value.tolist() == [True, False]
    _, result = run({"op": "in_set", "input": src("mode"), "choices": ["stopped", "running"]}, sources,
                    {"mode": [1, 2]})
    assert result.value.tolist() == [True, False]
    with pytest.raises(ValueError, match="unknown category"):
        infer_evaluation({"success": {"op": "equal", "a": src("mode"), "b": val("runnnning")}}, sources)
    _, result = run({"op": "not_equal", "a": src("mode"), "b": val("running")}, sources,
                    {"mode": [99, 1.5]})
    assert not result.valid.any()


def test_declared_shape_and_boolean_dtype_checked_on_actual_read():
    with pytest.raises(ValueError, match="declared shape"):
        run(src("a"), {"a": Information("position", (3,))}, {"a": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="boolean data"):
        run(src("a"), {"a": Information("boolean", (), "bool")}, {"a": [1, 0]})


def test_unknown_metadata_is_reported_not_guessed():
    report = infer_evaluation({"metrics": {"x": {"op": "norm", "input": src("legacy")}}}, {})
    assert report["untyped_signals"] == ["legacy"]


def test_native_command_metadata_comes_from_config_type():
    from shared.backends.isaac.entity_observations import command_information
    native = type("UniformVelocityCommandCfg", (), {"asset_name": "dog"})()
    info = command_information(native)
    assert info.fields["linear_velocity"].info.frame == "dog/root"
    assert info.fields["angular_velocity"].info.unit == "rad/s"


def test_generated_object_schema_includes_source_types():
    from pathlib import Path
    from shared.assembly.object_schema import generate
    for project in ("assembly_lift", "assembly_go2"):
        path = Path(__file__).resolve().parents[1] / "projects" / project / "scene.yaml"
        source = generate(path, write=False)
        assert 'class SceneInfo(ObjectView):' in source
        assert 'joint_position: Tensor' in source
        assert 'position: Tensor' in source
