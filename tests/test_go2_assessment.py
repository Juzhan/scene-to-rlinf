"""Independent route acceptance cannot be inflated by easier seeds or descents."""

import sys
from pathlib import Path

import pytest

from tools.go2_evaluation import acceptance, load_suite, parse_options, route_cases, summarize


def options(*args):
    return parse_options(["go2_stairs", "--checkpoint", "weights.pt", "--output", "report.json", *args])


def records(up_success, down_success, *, seed=43):
    rows = []
    for motion, count in (("up", up_success), ("down", down_success)):
        for index in range(10):
            rows.append({"seed": seed, "motion": motion, "success": index < count, "fell": False,
                         "terrain": "stairs", "level": 0, "world_direction_deg": 0, "body_heading_offset_deg": 0})
    return rows


def test_suite_is_project_local_and_cli_changes_are_recorded():
    root = Path(__file__).resolve().parents[1] / "projects/go2_stairs"
    selected = options("--speed", "0.4")
    report = load_suite(selected, root)
    assert selected.seeds == [43, 44]
    assert selected.levels == [0, 4]
    assert selected.directions == [0, 90, 180, 270]
    assert selected.body_headings == [0]
    assert selected.steps == 600 and selected.minimum_success == 0.8
    assert selected.speed == 0.4
    assert report["overrides"] == {"speed": 0.4}
    assert len(report["sha256"]) == 64


def test_aggregate_eighty_percent_cannot_hide_uphill_failure():
    rows = records(6, 10)
    assert summarize(rows)["success_rate"] == 0.8
    result = acceptance(rows, [43], 0.8)
    assert not result["accepted"]
    assert next(check for check in result["checks"] if check["motion"] == "up")["passed"] is False
    assert summarize(rows)["by_motion"]["up"]["success_rate"] == 0.6


def test_each_seed_and_both_motions_must_pass():
    assert acceptance(records(8, 8) + records(8, 8, seed=44), [43, 44], 0.8)["accepted"]
    assert not acceptance(records(10, 10) + records(6, 6, seed=44), [43, 44], 0.8)["accepted"]
    assert not acceptance(records(10, 10), [43, 44], 0.8)["accepted"]
    assert not acceptance([row for row in records(10, 10) if row["motion"] == "down"], [43], 0.8)["accepted"]


@pytest.mark.parametrize("args", [
    ["--seeds", "43", "43"], ["--speed", "0"], ["--minimum-success", "1.1"],
    ["--steps", "0"], ["--directions", "nan"], ["--levels", "-1"],
])
def test_invalid_suite_options_are_rejected_before_simulation(tmp_path, args):
    with pytest.raises(ValueError):
        load_suite(options(*args), tmp_path)


def test_assess_cli_delegates_to_the_existing_evaluator(monkeypatch):
    import run
    import tools.go2_evaluation
    received = []
    monkeypatch.setattr(tools.go2_evaluation, "main", lambda args: received.append(args))
    monkeypatch.setattr(sys, "argv", ["run.py", "assess", "go2_stairs", "--snapshot", "saved-run"])
    run.main()
    assert received == [["go2_stairs", "--snapshot", "saved-run"]]
