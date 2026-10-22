import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mjlab")
import mujoco
from projects.rax_tabletop.env import Tabletop
from projects.rax_tabletop.task import Task
from shared.assembly.backend_scene import prepare
from shared.assembly.objects import ObjectView

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def simulator():
    env = Tabletop()
    yield env
    env.close()


@pytest.fixture
def env(simulator):
    simulator.mode = "stack"
    simulator.horizon = 60.0
    simulator.reset(0)
    yield simulator


def place_for_test(env, name, position):
    """Evaluator fixture only: never exposed by the Agent tools."""
    joint = env.model.joint(f"{name}/root").id
    qid, dof = env.model.jnt_qposadr[joint], env.model.jnt_dofadr[joint]
    env.data.qpos[qid : qid + 3] = position
    env.data.qpos[qid + 3 : qid + 7] = [1, 0, 0, 0]
    env.data.qvel[dof : dof + 6] = 0
    mujoco.mj_forward(env.model, env.data)
    env._forces_time = None


def test_scene_is_shared_and_drawer_is_passive(env):
    _, _, factories, report = prepare(
        ROOT / "projects/rax_tabletop/scene.yaml", "mjlab"
    )
    assert "tabletop_drawer" in factories
    assert all("mjlab" in item["supported_backends"] for item in report["entities"])
    assert env.model.nu == 8
    assert env.model.jnt_type[env.slide] == mujoco.mjtJoint.mjJNT_SLIDE
    assert env.model.jnt_range[env.slide].tolist() == [0, 0.18]
    assert not any(
        env.model.actuator(i).name.startswith("drawer/") for i in range(env.model.nu)
    )


def test_reset_and_observation_are_deterministic(env):
    first = env.data.qpos.copy()
    env.wait(0.2)
    env.reset(0)
    np.testing.assert_allclose(first, env.data.qpos, atol=1e-12)
    env.reset(1)
    assert not np.array_equal(first, env.data.qpos)
    result = json.dumps(env.observe(), allow_nan=False)
    for _ in range(10):
        assert json.dumps(env.observe(), allow_nan=False) == result


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("move_tcp", {"position": [float("nan"), 0, 0]}),
        ("move_tcp", {"position": [0.6, 0.4, 0.6]}),
        ("move_tcp", {"position": [0.3, 0, 0.44], "orientation": [0, 0, 0, 0]}),
        ("move_tcp", {"position": [0.3, 0, 0.44], "duration": float("nan")}),
        ("gripper", {"closed_fraction": -1}),
        ("gripper", {"closed_fraction": True}),
        ("wait", {"duration": 5}),
    ],
)
def test_rejected_commands_do_not_move(env, method, kwargs):
    before = env.data.qpos.copy()
    time = env.data.time
    with pytest.raises((ValueError, TypeError)):
        getattr(env, method)(**kwargs)
    np.testing.assert_array_equal(env.data.qpos, before)
    assert env.data.time == time


def test_pose_control_and_closed_loop_hold(env):
    current = env.data.site_xpos[env.servo.site].copy()
    target = current + [0.03, -0.02, -0.03]
    desired = np.array([0.0, 1.0, 0.0, 0.0])
    env.move_tcp(target.tolist(), orientation=desired.tolist(), duration=1)
    assert np.linalg.norm(env.data.site_xpos[env.servo.site] - target) < 0.008
    assert abs(env.tcp_quaternion() @ desired) > 0.999
    env.wait(0.4)
    assert np.linalg.norm(env.data.site_xpos[env.servo.site] - target) < 0.008
    assert not env.result["success"][0]


def test_physical_stacking_predicate(env):
    env.wait(0.2)
    bottom = env.data.body("support/body").xpos.copy()
    place_for_test(env, "cube", bottom + [0, 0, 0.041])
    env.wait(1)
    assert env.result["success"][0]
    assert env.contact_forces()["cube_support"] > 0.05
    assert env.result["metrics"]["hold_seconds"][0] >= 0.5 - 1e-9
    with pytest.raises(RuntimeError, match="ended"):
        env.wait(0.1)


@pytest.mark.parametrize("mode", ["drawer_open", "drawer_closed"])
def test_drawer_physics_and_containment(env, mode):
    env.mode = mode
    env.reset(0)
    qid = env.model.jnt_qposadr[env.slide]
    if mode == "drawer_closed":
        assert abs(env.data.qpos[qid]) < 0.001
        # A physical force moves the passive slider; this is an asset test,
        # not a robot policy success.
        tray = env.model.body("drawer/tray").id
        env.data.xfrc_applied[tray, 1] = -2
        env.wait(0.8)
        env.data.xfrc_applied[:] = 0
        assert env.data.qpos[qid] > 0.10
        env.wait(0.8)
    assert env.data.qpos[qid] > 0.8 * env.model.jnt_range[env.slide, 1]
    tray_pos = env.data.body("drawer/tray").xpos.copy()
    place_for_test(env, "cube", tray_pos + [0, -0.04, 0.026])
    env.wait(1)
    assert env.result["success"][0]
    assert env.contact_forces()["cube_tray"] > 0.05


def test_held_hovering_outside_and_unopened_are_not_success(env):
    values = env.signals.read_many(env.signals.describe())
    values["cube.position"][:] = values["support.position"] + values[
        "cube.size"
    ] * np.array([0, 0, 1])
    # Geometry alone is insufficient without contact.
    result = Task().evaluate(ObjectView(values))
    assert not result["metrics"]["placed"][0]
    env.mode = "drawer_open"
    env.reset(0)
    tray = env.data.body("drawer/tray").xpos.copy()
    place_for_test(env, "cube", tray + [0, -0.04, 0.026])
    env.wait(0.3)
    values = env.signals.read_many(env.signals.describe())
    values["contacts.cube_gripper"][:] = 1.0
    assert not Task("drawer_open").evaluate(ObjectView(values))["metrics"]["released"][
        0
    ]
    values["cube.position"][:, 0] += 0.2
    assert not Task("drawer_open").evaluate(ObjectView(values))["metrics"]["inside"][0]
    values["cube.position"][:, 0] -= 0.2
    values["drawer.joint_position"][:] = 0
    assert not Task("drawer_closed").evaluate(ObjectView(values))["metrics"]["placed"][
        0
    ]


def test_task_predicates_batch_and_hold_reset(env):
    import torch

    values = {
        key: value.repeat(2, *([1] * (value.ndim - 1)))
        for key, value in env.signals.read_many(env.signals.describe()).items()
    }
    values["cube.position"] = values["support.position"] + torch.tensor([[0, 0, 0.04]])
    for key in ("cube.orientation", "support.orientation"):
        values[key][:] = torch.tensor([1, 0, 0, 0])
    for key in ("contacts.cube_support", "contacts.support_table"):
        values[key][:] = 1
    for key in (
        "cube.linear_velocity",
        "cube.angular_velocity",
        "support.linear_velocity",
        "support.angular_velocity",
    ):
        values[key][:] = 0
    task = Task()
    for _ in range(24):
        assert not task.evaluate(ObjectView(values))["success"].any()
    values["contacts.cube_support"][1] = 0
    result = task.evaluate(ObjectView(values))
    assert result["success"].tolist() == [True, False]
    assert result["metrics"]["hold_seconds"][1] == 0


def test_actual_camera_frames(env):
    front, wrist = env.image(), env.image("wrist")
    assert front.shape == wrist.shape == (480, 640, 3)
    assert front.dtype == np.uint8
    assert front.std() > 10 and wrist.std() > 10
    assert not np.array_equal(front, wrist)


def test_time_budget(env):
    env.horizon = 0.1
    with pytest.raises(RuntimeError, match="budget"):
        env.wait(0.2)


def test_schema_metadata_cannot_override_existing_fields():
    from shared.assembly.object_schema import generate

    scene = ROOT / "projects/rax_tabletop/scene.yaml"
    _, _, factories, _ = prepare(scene, "mjlab")
    with pytest.raises(ValueError, match="Duplicate observation metadata"):
        generate(
            scene, factories=factories, extra_fields={"robot.position": {}}, write=False
        )


def test_time_budget_checks_rounded_control_ticks(env):
    env.horizon = env.data.time + 0.03
    before = env.data.time
    with pytest.raises(RuntimeError, match="budget"):
        env.wait(0.021)
    assert env.data.time == before


def test_drawer_stack_physical_support_sequence(env):
    env.mode = "drawer_stack"
    env.reset(0)
    assert abs(env.data.qpos[env.model.jnt_qposadr[env.slide]]) < 0.001
    # Predicate fixture, not a robot policy: set an open drawer and place the bottom.
    env.data.qpos[env.model.jnt_qposadr[env.slide]] = 0.179
    mujoco.mj_forward(env.model, env.data)
    tray = env.data.body("drawer/tray").xpos.copy()
    place_for_test(env, "support", tray + [0, -0.04, 0.026])
    env.wait(0.8)
    assert env.result["metrics"]["bottom_placed"][0]
    assert not env.result["success"][0]
    place_for_test(env, "cube", env.data.body("support/body").xpos + [0, 0, 0.041])
    env.wait(1)
    assert env.result["success"][0]
    assert env.contact_forces()["support_tray"] > 0.05
    assert env.contact_forces()["cube_support"] > 0.05


def drawer_stack_values(env):
    import torch

    values = env.signals.read_many(env.signals.describe())
    values["drawer.joint_position"][:] = 0.179
    values["drawer.joint_velocity"][:] = 0
    values["support.position"] = values["drawer.tray_position"] + torch.tensor(
        [[0, -0.04, 0.025]], dtype=torch.float64
    )
    values["cube.position"] = values["support.position"] + torch.tensor(
        [[0, 0, 0.04]], dtype=torch.float64
    )
    for name in ("cube", "support"):
        values[f"{name}.orientation"][:] = torch.tensor([1, 0, 0, 0])
        values[f"{name}.linear_velocity"][:] = 0
        values[f"{name}.angular_velocity"][:] = 0
        values[f"contacts.{name}_gripper"][:] = 0
    values["robot.gripper_closed_fraction"][:] = 0
    values["contacts.support_tray"][:] = 1
    values["contacts.cube_support"][:] = 0
    task = Task("drawer_stack")
    task.evaluate(ObjectView(values))
    assert task.bottom_placed[0]
    values["contacts.cube_support"][:] = 1
    return task, values


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "closed",
        "held-bottom",
        "held-top",
        "no-floor-contact",
        "hovering",
        "overhang",
        "moving-bottom",
        "fallen-bottom",
        "prestacked",
    ],
)
def test_drawer_stack_rejects_incomplete_or_invalid_outcomes(env, case):
    import torch

    task, values = drawer_stack_values(env)
    if case == "closed":
        values["drawer.joint_position"][:] = 0
    elif case == "held-bottom":
        values["contacts.support_gripper"][:] = 1
    elif case == "held-top":
        values["contacts.cube_gripper"][:] = 1
    elif case == "no-floor-contact":
        values["contacts.support_tray"][:] = 0
    elif case == "hovering":
        values["cube.position"][:, 2] += 0.02
        values["contacts.cube_support"][:] = 0
    elif case == "overhang":
        values["cube.position"][:, 0] += 0.075
        values["support.position"][:, 0] += 0.075
        values["cube.orientation"][:] = torch.tensor(
            [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]
        )
    elif case == "moving-bottom":
        values["support.linear_velocity"][:, 0] = 0.03
    elif case == "fallen-bottom":
        values["support.position"][:, 2] = -0.1
    elif case == "prestacked":
        task = Task("drawer_stack")
    for _ in range(24):
        assert not task.evaluate(ObjectView(values))["success"][0]
    result = task.evaluate(ObjectView(values))
    assert bool(result["success"][0]) == (case == "valid")
    if case == "overhang":
        assert not result["metrics"]["cube_inside"][0]
    if case == "fallen-bottom":
        assert result["failure"][0]
    values["contacts.cube_support"][:] = 0
    assert task.evaluate(ObjectView(values))["metrics"]["hold_seconds"][0] == 0
