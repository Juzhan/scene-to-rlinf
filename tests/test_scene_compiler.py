"""Common scene authoring compiles to native configuration without a GPU."""

from copy import deepcopy
import ast
from importlib.util import find_spec, module_from_spec, spec_from_file_location
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from shared.assembly.spec import Entity, FixedJoint, Mount, Pose, Robot, RootReset, Scene, Sensor, WorldSpec
from shared.backends.isaac.scene import build_scene_config, event_configuration, scene_metadata


_ISAAC_SPEC = find_spec("isaaclab")


def installed_source():
    if _ISAAC_SPEC is None or _ISAAC_SPEC.origin is None:
        pytest.skip("Installed Isaac Lab source is unavailable")
    package = Path(_ISAAC_SPEC.origin).parent
    source = next((path for path in (package / "source/isaaclab/isaaclab", package)
                   if (path / "managers/event_manager.py").exists()), None)
    if source is None:
        pytest.skip("Installed Isaac Lab source is unavailable")
    return source


class Config(SimpleNamespace):
    pass


class SensorConfig(Config):
    pass


class CameraConfig(SensorConfig):
    class_type = object


class RigidConfig(Config):
    pass


class ArticulationConfig(Config):
    pass


@pytest.fixture
def native(monkeypatch):
    root = ModuleType("isaaclab")
    root.__path__ = []
    monkeypatch.setitem(sys.modules, root.__name__, root)
    for name, values in {
        "scene": {"InteractiveSceneCfg": Config},
        "sensors": {"SensorBaseCfg": SensorConfig, "CameraCfg": CameraConfig},
        "assets": {"RigidObjectCfg": RigidConfig, "ArticulationCfg": ArticulationConfig},
        "managers": {"EventTermCfg": Config, "SceneEntityCfg": lambda name: Config(name=name)},
    }.items():
        module = ModuleType("isaaclab." + name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        setattr(root, name, module)


def asset(position=(0, 0, .4)):
    return RigidConfig(prim_path="/old", init_state=Config(pos=position, rot=(1, 0, 0, 0)),
                       spawn=None, collision_group=0)


def sensor():
    return SensorConfig(prim_path="/old", offset=Config(pos=(0, 0, 0), rot=(1, 0, 0, 0)), mesh_prim_paths=[])


def test_common_project_scenes_load_as_one_schema():
    from shared.assembly.layout import load_world_spec

    root = Path(__file__).resolve().parents[1] / "projects"
    lift = load_world_spec(root / "assembly_lift/scene.yaml")
    go2 = load_world_spec(root / "assembly_go2/scene.yaml")
    assert set(scene_metadata(lift).keys.values()) >= {"robot", "object", "ee_frame"}
    metadata = scene_metadata(go2)
    assert metadata.prim_paths["world/robot"] == "{ENV_REGEX_NS}/Robot"
    assert metadata.prim_paths["world/terrain"] == "/World/ground"
    assert metadata.prim_paths["world/contact_forces"] == "{ENV_REGEX_NS}/Robot/.*"


def test_omitted_pose_keeps_factory_height_and_explicit_identity_overrides(native):
    world = WorldSpec(Scene("world", children=(
        Robot("default", factory="body", pose_authored=False),
        Robot("identity", factory="body"),
        Scene("group", pose=Pose((1, 2, 0)), children=(
            Robot("nested", factory="body", pose_authored=False),)),
    )))
    prototype = asset()
    config = build_scene_config(world, factories={"body": lambda: prototype})
    assert config.default.init_state.pos == (0, 0, .4)
    assert config.identity.init_state.pos == (0, 0, 0)
    assert config.group__nested.init_state.pos == (1, 2, .4)
    assert prototype.prim_path == "/old"
    assert prototype.init_state.pos == (0, 0, .4)


def test_native_sensors_mount_regex_and_mesh_references(native):
    world = WorldSpec(Scene("world", children=(
        Sensor("scanner", factory="scan", mount=Mount("world/robot", "base"),
               pose=Pose((0, 0, 20)), meshes=("world/terrain",)),
        Sensor("contacts", factory="scan", mount=Mount("world/robot", ".*")),
        Robot("robot", factory="body", prim_name="Robot"),
        Entity("terrain", factory="ground", prim_name="ground", scope="global"),
    )))
    config = build_scene_config(world, factories={"body": asset, "scan": sensor, "ground": lambda: Config(prim_path="")})
    assert config.scanner.prim_path == "{ENV_REGEX_NS}/Robot/base"
    assert config.scanner.offset.pos == (0, 0, 20)
    assert config.scanner.mesh_prim_paths == ["/World/ground"]
    assert config.contacts.prim_path == "{ENV_REGEX_NS}/Robot/.*"


def test_camera_tracks_body_independently_and_frame_targets_relocate(native):
    world = WorldSpec(Scene("world", children=(
        Robot("arm", factory="body", children=(
            Sensor("camera", factory="camera", mount=Mount("..", "hand")),
            Sensor("frame", factory="frame", mount=Mount("..", "base")),
        )),
    )))
    frame = SensorConfig(prim_path="", source_frame_offset=Config(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
                         target_frames=[Config(prim_path="{SOURCE_ASSET}/hand")])
    config = build_scene_config(world, factories={
        "body": asset, "camera": lambda: CameraConfig(**vars(sensor())), "frame": lambda: frame})
    assert config.arm__camera.prim_path == "{ENV_REGEX_NS}/arm__camera"
    assert config.arm__camera.update_latest_camera_pose is True
    assert config.arm__camera.class_type.__name__ == "MountedCamera"
    assert config.arm__frame.prim_path == "{ENV_REGEX_NS}/arm/base"
    assert config.arm__frame.target_frames[0].prim_path == "{ENV_REGEX_NS}/arm/hand"
    assert frame.target_frames[0].prim_path == "{SOURCE_ASSET}/hand"


def test_mounted_camera_samples_current_body_after_renderer_pose_propagation(native, monkeypatch):
    import torch
    from shared.backends.isaac.camera import mounted_camera_class

    math_spec = spec_from_file_location("_scene_camera_math", installed_source() / "utils/math.py")
    math_module = module_from_spec(math_spec)
    math_spec.loader.exec_module(math_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)
    transforms = torch.tensor([[20., 0, 0, 0, 0, 0, 1], [10., 0, 0, 0, 0, 0, 1]])
    body = Config(prim_paths=["/World/envs/env_1/robot/hand", "/World/envs/env_0/robot/hand"],
                  get_transforms=lambda: transforms)
    queued = rendered = torch.zeros((2, 3))
    render_count = 0
    camera = None

    def render():
        nonlocal queued, rendered, render_count
        render_count += 1
        rendered, queued = queued, camera.pending.clone()

    class BaseCamera:
        def __init__(self):
            self.device = "cpu"
            self.cfg = Config(prim_path="/World/envs/env_.*/camera", offset=Config(
                pos=(.2, 0., .06), rot=(1., 0., 0., 0.), convention="opengl"))
            self._view = Config(prim_paths=["/World/envs/env_0/camera", "/World/envs/env_1/camera"])

        def _initialize_impl(self):
            pass

        def set_world_poses(self, positions, orientations, *, convention):
            self.pending = positions.clone()

        def _update_buffers_impl(self, env_ids):
            self.sampled_image_position = rendered.clone()

    sim = ModuleType("isaaclab.sim")
    sim.SimulationContext = Config(instance=lambda: Config(render=render))
    manager = ModuleType("isaacsim.core.simulation_manager")
    manager.SimulationManager = Config(get_physics_sim_view=lambda: Config(create_rigid_body_view=lambda expr: body))
    monkeypatch.setitem(sys.modules, sim.__name__, sim)
    monkeypatch.setitem(sys.modules, manager.__name__, manager)
    camera = mounted_camera_class(BaseCamera, "{ENV_REGEX_NS}/robot/hand")()
    camera._initialize_impl()
    camera._update_buffers_impl([0, 1])
    torch.testing.assert_close(camera.sampled_image_position, torch.tensor([[10.2, 0, .06], [20.2, 0, .06]]))
    transforms[:, 0] += 3
    camera._update_buffers_impl([0, 1])
    torch.testing.assert_close(camera.sampled_image_position, torch.tensor([[13.2, 0, .06], [23.2, 0, .06]]))
    assert render_count == 4


def test_nested_group_scope_inherits_through_unspecified_groups():
    world = WorldSpec(Scene("world", children=(Scene("hall", scope="global", children=(
        Scene("station", children=(Entity("box", factory="body"),)),)),)))
    assert scene_metadata(world).prim_paths["world/hall/station/box"] == "/World/hall__station__box"


@pytest.mark.parametrize("children, message", [
    ((Entity("a__b", factory="body"), Scene("a", children=(Entity("b", factory="body"),))), "key collision"),
    ((Entity("a", factory="body", prim_name="same"), Entity("b", factory="body", prim_name="same")), "prim collision"),
    ((Entity("a", factory="body"), Sensor("s", factory="scan", mount=Mount("world/a"), scope="global")), "scope must match"),
    ((Sensor("s", factory="scan", meshes=("world/missing",)),), "Mount must reference"),
])
def test_ambiguous_scene_mapping_is_rejected(children, message):
    with pytest.raises(ValueError, match=message):
        scene_metadata(WorldSpec(Scene("world", children=children)))


def test_body_patterns_are_for_sensors_not_physical_connections():
    world = WorldSpec(Scene("world", children=(Entity("box", factory="body"),)), connections=(
        FixedJoint("anchor", child=Mount("world/box", ".*")),))
    with pytest.raises(ValueError, match="exact body paths"):
        world.index()


def test_only_the_attachment_root_owns_a_pose_reset(native):
    rule = RootReset(pose_range={"x": (-.1, .1)})
    joint = (FixedJoint("arm", child=Mount("world/arm"), parent=Mount("world/dog")),)
    carried = WorldSpec(Scene("world", children=(Entity("dog", factory="body", reset=rule),
                                                 Entity("arm", factory="body"))), connections=joint)
    carried.index()  # Roots are randomized first; attachment poses are placed after them.
    events = event_configuration(carried)
    assert list(events).index("assembly_root_poses") < list(events).index("assembly_attachment_poses")
    loose = WorldSpec(Scene("world", children=(Entity("dog", factory="body"),
                                               Entity("arm", factory="body", reset=rule))), connections=joint)
    with pytest.raises(ValueError, match="placed by its parent"):
        loose.index()


def test_scene_event_terms_are_native_and_serializable(native):
    import json

    world = WorldSpec(Scene("world", children=(
        Entity("table", factory="body"), Entity("tool", factory="body"),
        Entity("box", factory="body", reset=RootReset(pose_range={"x": (-.1, .1)})),
    )), connections=(FixedJoint("attachment", child=Mount("world/tool"), parent=Mount("world/table")),))
    config = build_scene_config(world, factories={"body": asset})
    events = event_configuration(world)
    assert list(events) == ["assembly_root_poses", "assembly_connections", "assembly_attachment_poses"]
    assert events["assembly_connections"].mode == "prestartup"
    assert events["assembly_attachment_poses"].mode == "reset"
    assert events["assembly_root_poses"].params["rules"]["box"]["pose_range"] == {"x": (-.1, .1)}
    for event in events.values():
        json.dumps(event.params)
    assert config.tool.prim_path == "{ENV_REGEX_NS}/tool"

    # The native resolver reorders optional arguments before validating params.
    # In particular, env_ids must remain a required positional event argument.
    source = installed_source() / "managers/manager_base.py"
    tree = ast.parse(source.read_text())
    resolver = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "_resolve_common_term_cfg")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              resolver], type_ignores=[])
    context = {"inspect": inspect, "ManagerTermBaseCfg": Config}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), context)
    manager = Config(_env=Config(sim=Config(is_playing=lambda: False)))
    for name, term in events.items():
        context["_resolve_common_term_cfg"](manager, name, term, min_argc=2)


def test_fixed_connections_disable_replication_for_installed_sdk_guard(native):
    # Execute the installed SDK's guard, without importing Kit. This validates
    # the integration contract rather than copying the guard into a fake SDK.
    source = installed_source() / "managers/event_manager.py"
    tree = ast.parse(source.read_text())
    guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and any(isinstance(part, ast.Attribute) and part.attr == "replicate_physics"
                         for part in ast.walk(node.test)))
    code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), str(source), "exec")
    term = Config(mode="prestartup")
    with pytest.raises(RuntimeError, match="Scene replication is enabled"):
        exec(code, {"term_cfg": term, "self": Config(_env=Config(scene=Config(cfg=Config(replicate_physics=True))))})
    world = WorldSpec(Scene("world", children=(Entity("tool", factory="body"),)), connections=(
        FixedJoint("anchor", child=Mount("world/tool")),))
    config = build_scene_config(world, factories={"body": asset}, replicate_physics=True, clone_in_fabric=True)
    assert config.replicate_physics is False
    assert config.clone_in_fabric is False
    exec(code, {"term_cfg": term, "self": Config(_env=Config(scene=Config(cfg=config)))})


def test_shared_assets_collide_with_each_cloned_environment(native):
    world = WorldSpec(Scene("world", children=(
        Entity("floor", factory="body", scope="global"), Entity("box", factory="body"),)))
    config = build_scene_config(world, factories={"body": asset})
    assert config.floor.collision_group == -1
    assert config.box.collision_group == 0


@pytest.mark.parametrize("config", [
    Config(prim_path="", init_state=Config(pos=(0, 0, 0), rot=(1, 0, 0, 0))),
    ArticulationConfig(prim_path="", init_state=Config(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
                       spawn=Config(articulation_props=Config(fix_root_link=True))),
    RigidConfig(prim_path="", init_state=Config(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
                spawn=Config(rigid_props=Config(kinematic_enabled=True))),
])
def test_pose_reset_rejects_non_dynamic_rigid_targets(native, config):
    world = WorldSpec(Scene("world", children=(Entity(
        "target", factory="target", reset=RootReset(pose_range={"x": (-.1, .1)})),)))
    with pytest.raises(ValueError, match="pose reset requires a dynamic rigid object"):
        build_scene_config(world, factories={"target": lambda: config})


def test_pose_reset_accepts_a_free_root_articulation(native):
    world = WorldSpec(Scene("world", children=(Entity(
        "target", factory="target", reset=RootReset(pose_range={"x": (-.1, .1)})),)))
    config = ArticulationConfig(prim_path="", init_state=Config(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
                                spawn=Config(articulation_props=Config(fix_root_link=False)))
    build_scene_config(world, factories={"target": lambda: config})


def test_native_training_does_not_silently_drop_semantic_initial_state(native):
    world = WorldSpec(Scene("world", children=(Entity("switch", factory="body", initial_state={"on": False}),)))
    with pytest.raises(NotImplementedError, match="initialize custom state through a project task event"):
        build_scene_config(world, factories={"body": asset})


def test_pose_list_reset_changes_only_selected_environments(native, monkeypatch):
    import torch
    from shared.backends.isaac.scene import reset_root_poses

    mdp = ModuleType("isaaclab.envs.mdp")
    mdp.reset_root_state_uniform = lambda *args, **kwargs: pytest.fail("Unexpected range reset")
    monkeypatch.setitem(sys.modules, mdp.__name__, mdp)
    written = torch.zeros((3, 13))
    defaults = written.clone()
    defaults[:, 3] = 1

    class Body:
        data = Config(default_root_state=defaults)

        def write_root_state_to_sim(self, states, *, env_ids):
            written[env_ids] = states

    class NativeScene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.]])

    env = Config(scene=NativeScene(box=Body()), device="cpu", num_envs=3)
    reset_root_poses(env, torch.tensor([1]), rules={"box": {"states": [
        {"position": (1., 2., .3), "orientation": (1., 0., 0., 0.)}], "pose_range": {}}})
    torch.testing.assert_close(written[1, :3], torch.tensor([5., 2., .3]))
    assert not written[[0, 2]].any()


def test_reused_config_instances_remain_independent(native):
    prototype = asset()
    before = deepcopy(prototype)
    world = WorldSpec(Scene("world", children=(Entity("left", factory="body"), Entity("right", factory="body"))))
    config = build_scene_config(world, factories={"body": lambda: prototype})
    config.left.init_state.pos = (7, 0, 0)
    assert config.right.init_state.pos == (0, 0, 0)
    assert vars(prototype.init_state) == vars(before.init_state)


def test_authored_physical_initial_state_reaches_native_config(native):
    preset = ArticulationConfig(prim_path='/old', init_state=Config(pos=(0,0,.4), rot=(1,0,0,0),
        lin_vel=(0,0,0), ang_vel=(0,0,0), joint_pos={'.*': 0}, joint_vel={'.*': 0}), spawn=None)
    world = WorldSpec(Scene('world', children=(Robot('arm', factory='arm', initial_state={
        'joint_positions': {'joint1': .6, 'joint2': -.2}, 'joint_velocities': {'joint1': .1, 'joint2': 0},
        'linear_velocity': [1,0,0], 'angular_velocity': [0,0,.5]}),)))
    cfg = build_scene_config(world, factories={'arm': lambda: preset})
    assert cfg.arm.init_state.joint_pos == {'joint1': .6, 'joint2': -.2}
    assert cfg.arm.init_state.lin_vel == (1,0,0)
    assert cfg.arm.init_state.ang_vel == (0,0,.5)
    assert preset.init_state.joint_pos == {'.*': 0}
