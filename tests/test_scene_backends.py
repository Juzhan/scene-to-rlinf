from pathlib import Path
import subprocess
import sys
import pytest
from shared.assembly.backend_scene import prepare
from shared.assembly.layout import world_spec_from_config
from shared.backends.mjlab.scene import validate

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'projects/dual_backend_scene/scene.yaml'


def test_shared_layout_and_lazy_imports():
    reports = [prepare(SCENE, backend)[3] for backend in ('isaac', 'mjlab')]
    assert reports[0]['entities'] == reports[1]['entities']
    assert len(reports[0]['entities']) == 5
    assert not reports[1]['rlinf_training_connected']
    subprocess.run([sys.executable, '-c',
        "from shared.assembly.backend_scene import prepare; import sys; "
        f"prepare({str(SCENE)!r}, 'mjlab'); "
        "assert not {'mujoco','mjlab','isaaclab','torch'} & sys.modules.keys()"],
        cwd=ROOT, check=True)


def test_unsupported_features_fail_before_simulator():
    settings, world, _, _ = prepare(SCENE, 'mjlab')
    with pytest.raises(NotImplementedError, match='unknown_option'):
        validate(world, {**settings, 'unknown_option': {}})
    world.root.children[1].prim_name = 'custom_prim'  # only renames the Isaac prim
    validate(world, settings)
    from shared.assembly.spec import BodyFrame, Pose
    robot = next(node for node in world.root.children if node.kind == 'robot')
    other = next(node for node in world.root.children if node.kind != 'robot' and node.factory)
    # Robot frames are read through the shared robot interface; nothing else carries frames in mjlab yet.
    robot.frames = (BodyFrame('tip', 'base', Pose()),)
    validate(world, settings)
    other.frames = (BodyFrame('tip', '', Pose()),)
    with pytest.raises(NotImplementedError, match='robot entities only'):
        validate(world, settings)


def test_native_mjlab_initial_state_and_layout():
    pytest.importorskip('mjlab')
    import numpy as np
    import mujoco
    from shared.backends.mjlab.scene import compile_scene
    settings, world, factories, _ = prepare(SCENE, 'mjlab')
    scene, model, data, keys = compile_scene(world, settings, factories)
    assert model.nv == 18 and model.nu == 12
    for path, expected in [('world/robot', [-1, 0, .42]),
                           ('world/steps/second', [.7, 0, .08]),
                           ('world/steps/platform', [1.3, 0, .12])]:
        body = model.body(scene.entities[keys[path]].root_body.name)
        np.testing.assert_allclose(data.xpos[body.id], expected, atol=1e-7)
    for j in range(model.njnt):
        name = model.joint(j).name
        for suffix, expected in [('hip_joint', 0), ('thigh_joint', .8), ('calf_joint', -1.5)]:
            if name.endswith(suffix):
                assert data.qpos[model.jnt_qposadr[j]] == pytest.approx(expected)
    # CPU reference stepping verifies the compiled model, not GPU or RL training.
    mujoco.mj_step(model, data, nstep=10)
    assert np.isfinite(data.qpos).all()
    assert data.time == pytest.approx(.05)


def test_native_free_box_and_rotated_parent():
    pytest.importorskip('mjlab')
    import numpy as np
    from shared.assembly.assets import factories_for
    from shared.backends.mjlab.scene import compile_scene
    settings = {'name':'world', 'entities':[{'name':'group',
        'pose':{'position':[1,2,0], 'orientation':[.7071067811865476,0,0,.7071067811865476]},
        'children':[{'name':'box', 'factory':'box', 'parameters':{'mass':2},
                     'pose':{'position':[1,0,1]}}]}]}
    world = world_spec_from_config(settings, directory=ROOT, backend='mjlab')
    scene, model, data, keys = compile_scene(world, settings, factories_for('mjlab'))
    body = model.body(scene.entities[keys['world/group/box']].root_body.name)
    np.testing.assert_allclose(data.xpos[body.id], [1,3,1], atol=1e-7)
    assert model.nv == 6


def test_mounted_camera_hides_only_its_enclosing_housing_and_frames_compile():
    pytest.importorskip('mjlab')
    from shared.assembly.assets import factories_for
    from shared.backends.mjlab.scene import HIDDEN_GEOM_GROUP, compile_scene, event_configuration
    settings = {'name': 'world', 'entities': [
        {'name': 'housing', 'factory': 'box', 'parameters': {'size': [.1, .1, .1]}, 'pose': {'position': [0, 0, 1]}},
        {'name': 'table', 'factory': 'box', 'parameters': {'size': [1, 1, .1]}},
        {'name': 'area', 'kind': 'abstract', 'shape': 'rectangle', 'size': [.2, .2], 'pose': {'position': [0, 0, .1]}},
        {'name': 'cube', 'factory': 'box', 'parameters': {'size': [.05, .05, .05], 'mass': .1},
         'pose': {'position': [0, 0, .2]}, 'reset': {'region': 'world/area'}},
        {'name': 'eye', 'factory': 'camera', 'kind': 'sensor', 'mount': {'entity': 'world/housing', 'body': 'body'}},
        {'name': 'tip', 'factory': 'frame_transformer', 'kind': 'sensor', 'parameters': {'target_body': 'body'},
         'mount': {'entity': 'world/housing', 'body': 'body'}}]}
    world = world_spec_from_config(settings, directory=ROOT, cameras=True, backend='mjlab')
    scene, model, data, keys = compile_scene(world, settings, factories_for('mjlab'))
    groups = {model.body(model.geom_bodyid[geom]).name: int(model.geom_group[geom]) for geom in range(model.ngeom)}
    # Only the box enclosing the camera origin leaves the rendered groups; its contact is unchanged.
    assert groups['housing/body'] == HIDDEN_GEOM_GROUP and groups['table/body'] == 0
    assert model.geom_contype[model.body('housing/body').geomadr[0]] == 1
    assert model.camera('eye').bodyid == model.body('housing/body').id
    assert model.sensor('tip').objid == model.site(model.sensor('tip').objid[0]).id
    events = event_configuration(world, keys, scene.entities)['assembly_root_resets'].params
    assert list(events['rules']) == ['cube'] and list(events['areas']) == ['world/area']


def test_mjlab_depth_follows_isaac_clipping():
    import math
    from types import SimpleNamespace
    import torch
    from shared.backends.mjlab.readers import depth, image_plane_cosine
    cfg = SimpleNamespace(width=4, height=2, fovy=90., clipping_range=(.02, 3.), depth_clipping='max')
    cosine = image_plane_cosine(cfg, 'cpu')
    # Pixel centres at x = +-0.5, +-1.5 (scaled by aspect 2) and y = +-0.5 in units of the focal length.
    assert cosine[0, 0, 0] == pytest.approx(1 / math.sqrt(1 + 1.5 ** 2 + .5 ** 2))
    ray = torch.tensor([[2., 0., 4., 1.], [1., 1., 1., 1.]])[None, ..., None]
    assert depth(ray, cfg, 'distance_to_camera')[0, 0, :, 0].tolist() == [2., 3., 3., 1.]
    plane = depth(ray, cfg, 'distance_to_image_plane', cosine)
    assert plane[0, 0, 0, 0] == pytest.approx(2 * float(cosine[0, 0, 0]))
    assert depth(ray, SimpleNamespace(**{**vars(cfg), 'depth_clipping': 'zero'}), 'distance_to_camera')[0, 0, 1, 0] == 0
    assert math.isinf(depth(ray, SimpleNamespace(**{**vars(cfg), 'depth_clipping': 'none'}), 'distance_to_camera')[0, 0, 2, 0])
