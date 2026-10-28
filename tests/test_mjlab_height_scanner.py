"""mjlab height_scanner against the Isaac RayCaster definition: grid order, offset, alignment, hits.

Isaac's expected values come from Isaac Lab's own grid_pattern / quat_apply source files (loaded
standalone, no Kit) and the RayCaster update formulas; GPU ray casts are opt-in (RUN_MJLAB_GPU=1).
"""
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

GPU = pytest.mark.skipif(os.environ.get('RUN_MJLAB_GPU') != '1', reason='requires mjlab CUDA environment')


def isaac_source(relative):
    """A self-contained Isaac Lab source file (torch/numpy imports only), loaded without Isaac Sim."""
    spec = importlib.util.find_spec('isaaclab')
    path = Path(spec.submodule_search_locations[0]) / relative if spec and spec.submodule_search_locations else None
    if path is None or not path.is_file():
        pytest.skip('Isaac Lab source tree not found')
    module_spec = importlib.util.spec_from_file_location(f'isaac_reference_{path.stem}', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def isaac_rays(size, resolution, ordering='xy', direction=(0., 0., -1.), position=(0., 0., 0.),
               orientation=(1., 0., 0., 0.)):
    """RayCaster._initialize_rays_impl: Isaac grid, offset rotation on directions, offset position on starts."""
    patterns = isaac_source('sensors/ray_caster/patterns/patterns.py')
    maths = isaac_source('utils/math.py')
    starts, directions = patterns.grid_pattern(SimpleNamespace(size=size, resolution=resolution, ordering=ordering,
                                                               direction=direction), 'cpu')
    rotation = torch.tensor(orientation).repeat(len(directions), 1)
    return starts + torch.tensor(position), maths.quat_apply(rotation, directions)


def isaac_world_rays(starts, directions, pos_w, quat_w, alignment):
    """RayCaster._update_buffers_impl for each environment (no drift)."""
    maths = isaac_source('utils/math.py')
    count = starts.shape[0]
    rows = []
    for pos, quat in zip(pos_w, quat_w):
        quats = quat.repeat(count, 1)
        if alignment == 'world':
            rows.append((starts + pos, directions))
        elif alignment == 'yaw':
            rows.append((maths.quat_apply_yaw(quats, starts) + pos, directions))
        else:
            rows.append((maths.quat_apply(quats, starts) + pos, maths.quat_apply(quats, directions)))
    return torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows])


@pytest.mark.parametrize('size, resolution, ordering', [
    ((1.6, 1.0), .1, 'xy'), ((1.6, 1.0), .1, 'yx'), ((1.65, .55), .1, 'xy'), ((.3, .25), .07, 'yx'), ((0., 0.), .1, 'xy')])
def test_grid_matches_isaac_pattern(size, resolution, ordering):
    from shared.backends.mjlab.assets import isaac_grid
    position, orientation = (.05, -.02, 20.), (math.cos(.1), 0., math.sin(.1), 0.)
    expected_starts, expected_directions = isaac_rays(size, resolution, ordering, (0., 0., -2.), position, orientation)
    starts, directions = isaac_grid(size, resolution, (0., 0., -2.), ordering, position, orientation, 'cpu')
    torch.testing.assert_close(starts, expected_starts)
    # mjlab casts unit directions; the factory scales max_distance by the direction length instead.
    torch.testing.assert_close(directions, expected_directions / expected_directions.norm(dim=-1, keepdim=True))


def test_grid_order_is_isaac_meshgrid_order():
    from shared.backends.mjlab.assets import isaac_grid
    starts, _ = isaac_grid((.2, .1), .1, (0., 0., -1.), 'xy', (0., 0., 0.), (1., 0., 0., 0.), 'cpu')
    torch.testing.assert_close(starts[:, :2], torch.tensor(
        [[-.1, -.05], [0., -.05], [.1, -.05], [-.1, .05], [0., .05], [.1, .05]]))
    starts, _ = isaac_grid((.2, .1), .1, (0., 0., -1.), 'yx', (0., 0., 0.), (1., 0., 0., 0.), 'cpu')
    torch.testing.assert_close(starts[:, :2], torch.tensor(
        [[-.1, -.05], [-.1, .05], [0., -.05], [0., .05], [.1, -.05], [.1, .05]]))


def test_factory_maps_and_rejects_isaac_parameters():
    pytest.importorskip('mjlab')
    from shared.backends.mjlab.assets import height_scanner
    pattern = {'resolution': .1, 'size': [1.6, 1.0]}
    cfg = height_scanner(pattern={**pattern, 'direction': [0, 0, -2]}, max_distance=3., attach_yaw_only=True,
                         ray_alignment='base', update_period=.02)
    assert cfg.ray_alignment == 'yaw' and cfg.max_distance == pytest.approx(6.) and cfg.update_period == .02
    assert height_scanner(pattern=pattern, attach_yaw_only=False, ray_alignment='world').ray_alignment == 'base'
    for parameters, error in [({'history_length': 1}, NotImplementedError),
                              ({'drift_range': [-.1, .1]}, NotImplementedError),
                              ({'ray_cast_drift_range': {'x': [0., .1]}}, NotImplementedError),
                              ({'visualizer_cfg': {}}, NotImplementedError),
                              ({'offset': {'pos': [0, 0, 20]}}, ValueError),
                              ({'mesh_prim_paths': ['/World/ground']}, ValueError),
                              ({'ray_alignment': 'roll'}, ValueError),
                              ({'max_distance': 0}, ValueError)]:
        with pytest.raises(error):
            height_scanner(pattern=pattern, **parameters)
    with pytest.raises(NotImplementedError):
        height_scanner(pattern={**pattern, 'type': 'lidar'})
    with pytest.raises(ValueError):
        height_scanner(pattern={'resolution': .1})


def scanner(**changes):
    node = {'name': 'height_scanner', 'factory': 'height_scanner', 'kind': 'sensor',
            'mount': {'entity': 'world/robot', 'body': 'base'}, 'pose': {'position': [0., 0., 20.]},
            'meshes': ['world/ground'],
            'parameters': {'ray_alignment': 'yaw', 'pattern': {'resolution': .1, 'size': [1.6, 1.0]}}}
    node.update(changes)
    return {key: value for key, value in node.items() if value is not None}


def scene_settings(*sensors, ground_pose=None, step=None):
    ground = {'name': 'ground', 'factory': 'ground_plane'}
    if ground_pose:
        ground['pose'] = ground_pose
    entities = [ground, {'name': 'robot', 'factory': 'go2', 'kind': 'robot', 'pose': {'position': [0., 0., .5]}}]
    if step is not None:
        entities.append({'name': 'step', 'factory': 'box', 'pose': {'position': list(step['position'])},
                         'parameters': {'size': list(step['size']), **({'mass': 1.} if step.get('free') else {})}})
    return {'name': 'world', 'entities': entities + list(sensors)}


def world(settings):
    from shared.assembly.layout import world_spec_from_config
    return world_spec_from_config(settings, directory=Path('.'), cameras=True, backend='mjlab')


@pytest.mark.parametrize('settings, step_dt, message', [
    (scene_settings(scanner(pose={'position': [0, 0, 20], 'orientation': [math.cos(.1), 0, math.sin(.1), 0]})),
     None, 'vertical'),
    (scene_settings(scanner(meshes=['world/ground', 'world/step']), step={'position': [1, 0, .1], 'size': [.4, .4, .2]}),
     None, 'exactly one'),
    (scene_settings(scanner(), ground_pose={'position': [0, 0, .1]}), None, 'z = 0'),
    (scene_settings(scanner(parameters={'update_period': .05, 'pattern': {'resolution': .1, 'size': [1., 1.]}})),
     .02, 'update_period'),
    (scene_settings({'name': 'eye', 'factory': 'camera', 'kind': 'sensor', 'meshes': ['world/ground'],
                     'mount': {'entity': 'world/robot', 'body': 'base'}}), None, 'only from'),
])
def test_scene_rejects_scanners_mjlab_cannot_reproduce(settings, step_dt, message):
    from shared.backends.mjlab.scene import validate
    with pytest.raises(NotImplementedError, match=message):
        validate(world(settings), settings, step_dt=step_dt)


def test_scene_requires_mount_and_static_single_target():
    pytest.importorskip('mjlab')
    from shared.assembly.assets import factories_for
    from shared.backends.mjlab.scene import build_scene_config, validate
    settings = scene_settings(scanner(mount=None))
    with pytest.raises(ValueError, match='mount'):
        validate(world(settings), settings)
    settings = scene_settings(scanner(meshes=['world/step']), step={'position': [1, 0, .5], 'size': [.4, .4, .2], 'free': True})
    with pytest.raises(NotImplementedError, match='free-moving'):
        build_scene_config(world(settings), settings, factories_for('mjlab'))
    other = scanner(name='front', meshes=['world/step'])
    settings = scene_settings(scanner(), other, step={'position': [1, 0, .1], 'size': [.4, .4, .2]})
    with pytest.raises(NotImplementedError, match='share one meshes target'):
        build_scene_config(world(settings), settings, factories_for('mjlab'))


def test_ray_target_copies_keep_geometry_and_skip_contacts():
    pytest.importorskip('mjlab')
    from shared.assembly.assets import factories_for
    from shared.backends.mjlab.scene import RAY_TARGET_GROUP, compile_scene
    settings = scene_settings(scanner(meshes=['world/step']), step={'position': [.3, 0, .1], 'size': [.4, .4, .2]})
    scene, model, data, keys = compile_scene(world(settings), settings, factories_for('mjlab'))
    copies = [geom for geom in range(model.ngeom) if model.geom_group[geom] == RAY_TARGET_GROUP]
    assert [model.body(model.geom_bodyid[geom]).name for geom in copies] == ['step/body']
    original = model.geom('step/geometry').id
    copy, = copies
    assert model.geom_contype[copy] == model.geom_conaffinity[copy] == 0
    assert model.geom_type[copy] == model.geom_type[original]
    for field in (data.geom_xpos, data.geom_xmat, model.geom_size):
        assert (field[copy] == field[original]).all()
    sensor = scene.sensors['height_scanner']
    assert sensor.cfg.include_geom_groups == (RAY_TARGET_GROUP,) and not sensor.cfg.exclude_parent_body


def gpu_scene(settings, num_envs=2):
    from mjlab.scene import Scene
    from mjlab.sim import Simulation, SimulationCfg
    from shared.assembly.assets import factories_for
    from shared.backends.mjlab.scene import build_scene_config
    cfg, _ = build_scene_config(world(settings), settings, factories_for('mjlab'))
    cfg.num_envs = num_envs
    scene = Scene(cfg, device='cuda:0')
    sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=scene.compile(), device='cuda:0')
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    sim.set_sensor_context(scene.sensor_context)
    return scene, sim


def quaternion(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def place(scene, sim):
    """Robot bases: level at 0.5 m in env 0; rolled, pitched and turned at 0.6 m in env 1."""
    from shared.backends.mjlab.readers import bind_reader
    origins = scene.env_origins
    for entity in scene.entities.values():
        if entity.is_fixed_base and entity.is_mocap:
            # As mjlab's reset event: the default pose in each environment (static entities).
            mocap = entity.indexing.mocap_id
            sim.data.mocap_pos[:, mocap] = entity.data.default_root_state[:, :3] + origins
            sim.data.mocap_quat[:, mocap] = entity.data.default_root_state[:, 3:7]
    robot = scene['robot']
    state = torch.zeros(2, 13, device='cuda:0')
    state[:, :3] = torch.tensor([[0., 0., .5], [.2, -.1, .6]], device='cuda:0') + origins
    state[:, 3:7] = torch.tensor([[1., 0., 0., 0.], quaternion(.15, -.1, .7)], device='cuda:0')
    robot.write_root_state_to_sim(state)
    sim.forward()
    sim.sense()
    env = SimpleNamespace(scene=scene, device='cuda:0', num_envs=2)
    read = lambda path: bind_reader(env, {}, path)()
    return state[:, :3].cpu(), state[:, 3:7].cpu(), origins.cpu(), read


@GPU
@pytest.mark.parametrize('alignment, orientation', [
    ('yaw', (1., 0., 0., 0.)), ('world', (1., 0., 0., 0.)), ('base', (math.cos(.1), 0., math.sin(.1), 0.))])
def test_ground_hits_match_isaac_ray_caster(alignment, orientation):
    position = (.05, -.02, 20.)
    # A non-target box under the robot must not block rays: Isaac casts only at its one mesh.
    settings = scene_settings(scanner(pose={'position': list(position), 'orientation': list(orientation)},
                                      parameters={'ray_alignment': alignment, 'max_distance': 40.,
                                                  'pattern': {'resolution': .1, 'size': [1.6, 1.0]}}),
                              step={'position': [0., 0., .1], 'size': [.6, .6, .2]})
    scene, sim = gpu_scene(settings)
    pos_w, quat_w, origins, read = place(scene, sim)
    hits = read('height_scanner.ray_hit_positions').cpu()
    sensor_position = read('height_scanner.position').cpu()
    assert hits.shape == (2, 187, 3)  # Isaac: 17 x 11 grid for 1.6 m x 1.0 m at 0.1 m
    torch.testing.assert_close(sensor_position, pos_w - origins)  # RayCasterData.pos_w = body origin
    starts, directions = isaac_world_rays(*isaac_rays((1.6, 1.0), .1, 'xy', (0., 0., -1.), position, orientation),
                                          pos_w, quat_w, alignment)
    distance = -starts[..., 2] / directions[..., 2]  # infinite ground plane z = 0, unit directions
    expected = torch.where(((distance >= 0) & (distance <= 40.))[..., None],
                           starts + distance[..., None] * directions, torch.inf) - origins[:, None, :]
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(hits, expected, atol=2e-4, rtol=0)
    # Isaac mdp.height_scan: sensor height - hit height - offset; over flat ground = base height - offset.
    height = sensor_position[:, 2:3] - hits[..., 2] - .5
    torch.testing.assert_close(height, (pos_w[:, 2:3] - .5).expand_as(height), atol=2e-4, rtol=0)


@GPU
def test_step_target_hits_and_misses():
    settings = scene_settings(scanner(meshes=['world/step']), step={'position': [.33, .04, .1], 'size': [.4, .4, .2]})
    scene, sim = gpu_scene(settings)
    pos_w, quat_w, origins, read = place(scene, sim)
    hits = read('height_scanner.ray_hit_positions').cpu()
    starts, _ = isaac_world_rays(*isaac_rays((1.6, 1.0), .1, 'xy', (0., 0., -1.), (0., 0., 20.)),
                                 pos_w, quat_w, 'yaw')
    local = starts[..., :2] - origins[:, None, :2] - torch.tensor([.33, .04])
    inside = (local.abs() < .2 - 1e-3).all(-1)
    outside = (local.abs() > .2 + 1e-3).any(-1)
    assert inside.sum() > 10 and outside.sum() > 10
    torch.testing.assert_close(hits[inside][:, 2], torch.full((int(inside.sum()),), .2), atol=2e-4, rtol=0)
    torch.testing.assert_close(hits[inside][:, :2], (starts[..., :2] - origins[:, None, :2])[inside], atol=2e-4, rtol=0)
    assert torch.isinf(hits[outside]).all()  # Isaac: misses are inf (mjlab reports -1)
