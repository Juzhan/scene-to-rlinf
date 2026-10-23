import json

import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('pxr')
pytest.importorskip('trimesh')

import trimesh

from shared.asset_formats import mjcf_for
from shared.terrain_assets import build, tile_pieces

SIZE = 4.


def tile_mesh(mesh, offset=(0., 0., 0.)):
    mesh = mesh.copy()
    mesh.apply_translation(np.array(offset) + [SIZE / 2, SIZE / 2, 0.])
    return mesh.vertices, mesh.faces


def box(extents, centre):
    mesh = trimesh.creation.box(extents)
    mesh.apply_translation(centre)
    return tile_mesh(mesh)


def grid(heights, size=SIZE):
    n = len(heights)
    xs = np.linspace(-size / 2, size / 2, n)
    vertices = np.array([(x, y, heights[j][i]) for j, y in enumerate(xs) for i, x in enumerate(xs)])
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    return tile_mesh(trimesh.Trimesh(vertices, faces, process=False))


def test_closed_pieces_become_boxes_and_convex_hulls_and_open_flat_becomes_slab():
    ground = tile_mesh(trimesh.Trimesh([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], [[0, 1, 2], [0, 2, 3]]))
    pyramid = trimesh.creation.cone(0.3, 0.2, sections=6)
    pyramid.apply_translation((1., 0., 0.))
    _, boxes, convex, surface = tile_pieces([ground, box((.5, .5, .2), (-1, 0, .1)), tile_mesh(pyramid)], SIZE)
    assert (len(boxes), len(convex), surface) == (2, 1, None)
    assert min(half[2] for _, half in boxes) == pytest.approx(.1)


def test_closed_non_convex_piece_is_rejected():
    ring = trimesh.creation.annulus(0.5, 1., 0.2)
    with pytest.raises(NotImplementedError, match='non-convex'):
        tile_pieces([tile_mesh(ring)], SIZE)


def test_irregular_open_surface_is_rejected():
    vertices = np.array([[0, 0, 0], [1, 0, 0.1], [0, 1, 0], [1.3, 1, 0.2], [0.5, 0.5, 0.3]], float)
    faces = [[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]]
    with pytest.raises(NotImplementedError, match='regular height grid'):
        tile_pieces([tile_mesh(trimesh.Trimesh(vertices, faces, process=False))], SIZE)


def test_height_grid_tile_is_written_for_both_formats_and_found_as_an_ordinary_asset(tmp_path, monkeypatch):
    import mujoco

    monkeypatch.setattr('shared.asset_formats.ROOT', tmp_path)
    heights = [[0.1 * ((i + j) % 3) for i in range(9)] for j in range(9)]
    record = build([grid(heights)], np.array([SIZE / 2, SIZE / 2, 0.1]), SIZE, tmp_path / 'tile', 'tile',
                   {'variant': 'synthetic'})
    assert record['steps'][0]['mjcf_collision'] == {'boxes': 0, 'convex_meshes': 0, 'height_field': True}
    assert record['steps'][0]['max_height_difference'] < 2e-3
    assert record['terrain']['spawn_origin'] == pytest.approx([0., 0., 0.1])
    assert json.loads((tmp_path / 'tile/ASSET.json').read_text())['formats']['mjcf'].endswith('mjcf/tile.xml')
    assert mjcf_for(tmp_path / 'tile/usd/tile.usda') == tmp_path / 'tile/mjcf/tile.xml'
    assert mujoco.MjModel.from_xml_path(str(tmp_path / 'tile/mjcf/tile.xml')).nhfield == 1
    with pytest.raises(FileExistsError):
        build([grid(heights)], np.zeros(3), SIZE, tmp_path / 'tile', 'tile', {})


def test_sphere_rests_on_a_box_step_in_the_generated_mjcf(tmp_path, monkeypatch):
    import mujoco

    monkeypatch.setattr('shared.asset_formats.ROOT', tmp_path)
    ground = tile_mesh(trimesh.Trimesh([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], [[0, 1, 2], [0, 2, 3]]))
    build([ground, box((1., 1., .3), (0, 0, .15))], np.zeros(3), SIZE, tmp_path / 'step', 'step', {})
    spec = mujoco.MjSpec.from_file(str(tmp_path / 'step/mjcf/step.xml'))
    body = spec.worldbody.add_body(pos=[0., 0., 1.])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.05, 0, 0], mass=1.)
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert data.qpos[2] - .05 == pytest.approx(.3, abs=2e-3)


def test_height_grid_split_along_the_other_diagonal_is_rejected(tmp_path):
    heights = [[1. if (i, j) == (4, 4) else 0. for i in range(9)] for j in range(9)]
    n, size = 9, SIZE
    xs = np.linspace(-size / 2, size / 2, n)
    vertices = np.array([(x, y, heights[j][i]) for j, y in enumerate(xs) for i, x in enumerate(xs)])
    faces = [f for j in range(n - 1) for i in range(n - 1)
             for f in ((j * n + i, j * n + i + 1, j * n + i + n), (j * n + i + 1, j * n + i + n + 1, j * n + i + n))]
    mesh = tile_mesh(trimesh.Trimesh(vertices, faces, process=False))
    with pytest.raises(ValueError, match='differs from the generated triangles'):
        build([mesh], np.zeros(3), SIZE, tmp_path / 'flipped', 'flipped', {})


def test_mjlab_scene_uses_generated_tile_as_ordinary_static_object(tmp_path, monkeypatch):
    pytest.importorskip('mjlab')
    import mujoco
    from shared.assembly.assets import factories_for
    from shared.assembly.layout import world_spec_from_config
    from shared.backends.mjlab.scene import compile_scene

    monkeypatch.setattr('shared.asset_formats.ROOT', tmp_path)
    flat = [[0.] * 9 for _ in range(9)]
    for j in range(3, 6):
        for i in range(3, 6):
            flat[j][i] = .2
    build([grid(flat), box((.2, .2, .1), (1.5, 1.5, -.05))], np.zeros(3), SIZE, tmp_path / 'tile', 'tile', {})
    settings = {'name': 'world', 'entities': [
        {'name': 'ground', 'factory': 'usd_object', 'parameters': {'usd_path': str(tmp_path / 'tile/usd/tile.usda')}},
        {'name': 'cube', 'factory': 'box', 'parameters': {'size': [.1, .1, .1], 'mass': 1.},
         'pose': {'position': [0., 0., .6]}}]}
    world = world_spec_from_config(settings, directory=tmp_path, backend='mjlab')
    scene, model, data, keys = compile_scene(world, settings, factories_for('mjlab'))
    cube = model.body(scene.entities[keys['world/cube']].root_body.name)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert data.xpos[cube.id][2] == pytest.approx(.2 + .05, abs=3e-3)


def test_presets_cover_every_terrain_variant():
    from shared.backends.isaac.assets import TERRAIN_VARIANTS
    from shared.terrain_assets import LABELS, PRESETS

    assert set(PRESETS) == set(TERRAIN_VARIANTS) == set(LABELS)
