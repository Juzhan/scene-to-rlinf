"""Editor decimation preserves source assets, separate parts and native frames."""
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from shared.assembly.mesh_preview import MIN_PART_TRIANGLES, TRIANGLE_BUDGET, write_mesh_source
from shared.assembly.visuals import _ensure_lightweight


def sphere(name, segments=256, rings=128, radius=.3):
    theta, phi = np.meshgrid(np.linspace(0, np.pi, rings + 1), np.linspace(0, 2 * np.pi, segments, endpoint=False), indexing='ij')
    vertices = radius * np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1).reshape(-1, 3)
    index = np.arange(len(vertices)).reshape(rings + 1, segments)
    a, b = index[:-1], np.roll(index[:-1], -1, axis=1)
    c, d = index[1:], np.roll(index[1:], -1, axis=1)
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    return {'name': name, 'vertices': vertices, 'faces': faces}


def cube(name, size=.002):
    vertices = (np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float) - .5) * size
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return {'name': name, 'vertices': vertices, 'faces': faces}


@pytest.mark.skipif(not shutil.which('blender'), reason='Blender unavailable')
@pytest.mark.parametrize('suffix', ['.mesh.npz', '.mesh.json.gz'])  # current and older full-mesh caches
def test_light_mesh_retains_frames_parts_and_original(tmp_path, monkeypatch, suffix):
    monkeypatch.delenv('RLINF_PREVIEW_SIMPLIFIER', raising=False)
    parts = [sphere('link0'), sphere('link1'), cube('small_attachment')]
    frames = {'wrist': {'position': [.3, .1, .7], 'orientation': [1, 0, 0, 0]}}
    source = tmp_path / ('asset' + suffix)
    if suffix == '.mesh.npz':
        with source.open('wb') as stream:
            write_mesh_source(stream, parts, frames, [])
    else:
        with gzip.open(source, 'wt') as stream:
            json.dump({'parts': [{'name': p['name'], 'vertices': p['vertices'].tolist(), 'faces': p['faces'].tolist()}
                                 for p in parts], 'frames': frames}, stream)
    original = source.read_bytes()
    manifest = tmp_path / 'scene_visuals.json'
    manifest.write_text(json.dumps({'entities': {'world/arm': {
        'key': 'unchanged', 'path': str(source), 'initial_state': {'joint_positions': {'wrist': .3}}}}}))
    _ensure_lightweight(manifest)
    record = json.loads(manifest.read_text())['entities']['world/arm']
    assert source.read_bytes() == original
    assert record['key'] == 'unchanged' and record['initial_state']['joint_positions']['wrist'] == .3
    assert record['triangle_count'] <= TRIANGLE_BUDGET + 3 * MIN_PART_TRIANGLES
    assert record['vertex_count'] < record['source_vertex_count']
    assert not list(tmp_path.glob('*.light-*.npz'))  # job results are merged and removed
    with gzip.open(record['path'], 'rt') as stream:
        light = json.load(stream)
    assert light['frames'] == frames
    assert [p['name'] for p in light['parts']] == [p['name'] for p in parts]
    for before, after in zip(parts, light['parts']):
        after = np.asarray(after['vertices'])
        assert np.abs(before['vertices'].min(0) - after.min(0)).max() < .003
        assert np.abs(before['vertices'].max(0) - after.max(0)).max() < .003
    stamp = Path(record['path']).stat().st_mtime_ns
    _ensure_lightweight(manifest)
    assert Path(record['path']).stat().st_mtime_ns == stamp


def test_whole_scene_previews_merge_parts_per_body_and_ignore_a_sky_dome(tmp_path):
    from shared.assembly.mesh_preview import MAX_PARTS
    rows = int(np.ceil(np.sqrt(MAX_PARTS + 100)))
    parts = [sphere(f'/World/props/item_{i}', segments=16, rings=8, radius=.1) for i in range(rows * rows)]
    for i, part in enumerate(parts):
        part['vertices'] = part['vertices'] + (i % rows, i // rows, 0)
    parts.append(sphere('/World/sky', segments=32, rings=16, radius=5000.))
    source = tmp_path / 'scene.mesh.npz'
    with source.open('wb') as stream:
        write_mesh_source(stream, parts, {}, [])
    manifest = tmp_path / 'scene_visuals.json'
    manifest.write_text(json.dumps({'entities': {'world/environment': {'path': str(source)}}}))
    _ensure_lightweight(manifest)  # thousands of parts take the numpy path, with or without Blender
    record = json.loads(manifest.read_text())['entities']['world/environment']
    with gzip.open(record['path'], 'rt') as stream:
        light = json.load(stream)
    assert len(light['parts']) == 1 and record['triangle_count'] <= TRIANGLE_BUDGET + MIN_PART_TRIANGLES
    vertices = np.asarray(light['parts'][0]['vertices'])
    # the grid follows the props, not the dome: props keep sub-decimetre detail
    near = vertices[np.abs(vertices).max(1) < 100]
    assert len(near) > 1000 and near.min(0)[0] < -.05 and near.max(0)[0] > rows - 1 + .05


def test_part_colours_survive_the_mesh_source_and_join_by_triangles(tmp_path):
    from shared.assembly.mesh_preview import mean_color, read_mesh_source
    parts = [{**cube('wood'), 'color': [.8, .6, .4]}, {**sphere('glass', 8, 4), 'color': None}]
    with (tmp_path / 'scene.mesh.npz').open('wb') as stream:
        write_mesh_source(stream, parts, {}, [], objects=True)
    read = read_mesh_source(tmp_path / 'scene.mesh.npz')['parts']
    assert read[0]['color'] == [.8, .6, .4] and read[1]['color'] is None
    red = {'color': [1., 0., 0.], 'faces': np.zeros((3, 3))}
    blue = {'color': [0., 0., 1.], 'faces': np.zeros((1, 3))}
    assert mean_color([red, blue, {'color': None, 'faces': np.zeros((9, 3))}]) == [.75, 0., .25]
    assert mean_color([{'color': None, 'faces': np.zeros((1, 3))}]) is None


def test_scene_objects_keep_their_own_detail_up_to_a_cap(monkeypatch):
    import shared.assembly.mesh_preview as preview
    from shared.assembly.mesh_preview import OBJECT_TRIANGLES, part_targets
    sofa = sphere('sofa', 256, 128, radius=1.)      # 65 536 triangles: capped
    kettle = sphere('kettle', 64, 32, radius=.1)    # 4 096: kept whole, however large the scene
    wall = cube('wall', size=4.)
    targets = part_targets({'parts': [sofa, kettle, wall], 'objects': True})
    assert targets == [OBJECT_TRIANGLES, len(kettle['faces']), 12]
    monkeypatch.setattr(preview, 'SCENE_CEILING', 12000)  # only a scene beyond the ceiling scales down
    sofa_target, kettle_target, _ = part_targets({'parts': [sofa, kettle, wall], 'objects': True})
    assert sofa_target + kettle_target <= 12000 and sofa_target / kettle_target == pytest.approx(OBJECT_TRIANGLES / 4096, rel=.01)


@pytest.mark.skipif(not __import__('importlib').util.find_spec('fast_simplification'), reason='no fast_simplification')
def test_quadric_simplification_reaches_the_budget_of_a_part_made_of_many_pieces():
    from shared.assembly.mesh_preview import simplify_part
    body = sphere('body', 128, 64, radius=.5)
    # 2 000 screws: separate pieces that collapse cannot join, each keeping its triangles.
    screws = [cube(f's{i}', size=.004) for i in range(2000)]
    offsets = np.random.default_rng(0).uniform(-.5, .5, (2000, 3))
    vertices = [body['vertices']] + [s['vertices'] + o for s, o in zip(screws, offsets)]
    starts = np.cumsum([0] + [len(v) for v in vertices[:-1]])
    faces = np.concatenate([body['faces']] + [s['faces'] + start for s, start in zip(screws, starts[1:])])
    reduced, kept = simplify_part(np.concatenate(vertices), faces, 1500)
    assert len(kept) <= 1500 * 2
    # The body is kept: its points stay on the sphere.
    radius = np.linalg.norm(reduced[np.unique(kept)], axis=1)
    assert (np.abs(radius - .5) < .02).mean() > .9
