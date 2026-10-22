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
