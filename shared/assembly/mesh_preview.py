"""Blender-free lightweight editor meshes; sources and native frames are unchanged.

Vertex clustering per part keeps link identity and exported frames, like the
Blender decimation path in ``tools/simplify_scene_visuals.py``. It is only an
editor preview: simulation always loads the original assets.
"""
import gzip
import json
from pathlib import Path

import numpy as np

VERSION = 1
TRIANGLE_BUDGET = 60000
MIN_PART_TRIANGLES = 256
# Beyond this many parts (a whole indoor scene exports ~20 000) parts of one body are merged before simplifying.
MAX_PARTS = 512
NOTE = 'Lightweight editor mesh; exact native joint and mounting frames. Simulation uses original assets.'
# Full-resolution export, only ever read by the simplifiers. Older caches used gzip JSON.
MESH_SOURCE_SUFFIX = '.mesh.npz'
SOURCE_SUFFIXES = (MESH_SOURCE_SUFFIX, '.mesh.json.gz')


def write_mesh_source(stream, parts, frames, joints):
    """parts: [{'name', 'vertices' (N, 3), 'faces' (M, 3) triangles}]."""
    arrays = {}
    for index, part in enumerate(parts):
        arrays[f'vertices_{index}'] = np.asarray(part['vertices'], dtype=np.float32).reshape(-1, 3)
        arrays[f'faces_{index}'] = np.asarray(part['faces'], dtype=np.int32).reshape(-1, 3)
    header = {'names': [part['name'] for part in parts], 'frames': frames, 'joints': joints}
    np.savez(stream, header=np.array(json.dumps(header)), **arrays)


def read_mesh_source(filename):
    """Geometry dict with array parts, from either source format."""
    if str(filename).endswith('.mesh.json.gz'):
        with gzip.open(filename, 'rt') as stream:
            geometry = json.load(stream)
        for part in geometry['parts']:
            part['vertices'] = np.asarray(part['vertices'], dtype=float).reshape(-1, 3)
            part['faces'] = triangles(part['faces'])
        return geometry
    with np.load(filename) as data:
        header = json.loads(str(data['header']))
        parts = [{'name': name, 'vertices': data[f'vertices_{index}'].astype(float), 'faces': data[f'faces_{index}'].astype(np.int64)}
                 for index, name in enumerate(header['names'])]
    return {'parts': parts, 'frames': header['frames'], 'joints': header['joints']}


def light_name(source):
    source = Path(source)
    suffix = next(suffix for suffix in SOURCE_SUFFIXES if source.name.endswith(suffix))
    return source.with_name(source.name[:-len(suffix)] + f'.light-v{VERSION}.mesh.json.gz')


def triangles(faces):
    if isinstance(faces, np.ndarray):
        return faces.astype(np.int64).reshape(-1, 3)
    result = [(face[0], face[i], face[i + 1]) for face in faces for i in range(1, len(face) - 1)]
    return np.asarray(result, dtype=np.int64).reshape(-1, 3)


def _cluster(vertices, faces, cells):
    # Cells span where most vertices are, so one far-off piece (a sky dome) does not coarsen the rest.
    low, high = np.percentile(vertices, 0.5, axis=0), np.percentile(vertices, 99.5, axis=0)
    size = max(float((high - low).max()), 1e-9) / cells
    keys = np.floor((vertices - low) / size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    # The vertex farthest from the part centre represents its cell, so the
    # outline (and reach / bounding box) shrinks less than with cell means.
    distance = ((vertices - vertices.mean(axis=0)) ** 2).sum(axis=1)
    order = np.lexsort((-distance, inverse))
    first = np.ones(len(order), dtype=bool)
    first[1:] = inverse[order[1:]] != inverse[order[:-1]]
    representative = order[first]
    remapped = inverse[faces]
    valid = ((remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2])
             & (remapped[:, 0] != remapped[:, 2]))
    remapped = remapped[valid]
    if len(remapped):
        _, unique = np.unique(np.sort(remapped, axis=1), axis=0, return_index=True)
        remapped = remapped[np.sort(unique)]
    used, compact = np.unique(remapped, return_inverse=True)
    return vertices[representative][used], compact.reshape(-1, 3)


def simplify_part(vertices, faces, target):
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = triangles(faces)
    if len(faces) <= target:
        return vertices, faces
    best, low, high = None, 1, 4096
    while low <= high:
        cells = (low + high) // 2
        candidate = _cluster(vertices, faces, cells)
        if len(candidate[1]) <= target:
            best, low = (candidate if len(candidate[1]) else best), cells + 1
        else:
            high = cells - 1
    if best is None or not len(best[1]):
        # Tiny budget: keep the coarsest clustering that still has faces.
        best = next(candidate for candidate in (_cluster(vertices, faces, cells) for cells in (2, 4, 8, 16, 32, 64))
                    if len(candidate[1])) if len(faces) else (vertices[:0], faces)
    return best


def part_targets(geometry):
    """Triangle budget per part, proportional to its share of the source; 0 for face-less parts."""
    counts = [len(part['faces']) for part in geometry['parts']]
    total = max(sum(counts), 1)
    return [max(MIN_PART_TRIANGLES, int(TRIANGLE_BUDGET * count / total)) if count else 0 for count in counts]


def assign(geometries, jobs):
    """Deterministic {(geometry, part): job}, balancing source triangles, largest parts first."""
    load, result = [0] * jobs, {}
    items = sorted(((len(part['faces']), g, i) for g, geometry in enumerate(geometries)
                    for i, part in enumerate(geometry['parts']) if len(part['faces'])), reverse=True)
    for count, g, i in items:
        job = load.index(min(load))
        load[job] += count
        result[g, i] = job
    return result


def write_light(destination, geometry, simplified, method):
    """``simplified[i] = (vertices, triangles)`` for every part that has faces."""
    original_vertices = sum(len(part['vertices']) for part in geometry['parts'])
    for i, part in enumerate(geometry['parts']):
        if i not in simplified:
            part['vertices'], part['faces'] = np.asarray(part['vertices']).tolist(), []
            continue
        vertices, faces = simplified[i]
        if not np.isfinite(vertices).all():
            raise ValueError('Preview contains non-finite coordinates')
        if not len(faces):
            raise ValueError('Preview simplification removed an entire part')
        part['vertices'] = np.round(np.asarray(vertices, dtype=float), 7).tolist()
        part['faces'] = np.asarray(faces).tolist()
    geometry['preview'] = {'version': VERSION, 'source_vertex_count': original_vertices,
                           'triangle_budget': TRIANGLE_BUDGET, 'method': method}
    temporary = destination.with_suffix('.tmp')
    with gzip.open(temporary, 'wt', compresslevel=3) as stream:
        json.dump(geometry, stream, separators=(',', ':'))
    temporary.replace(destination)
    return {'vertex_count': sum(len(p['vertices']) for p in geometry['parts']),
            'triangle_count': sum(sum(max(0, len(f) - 2) for f in p['faces']) for p in geometry['parts']),
            'source_vertex_count': original_vertices}


def part_count(filename):
    """Number of parts of a mesh source, reading only its header where possible."""
    if str(filename).endswith('.mesh.json.gz'):
        return len(read_mesh_source(filename)['parts'])
    with np.load(filename) as data:
        return len(json.loads(str(data['header']))['names'])


def merged_parts(geometry, cells=2048):
    """`geometry` with the parts of each body (as ``mesh_parts`` assigns them) merged into one, each first
    snapped to a grid of `cells` along the largest extent, so tens of millions of triangles become a few
    meshes small enough to simplify."""
    parts = [part for part in geometry['parts'] if len(part['faces'])]
    if not parts:
        return geometry
    # Grid over where most vertices are: a sky dome kilometres wide must not set the cell size.
    sample = np.concatenate([part['vertices'][::max(1, len(part['vertices']) // 1000)] for part in parts])
    low, high = np.percentile(sample, 0.5, axis=0), np.percentile(sample, 99.5, axis=0)
    size = max(float((high - low).max()), 1e-9) / cells
    groups = {}
    for part in parts:
        body = next((body for body in geometry['frames'] if '/' + body + '/' in part['name']), None)
        keys = np.floor((part['vertices'] - low) / size).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        faces = inverse[triangles(part['faces'])]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
        if not len(faces):
            continue
        vertices = np.zeros((inverse.max() + 1, 3))
        vertices[inverse] = part['vertices']  # any vertex of its cell represents it
        groups.setdefault(body, []).append((part['name'], vertices, faces))
    merged = []
    for members in groups.values():
        offsets = np.cumsum([0] + [len(vertices) for _, vertices, _ in members])
        merged.append({'name': members[0][0], 'vertices': np.concatenate([vertices for _, vertices, _ in members]),
                       'faces': np.concatenate([faces + offset for (_, _, faces), offset in zip(members, offsets)])})
    return {**geometry, 'parts': merged}


def simplify(source, destination):
    geometry = read_mesh_source(source)
    if len(geometry['parts']) > MAX_PARTS:
        geometry = merged_parts(geometry)
    simplified = {i: simplify_part(part['vertices'], part['faces'], target)
                  for i, (part, target) in enumerate(zip(geometry['parts'], part_targets(geometry))) if target}
    return write_light(destination, geometry, simplified, 'vertex_clustering')


def needs_light(record):
    filename = record.get('source_path') or record.get('path') or ''
    if not filename.endswith(SOURCE_SUFFIXES):
        return False
    return record.get('preview_version') != VERSION or not Path(record['path']).is_file()


def pending(data):
    """Source meshes of a manifest that still lack a current lightweight mesh."""
    return [(record, Path(record.get('source_path') or record['path']))
            for record in data['entities'].values() if needs_light(record)]


def _finish(manifest, original, data):
    if manifest.read_bytes() != original:
        raise RuntimeError('Geometry manifest changed during simplification; retry')
    temporary = manifest.with_suffix('.light.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(manifest)


def _record_light(record, source, counts):
    record.update(counts, source_path=str(source), path=str(light_name(source)), preview_version=VERSION,
                  exact=False, note=NOTE)


def simplify_manifest(manifest):
    """Blender-free path, same manifest contract as ``merge_partials``."""
    manifest = Path(manifest)
    original = manifest.read_bytes()
    data = json.loads(original)
    for record, source in pending(data):
        _record_light(record, source, simplify(source, light_name(source)))
    _finish(manifest, original, data)


def merge_partials(manifest, partials, method):
    """Combine part results of parallel simplifier jobs (``assign``) into lightweight meshes."""
    manifest = Path(manifest)
    original = manifest.read_bytes()
    data = json.loads(original)
    results = {}
    for partial in partials:
        with np.load(partial) as arrays:
            for name in arrays.files:
                if name.startswith('v_'):
                    results[name[2:]] = (arrays[name], arrays['f_' + name[2:]])
    for g, (record, source) in enumerate(pending(data)):
        geometry = read_mesh_source(source)
        simplified = {i: results[f'{g}_{i}'] for i, part in enumerate(geometry['parts']) if len(part['faces'])}
        _record_light(record, source, write_light(light_name(source), geometry, simplified, method))
    _finish(manifest, original, data)
