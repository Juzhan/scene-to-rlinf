"""Blender-free lightweight editor meshes; sources and native frames are unchanged.

Vertex clustering per part keeps link identity and exported frames, like the
Blender decimation path in ``tools/simplify_scene_visuals.py``. It is only an
editor preview: simulation always loads the original assets.
"""
import gzip
import json
import os
from pathlib import Path

import numpy as np

VERSION = 3  # 2: quadric collapse; 3: a triangle cap per object, not a scene budget
TRIANGLE_BUDGET = 60000
MIN_PART_TRIANGLES = 256
# Beyond this many parts (a whole indoor scene exports ~20 000) parts of one body are merged before simplifying.
MAX_PARTS = 512
# A scene exported object by object (shared/assembly/composition.py) keeps one part per object, so the
# editor can select and remove each; the whole scene shares this budget, each object at least the minimum.
OBJECT_TRIANGLES = 20000  # per object of an imported scene: its own detail, up to this
SCENE_CEILING = 4_000_000  # all objects of a scene; only very large scenes reach it
MIN_OBJECT_TRIANGLES = 48
NOTE = 'Lightweight editor mesh; exact native joint and mounting frames. Simulation uses original assets.'
# Full-resolution export, only ever read by the simplifiers. Older caches used gzip JSON.
MESH_SOURCE_SUFFIX = '.mesh.npz'
SOURCE_SUFFIXES = (MESH_SOURCE_SUFFIX, '.mesh.json.gz')


def mean_color(parts):
    """Base colour of parts drawn as one: their ``color`` (r, g, b in 0-1) weighted by triangles; None if none has one."""
    coloured = [(part['color'], len(part['faces'])) for part in parts if part.get('color') is not None]
    total = sum(weight for _, weight in coloured)
    if not total:
        return coloured[0][0] if coloured else None
    return [round(sum(color[i] * weight for color, weight in coloured) / total, 4) for i in range(3)]


def write_mesh_source(stream, parts, frames, joints, objects=False):
    """parts: [{'name', 'vertices' (N, 3), 'faces' (M, 3) triangles, optional 'color' (r, g, b in 0-1, from the
    material) and 'instance' ({'of': part name, 'matrix': 4x4}, a copy of another part, without geometry of
    its own)}]; ``objects``: one part per scene object, named by its path relative to the entity's default prim."""
    arrays = {}
    for index, part in enumerate(parts):
        arrays[f'vertices_{index}'] = np.asarray(part['vertices'], dtype=np.float32).reshape(-1, 3)
        arrays[f'faces_{index}'] = np.asarray(part['faces'], dtype=np.int32).reshape(-1, 3)
    header = {'names': [part['name'] for part in parts], 'frames': frames, 'joints': joints, 'objects': objects,
              'colors': [part.get('color') for part in parts], 'instances': [part.get('instance') for part in parts]}
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
        colors = header.get('colors') or [None] * len(header['names'])
        instances = header.get('instances') or [None] * len(header['names'])
        parts = [{'name': name, 'vertices': data[f'vertices_{index}'].astype(float), 'faces': data[f'faces_{index}'].astype(np.int64),
                  'color': color, **({'instance': instance} if instance else {})}
                 for index, (name, color, instance) in enumerate(zip(header['names'], colors, instances))]
    return {'parts': parts, 'frames': header['frames'], 'joints': header['joints'], 'objects': header.get('objects', False)}


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


PIECE_TRIANGLES = 8  # triangles a separate piece keeps, about, after collapse (a box: 12)
SMALL_PIECE = .03    # pieces under this fraction of their part's size may be left out of the preview


def _pieces(vertices, faces):
    """(piece of each face, size (bounding-box diagonal) of each piece) for the connected pieces of a mesh."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    graph = coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
    _, label = connected_components(graph, directed=False)
    face_label = label[faces[:, 0]]
    order = np.argsort(face_label, kind='stable')
    points = vertices[faces[order]].reshape(len(faces), 9)
    starts = np.flatnonzero(np.r_[True, face_label[order][1:] != face_label[order][:-1]])
    low = np.minimum.reduceat(np.minimum(np.minimum(points[:, 0:3], points[:, 3:6]), points[:, 6:9]), starts)
    high = np.maximum.reduceat(np.maximum(np.maximum(points[:, 0:3], points[:, 3:6]), points[:, 6:9]), starts)
    pieces = np.empty(len(faces), dtype=np.int64)
    pieces[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(faces)]))
    return pieces, np.linalg.norm(high - low, axis=1)


def _quadric(vertices, faces, target):
    """Quadric-error edge collapse (``fast_simplification``) of the welded mesh, or None without it:
    edges collapse where they change the surface least, so outlines and flat walls keep their shape.
    Collapse never joins separate pieces, and each keeps a few triangles: a modelled cabinet is ten
    thousand screws, handles and trims. So pieces under ``SMALL_PIECE`` of the part are left out as far as
    the budget needs (``target / PIECE_TRIANGLES`` pieces in all), largest kept first; what is still over
    the budget is clustered (``_cluster``) from the collapsed mesh."""
    try:
        import fast_simplification
    except ImportError:
        return None
    # Exported meshes repeat vertices along UV and normal seams; welded, the seams can collapse too.
    low = vertices.min(axis=0)
    scale = max(float(np.ptp(vertices, axis=0).max()), 1e-9)
    grid = np.round((vertices - low) / scale * (2 ** 20)).astype(np.int64)  # 21 bits per axis: one int64 key
    _, first, inverse = np.unique((grid[:, 0] << 42) | (grid[:, 1] << 21) | grid[:, 2],
                                  return_index=True, return_inverse=True)
    welded, remapped = vertices[first], inverse.reshape(-1)[faces]
    remapped = remapped[(remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2])
                        & (remapped[:, 0] != remapped[:, 2])]
    if len(remapped) <= target:
        return welded, remapped

    def collapse(kept):
        # Gentle passes: an aggressive one (agg 7) reaches the count by collapsing whole cushions (25 cm
        # off on an armchair at 1 % of its triangles, against 3 cm for two gentle passes).
        reduced, out = welded.astype(np.float32), kept.astype(np.int32)
        for _ in range(3):
            if len(out) <= target * 1.25:
                break
            reduced, out = fast_simplification.simplify(reduced, out, target_reduction=1 - target / len(out), agg=3)
        return np.asarray(reduced, dtype=float), np.asarray(out, dtype=np.int64)
    pieces, sizes = _pieces(welded, remapped)
    room = max(1, target // PIECE_TRIANGLES)
    if len(sizes) > room:
        # Pieces of a few per cent of the part (cushions, legs, panels) all stay; of the smaller ones the
        # largest fill what room is left.
        large = sizes >= SMALL_PIECE * float(np.linalg.norm(np.ptp(welded, axis=0)))
        small = np.flatnonzero(~large)
        keep = large.copy()
        keep[small[np.argsort(-sizes[small], kind='stable')[:max(0, room - int(large.sum()))]]] = True
        remapped = remapped[keep[pieces]]
    result = collapse(remapped)
    if len(result[1]) > target * 2:
        result = _clustered(*result, target)
    return result if len(result[1]) else None


def _clustered(vertices, faces, target):
    """The finest vertex clustering (``_cluster``) within ``target`` triangles."""
    best, low, high = None, 1, 4096
    while low <= high:
        cells = (low + high) // 2
        candidate = _cluster(vertices, faces, cells)
        if len(candidate[1]) <= target:
            best, low = (candidate if len(candidate[1]) else best), cells + 1
        else:
            high = cells - 1
    return best if best is not None else (vertices[:0], faces[:0])


def simplify_part(vertices, faces, target):
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = triangles(faces)
    if len(faces) <= target:
        return vertices, faces
    quadric = _quadric(vertices, faces, target)
    if quadric is not None and len(quadric[1]) <= target * 2:
        return quadric
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
    """Triangle budget per part; 0 for face-less parts. Parts of one asset share ``TRIANGLE_BUDGET`` by their
    share of the source. Objects of a scene are models of their own: each keeps its source up to
    ``OBJECT_TRIANGLES`` (a kettle stays a kettle however large the house), and only a scene beyond
    ``SCENE_CEILING`` in all (a furnished shop, a performance test) scales them down together."""
    counts = [len(part['faces']) for part in geometry['parts']]
    total = max(sum(counts), 1)
    if not geometry.get('objects'):
        return [max(MIN_PART_TRIANGLES, int(TRIANGLE_BUDGET * count / total)) if count else 0 for count in counts]
    targets = [min(count, OBJECT_TRIANGLES) for count in counts]
    scale = min(1., SCENE_CEILING / max(sum(targets), 1))
    return [min(count, max(MIN_OBJECT_TRIANGLES, int(target * scale))) if count else 0
            for count, target in zip(counts, targets)]


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
        groups.setdefault(body, []).append((part['name'], vertices, faces, part.get('color')))
    merged = []
    for members in groups.values():
        offsets = np.cumsum([0] + [len(vertices) for _, vertices, _, _ in members])
        merged.append({'name': members[0][0], 'vertices': np.concatenate([vertices for _, vertices, _, _ in members]),
                       'faces': np.concatenate([faces + offset for (_, _, faces, _), offset in zip(members, offsets)]),
                       'color': mean_color([{'color': color, 'faces': faces} for _, _, faces, color in members])})
    return {**geometry, 'parts': merged}


def simplify(source, destination):
    geometry = read_mesh_source(source)
    if len(geometry['parts']) > MAX_PARTS and not geometry.get('objects'):
        geometry = merged_parts(geometry)
    work = [(i, part['vertices'], part['faces'], target)
            for i, (part, target) in enumerate(zip(geometry['parts'], part_targets(geometry))) if target]
    simplified = dict(_simplify_all(work))
    return write_light(destination, geometry, simplified, 'quadric' if _has_quadric() else 'vertex_clustering')


def _has_quadric():
    try:
        import fast_simplification  # noqa: F401
    except ImportError:
        return False
    return True


def _simplify_one(item):
    index, vertices, faces, target = item
    return index, simplify_part(vertices, faces, target)


def _simplify_all(work):
    """``[(index, (vertices, faces))]`` of ``work`` items, in worker processes when it is large: a furnished
    house is 15 M source triangles in 700 objects, about two minutes on one core."""
    heavy = sum(len(faces) for _, _, faces, target in work if len(faces) > target)
    if heavy < 2_000_000 or len(work) < 8:
        return [_simplify_one(item) for item in work]
    import concurrent.futures
    import multiprocessing
    workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    work = sorted(work, key=lambda item: -len(item[2]))  # big parts first: shorter tail
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        return list(pool.map(_simplify_one, work, chunksize=4))


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


if __name__ == '__main__':  # shared.assembly.visuals runs the quadric simplifier in a process of its own
    import sys
    simplify_manifest(sys.argv[1])
