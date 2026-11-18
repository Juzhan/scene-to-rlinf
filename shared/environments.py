"""Indoor and outdoor static environments: a catalog of known scenes, fetched and wrapped on demand.

Like shared/simready.py for objects: ``prepare(key)`` downloads a scene's files when they are missing,
wraps it with shared.asset_formats.environment (static colliders, the fixes Newton needs) into
assets/<name>/ and returns the ASSET.json record. Scenes then use it as a ``usd_object`` with
``scope: global`` at the recorded ``environment.pose`` (which puts its floor at z = 0).

    isaac/<name>                Isaac Sim 5.1 environments on NVIDIA's public bucket (local use only)
    grscenes/<part>/<scene id>  GRScenes-100 navigation scenes (CC BY-NC-SA 4.0), from one ModelScope zip
                                per part, downloaded and extracted under assets/grscenes/ by
                                ``download_grscenes`` (run.py asset environment grscenes/<part>)
"""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import difflib
import json
import re
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
CACHE = ROOT / '.cache' / 'environments'
ISAAC_LICENSE = 'NVIDIA Omniverse 资产，仅限本地使用'
GRSCENES_LICENSE = 'CC BY-NC-SA 4.0（非商用）'
VEGETATION_URL = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets'

# Rivermark files missing from the public bucket -> closest Isaac Vegetation model (below VEGETATION_URL).
RIVERMARK_VEGETATION = {
    'tree_ash_red_01.usd': 'Vegetation/Trees/Red_Ash.usd', 'tree_ash_red_02.usd': 'Vegetation/Trees/Red_Ash.usd',
    'tree_ash_red_03.usd': 'Vegetation/Trees/Red_Ash_Fall.usd',
    'tree_ash_white_dark.usd': 'Vegetation/Trees/White_Ash.usd',
    **{f'tree_beech_american_0{i}.usd': 'Vegetation/Trees/American_Beech.usd' for i in (1, 2, 3)},
    'tree_broadleaf_01.usd': 'Vegetation/Trees/Sugar_Maple.usd',
    'tree_broadleaf_02.usd': 'Vegetation/Trees/Red_Maple.usd', 'tree_broadleaf_03.usd': 'Vegetation/Trees/Red_Oak.usd',
    'tree_fir_douglas.usd': 'Vegetation/Trees/Douglas_Fir.usd', 'tree_fir_douglas_01.usd': 'Vegetation/Trees/Douglas_Fir.usd',
    **{f'tree_lilac_0{i}.usd': 'Vegetation/Shrub/Lilac.usd' for i in range(1, 6)},
    'tree_poplar_lombardy_02.usd': 'Vegetation/Trees/Lombardy_Poplar.usd',
    'bush_gen_06.usd': 'Vegetation/Shrub/Privet.usd', 'bush_gen_06_dark.usd': 'Vegetation/Shrub/Privet.usd',
    'bush_small_01.usd': 'Vegetation/Shrub/Boxwood.usd', 'hedge_round_01.usd': 'Vegetation/Shrub/Boxwood.usd',
    'grass_clump_01.usd': 'Vegetation/Shrub/Grass_Short_A.usd', 'grass_clump_02.usd': 'Vegetation/Shrub/Grass_Short_B.usd',
    'grass_clump_03.usd': 'Vegetation/Shrub/Grass_Short_C.usd', 'grass_clump_04.usd': 'Vegetation/Shrub/Grass_Short_A.usd',
    **{f'plant_swordleaf_0{i}.usd': 'Vegetation/Shrub/Fountain_Grass_Tall.usd' for i in (1, 2, 3)},
    **{f'plant_swordleaf_floral_01{suffix}.usd': 'Vegetation/Shrub/Fountain_Grass_Short.usd'
       for suffix in ('', '_dark', '_yellow')},
}

# `folder`: assets/<folder>/source/ holds the vendored files (shared by scenes of one Isaac directory);
# `pose`: where the scene goes so that its floor near the origin is at z = 0; `verified`: smoke-tested.
ISAAC = {
    'rivermark': {'label': 'Rivermark 室外住宅区', 'kind': 'outdoor', 'folder': 'rivermark', 'verified': True,
                  'path': 'Isaac/Environments/Outdoor/Rivermark/rivermark.usd', 'pose': [0, 0, -5.75],
                  'substitutes': RIVERMARK_VEGETATION,
                  'footprint_colliders': ['/World/main/props', '/World/main/sub_traffic_signs'],
                  'note': '街道、联排住宅、广场、停车场、树木（约 1 km 见方）；建筑按占地方块碰撞，树木无碰撞体'},
    'simple_warehouse': {'label': '简单仓库', 'kind': 'indoor', 'folder': 'simple_warehouse', 'verified': True,
                         'path': 'Isaac/Environments/Simple_Warehouse/warehouse.usd', 'pose': [0, 0, 0]},
    'full_warehouse': {'label': '完整仓库（货架）', 'kind': 'indoor', 'folder': 'simple_warehouse', 'verified': True,
                       'path': 'Isaac/Environments/Simple_Warehouse/full_warehouse.usd', 'pose': [0, 0, 0]},
    'warehouse_multiple_shelves': {'label': '多货架仓库', 'kind': 'indoor', 'folder': 'simple_warehouse',
                                   'path': 'Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd',
                                   'pose': [0, 0, 0]},
    'warehouse_with_forklifts': {'label': '叉车仓库', 'kind': 'indoor', 'folder': 'simple_warehouse',
                                 'path': 'Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd',
                                 'pose': [0, 0, 0], 'note': '叉车作为静态障碍物'},
    'simple_room': {'label': '简单房间', 'kind': 'indoor', 'folder': 'simple_room', 'verified': True,
                    'path': 'Isaac/Environments/Simple_Room/simple_room.usd', 'pose': [0, 0, 0.77]},
    'office': {'label': '办公室', 'kind': 'indoor', 'folder': 'office', 'verified': True,
               'path': 'Isaac/Environments/Office/office.usd', 'pose': [0, 0, 0],
               'spawn_exclude': ['/Root/SM_Buildings'], 'note': '窗外城市背景也有碰撞体'},
    'hospital': {'label': '医院', 'kind': 'indoor', 'folder': 'hospital', 'verified': True,
                 'path': 'Isaac/Environments/Hospital/hospital.usd', 'pose': [0, 0, 0]},
    'digital_twin_warehouse': {'label': '数字孪生小仓库', 'kind': 'indoor', 'folder': 'digital_twin_warehouse',
                               'path': 'Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd',
                               'pose': [0, 0, 0]},
}

MODELSCOPE = 'https://www.modelscope.cn/api/v1/datasets/Shanghai_AI_Laboratory/GRScenes/repo?Revision=master&FilePath='
GRSCENES = {'commercial': {'label': '商业场所', 'zip': 'scenes/GRScenes-100/commercial_scenes.zip', 'bytes': 23792807959},
            'home': {'label': '家居', 'zip': 'scenes/GRScenes-100/home_scenes.zip', 'bytes': 50949485250}}
GRSCENES_ROOT = ASSETS / 'grscenes'
# Smoke-tested on isaaclab3 + Newton with a Go2 at the first recorded spawn point (simplified colliders).
GRSCENES_VERIFIED = {'MV7J6NIKTKJZ2AABAAAAADA8', 'MV7J6NIKTKJZ2AABAAAAADI8', 'MWAX5JYKTKJZ2AABAAAAAAQ8',
                     'MWBGLKQKTKJZ2AABAAAAAAA8', 'MWHLEPQKTIFZIAABAAAAAAA8'}


def _isaac_url(path):
    from shared.asset_formats import NUCLEUS_URL
    return f'{NUCLEUS_URL}/{path}'


def _grscenes_scenes(part):
    folder = GRSCENES_ROOT / f'{part}_scenes' / 'scenes'
    return sorted(path.name[:-4] for path in folder.glob('*_usd') if (path / 'start_result_navigation.usd').is_file()) \
        if folder.is_dir() else []


def entry(key):
    """Catalog entry for `key`: label, kind, source USD (local path), output directory, pose and wrap options."""
    source, _, rest = key.partition('/')
    if source == 'isaac' and rest in ISAAC:
        item = ISAAC[rest]
        return {**item, 'key': key, 'source': 'isaac', 'license': ISAAC_LICENSE, 'url': _isaac_url(item['path']),
                'usd': ASSETS / item['folder'] / 'source' / item['path'], 'output': ASSETS / rest, 'name': rest}
    part, _, scene = rest.partition('/')
    if source == 'grscenes' and part in GRSCENES and scene:
        folder = GRSCENES_ROOT / f'{part}_scenes' / 'scenes' / f'{scene}_usd'
        return {'key': key, 'source': 'grscenes', 'license': GRSCENES_LICENSE, 'kind': 'indoor',
                'verified': scene in GRSCENES_VERIFIED,
                'label': f'GRScenes {GRSCENES[part]["label"]} {scene[:8]}-{scene[-4:]}',
                'usd': folder / 'start_result_navigation.usd',
                'output': ASSETS / f'grscenes_{part}_{scene}', 'name': 'scene', 'pose': [0, 0, 0],
                'simplify_colliders': 0.03,
                # Visible triangles in all, and texture copies: Newton's GL viewer puts every triangle and one
                # texture per mesh on the GPU (a shop: 37 M triangles, 20 k textures of ~600 px; beyond 16 GB).
                'simplify_visuals': 15_000_000, 'texture_size': 128,
                # GRScenes' own sky sphere (52 km across) and physics scene: the project provides both.
                'deactivate': ['/Root/__default_setting', '/Root/physicsScene'],
                'note': '导航版（静态碰撞体）；厘米单位已换算为米'}
    raise KeyError(f'未知的场景环境 {key!r}；可选见 run.py asset environment --list')


def catalog():
    """Every known scene: {key, label, kind, source, local, wrapped, verified, note, license}; GRScenes scenes
    appear once their part is extracted, plus one {key: grscenes/<part>, part: True} entry per part that is not."""
    entries = []
    for key in [f'isaac/{name}' for name in ISAAC] + [f'grscenes/{part}/{scene}' for part in GRSCENES
                                                      for scene in _grscenes_scenes(part)]:
        item = entry(key)
        entries.append({'key': key, 'label': item['label'], 'kind': item['kind'], 'source': item['source'],
                        'local': item['usd'].is_file(), 'wrapped': (item['output'] / 'ASSET.json').is_file(),
                        'verified': item.get('verified', False), 'note': item.get('note', ''),
                        'license': item['license']})
    for part, item in GRSCENES.items():
        if not _grscenes_scenes(part):
            entries.append({'key': f'grscenes/{part}', 'label': f'GRScenes {item["label"]}（整包）', 'kind': 'indoor',
                            'source': 'grscenes', 'local': False, 'wrapped': False, 'verified': False, 'part': True,
                            'note': f'下载并解压整包 {item["bytes"] / 1e9:.0f} GB 后列出其中各场景', 'license': GRSCENES_LICENSE})
    return entries


def resolve(text):
    """Catalog key for a key, an Isaac scene name, or a GRScenes scene id (or its first 8 characters)."""
    text = text.strip().strip('/')
    keys = [item['key'] for item in catalog()]
    if text in keys:
        return text
    found = [key for key in keys if key.split('/')[-1] == text or
             (key.startswith('grscenes/') and len(text) >= 8 and key.split('/')[-1].startswith(text))]
    if len(found) == 1:
        return found[0]
    close = difflib.get_close_matches(text, keys + [key.split('/')[-1] for key in keys], n=5)
    raise KeyError(f'找不到场景环境 {text!r}' + (f'；有多个匹配：{", ".join(found)}' if found else '')
                   + (f'；相近：{", ".join(close)}' if close and not found else ''))


def thumbnail(key, online=True, draw=True):
    """A preview image of the scene: the publisher's own render, else a floor plan drawn from the local USD
    (``sketch``; seconds and several GB for a furnished GRScenes home, so `draw=False` skips it), else None."""
    item = entry(key)
    if item['source'] == 'grscenes':
        found = sorted((item['usd'].parent / '.thumbs' / '256x256').glob('*.png'))
        if found:
            return found[0]
    target = CACHE / 'thumbs' / f'{key.replace("/", "_")}.png'
    if not target.is_file() and online and item['source'] == 'isaac':
        directory, file = item['url'].rsplit('/', 1)
        from shared.asset_formats import _download
        try:
            data = _download(f'{directory}/.thumbs/256x256/{file}.png', attempts=2)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError:
            pass
    if not target.is_file() and draw and item['usd'].is_file():
        sketch(item['usd'], target, cut=item['kind'] == 'indoor')
    return target if target.is_file() else None


# Outward triangles of a box whose corner i has x, y, z from bits 0, 1, 2 of i.
_BOX_FACES = ((0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (1, 3, 7), (1, 7, 5),
              (0, 4, 6), (0, 6, 2), (2, 6, 7), (2, 7, 3), (0, 1, 5), (0, 5, 4))


def _meshes(stage, colliders_only=False, skip=()):
    """(world points [N, 3] in metres Z-up, triangles [T, 3] indices, collides) of every drawn mesh, or with
    `colliders_only` of every collider (invisible ones included); prims under the paths `skip` are left out."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    unit, y_up = UsdGeom.GetStageMetersPerUnit(stage) or 1., UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    cache = UsdGeom.XformCache()
    traversal = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in traversal:
        if str(prim.GetPath()) in skip:
            traversal.PruneChildren()
            continue
        if colliders_only and prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Mesh) and \
                prim.HasAPI(UsdPhysics.CollisionAPI):
            # A collider shape (cube, sphere, ...): its extent box.
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
            if extent:
                low, high = np.array(extent[0], float), np.array(extent[1], float)
                corners = np.array([[(low, high)[(i >> axis) & 1][axis] for axis in range(3)] for i in range(8)])
                matrix = np.array(cache.GetLocalToWorldTransform(prim))
                corners = ((corners @ matrix[:3, :3] + matrix[3, :3]) * unit).astype(np.float32)
                if y_up:
                    corners = corners[:, [0, 2, 1]] * np.float32((1, -1, 1))
                yield corners, np.array(_BOX_FACES), True
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        # GRScenes puts the collider on the Xform above its meshes.
        collides = any(stage.GetPrimAtPath(prefix).HasAPI(UsdPhysics.CollisionAPI)
                       for prefix in prim.GetPath().GetPrefixes()[-3:])
        imageable = UsdGeom.Imageable(prim)
        if colliders_only:
            if not collides:
                continue
        elif imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or \
                imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        mesh = UsdGeom.Mesh(prim)
        points, counts, indices = (mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(),
                                   mesh.GetFaceVertexIndicesAttr().Get())
        if not points or not counts or not indices:
            continue
        matrix = np.array(cache.GetLocalToWorldTransform(prim))
        points = ((np.array(points, float) @ matrix[:3, :3] + matrix[3, :3]) * unit).astype(np.float32)
        if y_up:
            points = points[:, [0, 2, 1]] * np.float32((1, -1, 1))
        counts, indices = np.array(counts), np.array(indices)
        fans = np.maximum(counts - 2, 0)
        first = np.repeat(np.cumsum(counts) - counts, fans)
        step = np.arange(fans.sum()) - np.repeat(np.cumsum(fans) - fans, fans)
        yield points, indices[np.stack([first, first + step + 1, first + step + 2], 1)], collides


def sketch(path, target, size=256, cut=True, density=100., seed=0, framing_points=2000):
    """Write a top-down floor plan of the scene USD at `path` to `target` (PNG): `density` points per m² sampled
    over its drawn meshes, the highest per pixel, shaded by height above the floor; with `cut`, geometry more
    than 1.8 m above the floor (ceilings, roofs) is left out so rooms stay visible. Two passes over the meshes
    (framing, then drawing) keep only one mesh in memory at a time."""
    import numpy as np
    from PIL import Image
    from pxr import Usd
    stage = Usd.Stage.Open(str(path))
    random = np.random.default_rng(seed)
    solid, every = [], []
    for points, _, collides in _meshes(stage):
        picked = points[random.choice(len(points), min(len(points), framing_points), replace=False)]
        (solid if collides else every).append(picked)
    if not solid and not every:
        return None
    # Framed by what collides (floors, walls): visual backdrops such as sky domes reach far beyond the rooms.
    framing = np.concatenate(solid or every)
    low, high = np.percentile(framing[:, :2], 0.5, axis=0), np.percentile(framing[:, :2], 99.5, axis=0)
    floor = float(np.percentile(framing[:, 2], 2))
    del solid, every, framing
    scale = (2 * size - 1) / max(float(np.max(high - low)), 1e-6)
    shape = (np.ceil((high - low) * scale).astype(int) + 1)[::-1]
    heights = np.full(shape, -np.inf, np.float32)
    density = min(density, 4 * scale ** 2)  # a few samples per pixel is enough for large scenes
    for points, triangles, _ in _meshes(stage):
        corners = points[triangles]
        # Only triangles over the picture and in the drawn height band (a sky dome would ask for 1e8 samples).
        inside = np.all((corners[..., :2].max(1) >= low) & (corners[..., :2].min(1) <= high), axis=1) & \
            (corners[..., 2].max(1) > floor - 0.5)
        if cut:
            inside &= corners[..., 2].min(1) < floor + 1.8
        corners = corners[inside]
        area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2
        expected = area * density
        if expected.sum() > 4 * heights.size:
            expected *= 4 * heights.size / expected.sum()
        chosen = np.repeat(np.arange(len(corners)), random.poisson(expected))
        u, v = random.random((2, len(chosen)), dtype=np.float32)
        flip = u + v > 1
        u, v = np.where(flip, 1 - u, u)[:, None], np.where(flip, 1 - v, v)[:, None]
        corners = corners[chosen]
        samples = np.concatenate([points, corners[:, 0] + u * (corners[:, 1] - corners[:, 0])
                                  + v * (corners[:, 2] - corners[:, 0])])
        height = samples[:, 2] - floor
        columns, rows = ((samples[:, :2] - low) * scale).astype(int).T
        keep = (columns >= 0) & (rows >= 0) & (columns < shape[1]) & (rows < shape[0]) & (height > -0.5)
        if cut:
            keep &= height < 1.8
        np.maximum.at(heights, (shape[0] - 1 - rows[keep], columns[keep]), height[keep])
    from scipy import ndimage
    heights = np.where(np.isfinite(heights), heights, ndimage.maximum_filter(heights, size=3))  # sampling gaps
    empty = ~np.isfinite(heights)
    top = 1.8 if cut else max(float(np.percentile(heights[~empty], 99)), 1.)
    level = np.clip(np.where(empty, 0, heights) / top, 0, 1)[..., None]
    gy, gx = np.gradient(np.where(empty, 0, heights))
    shade = np.clip(1 - 0.6 * (gx - gy) * scale / 4, 0.55, 1.15)[..., None]
    colour = ((1 - level) * np.array([232, 226, 214]) + level * np.array([70, 92, 128])) * shade
    image = np.where(empty[..., None], 255, np.clip(colour, 0, 255)).astype(np.uint8)
    picture = Image.fromarray(image)
    picture.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGB', (size, size), 'white')
    canvas.paste(picture, ((size - picture.width) // 2, (size - picture.height) // 2))
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return target


SPAWN_METHOD = 4  # bump when spawn_points changes, so recorded spawn points are recomputed


def spawn_points(path, offset=(0, 0, 0), radius=0.45, clear=(0.1, 0.8), count=8, seed=0, skip=()):
    """Up to `count` spread-out places to start a robot in the scene USD at `path`, placed at `offset`: on the
    lowest floor (upward-facing collider level holding at least 5% of the largest such area) that has a spot at
    least `radius` m from any collider between `clear` m above it. [{position, clearance}] in the scene frame."""
    import numpy as np
    from pxr import Usd
    stage = Usd.Stage.Open(str(path))
    levels, flats = {}, []
    for points, triangles, _ in _meshes(stage, colliders_only=True, skip=skip):
        corners = points[triangles]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        area = np.linalg.norm(normal, axis=1) / 2
        flat = normal[:, 2] > 0.9 * np.maximum(2 * area, 1e-12)  # facing up: not a slab's underside
        if not flat.any():
            continue
        heights = np.round(corners[flat, :, 2].mean(1) / 0.05).astype(int)
        for level in np.unique(heights):
            levels[level] = levels.get(level, 0.) + float(area[flat][heights == level].sum())
        every = max(1, int(flat.sum()) // 500)  # enough corners to find the floor's extent
        flats.append((corners[flat][::every, :, :2].reshape(-1, 2), heights[::every].repeat(3)))
    if not levels:
        return []
    # Pooled over ±0.25 m for gently sloped streets; a roof, ceiling or upper storey may be larger than the floor.
    pooled = {level: sum(levels.get(level + step, 0.) for step in range(-5, 6)) for level in levels}
    largest = max(pooled.values())
    tried = []
    for level in sorted(level for level, area in pooled.items() if area >= 0.05 * largest):
        if any(abs(level - done) <= 5 for done in tried):
            continue
        tried.append(level)
        found = _open_floor(stage, flats, level * 0.05, offset, radius, clear, count, seed, skip)
        if found or len(tried) == 4:
            return found
    return []


def _open_floor(stage, flats, floor, offset, radius, clear, count, seed, skip):
    import numpy as np
    from scipy import ndimage
    random = np.random.default_rng(seed)
    near = np.concatenate([xy[np.abs(level * 0.05 - floor) <= 0.1] for xy, level in flats if len(xy)])
    low, high = near.min(0) - radius, near.max(0) + radius
    cell = max(0.1, float(np.max(high - low)) / 1500)
    shape = (np.ceil((high - low) / cell).astype(int) + 1)
    ground, blocked = np.zeros(shape, bool), np.zeros(shape, bool)
    surface = np.full(shape, -np.inf)  # floor height per cell, relative to `floor`
    for points, triangles, _ in _meshes(stage, colliders_only=True, skip=skip):
        corners = points[triangles]
        inside = np.all((corners[..., :2].max(1) >= low) & (corners[..., :2].min(1) <= high), axis=1) & \
            (corners[..., 2].max(1) > floor - 0.1) & (corners[..., 2].min(1) < floor + clear[1])
        corners = corners[inside]
        if not len(corners):
            continue
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        area = np.linalg.norm(normal, axis=1) / 2
        flat = normal[:, 2] > 0.9 * np.maximum(2 * area, 1e-12)
        expected = np.minimum(area * 4 / cell ** 2, 4 * shape.prod())
        chosen = np.repeat(np.arange(len(corners)), random.poisson(expected))
        u, v = random.random((2, len(chosen)))
        swap = u + v > 1
        u, v = np.where(swap, 1 - u, u)[:, None], np.where(swap, 1 - v, v)[:, None]
        picked = corners[chosen]
        samples = np.concatenate([corners.reshape(-1, 3), picked[:, 0] + u * (picked[:, 1] - picked[:, 0])
                                  + v * (picked[:, 2] - picked[:, 0])])
        on_flat = np.concatenate([np.repeat(flat, 3), flat[chosen]])
        index = np.clip(np.floor((samples[:, :2] - low) / cell).astype(int), 0, shape - 1)
        height = samples[:, 2] - floor
        floor_hit = on_flat & (np.abs(height) <= 0.15)
        ground[index[floor_hit, 0], index[floor_hit, 1]] = True
        np.maximum.at(surface, (index[floor_hit, 0], index[floor_hit, 1]), height[floor_hit])
        hit = (height > clear[0]) & (height < clear[1])
        blocked[index[hit, 0], index[hit, 1]] = True
    ground = ndimage.binary_closing(ground, iterations=2)  # sampling gaps in the floor
    if np.isfinite(surface).any():
        surface = np.where(np.isfinite(surface), surface, np.median(surface[np.isfinite(surface)]))
    room = ndimage.distance_transform_edt(ground & ~blocked) * cell
    candidates = np.argwhere(room >= radius)
    if not len(candidates):
        return []
    # The roomiest place first, then each next as far as possible from those chosen (room past 1 m counts the same).
    chosen, nearest = [], np.full(len(candidates), np.inf)
    score = room[tuple(candidates.T)]
    for _ in range(min(count, len(candidates))):
        best = int(np.argmax(nearest * np.minimum(score, 1.0) if chosen else score))
        chosen.append(best)
        nearest = np.minimum(nearest, np.linalg.norm((candidates - candidates[best]) * cell, axis=1))
        if np.max(nearest) < 2 * radius:
            break
    return [{'position': [round(float(low[0] + (candidates[i][0] + .5) * cell + offset[0]), 3),
                          round(float(low[1] + (candidates[i][1] + .5) * cell + offset[1]), 3),
                          round(float(floor + surface[tuple(candidates[i])] + offset[2]), 3)],
             'clearance': round(float(score[i]), 2)} for i in chosen]


LARGE_SCENE = 1000  # objects: a scene this rich (a furnished shop) is a performance test, not a working scene


def wrapped_usd(key):
    """The wrapper USD of a wrapped scene, or None before it is wrapped."""
    item = entry(key)
    usd = item['output'] / 'usd' / f"{item['name']}.usda"
    return usd if usd.is_file() else None


def object_count(key, analyse=True):
    """Objects of a wrapped scene (shared/assembly/composition.py), or None before it is wrapped; with
    ``analyse=False`` only when the analysis is cached (a large outdoor scene takes 15 s to analyse)."""
    usd = wrapped_usd(key)
    if usd is None:
        return None
    from shared.assembly.composition import cached_scene_objects, scene_objects
    objects = scene_objects(usd) if analyse else cached_scene_objects(usd)
    return None if objects is None else sum(kind == 'object' for _, kind in objects)


def recorded_spawn(key):
    """Spawn points recorded when `key` was wrapped, or None before it is wrapped."""
    path = entry(key)['output'] / 'ASSET.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get('environment', {}).get('spawn')


def _options(item):
    """What the wrapper depends on besides the source: changing any of it rewraps."""
    # 'colliders': simplified colliders kept per scene object (removable with usd_object.exclude).
    return {'pose': item['pose'], 'substitutes': item.get('substitutes') or {}, 'colliders': 'per_object',
            'footprint_colliders': item.get('footprint_colliders') or [],
            'simplify_colliders': item.get('simplify_colliders'), 'deactivate': item.get('deactivate') or [],
            'simplify_visuals': item.get('simplify_visuals'), 'texture_size': item.get('texture_size')}


def prepare(key, progress=None):
    """Download what is missing and wrap: the ASSET.json record for usd_object. An existing wrapper of the
    same source and options is reused; one of other options (a newer catalog) is rewrapped. Calls from several
    threads run one after another (scenes of one Isaac directory share its downloaded files)."""
    say = progress or (lambda message: None)
    if not _preparing.acquire(blocking=False):
        say(f'{key}：等待前一个场景准备完成…')
        _preparing.acquire()
    try:
        record = _prepare(key, say)
    finally:
        _preparing.release()
    try:  # the editor's mesh, ahead: adding the scene to any project then shows it at once
        prepare_editor_mesh(key, say)
    except Exception as error:  # noqa: BLE001 - the editor exports it itself when it is added
        say(f'{key}：编辑器网格未预先生成（{str(error)[-300:]}），添加后由编辑器生成')
    return record


_preparing = threading.Lock()
EDITOR_SCENES = CACHE / 'editor'


def prepare_editor_mesh(key, progress=None, backend='isaaclab3'):
    """Export the web editor's lightweight mesh of a wrapped scene into the shared cache
    (shared/assembly/visuals.py), from a scene of that environment alone: the cache key depends on the
    asset and its parameters only, so the editor of any project adding the scene (as the shelf does, without
    ``exclude``) finds it. A furnished house takes about a minute, once; a cached one returns at once."""
    from shared.assembly.visuals import ensure_visuals
    say = progress or (lambda message: None)
    usd = wrapped_usd(key)
    if usd is None:
        raise ValueError(f'{key} is not wrapped yet')
    folder = EDITOR_SCENES / re.sub(r'[^A-Za-z0-9_.-]', '_', key)
    folder.mkdir(parents=True, exist_ok=True)
    scene = folder / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities:\n'
                     '  - {name: environment, factory: usd_object, kind: entity, scope: global,\n'
                     f'     parameters: {{usd_path: {json.dumps(str(Path(os.path.relpath(usd, ROOT))))}}}}}\n')
    (folder / 'train.yaml').write_text(f'env:\n  train:\n    init_params:\n      backend: {backend}\n')
    say(f'{key}：生成编辑器网格（首次约一分钟）…')
    return ensure_visuals(scene)


def _prepare(key, say):
    from shared.asset_formats import environment, vendor_url, workspace_path
    item = entry(key)
    if not item['usd'].is_file():
        if item['source'] != 'isaac':
            raise FileNotFoundError(f'{workspace_path(item["usd"])} 不存在；先下载整包：run.py asset environment '
                                    f'grscenes/{key.split("/")[1]}')
        from shared.asset_formats import NUCLEUS_URL
        say(f'下载 {item["url"]} 及其引用的文件…')
        vendor_url(item['url'], ASSETS / item['folder'], NUCLEUS_URL)
    substitutes = {}
    for missing, path in (item.get('substitutes') or {}).items():
        target = ASSETS / 'vegetation' / 'source' / path
        if not target.is_file():
            say(f'下载替代植被 {path}…')
            vendor_url(f'{VEGETATION_URL}/{path}', ASSETS / 'vegetation', VEGETATION_URL)
        substitutes[missing] = target
    options = _options(item)
    record_path = item['output'] / 'ASSET.json'
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        if record.get('source') != workspace_path(item['usd']):
            raise FileExistsError(f'{workspace_path(item["output"])} 已是另一个资产（{record.get("source")}）')
        if (record.get('environment') or {}).get('options') == options:
            return _with_spawn(record, record_path, say, item.get('spawn_exclude', []))
        say(f'{key} 的包装选项已变，重新包装…')
        record_path.unlink()
        shutil.rmtree(item['output'] / 'usd', ignore_errors=True)
    say(f'包装 {key}（静态碰撞体、Newton 兼容修正）…')
    record = environment(item['usd'], item['output'], item['name'], substitutes, item.get('footprint_colliders') or (),
                         catalog={'key': key, 'label': item['label'], 'pose': item['pose'], 'license': item['license'],
                                  'options': options}, simplify_colliders=item.get('simplify_colliders'),
                         deactivate=item.get('deactivate') or (), simplify_visuals=item.get('simplify_visuals'),
                         texture_size=item.get('texture_size'))
    return _with_spawn(record, record_path, say, item.get('spawn_exclude', []))


def _with_spawn(record, record_path, say, skip):
    """The record with `environment.spawn` (spawn_points, leaving out the prims `skip`), computed once."""
    from shared.asset_formats import ROOT as WORKSPACE
    found = record['environment']
    if found.get('spawn_method') != SPAWN_METHOD or found.get('spawn_exclude', []) != skip:
        say('查找可放机器人的空地…')
        found['spawn'] = spawn_points(WORKSPACE / record['formats']['usd'], found['pose'], skip=skip)
        found['spawn_exclude'], found['spawn_method'] = skip, SPAWN_METHOD
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return record


def download_grscenes(part, threads=16, progress=None, chunk=64 << 20):
    """Download one GRScenes part's zip (resumable range requests) and extract it under assets/grscenes/."""
    from urllib.request import Request, urlopen
    item = GRSCENES[part]
    url, archive = MODELSCOPE + item['zip'], GRSCENES_ROOT / 'download' / Path(item['zip']).name
    archive.parent.mkdir(parents=True, exist_ok=True)
    state = archive.with_name(archive.name + '.chunks.json')
    done = set(json.loads(state.read_text())) if state.is_file() else set()
    total, count = item['bytes'], (item['bytes'] + chunk - 1) // chunk
    lock = threading.Lock()
    if not archive.is_file() or archive.stat().st_size != total:
        with open(archive, 'ab') as handle:
            handle.truncate(total)

    def fetch(index):
        start, end = index * chunk, min(total, (index + 1) * chunk) - 1
        for attempt in range(10):
            try:
                with urlopen(Request(url, headers={'Range': f'bytes={start}-{end}'}), timeout=120) as response:
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError(f'short read {len(data)}')
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(min(60, 2 ** attempt))
        with open(archive, 'r+b') as handle:
            handle.seek(start)
            handle.write(data)
        with lock:
            done.add(index)
            state.write_text(json.dumps(sorted(done)))
            if progress:
                progress(f'GRScenes {part}: {len(done)}/{count} 块')

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(fetch, [index for index in range(count) if index not in done]))
    if progress:
        progress(f'解压 {archive.name}…')
    subprocess.run(['unzip', '-q', '-n', str(archive), '-d', str(GRSCENES_ROOT)], check=True)
    return _grscenes_scenes(part)


if __name__ == '__main__':  # python -m shared.environments [key ...]: draw the missing previews, one line per key
    import fcntl
    import sys
    # One drawer per machine, niced and the first the kernel kills when memory runs out (not the editor or IDE).
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = open(CACHE / 'drawing.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit('another process is drawing environment previews')
    os.nice(10)
    with contextlib.suppress(OSError):
        Path('/proc/self/oom_score_adj').write_text('1000')
    for name in sys.argv[1:] or [item['key'] for item in catalog() if not item.get('part')]:
        try:
            print(name, thumbnail(name) or '', flush=True)
        except Exception as error:  # noqa: BLE001 - one scene's failure leaves the others
            print(name, '', f'{type(error).__name__}: {error}', flush=True)
