"""Procedural terrain tiles as ordinary static assets usable by every backend.

A tile is generated once by an Isaac Lab sub-terrain generator (variant names and
parameters as in shared/backends/isaac/assets.py TERRAIN_VARIANTS), then written as a
USD and an MJCF whose collision geometry matches, plus ASSET.json. Scenes use it like any
other prop (`usd_object` pointing at the USD); the tile is centred on the entity origin.

    run.py asset terrain stairs assets/terrains/stairs --size 8 --difficulty 0.6 \\
        --param '{"step_height_range": [0.1, 0.2], "step_width": 0.3, "platform_width": 2.0}'

USD keeps the generated triangles as one static mesh collider. MuJoCo collides meshes as
convex hulls, so the MJCF is rebuilt from pieces it can represent exactly: boxes and
convex hulls for closed pieces (stairs, pits, boxes, rails, gaps, repeated objects, grids
of columns) and an hfield for a regular height-field surface (rough, slope, waves,
obstacles, stepping stones). A flat open tile becomes a thin slab. Anything else (a
closed non-convex piece, an irregular open surface) is rejected rather than approximated.
After writing, the MJCF is ray-cast against the generated triangles and the difference is
stored in ASSET.json; the tile is rejected above HEIGHT_TOLERANCE. MuJoCo splits hfield cells
along the (i, j)-(i+1, j+1) diagonal, as Isaac Lab's height-field meshes do, so they agree;
a mesh split the other way would fail that check.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from shared.asset_formats import ROOT, compare, digest, usd_layers, workspace_path

# Parameters each tile is generated with when none are given (and by `run.py asset terrain all`).
PRESETS = {
    'flat': {},
    'stairs': dict(step_height_range=(.1, .2), step_width=.3, platform_width=2., border_width=.5),
    'inverted_stairs': dict(step_height_range=(.1, .2), step_width=.3, platform_width=2., border_width=.5),
    'random_grid': dict(grid_width=.45, grid_height_range=(.05, .2), platform_width=1.),
    'rails': dict(rail_thickness_range=(.1, .2), rail_height_range=(.1, .2), platform_width=2.),
    'pit': dict(pit_depth_range=(.1, .3), platform_width=2.),
    'box': dict(box_height_range=(.1, .3), platform_width=2.),
    'gap': dict(gap_width_range=(.2, .5), platform_width=2.),
    'floating_ring': dict(ring_width_range=(.3, .5), ring_height_range=(.3, .5), ring_thickness=.1, platform_width=2.),
    'star': dict(num_bars=4, bar_width_range=(.2, .4), bar_height_range=(.1, .2), platform_width=1.5),
    'repeated_pyramids': dict(object_params_start=dict(num_objects=4, height=.05, radius=.5, max_yx_angle=0.),
                              object_params_end=dict(num_objects=10, height=.2, radius=.4, max_yx_angle=0.),
                              platform_width=2.),
    'repeated_boxes': dict(object_params_start=dict(num_objects=4, height=.05, size=(.5, .5)),
                           object_params_end=dict(num_objects=10, height=.2, size=(.4, .4)), platform_width=2.),
    'repeated_cylinders': dict(object_params_start=dict(num_objects=4, height=.05, radius=.3),
                               object_params_end=dict(num_objects=10, height=.2, radius=.25), platform_width=2.),
    'rough': dict(noise_range=(.02, .08), noise_step=.02, border_width=.25),
    'slope': dict(slope_range=(.1, .4), platform_width=2., border_width=.25),
    'inverted_slope': dict(slope_range=(.1, .4), platform_width=2., border_width=.25),
    'hf_stairs': dict(step_height_range=(.05, .15), step_width=.4, platform_width=2., border_width=.25),
    'hf_inverted_stairs': dict(step_height_range=(.05, .15), step_width=.4, platform_width=2., border_width=.25),
    'obstacles': dict(obstacle_height_mode='choice', obstacle_width_range=(.3, .8), obstacle_height_range=(.05, .2),
                      num_obstacles=20, platform_width=2., border_width=.25),
    'waves': dict(amplitude_range=(.1, .2), num_waves=3, border_width=.25),
    'stepping_stones': dict(stone_height_max=.1, stone_width_range=(.4, .8), stone_distance_range=(.1, .3),
                            holes_depth=-1., platform_width=2., border_width=.25),
}
# Shelf names, recorded in ASSET.json.
LABELS = {
    'flat': '平地', 'stairs': '台阶（上）', 'inverted_stairs': '台阶（下）', 'random_grid': '随机方格',
    'rails': '横杆', 'pit': '坑', 'box': '高台', 'gap': '沟', 'floating_ring': '悬空环', 'star': '星形窄桥',
    'repeated_pyramids': '散布锥体', 'repeated_boxes': '散布方块', 'repeated_cylinders': '散布圆柱',
    'rough': '粗糙地面', 'slope': '坡（上）', 'inverted_slope': '坡（下）', 'hf_stairs': '高度场台阶（上）',
    'hf_inverted_stairs': '高度场台阶（下）', 'obstacles': '离散障碍', 'waves': '波浪', 'stepping_stones': '梅花桩',
}
SLAB = 0.2            # m, thickness standing in for an open flat tile
HEIGHT_TOLERANCE = 2e-3  # m, MJCF vs generated triangles at the sampled points
GREY = (.55, .55, .55)


def tile_pieces(meshes, size):
    """Split generated triangle meshes into boxes, convex hulls and at most one height grid.

    Coordinates are shifted so the tile centre is the origin."""
    import trimesh

    shift = np.array([size / 2, size / 2, 0.])
    boxes, convex, grids = [], [], []
    whole = trimesh.util.concatenate([trimesh.Trimesh(v - shift, f, process=True) for v, f in meshes])
    for part in whole.split(only_watertight=False, repair=False):
        if part.is_watertight and abs(part.volume) > 1e-12:
            volume = abs(part.volume)
            if abs(part.bounding_box.volume - volume) <= 1e-6 * volume:
                boxes.append((part.bounds.mean(axis=0), part.extents / 2))
            elif part.convex_hull.volume <= volume * (1 + 1e-4):
                convex.append(part)
            else:
                raise NotImplementedError('closed non-convex terrain piece: MuJoCo would collide its convex hull')
        elif np.ptp(part.vertices[:, 2]) < 1e-9 and abs(part.area - np.prod(part.extents[:2])) < 1e-6 * part.area:
            top = part.vertices[0, 2]
            boxes.append((np.array([*part.bounds.mean(axis=0)[:2], top - SLAB / 2]),
                          np.array([*part.extents[:2] / 2, SLAB / 2])))
        else:
            grids.append(_height_grid(part))
    if len(grids) > 1:
        raise NotImplementedError('more than one height-field surface in a tile')
    return whole, boxes, convex, grids[0] if grids else None


def _height_grid(part):
    xs, ys = (np.unique(np.round(part.vertices[:, axis], 6)) for axis in (0, 1))
    steps = [np.diff(values) for values in (xs, ys)]
    if len(xs) * len(ys) != len(part.vertices) or len(part.faces) != 2 * (len(xs) - 1) * (len(ys) - 1) \
            or any(len(s) == 0 or not np.allclose(s, s[0], atol=1e-5) for s in steps):
        raise NotImplementedError('open terrain surface is not a regular height grid')
    heights = np.empty((len(ys), len(xs)))
    heights[np.searchsorted(ys, np.round(part.vertices[:, 1], 6)),
            np.searchsorted(xs, np.round(part.vertices[:, 0], 6))] = part.vertices[:, 2]
    return xs, ys, heights


def write_usd(path, whole):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/terrain')
    stage.SetDefaultPrim(root.GetPrim())
    surface = UsdGeom.Mesh.Define(stage, '/terrain/surface')
    surface.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in whole.vertices])
    surface.CreateFaceVertexCountsAttr([3] * len(whole.faces))
    surface.CreateFaceVertexIndicesAttr([int(index) for index in whole.faces.ravel()])
    surface.CreateSubdivisionSchemeAttr('none')
    surface.CreateDoubleSidedAttr(True)
    surface.CreateDisplayColorAttr([Gf.Vec3f(*GREY)])
    UsdPhysics.CollisionAPI.Apply(surface.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(surface.GetPrim()).CreateApproximationAttr('none')
    stage.Save()


def write_mjcf(directory, name, boxes, convex, grid):
    assets, geoms, files = [], [], directory / 'assets'
    files.mkdir(parents=True)
    rgba = ' '.join(f'{value:g}' for value in (*GREY, 1.))
    for index, (centre, half) in enumerate(boxes):
        geoms.append(f'<geom name="box{index}" type="box" pos="{_text(centre)}" size="{_text(half)}" rgba="{rgba}"/>')
    for index, part in enumerate(convex):
        part.export(files / f'convex{index}.obj')
        assets.append(f'<mesh name="convex{index}" file="assets/convex{index}.obj"/>')
        geoms.append(f'<geom name="convex{index}" type="mesh" mesh="convex{index}" rgba="{rgba}"/>')
    if grid is not None:
        xs, ys, heights = grid
        low, span = heights.min(), max(np.ptp(heights), 1e-6)
        (files / 'surface.bin').write_bytes(np.array([len(ys), len(xs)], '<i4').tobytes()
                                            + ((heights - low) / span).astype('<f4').tobytes())
        assets.append(f'<hfield name="surface" file="assets/surface.bin" nrow="{len(ys)}" ncol="{len(xs)}" '
                      f'size="{(xs[-1] - xs[0]) / 2:.9g} {(ys[-1] - ys[0]) / 2:.9g} {span:.9g} {SLAB:g}"/>')
        centre = ((xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2, low)
        geoms.append(f'<geom name="surface" type="hfield" hfield="surface" pos="{_text(centre)}" rgba="{rgba}"/>')
    path = directory / f'{name}.xml'
    path.write_text('<mujoco model="{}">\n  <compiler meshdir="./"/>\n  <asset>\n    {}\n  </asset>\n'
                    '  <worldbody>\n    <body name="terrain">\n      {}\n    </body>\n  </worldbody>\n</mujoco>\n'
                    .format(name, '\n    '.join(assets), '\n      '.join(geoms)))
    return path


def _text(values):
    return ' '.join(f'{float(value):.9g}' for value in values)


def height_error(whole, mjcf, size):
    """Largest difference of the top surface between the generated triangles and the MJCF."""
    import mujoco
    import trimesh

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    inside = np.linspace(-size / 2 * 0.95, size / 2 * 0.95, 41) + 1.234e-3
    points = np.array([(x, y) for x in inside for y in inside])
    top = whole.bounds[1][2] + 1.
    origins = np.column_stack([points, np.full(len(points), top)])
    locations, rays, _ = whole.ray.intersects_location(origins, np.tile([0., 0., -1.], (len(points), 1)),
                                                      multiple_hits=False)
    expected = np.full(len(points), np.nan)
    expected[rays] = locations[:, 2]
    worst = 0.
    for point, height in zip(origins, expected):
        distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]), None, 1, -1, np.zeros(1, 'i4'))
        if np.isnan(height) != (distance < 0):
            raise ValueError(f'MJCF and USD disagree on whether the terrain exists at {point[:2]}')
        if distance >= 0:
            worst = max(worst, abs(top - distance - height))
    return worst


def build(meshes, origin, size, output, name, generator):
    """Write usd/, mjcf/ and ASSET.json under `output` from (vertices, faces) meshes of one tile."""
    output = Path(output).resolve()
    if (output / 'ASSET.json').exists():
        raise FileExistsError(f'{output / "ASSET.json"} exists; remove it or choose another output')
    whole, boxes, convex, grid = tile_pieces(meshes, size)
    (output / 'usd').mkdir(parents=True, exist_ok=True)
    usd = output / 'usd' / f'{name}.usda'
    write_usd(usd, whole)
    mjcf = write_mjcf(output / 'mjcf', name, boxes, convex, grid)
    error = height_error(whole, mjcf, size)
    if error > HEIGHT_TOLERANCE:
        raise ValueError(f'MJCF terrain differs from the generated triangles by {error:.4f} m')
    record = {
        'source': workspace_path(usd), 'source_format': 'usd', 'source_sha256': digest(usd),
        'source_layers_sha256': usd_layers(usd),
        'formats': {'usd': workspace_path(usd), 'mjcf': workspace_path(mjcf)},
        'steps': [{'from': 'isaaclab terrain generator', 'to': 'usd + mjcf', 'generator': generator,
                   'mjcf_collision': {'boxes': len(boxes), 'convex_meshes': len(convex),
                                      'height_field': grid is not None},
                   'max_height_difference': error}],
        'terrain': {'variant': generator.get('variant'), 'label': LABELS.get(generator.get('variant'), name),
                    'size': size, 'centre': 'tile centre is the entity origin',
                    'spawn_origin': (np.asarray(origin) - [size / 2, size / 2, 0.]).tolist()},
        'alignment': compare(usd, mjcf),
        'sha256': {path.relative_to(output).as_posix(): digest(path)
                   for path in sorted(output.rglob('*')) if path.is_file()},
    }
    (output / 'ASSET.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return record


def generate(variant, output, *, name=None, size=8.0, difficulty=0.5, seed=0, parameters=None):
    """Generate a tile with Isaac Lab (in the Isaac interpreter) and write it with build().

    Without `parameters` the variant's PRESETS entry is used."""
    parameters = PRESETS.get(variant, {}) if parameters is None else parameters
    python = os.environ.get('ISAAC_PYTHON', str(ROOT / '.venv/bin/python'))
    with tempfile.TemporaryDirectory() as scratch:
        dump = Path(scratch) / 'tile.npz'
        subprocess.run([python, '-m', 'shared.backends.isaac.terrain', variant, str(dump), str(size),
                        str(difficulty), str(seed), json.dumps(parameters)],
                       check=True, cwd=ROOT, env={
                           **os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES',
                           'PYTHONPATH': os.pathsep.join(filter(None, (str(ROOT), os.environ.get('PYTHONPATH'))))})
        if not dump.exists():
            raise RuntimeError(f"Isaac generated no tile for '{variant}' (see the error printed above)")
        with np.load(dump) as tile:
            count = len([key for key in tile.files if key.startswith('vertices')])
            meshes = [(tile[f'vertices{index}'], tile[f'faces{index}']) for index in range(count)]
            origin = tile['origin']
    generator = {'variant': variant, 'parameters': parameters, 'size': size, 'difficulty': difficulty, 'seed': seed}
    return build(meshes, origin, size, output, name or variant, generator)
