from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

from shared import environments


def test_catalog_scene_is_wrapped_once_and_rewrapped_when_its_options_change(tmp_path, monkeypatch):
    source = tmp_path / 'assets/demo/source/Isaac/Environments/Demo/demo.usda'
    source.parent.mkdir(parents=True)
    (tmp_path / 'assets/demo/source/SOURCE.json').write_text('{}')
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, '/World/floor').GetPrim())
    stage.Save()
    monkeypatch.setattr(environments, 'ASSETS', tmp_path / 'assets')
    monkeypatch.setattr(environments, 'GRSCENES_ROOT', tmp_path / 'assets/grscenes')
    monkeypatch.setattr(environments, 'ISAAC', {'demo': {'label': 'Demo', 'kind': 'indoor', 'folder': 'demo',
                                                         'path': 'Isaac/Environments/Demo/demo.usda', 'pose': [0, 0, 1]}})

    assert environments.resolve('demo') == 'isaac/demo'
    assert [(item['key'], item['local'], item['wrapped']) for item in environments.catalog() if not item.get('part')] \
        == [('isaac/demo', True, False)]
    record = environments.prepare('isaac/demo')
    assert record['environment']['key'] == 'isaac/demo' and record['environment']['pose'] == [0, 0, 1]
    written = (tmp_path / 'assets/demo/ASSET.json').stat().st_mtime_ns
    assert environments.prepare('isaac/demo') == record
    assert (tmp_path / 'assets/demo/ASSET.json').stat().st_mtime_ns == written
    environments.ISAAC['demo']['pose'] = [0, 0, 2]
    assert environments.prepare('isaac/demo')['environment']['pose'] == [0, 0, 2]


def _room(path, units=1.0):
    """A 6 m x 6 m floor of 60 x 60 quads (top at z = 0) with a 1 m box in its middle, both colliders."""
    import numpy as np
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, units)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
    grid = np.stack(np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61), indexing='ij'), -1).reshape(-1, 2)
    floor = UsdGeom.Mesh.Define(stage, '/World/floor')
    floor.CreatePointsAttr([(x / units, y / units, 0) for x, y in grid])
    quads = [(i * 61 + j, (i + 1) * 61 + j, (i + 1) * 61 + j + 1, i * 61 + j + 1) for i in range(60) for j in range(60)]
    floor.CreateFaceVertexCountsAttr([4] * len(quads))
    floor.CreateFaceVertexIndicesAttr([index for quad in quads for index in quad])
    box = UsdGeom.Cube.Define(stage, '/World/box')
    box.CreateSizeAttr(1 / units)
    box.AddTranslateOp().Set((0, 0, 0.5 / units))
    for prim in (floor.GetPrim(), box.GetPrim()):
        UsdPhysics.CollisionAPI.Apply(prim)
    UsdGeom.Sphere.Define(stage, '/World/sky').CreateRadiusAttr(5000 / units)  # a visual sky dome
    stage.Save()


def test_simplified_colliders_replace_dense_meshes_in_any_source_unit(tmp_path):
    from shared.asset_formats import environment, usd_bounds
    _room(tmp_path / 'room.usda', units=0.01)
    record = environment(tmp_path / 'room.usda', tmp_path / 'room', simplify_colliders=0.5, deactivate=['/World/sky'])
    report = record['steps'][0]
    simplified = report['simplified_colliders']
    assert simplified['meshes'] == 1 and simplified['triangles_before'] == 7200
    assert 0 < simplified['triangles_after'] < 7200  # vertices snapped to 0.5 m, i.e. 50 source units
    out = Usd.Stage.Open(str(tmp_path / 'room/usd/room.usda'))
    assert report['deactivated'] == ['/World/sky'] and not out.GetPrimAtPath('/World/sky').IsActive()
    assert not out.GetPrimAtPath('/World/floor').HasAPI(UsdPhysics.CollisionAPI)
    assert out.GetPrimAtPath('/World/box').HasAPI(UsdPhysics.CollisionAPI)  # not a Mesh: kept as is
    # The floor is a scene object and a gprim: its simplified collider is beside it, removed with it.
    assert simplified['object_colliders'] == 1 and simplified['tiles'] == 0
    collider = out.GetPrimAtPath('/World/floor_simplified_collider')
    assert collider.HasAPI(UsdPhysics.CollisionAPI) and not out.GetPrimAtPath('/World/floor').GetChildren()
    assert UsdGeom.Imageable(collider).ComputePurpose() == UsdGeom.Tokens.guide
    # the editor placeholder spans the colliders in metres (guide ones included), not a sky dome
    low, high = usd_bounds(tmp_path / 'room/usd/room.usda')
    assert max(map(abs, low + high)) < 3.5 and high[2] > 0.9


def test_simplified_visuals_keep_uvs_and_material_subsets(tmp_path):
    import numpy as np
    from PIL import Image
    from pxr import Sdf, UsdShade, Vt
    from shared.asset_formats import environment
    _room(tmp_path / 'room.usda')
    Image.new('RGB', (512, 256), 'red').save(tmp_path / 'wood.png')
    stage = Usd.Stage.Open(str(tmp_path / 'room.usda'))
    texture = UsdShade.Shader.Define(stage, '/World/looks/wood/texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set('./wood.png')
    floor = UsdGeom.Mesh(stage.GetPrimAtPath('/World/floor'))
    corners = len(floor.GetFaceVertexIndicesAttr().Get())
    UsdGeom.PrimvarsAPI(floor).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying) \
        .Set(Vt.Vec2fArray.FromNumpy(np.arange(corners * 2, dtype=np.float32).reshape(-1, 2)))
    half = UsdGeom.Subset.CreateGeomSubset(floor, 'left', UsdGeom.Tokens.face, list(range(1800)), 'materialBind')
    UsdGeom.Subset.CreateGeomSubset(floor, 'right', UsdGeom.Tokens.face, list(range(1800, 3600)), 'materialBind')
    stage.Save()
    record = environment(tmp_path / 'room.usda', tmp_path / 'room', simplify_colliders=0.5, simplify_visuals=1000,
                         texture_size=64)
    assert record['steps'][0]['shrunk_textures'] == {'size': 64, 'files': 1, 'shrunk': 1}
    report = record['steps'][0]['simplified_visuals']
    assert report['simplified'] == 1 and 0 < report['triangles_after'] <= 1000
    assert set(record['sha256']) == {'usd/room.usda', 'usd/room.visuals.usdc'}
    out = Usd.Stage.Open(str(tmp_path / 'room/usd/room.usda'))
    mesh = UsdGeom.Mesh(out.GetPrimAtPath('/World/floor'))
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    assert len(faces) <= 1000 and faces.max() < len(mesh.GetPointsAttr().Get())
    assert len(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get()) == faces.size
    subsets = [np.asarray(subset.GetIndicesAttr().Get()) for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh)]
    assert sorted(np.concatenate(subsets).tolist()) == list(range(len(faces))) and all(map(len, subsets))
    left = np.asarray(mesh.GetPointsAttr().Get())[faces[np.asarray(UsdGeom.Subset(
        out.GetPrimAtPath(half.GetPath())).GetIndicesAttr().Get())]]
    assert left[..., 0].mean() < 0  # the first 1800 quads lie at x < 0
    copy = out.GetPrimAtPath('/World/looks/wood/texture').GetAttribute('inputs:file').Get().resolvedPath
    assert Path(copy).parent.name == 'room.textures' and Image.open(copy).size == (64, 32)
    assert Image.open(tmp_path / 'wood.png').size == (512, 256)


def test_spawn_points_stand_on_the_floor_clear_of_obstacles(tmp_path):
    import numpy as np
    _room(tmp_path / 'room.usda')
    found = environments.spawn_points(tmp_path / 'room.usda', offset=(10, 0, 1))
    assert len(found) >= 4
    for spawn in found:
        x, y, z = spawn['position']
        assert abs(z - 1) < 0.02 and spawn['clearance'] >= 0.45
        # 0.45 m from the 1 m box, inside the 6 m floor (one 0.1 m cell of slack)
        assert max(abs(x - 10), abs(y)) >= 0.5 + 0.45 - 0.1 and max(abs(x - 10), abs(y)) <= 3
    assert np.min([np.hypot(a['position'][0] - b['position'][0], a['position'][1] - b['position'][1])
                   for a in found for b in found if a is not b]) >= 0.9
