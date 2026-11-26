import pytest

pytest.importorskip('pxr')

from shared.assembly.composition import is_composite, members, object_of, objects_below, tree


def triangle(stage, path, offset=(0., 0., 0.), collider=False):
    from pxr import UsdGeom, UsdPhysics
    mesh = UsdGeom.Mesh.Define(stage, path)
    x, y, z = offset
    mesh.CreatePointsAttr([(x, y, z), (x + 1, y, z), (x, y + 1, z + 1)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    if collider:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def write_home(path, collider=False):
    """A GRScenes-like house: Scope folders, one Xform per object, a light, a material and an Xform
    holding more meshes of its own than an object would (a container, as Isaac's Root)."""
    from pxr import Usd, UsdGeom, UsdLux, UsdShade
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/Root').GetPrim())
    for folder in ('Meshes', 'Meshes/Base', 'Meshes/Base/wall', 'Meshes/Furnitures', 'Meshes/Furnitures/table', 'Lights'):
        UsdGeom.Scope.Define(stage, f'/Root/{folder}')
    for index in (1, 2):
        UsdGeom.Xform.Define(stage, f'/Root/Meshes/Base/wall/wall_{index}')
        triangle(stage, f'/Root/Meshes/Base/wall/wall_{index}/mesh', (index, 0, 0), collider)
        triangle(stage, f'/Root/Meshes/Base/wall/wall_{index}/trim', (index, 0, 1), collider)
    UsdGeom.Xform.Define(stage, '/Root/Meshes/Furnitures/table/model_0123456789abcdef0123_0')
    triangle(stage, '/Root/Meshes/Furnitures/table/model_0123456789abcdef0123_0/top', (5, 5, .8), collider)
    UsdLux.SphereLight.Define(stage, '/Root/Lights/lamp')
    UsdShade.Material.Define(stage, '/Root/Looks/wood')
    UsdGeom.Xform.Define(stage, '/Root/Floor')
    for index in range(45):
        triangle(stage, f'/Root/Floor/SM_floor{index}', (index, 9, 0), collider)
    stage.Save()
    return path


def test_objects_are_the_first_xform_or_gprim_below_containers(tmp_path):
    from pxr import Usd
    stage = Usd.Stage.Open(str(write_home(tmp_path / 'home.usda')))
    found = objects_below(stage.GetDefaultPrim())
    objects = dict(found)
    assert objects['Meshes/Base/wall/wall_1'] == 'object'  # its two meshes are parts of one object
    assert 'Meshes/Base/wall/wall_1/mesh' not in objects
    assert objects['Lights/lamp'] == 'light' and not any(name.startswith('Looks') for name in objects)
    assert objects['Floor/SM_floor0'] == 'object' and 'Floor' not in objects  # 45 meshes: a container
    assert len(objects) == 2 + 1 + 1 + 45 and is_composite(list(objects.items()))
    assert object_of('Meshes/Base/wall/wall_2/trim', found) == 'Meshes/Base/wall/wall_2'
    assert object_of('Lights', found) is None


def test_tree_groups_long_folders_and_shows_a_lone_object_by_its_folder(tmp_path):
    from pxr import Usd
    stage = Usd.Stage.Open(str(write_home(tmp_path / 'home.usda')))
    objects = objects_below(stage.GetDefaultPrim())
    rows = tree(objects, exclude=['Meshes/Base/wall/wall_2', 'Floor'])
    by_id = {row['id']: row for row in rows}
    table = 'Meshes/Furnitures/table/model_0123456789abcdef0123_0'
    assert 'Meshes/Furnitures/table' not in by_id
    assert by_id[table]['label'] == 'table' and by_id[table]['parent'] == 'Meshes/Furnitures'
    group = by_id['Floor/SM_floor ×']
    assert group['label'] == 'SM_floor ×45' and group['count'] == 45 and group['parent'] == 'Floor'
    assert by_id['Floor/SM_floor7']['parent'] == 'Floor/SM_floor ×'
    assert by_id['Meshes']['count'] == 3 and by_id['Meshes/Base/wall']['count'] == 2
    assert by_id['Meshes/Base/wall/wall_2']['excluded'] and not by_id['Meshes/Base/wall/wall_1']['excluded']
    assert by_id['Floor/SM_floor3']['excluded']  # below an excluded container
    assert sorted(members(rows, 'Meshes/Base')) == ['Meshes/Base/wall/wall_1', 'Meshes/Base/wall/wall_2']
    assert len(members(rows, 'Floor/SM_floor ×')) == 45 and members(rows, table) == [table]


def test_pruned_layer_deactivates_excluded_objects(tmp_path, monkeypatch):
    from pxr import Usd
    import shared.asset_formats as formats
    monkeypatch.setattr(formats, 'ROOT', tmp_path)
    source = write_home(tmp_path / 'home.usda')
    layer = formats.pruned_layer(source, ['Meshes/Base/wall/wall_2', '/Lights/'])
    assert layer.parent == tmp_path / '.cache/usd_pruned'
    stage = Usd.Stage.Open(str(layer))
    assert not stage.GetCompositionErrors() and stage.GetDefaultPrim().GetPath() == '/Root'
    assert not stage.GetPrimAtPath('/Root/Meshes/Base/wall/wall_2').IsActive()
    assert not stage.GetPrimAtPath('/Root/Lights').IsActive()
    assert stage.GetPrimAtPath('/Root/Meshes/Base/wall/wall_1').IsActive()
    assert formats.pruned_layer(source, ['Lights', 'Meshes/Base/wall/wall_2']) == layer
    with pytest.raises(ValueError, match='do not exist: Meshes/sofa'):
        formats.pruned_layer(source, ['Meshes/sofa'])
    with pytest.raises(ValueError, match='relative to the default prim'):
        formats.pruned_layer(source, ['../other'])


def test_simplified_colliders_stay_with_their_object(tmp_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from shared.asset_formats import environment_layer
    source = write_home(tmp_path / 'home.usda', collider=True)
    report = environment_layer(source, tmp_path / 'usd/home.usda', simplify_colliders=.01)
    assert report['simplified_colliders']['object_colliders'] == 2 + 1 + 45
    assert report['simplified_colliders']['tiles'] == 0
    stage = Usd.Stage.Open(str(tmp_path / 'usd/home.usda'))
    wall = stage.GetPrimAtPath('/Root/Meshes/Base/wall/wall_1/simplified_collider')
    assert wall.HasAPI(UsdPhysics.CollisionAPI) and len(UsdGeom.Mesh(wall).GetFaceVertexCountsAttr().Get()) == 2
    assert not stage.GetPrimAtPath('/Root/Meshes/Base/wall/wall_1/mesh').HasAPI(UsdPhysics.CollisionAPI)
    # The collider keeps the place of the geometry it replaces.
    world = UsdGeom.XformCache().GetLocalToWorldTransform(wall)
    corners = [world.Transform(Gf.Vec3d(*point)) for point in UsdGeom.Mesh(wall).GetPointsAttr().Get()]
    assert any(Gf.IsClose(corner, Gf.Vec3d(1, 0, 0), 1e-6) for corner in corners)
    # A gprim object (a floor tile below a container) gets its collider beside it, and the wrapper's
    # guide colliders are not scene objects.
    tile = stage.GetPrimAtPath('/Root/Floor/SM_floor3_simplified_collider')
    assert tile.HasAPI(UsdPhysics.CollisionAPI) and not stage.GetPrimAtPath('/Root/Floor/SM_floor3').GetChildren()
    corners = [UsdGeom.XformCache().GetLocalToWorldTransform(tile).Transform(Gf.Vec3d(*point))
               for point in UsdGeom.Mesh(tile).GetPointsAttr().Get()]
    assert any(Gf.IsClose(corner, Gf.Vec3d(3, 9, 0), 1e-6) for corner in corners)
    assert dict(objects_below(stage.GetDefaultPrim())).keys() == dict(objects_below(
        Usd.Stage.Open(str(source)).GetDefaultPrim())).keys()


def test_pruned_layer_removes_the_collider_beside_a_gprim_object(tmp_path, monkeypatch):
    from pxr import Usd
    import shared.asset_formats as formats
    from shared.asset_formats import environment_layer
    monkeypatch.setattr(formats, 'ROOT', tmp_path)
    wrapped = tmp_path / 'usd/home.usda'
    environment_layer(write_home(tmp_path / 'home.usda', collider=True), wrapped, simplify_colliders=.01)
    stage = Usd.Stage.Open(str(formats.pruned_layer(wrapped, ['Floor/SM_floor3'])))
    assert not stage.GetPrimAtPath('/Root/Floor/SM_floor3_simplified_collider').IsActive()
    assert stage.GetPrimAtPath('/Root/Floor/SM_floor4_simplified_collider').IsActive()


def test_display_parts_draw_the_largest_folders_as_one_part_past_the_cap(tmp_path):
    from pxr import Usd
    from shared.assembly.composition import display_parts
    stage = Usd.Stage.Open(str(write_home(tmp_path / 'home.usda')))
    objects = objects_below(stage.GetDefaultPrim())
    assert all(part == [name] for name, part in display_parts(objects).items())  # 49 objects: one part each
    parts = display_parts(objects, cap=10)
    assert len(parts['Floor/SM_floor ×']) == 45 and parts['Meshes/Base/wall/wall_1'] == ['Meshes/Base/wall/wall_1']
    assert len(parts) == 2 + 1 + 1 and 'Lights/lamp' not in parts  # lights have no mesh part


def test_one_off_names_beside_name_groups_form_a_rest_group(tmp_path):
    from pxr import Usd
    stage = Usd.Stage.Open(str(write_home(tmp_path / 'home.usda')))
    for index in range(45):
        triangle(stage, f'/Root/Floor/prop_{chr(65 + index % 26)}{chr(65 + index // 26)}x', (index, 20, 0))
    rows = {row['id']: row for row in tree(objects_below(stage.GetDefaultPrim()))}
    assert rows['Floor/其他 ×']['label'] == '其他 ×45' and rows['Floor/prop_AAx']['parent'] == 'Floor/其他 ×'
    assert rows['Floor/SM_floor ×']['count'] == 45


def test_a_group_holding_many_gprims_is_split_into_its_parts(tmp_path, monkeypatch):
    """Rivermark's ``main``: a handful of children (props, foliage, roads) with 100 000 meshes in all."""
    from pxr import Usd, UsdGeom
    import shared.assembly.composition as composition
    stage = Usd.Stage.CreateNew(str(tmp_path / 'town.usda'))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
    UsdGeom.Xform.Define(stage, '/World/main')
    for group in ('houses', 'trees'):
        UsdGeom.Xform.Define(stage, f'/World/main/{group}')
        for index in range(3):
            UsdGeom.Xform.Define(stage, f'/World/main/{group}/{group}_{index}')
            for part in range(2):
                triangle(stage, f'/World/main/{group}/{group}_{index}/mesh_{part}', (index, part, 0))
    assert [name for name, _ in objects_below(stage.GetDefaultPrim())] == ['main']  # 12 gprims: one object
    monkeypatch.setattr(composition, 'GROUP_GPRIMS', 5)
    names = [name for name, _ in objects_below(stage.GetDefaultPrim())]
    assert names[:2] == ['main/houses/houses_0', 'main/houses/houses_1'] and len(names) == 6


def test_object_layer_takes_one_object_out_in_place(tmp_path, monkeypatch):
    """A chair of a centimetre scene, turned and moved by its own transform and its folder's."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    import shared.asset_formats as formats
    monkeypatch.setattr(formats, 'ROOT', tmp_path)
    stage = Usd.Stage.CreateNew(str(tmp_path / 'room.usda'))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    top = UsdGeom.Xform.Define(stage, '/Root')
    top.AddScaleOp().Set(Gf.Vec3f(.01))
    stage.SetDefaultPrim(top.GetPrim())
    UsdGeom.Xform.Define(stage, '/Root/Meshes').AddTranslateOp().Set(Gf.Vec3d(100, 0, 0))
    chair = UsdGeom.Xform.Define(stage, '/Root/Meshes/chair')
    chair.AddTranslateOp().Set(Gf.Vec3d(0, 200, 0))
    chair.AddRotateZOp().Set(90.)
    triangle(stage, '/Root/Meshes/chair/seat', (10, 0, 45), collider=True)
    triangle(stage, '/Root/Meshes/table', (0, 0, 70))
    stage.Save()
    pose = formats.object_pose(tmp_path / 'room.usda', 'Meshes/chair')
    assert pose['position'] == pytest.approx([1., 2., 0.]) and pose['orientation'] == pytest.approx(
        [.7071068, 0, 0, .7071068], abs=1e-6)
    layer = formats.object_layer(tmp_path / 'room.usda', 'Meshes/chair', convex=True)
    out = Usd.Stage.Open(str(layer))
    assert not out.GetCompositionErrors() and out.GetDefaultPrim().GetPath() == '/Object'
    assert not out.GetPrimAtPath('/Object/frame/table') and out.GetPrimAtPath('/Object/frame/object/seat')
    seat = out.GetPrimAtPath('/Object/frame/object/seat')
    assert UsdPhysics.MeshCollisionAPI(seat).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull
    # Placed at the pose, the seat is where it was in the scene (metres; the scene at the origin).
    entity = Gf.Matrix4d().SetRotate(Gf.Quatd(*pose['orientation'])).SetTranslateOnly(Gf.Vec3d(*pose['position']))
    world = UsdGeom.XformCache().GetLocalToWorldTransform(seat) * entity
    original = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/Root/Meshes/chair/seat'))
    for point in UsdGeom.Mesh(seat).GetPointsAttr().Get():
        assert Gf.IsClose(world.Transform(Gf.Vec3d(*point)), original.Transform(Gf.Vec3d(*point)), 1e-6)
    with pytest.raises(ValueError, match='no such prim'):
        formats.object_layer(tmp_path / 'room.usda', 'Meshes/sofa')


def test_base_colour_comes_from_the_material_and_its_texture(tmp_path):
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdShade
    from shared.asset_formats import base_color
    Image.new('RGB', (16, 16), (128, 255, 0)).save(tmp_path / 'leaf.png')
    stage = Usd.Stage.CreateNew(str(tmp_path / 'look.usda'))
    meshes = [triangle(stage, f'/World/m{index}') for index in range(4)]

    def material(name, shader_id=None, **inputs):
        look = UsdShade.Material.Define(stage, f'/World/Looks/{name}')
        shader = UsdShade.Shader.Define(stage, f'/World/Looks/{name}/shader')
        if shader_id:
            shader.CreateIdAttr(shader_id)
            look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        else:  # an MDL material, as Isaac's and GRScenes' scenes author them
            look.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(), 'out')
        for key, (kind, value) in inputs.items():
            shader.CreateInput(key, kind).Set(value)
        return look, shader
    preview, _ = material('red', 'UsdPreviewSurface', diffuseColor=(Sdf.ValueTypeNames.Color3f, Gf.Vec3f(1, 0, 0)))
    textured, shader = material('leaf', 'UsdPreviewSurface')
    reader = UsdShade.Shader.Define(stage, '/World/Looks/leaf/texture')
    reader.CreateIdAttr('UsdUVTexture')
    reader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set('./leaf.png')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), 'rgb')
    mdl, _ = material('wall', Color=(Sdf.ValueTypeNames.Color4f, Gf.Vec4f(.5, .5, .5, 1)),
                      BaseColor=(Sdf.ValueTypeNames.Asset, Sdf.AssetPath('./leaf.png')))
    for mesh, look in zip(meshes, (preview, textured, mdl)):
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(look)
    meshes[3].CreateDisplayColorAttr([Gf.Vec3f(0, 0, 1)])
    stage.Save()
    colours = [base_color(mesh.GetPrim()) for mesh in meshes]
    assert colours[0] == pytest.approx([1, 0, 0]) and colours[3] == pytest.approx([0, 0, 1])
    assert colours[1] == pytest.approx([128 / 255, 1, 0], abs=1e-3)
    assert colours[2] == pytest.approx([64 / 255, .5, 0], abs=1e-3)  # Color x the average of BaseColor


def test_copies_of_one_model_keep_one_geometry_and_their_placement(tmp_path):
    """The exporter's object parts: a model referenced twice is stored once, the second as a matrix."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from shared.assembly.composition import model_key
    from tools.export_scene_visuals import _scene_objects
    model = Usd.Stage.CreateNew(str(tmp_path / 'chair.usda'))
    model.SetDefaultPrim(UsdGeom.Xform.Define(model, '/Chair').GetPrim())
    triangle(model, '/Chair/seat', (0, 0, .4))
    model.Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / 'room.usda'))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/Root').GetPrim())
    for name, position, yaw in (('chair_a', (1, 0, 0), 0.), ('chair_b', (3, 2, 0), 90.)):
        chair = UsdGeom.Xform.Define(stage, f'/Root/{name}')
        chair.AddTranslateOp().Set(Gf.Vec3d(*position))
        chair.AddRotateZOp().Set(yaw)
        chair.GetPrim().GetReferences().AddReference('./chair.usda')
    triangle(stage, '/Root/table', (5, 5, .7))
    stage.Save()
    top, cache = stage.GetDefaultPrim(), UsdGeom.XformCache()
    assert model_key(stage.GetPrimAtPath('/Root/chair_a')) == model_key(stage.GetPrimAtPath('/Root/chair_b')) is not None
    assert model_key(stage.GetPrimAtPath('/Root/table')) is None
    parts = []
    for prim in Usd.PrimRange(top):
        if prim.IsA(UsdGeom.Mesh):
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            parts.append({'name': str(prim.GetPath()), 'vertices': points @ matrix[:3, :3] + matrix[3, :3],
                          'faces': np.array([[0, 1, 2]])})
    node = type('Node', (), {'factory': 'usd_object'})()
    by_name = {part['name']: part for part in _scene_objects(node, {}, top, parts, cache.GetLocalToWorldTransform)}
    assert 'instance' not in by_name['chair_a'] and 'instance' not in by_name['table']
    copy = by_name['chair_b']
    assert copy['instance']['of'] == 'chair_a' and not len(copy['vertices'])
    source = by_name['chair_a']['vertices']
    moved = np.c_[source, np.ones(len(source))] @ np.asarray(copy['instance']['matrix'])
    expected = next(part['vertices'] for part in parts if part['name'] == '/Root/chair_b/seat')
    assert np.allclose(moved[:, :3], expected, atol=1e-6)


def test_scene_edit_layer_moves_an_object_and_its_collider(tmp_path, monkeypatch):
    """A chair of a centimetre scene put elsewhere: its new pose is in the entity frame, metres."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    import shared.asset_formats as formats
    monkeypatch.setattr(formats, 'ROOT', tmp_path)
    stage = Usd.Stage.CreateNew(str(tmp_path / 'room.usda'))
    top = UsdGeom.Xform.Define(stage, '/Root')
    top.AddScaleOp().Set(Gf.Vec3f(.01))
    stage.SetDefaultPrim(top.GetPrim())
    UsdGeom.Xform.Define(stage, '/Root/Meshes').AddTranslateOp().Set(Gf.Vec3d(100, 0, 0))
    chair = UsdGeom.Xform.Define(stage, '/Root/Meshes/chair')
    chair.AddTranslateOp().Set(Gf.Vec3d(0, 200, 0))
    triangle(stage, '/Root/Meshes/chair/seat', (10, 0, 45))
    triangle(stage, '/Root/Meshes/lamp', (0, 0, 70))
    triangle(stage, '/Root/Meshes/lamp' + formats.COLLIDER_SUFFIX, (0, 0, 70))
    stage.Save()
    source = tmp_path / 'room.usda'
    before = formats.object_pose(source, 'Meshes/chair')
    assert before['position'] == pytest.approx([1, 2, 0])
    target = {'position': [3., -1., 0.], 'orientation': [.7071068, 0, 0, .7071068]}
    lamp_target = {'position': [0., 0., 0.5], 'orientation': [1, 0, 0, 0]}
    layer = formats.scene_edit_layer(source, exclude=[], moved={'Meshes/chair': target, 'Meshes/lamp': lamp_target})
    assert formats.object_pose(layer, 'Meshes/chair')['position'] == pytest.approx(target['position'], abs=1e-6)
    assert formats.object_pose(layer, 'Meshes/chair')['orientation'] == pytest.approx(target['orientation'], abs=1e-6)
    edited = Usd.Stage.Open(str(layer))
    cache = UsdGeom.XformCache()

    def world(path):  # in the entity frame: the top prim's scale only, as Isaac Lab spawns it
        return cache.GetLocalToWorldTransform(edited.GetPrimAtPath(path)) * \
            cache.GetLocalToWorldTransform(edited.GetPrimAtPath('/Root')).GetInverse() * Gf.Matrix4d().SetScale(.01)
    # The seat keeps its place on the chair; the lamp's collider moves with the lamp.
    seat = world('/Root/Meshes/chair/seat').Transform(Gf.Vec3d(10, 0, 45))
    assert np.allclose(seat, [3, -1 + .1, .45], atol=1e-6)
    lamp, collider = world('/Root/Meshes/lamp'), world('/Root/Meshes/lamp' + formats.COLLIDER_SUFFIX)
    assert np.allclose(np.asarray(lamp), np.asarray(collider))
    with pytest.raises(ValueError, match='moved.Meshes/chair'):
        formats.scene_edit_layer(source, moved={'Meshes/chair': {'position': [1, 2]}})

pytestmark = pytest.mark.group_import  # runs in the [import] dependency group
