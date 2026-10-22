"""Native preview caches retain pose edits but reject stale geometry."""
import json
import shutil
from pathlib import Path
from shared.assembly.editing import SceneDocument
from shared.assembly.visuals import cache_path,geometry_keys,load_visuals

ROOT=Path(__file__).resolve().parents[1]

def test_native_preview_default_pose_and_geometry_invalidation(tmp_path):
    scene=tmp_path/'scene.yaml'
    shutil.copyfile(ROOT/'projects/assembly_go2/scene.yaml',scene)
    keys=geometry_keys(scene)
    mesh=tmp_path/'robot.mesh.json.gz';mesh.write_bytes(b'cached')
    record={'key':keys['world/robot'],'path':str(mesh),'default_pose':{'position':[0,0,.4],'orientation':[1,0,0,0]},'exact':True}
    record['initial_state'] = {'joint_positions': {'knee': -1.5}, 'linear_velocity': [0, 0, 0]}
    cache=cache_path(scene);cache.parent.mkdir()
    cache.write_text(json.dumps({'entities':{'world/robot':record}}))
    doc=SceneDocument(scene)
    robot=next(x for x in doc.snapshot()['entities'] if x['path']=='world/robot')
    assert robot['display_pose']['position']==[0,0,.4] and not robot['pose_authored']
    assert robot['initial_display']['joint_positions'] == {'knee': -1.5}
    assert robot['category'] == 'robot'
    before=scene.read_bytes();assert not doc.save();assert scene.read_bytes()==before
    doc.set_entity('world/robot',{'pose':{'position':[1,2,.5]}});doc.save()
    assert load_visuals(scene)['world/robot']['path']==str(mesh)
    robot=next(x for x in SceneDocument(scene).snapshot()['entities'] if x['path']=='world/robot')
    assert robot['display_pose']['position']==[1,2,.5]
    # Custom source changes can alter native geometry/defaults without YAML edits.
    (tmp_path/'assets.py').write_text('# changed geometry factory\n')
    assert load_visuals(scene)=={}


def test_mount_keeps_mesh_cache_but_joint_pose_invalidates(tmp_path):
    scene = tmp_path / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities:\n  - {name: arm, factory: piper_x}\n')
    before = geometry_keys(scene)
    doc = SceneDocument(scene)
    doc.set_entity('world/arm', {'parameters': {'fix_root_link': False}})
    doc.save()
    assert geometry_keys(scene) == before
    doc.set_entity('world/arm', {'initial_state': {'joint_positions': {'joint1': .5}}})
    doc.save()
    assert geometry_keys(scene) != before


def test_editor_opens_before_native_export(tmp_path, monkeypatch):
    from tools import blender_editor
    scene = tmp_path / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities: []\n')
    launched = []
    monkeypatch.setattr(blender_editor.shutil, 'which', lambda name: '/usr/bin/blender')
    monkeypatch.setattr(blender_editor.subprocess, 'call', lambda command, **kwargs: launched.append(command) or 0)
    import shared.assembly.visuals as visuals
    monkeypatch.setattr(visuals, 'ensure_visuals', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Open must not block on Isaac')))
    assert blender_editor.main(['open', '--scene', str(scene)]) == 0
    assert '--background-visuals' in launched[0]
    assert blender_editor.main(['open', '--scene', str(scene), '--no-native-preview']) == 0
    assert '--background-visuals' not in launched[1]


def test_shared_preview_reused_for_new_instance(tmp_path, monkeypatch):
    import shared.assembly.visuals as visuals
    shared = tmp_path / 'shared_cache'
    shared.mkdir()
    monkeypatch.setattr(visuals, 'SHARED_CACHE', shared)
    scene = tmp_path / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities:\n  - {name: first, factory: piper_x}\n  - {name: second, factory: piper_x}\n')
    keys = geometry_keys(scene)
    assert keys['world/first'] == keys['world/second']
    mesh = shared / 'mesh.json.gz'
    mesh.write_bytes(b'cached')
    record = {'key': keys['world/first'], 'path': str(mesh), 'preview_version': 1}
    (shared / (record['key'] + '.json')).write_text(json.dumps(record))
    assert set(load_visuals(scene)) == {'world/first', 'world/second'}
    monkeypatch.setattr(visuals.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Must reuse cache')))
    assert visuals.ensure_visuals(scene)['cached']
