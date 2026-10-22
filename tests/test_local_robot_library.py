"""Local robot discovery is lazy and does not turn compositions into assets."""
import json
from pathlib import Path
import subprocess
import sys
from shared.assembly.catalog import describe_scene
from shared.backends.isaac.assets import FACTORIES

ROOT=Path(__file__).resolve().parents[1]


def test_independent_local_robots_are_discoverable_without_sdk_imports():
    code='''import sys,json
from shared.assembly.catalog import describe_scene
r=describe_scene('projects/assembly_go2/scene.yaml')
assert not any(n.split('.')[0] in ('isaaclab','isaacsim','omni','torch') for n in sys.modules)
print(json.dumps({e['id']:e for e in r['assets']}))
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assets=json.loads(result.stdout)
    for name in ('piper','piper_x','m20','go2','anymal_c'):
        assert assets[name]['category']=='robot'
        assert 'joint_state' in assets[name]['capabilities']
    for name in ('piper','piper_x','m20'):
        assert 'fix_root_link' in assets[name]['parameters']
    assert assets['anymal_c']['label']=='ANYmal-C 四足机器人'
    assert 'm20_piper' not in assets and 'm20_piper_pendulum' not in assets


def test_new_robot_instances_get_independent_unselected_observations(tmp_path):
    scene={'schema_version':1,'name':'world','entities':[
        {'name':'arm','factory':'piper_x','kind':'robot'},
        {'name':'base','factory':'m20','kind':'robot'},
        {'name':'dog','factory':'anymal_c','kind':'robot'}]}
    report=describe_scene(scene,directory=tmp_path)
    terms={item['id']:item for item in report['observations']}
    for name in ('arm','base','dog'):
        entry=terms[f'{name}.joint_position']
        assert entry['selected_in']==[]
        assert entry['source_groups']['default']['group']==f'world/{name}'
    assert all(name in FACTORIES for name in ('piper','piper_x','m20','go2','anymal_c'))
