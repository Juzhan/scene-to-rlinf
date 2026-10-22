"""Generated object policies own their networks and remain portable project modules."""
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('project',['assembly_lift','assembly_go2','assembly_go2_navigation','planner_executor'])
def test_project_policy_has_no_other_project_or_simulator_dependency(project):
    script='''
import importlib, importlib.abc, sys
project=sys.argv[1]
allowed='projects.'+project
class OnlyThisProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('projects.') and fullname != allowed and not fullname.startswith(allowed+'.'):
            raise AssertionError(fullname)
        if fullname.startswith('isaaclab'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,OnlyThisProject())
module=importlib.import_module(allowed+'.policy')
assert module is not None
importlib.import_module(allowed+'.task')
importlib.import_module(allowed+'.observation')
'''
    result=subprocess.run([sys.executable,'-c',script,project],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
