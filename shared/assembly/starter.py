"""New environments in the browser and Blender editors: a draft copy of a maintained template.

It is the same skeleton as ``run.py new`` (``tools/create.py``), so a new environment
already has control targets, policy.py, task.py and train.yaml and can be checked and
trained. The draft lives in a temporary directory until the first save moves the whole
project to ``projects/<name>``; an existing project is never overwritten.
"""
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT / 'projects'


def create_draft(name, template='planner_executor'):
    """Copy ``template`` as project ``name`` into a temporary directory; returns its scene.yaml."""
    from tools.create import create_project
    if (PROJECTS / name).exists():
        raise FileExistsError(f'项目目录已经存在：projects/{name}')
    directory = Path(tempfile.mkdtemp(prefix='rlinf-environment-'))
    try:
        target, _ = create_project(name, template, destination=directory)
    except BaseException:
        shutil.rmtree(directory)
        raise
    return target / 'scene.yaml'


def publish_draft(scene, name):
    """Move the draft project of ``scene`` to projects/<name>; returns the new scene.yaml."""
    source, destination = Path(scene).parent, PROJECTS / name
    if destination.exists():
        raise FileExistsError(f'项目目录已经存在：projects/{name}')
    shutil.copytree(source, destination)
    shutil.rmtree(source.parent, ignore_errors=True)
    return destination / 'scene.yaml'
