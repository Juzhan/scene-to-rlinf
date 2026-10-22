from pathlib import Path
import shutil
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
from shared.assembly.object_schema import generate


def lift_copy(tmp_path):
    target = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', target, ignore=shutil.ignore_patterns('__pycache__'))
    return target


def test_observation_types_declare_function_contracts(tmp_path):
    project = lift_copy(tmp_path)
    text = generate(project / 'scene.yaml', write=False)
    assert "PlannerOutput = Tensor  # ['B', 3], 固定组件" in text
    assert "ExecutorOutput = Tensor  # ['B', 8], 训练组件 forward 返回" in text
    # Regex joint selections have no known count before inspect: left as None.
    assert "    arm: Tensor  # ['B', 7], joint_position, robot: panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5, panda_joint6, panda_joint7\n" in text
    assert 'class TaskResult(TypedDict, total=False)' in text
    assert [path.name for path in project.glob('*.md')] == ['README.md']
    namespace = {}
    exec(compile(text, 'observation.py', 'exec'), namespace)
    assert set(namespace['ExecutorTargets'].__annotations__) == {'arm', 'gripper'}


def test_auto_executor_dimension_counts_listed_joints(tmp_path):
    project = lift_copy(tmp_path)
    scene = project / 'scene.yaml'
    scene.write_text(scene.read_text().replace('dimension: 8', 'dimension: auto'))
    text = generate(scene, write=False)
    assert "ExecutorOutput = Tensor  # ['B', 9]" in text  # 7 arm + 2 finger joints, listed by name.
    assert "    output: Tensor  # ['B', 9], float, 本步输出（解码前动作）" in text


def test_regex_joint_selection_is_rejected():
    from shared.assembly.control import require_listed_joints
    listed = {'arm': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['panda_joint1']}}
    require_listed_joints(listed)
    with pytest.raises(ValueError, match='逐个列出'):
        require_listed_joints({'arm': {**listed['arm'], 'joint_names': ['panda_joint[1-7]']}})


def test_editor_save_refreshes_observation_types(tmp_path):
    from shared.assembly.editing import SceneDocument
    project = lift_copy(tmp_path)
    document = SceneDocument(project / 'scene.yaml')
    document.data['control']['output']['gripper']['joint_names'] = ['panda_finger_joint1']
    assert document.save()
    text = (project / 'observation.py').read_text()
    assert "    gripper: Tensor  # ['B', 1], joint_position, robot: panda_finger_joint1\n" in text
    assert not (project / 'PROJECT_CONTEXT.md').exists()


def test_starter_scene_uses_shared_factories(tmp_path):
    from shared.assembly.assets import factories_for
    from shared.assembly.layout import load_world_spec
    from shared.assembly.starter import write_starter_scene
    scene = write_starter_scene(tmp_path / 'scene.yaml')
    entries = yaml.safe_load(scene.read_text())['entities']
    assert [entry['factory'] for entry in entries] == ['ground_plane', 'dome_light', 'franka_panda', 'dex_cube']
    factories = factories_for('isaac')
    paths = load_world_spec(scene, factories=factories).index()
    assert {'world/ground', 'world/robot', 'world/cube'} <= set(paths)


def test_edit_opens_browser_by_default(monkeypatch):
    import run
    import tools.blender_editor
    launched = []
    monkeypatch.setattr(tools.blender_editor, 'main', lambda argv: launched.append(argv) or 0)
    for extra, command in (([], 'web'), (['--web'], 'web'), (['--blender'], 'open')):
        monkeypatch.setattr(sys, 'argv', ['run.py', 'edit', 'assembly_lift', *extra])
        with pytest.raises(SystemExit):
            run.main()
        assert launched.pop()[0] == command
