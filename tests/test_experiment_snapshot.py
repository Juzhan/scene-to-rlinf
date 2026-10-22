"""Project snapshots protect saved policies from silent scene/interface drift."""

from importlib import import_module
import json
from pathlib import Path
import sys
import uuid

from omegaconf import OmegaConf
import pytest

from shared.experiments import (
    compare_project, create_snapshot, find_snapshot, load_snapshot, prepare_replay,
    project_signature, read_manifest, verify_snapshot,
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    name = "snapshot_probe_" + uuid.uuid4().hex[:8]
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text(
        "from shared.assembly.project import Project\n"
        "project = Project(__file__, __name__, task_id='snapshot-test', model_type='snapshot-policy')\n"
        "def register():\n    pass\n"
    )
    (root / "policy.py").write_text("SCALE = 1.0\n")
    (root / "assemblies").mkdir()
    (root / "assemblies/robot.yaml").write_text(
        "schema_version: 1\nroot:\n  name: robot\n  kind: robot\n  factory: go2\n"
    )
    scene = {
        "schema_version": 1, "name": "world",
        "entities": [{"name": "robot", "assembly": "assemblies/robot.yaml", "pose": {"position": [0, 0, 0.4]}}],
        "observations": {"states": ["joint_pos", "joint_vel"]},
        "actions": {"joints": {"type": "joint_position", "asset_name": "robot", "joint_names": [".*"], "scale": 0.25}},
        "task": {"preset": "velocity"},
    }
    OmegaConf.save(OmegaConf.create(scene), root / "scene.yaml")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = import_module(name)
    cfg = OmegaConf.create({
        "runner": {"only_eval": False, "resume_dir": None, "logger": {"log_path": str(tmp_path / "outputs"), "experiment_name": "probe"}},
        "env": {"train": {"init_params": {"id": "snapshot-test", "obs_dim": 24, "action_dim": 12}},
                "eval": "${env.train}"},
        "actor": {"model": {"model_type": "snapshot-policy", "obs_dim": 24, "action_dim": 12}},
    })
    yield root, cfg, module
    for loaded in list(sys.modules):
        if loaded == name or loaded.startswith(name + "."):
            sys.modules.pop(loaded, None)


def save_scene(root, change):
    scene = OmegaConf.load(root / "scene.yaml")
    change(scene)
    OmegaConf.save(scene, root / "scene.yaml")


def test_snapshot_freezes_scene_subassemblies_source_and_recipe(project, tmp_path):
    root, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    _, manifest = read_manifest(snapshot)
    assert {"policy.py", "scene.yaml", "assemblies/robot.yaml"} <= manifest["source_files"].keys()
    assert manifest["shared_sources"] and manifest["dependencies"]["python"]
    assert not any(path.endswith("experiments.py") for path in manifest["shared_sources"])
    save_scene(root, lambda scene: scene.entities[0].pose.position.__setitem__(0, 7))
    frozen_cfg, frozen = load_snapshot(snapshot, register=False)
    assert frozen.project.scene_config != root / "scene.yaml"
    assert OmegaConf.load(frozen.project.scene_config).entities[0].pose.position[0] == 0
    from shared.assembly.layout import load_world_spec
    assert load_world_spec(frozen_cfg.env.train.init_params.scene_config).index()["world/robot"][0].factory == "go2"
    assert frozen_cfg.actor.model.obs_dim == 24
    assert Path(frozen_cfg.runner.logger.log_path) == tmp_path / "run"
    with pytest.raises(FileExistsError):
        create_snapshot(cfg, module, tmp_path / "run")


@pytest.mark.parametrize("change", [
    lambda scene: scene.observations.__setitem__("states", ["joint_vel", "joint_pos"]),
    lambda scene: setattr(scene.actions.joints, "scale", 0.5),
    lambda scene: setattr(scene.actions.joints, "joint_names", ["FR_.*", "FL_.*"]),
])
def test_same_dimension_semantic_changes_are_rejected(project, tmp_path, change):
    root, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    save_scene(root, change)
    for allow in (False, True):
        with pytest.raises(ValueError, match="semantics"):
            compare_project(snapshot, cfg, module, allow_scene_change=allow)


def test_scene_change_requires_explicit_choice_and_reports_it(project, tmp_path):
    root, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    save_scene(root, lambda scene: scene.entities[0].pose.position.__setitem__(0, 2))
    with pytest.raises(ValueError, match="Project differs"):
        compare_project(snapshot, cfg, module)
    report = compare_project(snapshot, cfg, module, allow_scene_change=True)
    assert report["changed"] == ["scene configuration"]
    assert report["policy_interface_matches"] is True


def test_large_assets_are_referenced_hashed_and_checked(project, tmp_path):
    root, cfg, module = project
    asset = root / "robot.usd"
    asset.write_bytes(b"asset contents")
    initial = tmp_path / "weights.pt"
    initial.write_bytes(b"checkpoint")
    cfg.actor.model.initial_checkpoint = str(initial)
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    _, manifest = read_manifest(snapshot)
    assert {item["path"] for item in manifest["assets"]} == {str(asset), str(initial)}
    frozen_asset = snapshot / "source" / manifest["package"] / "robot.usd"
    assert frozen_asset.is_symlink()
    asset.write_bytes(b"changed asset")
    with pytest.raises(ValueError, match="Referenced asset changed"):
        load_snapshot(snapshot, register=False)


def test_resume_uses_frozen_source_and_existing_rlinf_checkpoint(project, tmp_path):
    root, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    checkpoint = tmp_path / "run/probe/checkpoints/global_step_20"
    checkpoint.mkdir(parents=True)
    assert find_snapshot(checkpoint) == snapshot
    frozen_cfg, frozen = load_snapshot(snapshot, [f"runner.resume_dir={checkpoint}", "runner.max_epochs=30"], register=False)
    assert frozen_cfg.runner.resume_dir == str(checkpoint)
    assert frozen_cfg.runner.max_epochs == 30
    next_snapshot = create_snapshot(frozen_cfg, frozen, tmp_path / "resume")
    next_cfg, next_module = load_snapshot(next_snapshot, register=False)
    assert next_cfg.runner.resume_dir == str(checkpoint)
    assert project_signature(next_cfg, next_module)["interface_sha256"] == project_signature(cfg, module)["interface_sha256"]
    assert read_manifest(next_snapshot)[1]["project"] == root.name
    cfg.runner.resume_dir = str(checkpoint)
    cfg.runner.max_epochs = 40
    resumed, imported, _ = prepare_replay(cfg, module, checkpoint=checkpoint)
    assert resumed.runner.resume_dir == str(checkpoint)
    assert resumed.runner.max_epochs == 40
    assert imported.__name__.startswith("_experiment_")


def test_frozen_overrides_cannot_silently_change_interface(project, tmp_path):
    _, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    with pytest.raises(ValueError, match="policy/model semantics"):
        load_snapshot(snapshot, ["actor.model.obs_dim=12"], register=False)
    with pytest.raises(ValueError, match="scene or policy interface"):
        load_snapshot(snapshot, ["env.train.init_params.scene_config=/tmp/other.yaml"], register=False)


def test_frozen_files_and_shared_runtime_are_verified(project, tmp_path, monkeypatch):
    import shared.experiments as experiments
    _, cfg, module = project
    runtime = tmp_path / "backend.py"
    runtime.write_text("version = 1\n")
    monkeypatch.setattr(experiments, "_runtime_sources", lambda: {str(runtime): experiments._hash(runtime)})
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    runtime.write_text("version = 2\n")
    with pytest.raises(ValueError, match="Shared framework source changed"):
        verify_snapshot(snapshot)
    accepted = verify_snapshot(snapshot, allow_shared_change=True)
    assert accepted["shared_changes"] == [str(runtime)]
    (snapshot / "train.yaml").write_text("changed: true\n")
    with pytest.raises(ValueError, match="Frozen project file changed"):
        verify_snapshot(snapshot, allow_shared_change=True)


def test_rollout_worker_is_in_the_saved_runtime_fingerprint(project, tmp_path, monkeypatch):
    _, cfg, module = project
    rlinf = tmp_path / "rlinf"
    worker = rlinf / "rlinf/workers/rollout/hf/huggingface_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("sample_mode = 'eval'\n")
    monkeypatch.setenv("RLINF_ROOT", str(rlinf))
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    assert str(worker) in read_manifest(snapshot)[1]["shared_sources"]
    worker.write_text("sample_mode = 'train'\n")
    with pytest.raises(ValueError, match="huggingface_worker.py"):
        verify_snapshot(snapshot)


def test_legacy_checkpoint_is_not_silently_assumed_compatible(project, tmp_path):
    _, cfg, module = project
    checkpoint = tmp_path / "old.pt"
    checkpoint.write_bytes(b"weights")
    with pytest.raises(ValueError, match="no project snapshot"):
        prepare_replay(cfg, module, checkpoint=checkpoint)
    _, _, report = prepare_replay(cfg, module, checkpoint=checkpoint, allow_legacy_checkpoint=True)
    assert report["policy_interface_matches"] is None


def test_run_can_move_without_reading_the_old_assembly_directory(project, tmp_path):
    root, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    moved = tmp_path / "relocated"
    load_snapshot(snapshot, register=False)
    snapshot.parent.rename(moved)
    # The authored source can also disappear: only referenced large assets stay external.
    (root / "assemblies/robot.yaml").unlink()
    frozen_cfg, frozen = load_snapshot(moved / "snapshot", register=False)
    from shared.assembly.layout import load_world_spec
    description = load_world_spec(frozen_cfg.env.train.init_params.scene_config)
    assert description.index()["world/robot"][0].factory == "go2"


def test_actual_joint_and_action_order_are_checked(project, tmp_path):
    from copy import deepcopy
    from shared.experiments import verify_resolved_interface
    _, cfg, module = project
    interface = {"observations": {"states": {"shape": [24], "terms": ["joint_pos", "joint_vel"]}},
                 "actions": [{"name": "joints", "dimension": 12, "asset": "robot", "joint_names": ["FL", "FR"]}],
                 "articulations": {"robot": {"joint_names": ["FL", "FR"], "body_names": ["base", "leg"]}},
                 "action_dim": 12, "step_dt": 0.02}
    module.interface_metadata = interface
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    changed = deepcopy(interface)
    changed["actions"][0]["joint_names"].reverse()
    changed["articulations"]["robot"]["joint_names"].reverse()
    module.interface_metadata = changed
    with pytest.raises(ValueError, match="joint order changed"):
        compare_project(snapshot, cfg, module)
    with pytest.raises(ValueError, match="joint order changed"):
        verify_resolved_interface(module, changed)
    assert verify_resolved_interface(module, interface)
    # Another backend may order bodies natively; joints still have to match.
    converted = deepcopy(interface)
    converted["backend"] = "mjlab"
    converted["articulations"]["robot"]["body_names"] = ["mocap_base", "leg", "base"]
    assert verify_resolved_interface(module, converted)
    converted["articulations"]["robot"]["joint_names"].reverse()
    with pytest.raises(ValueError, match="joint order changed"):
        verify_resolved_interface(module, converted)
    renamed = deepcopy(interface)
    renamed["articulations"]["robot"]["body_names"] = ["base", "arm"]
    with pytest.raises(ValueError, match="joint order changed"):
        verify_resolved_interface(module, renamed)


def test_initial_weights_and_checkpoint_snapshot_pairing_are_validated(project, tmp_path):
    from shared.experiments import checkpoint_compatibility, effective_checkpoint
    root, cfg, module = project
    first = create_snapshot(cfg, module, tmp_path / "first")
    cfg.actor.model.initial_checkpoint = str(tmp_path / "legacy.pt")
    assert effective_checkpoint(cfg, "train") == str(tmp_path / "legacy.pt")
    assert effective_checkpoint(cfg, "view") == str(tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="allow-legacy-checkpoint"):
        checkpoint_compatibility(cfg, module, effective_checkpoint(cfg, "train"))
    save_scene(root, lambda scene: setattr(scene.actions.joints, "scale", 0.75))
    second = create_snapshot(cfg, module, tmp_path / "second")
    checkpoint = second.parent / "probe/checkpoints/global_step_20/actor/weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    frozen_cfg, frozen = load_snapshot(first, register=False)
    with pytest.raises(ValueError, match="semantics"):
        checkpoint_compatibility(frozen_cfg, frozen, checkpoint, allow_scene_change=True)


def test_only_action_component_order_is_semantic_not_parameter_field_order(project, tmp_path):
    root, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    scene = OmegaConf.to_container(OmegaConf.load(root / "scene.yaml"), resolve=True)
    scene["actions"]["joints"] = dict(reversed(list(scene["actions"]["joints"].items())))
    OmegaConf.save(OmegaConf.create(scene), root / "scene.yaml")
    # YAML dictionary field order is not a different controller.
    assert compare_project(snapshot, cfg, module, allow_scene_change=True)["policy_interface_matches"]


def test_cli_training_runs_frozen_source_and_preserves_regular_projects(project, tmp_path, monkeypatch):
    import run
    import tools.inspect_scene
    root, cfg, module = project
    monkeypatch.setattr(run, "compose_project", lambda *args, **kwargs: (cfg, module))
    monkeypatch.setattr(tools.inspect_scene, "resolve_model_dimensions", lambda *args: None)
    observed = []
    monkeypatch.setattr(run, "execute", lambda recipe, operation: observed.append((recipe, operation)))
    monkeypatch.setattr(sys, "argv", ["run.py", "train", root.name])
    run.main()
    recipe, operation = observed.pop()
    assert operation == "train"
    snapshot = find_snapshot(recipe.runner.logger.log_path)
    assert snapshot is not None
    assert Path(recipe.env.train.init_params.scene_config).is_relative_to(snapshot)

    from types import SimpleNamespace
    regular = SimpleNamespace(register=lambda: None)
    monkeypatch.setattr(run, "compose_project", lambda *args, **kwargs: (cfg, regular))
    original_path = cfg.runner.logger.log_path
    run.main()
    assert observed.pop()[0].runner.logger.log_path == original_path


def test_cli_evaluation_checks_same_size_semantics_before_dispatch(project, tmp_path, monkeypatch):
    import run
    import tools.inspect_scene
    root, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    checkpoint = snapshot.parent / "probe/checkpoints/global_step_20/actor/weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    cfg.runner.ckpt_path = str(checkpoint)
    save_scene(root, lambda scene: scene.observations.__setitem__("states", ["joint_vel", "joint_pos"]))
    monkeypatch.setattr(run, "compose_project", lambda *args, **kwargs: (cfg, module))
    monkeypatch.setattr(tools.inspect_scene, "resolve_model_dimensions", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["run.py", "evaluate", root.name])
    with pytest.raises(ValueError, match="semantics"):
        run.main()


@pytest.mark.parametrize('change', ['role', 'observations', 'period', 'mode', 'joints', 'decoder'])
def test_control_semantics_cannot_be_bypassed_by_allow_scene_change(project, tmp_path, change):
    root, cfg, module = project
    scene = OmegaConf.load(root / 'scene.yaml')
    scene.control = {'train': 'executor', 'components': {
        'planner': {'model': '.policy:build_planner', 'period': .2, 'dimension': 3,
                    'observations': ['robot.position']},
        'executor': {'model': '.policy:build_executor', 'period': .02, 'dimension': 2,
                     'observations': ['robot.joint_position', 'robot.joint_velocity'],
                     'decode': '.policy:decode_action'}},
        'output': {'legs': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['a', 'b']}}}
    OmegaConf.save(scene, root / 'scene.yaml')
    snapshot = create_snapshot(cfg, module, tmp_path / 'control_run')
    if change == 'role': scene.control.train = 'planner'
    if change == 'observations': scene.control.components.executor.observations = ['robot.joint_velocity', 'robot.joint_position']
    if change == 'period': scene.control.components.executor.period = .04
    if change == 'mode': scene.control.output.legs.type = 'joint_velocity'
    if change == 'joints': scene.control.output.legs.joint_names = ['b', 'a']
    if change == 'decoder': scene.control.components.executor.decode = '.policy:other_decode'
    OmegaConf.save(scene, root / 'scene.yaml')
    with pytest.raises(ValueError, match='semantics'):
        compare_project(snapshot, cfg, module, allow_scene_change=True)


def test_measured_named_control_order_is_checked(project, tmp_path):
    from shared.experiments import verify_resolved_interface
    root, cfg, module = project
    module.interface_metadata = {'control': {'output': {'arm': {
        'type': 'joint_position', 'resolved_joint_names': ['a', 'b']}}}}
    snapshot = create_snapshot(cfg, module, tmp_path / 'control_run')
    actual = {'control': {'output': {'arm': {
        'type': 'joint_position', 'resolved_joint_names': ['b', 'a']}}}}
    with pytest.raises(ValueError, match='joint order changed'):
        verify_resolved_interface(module, actual, snapshot=snapshot)

def test_backend_is_recorded_not_part_of_policy_interface(project, tmp_path):
    _, cfg, module = project
    before = project_signature(cfg, module)
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    cfg.env.train.init_params.backend = 'mjlab'
    after = project_signature(cfg, module)
    assert after['backends'] == {'train': 'mjlab', 'eval': 'mjlab'}
    assert before['interface_sha256'] == after['interface_sha256']
    report = compare_project(snapshot, cfg, module)
    assert report['backend_changed'] and report['backends']['checkpoint']['train'] == 'isaac'
    replay, _ = load_snapshot(snapshot, ["env.train.init_params.backend=mjlab"], register=False)
    assert replay.env.eval.init_params.backend == 'mjlab'


def test_snapshots_with_backend_in_interface_still_compare(project, tmp_path):
    _, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    signature = manifest["signature"]
    signature["interface"]["backends"] = signature.pop("backends")
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    assert compare_project(snapshot, cfg, module)['backend_changed'] is False


def test_editor_cache_files_are_not_snapshot_inputs(project, tmp_path):
    root, cfg, module = project
    (root / ".cache").mkdir()
    (root / ".cache/exporter.log").write_text("first")
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    _, manifest = read_manifest(snapshot)
    assert not any(".cache" in item["path"] for item in manifest["assets"])
    assert not any(".cache" in name for name in manifest["frozen_files"])
    (root / ".cache/exporter.log").write_text("rewritten by a running exporter")
    load_snapshot(snapshot, register=False)


def test_frozen_replay_reports_the_backend_it_runs_on(project, tmp_path):
    _, cfg, module = project
    snapshot = create_snapshot(cfg, module, tmp_path / "run")
    checkpoint = snapshot.parent / "probe/checkpoints/global_step_20/actor/weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    replay, frozen = load_snapshot(snapshot, ["env.train.init_params.backend=mjlab"], register=False)
    from shared.experiments import checkpoint_compatibility
    report = checkpoint_compatibility(replay, frozen, checkpoint)
    assert report["backends"]["current"] == {"train": "mjlab", "eval": "mjlab"}
    assert report["backend_changed"] and report["policy_interface_matches"]
