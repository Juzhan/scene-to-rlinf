"""Freeze local assembly projects beside RLinf's existing training artifacts.

Snapshots contain project code and configuration, not another runner. Large
assets stay in place and are verified by hash before a saved project is loaded.
"""

import ast
from datetime import datetime, timezone
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import uuid

from omegaconf import OmegaConf, open_dict


_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".rst"}
_ASSET_SUFFIXES = {".usd", ".usda", ".usdc", ".obj", ".stl", ".dae", ".ply", ".fbx",
                   ".glb", ".gltf", ".urdf", ".xml", ".mjcf", ".png", ".jpg", ".jpeg",
                   ".exr", ".hdr", ".pt", ".pth", ".onnx", ".blend", ".mtl"}
_SKIP = {"__pycache__", ".git", ".venv", ".pytest_cache", "node_modules", "outputs"}


def _derived_cache(path):
    """Editor mesh previews and tool logs under .cache are rewritten at will and never simulated."""
    return ".cache" in Path(path).parts and Path(path).suffix != ".json" and Path(path).suffix.lower() not in _ASSET_SUFFIXES


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    def ordered_actions(item):
        if isinstance(item, dict):
            return {key: ([[name, ordered_actions(config)] for name, config in child.items()]
                          if key == "actions" and isinstance(child, dict) else ordered_actions(child))
                    for key, child in item.items()}
        return [ordered_actions(child) for child in item] if isinstance(item, list) else item
    return hashlib.sha256(json.dumps(ordered_actions(value), ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _resolved_interface(value, *, bodies=True):
    # Policies read joints, roots, frames and sensors, never the articulation body array;
    # its order is still compared on one backend, while converted assets order bodies natively.
    if not value:
        return None
    keys = ("joint_names", "body_names") if bodies else ("joint_names",)
    return {"observations": value.get("observations"),
            "actions": [{key: term.get(key) for key in ("name", "dimension", "asset", "joint_names")}
                        for term in value.get("actions", [])],
            "articulations": {name: {key: asset.get(key) for key in keys}
                              for name, asset in value.get("articulations", {}).items()},
            "control": value.get("control"),
            "action_dim": value.get("action_dim"), "step_dt": value.get("step_dt")}


def _same_backend(before, after):
    return (before or {}).get("backend", "isaac") == (after or {}).get("backend", "isaac")


def _differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [item for key in dict.fromkeys([*before, *after])
                for item in _differences(before.get(key), after.get(key), f"{path}.{key}".lstrip("."))]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [item for index, (left, right) in enumerate(zip(before, after))
                for item in _differences(left, right, f"{path}[{index}]")]
    return [] if before == after else [{"path": path, "before": before, "after": after}]


def is_assembly_project(module):
    from shared.assembly.project import Project
    return isinstance(getattr(module, "project", None), Project)


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _map_strings(value, function):
    if isinstance(value, dict):
        return {key: _map_strings(item, function) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings(item, function) for item in value]
    return function(value) if isinstance(value, str) else value


def _project_root(module):
    return Path(module.project.scene_config).resolve().parent


def _scene_paths(cfg, module):
    return {name: Path(env.init_params.get("scene_config", module.project.scene_config)).expanduser().resolve()
            for name, env in (("train", cfg.env.train), ("eval", cfg.env.eval))}


def _configuration_files(scene_paths):
    """Follow reusable assembly files, including source-relative external ones."""
    documents = {}

    def visit(path):
        path = path.resolve()
        if path in documents:
            return
        data = _plain(OmegaConf.load(path))
        documents[path] = data

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "assembly" and isinstance(item, str):
                        visit(path.parent / item)
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(data)

    for path in scene_paths.values():
        visit(path)
    return documents


def project_signature(cfg, module):
    """Keep term order and control parameters; tensor dimensions alone are unsafe."""
    if getattr(module, "experiment_snapshot", None):
        _, saved = read_manifest(module.experiment_snapshot)
        if module.__name__ == saved["package"]:
            from copy import deepcopy
            signature = deepcopy(saved["signature"])
            signature["resolved_interface"] = getattr(module, "interface_metadata", signature.get("resolved_interface"))
            # A frozen replay may still choose where it runs.
            signature["backends"] = {name: cfg.env[name].init_params.get("backend", "isaac")
                                     for name in _backends(signature) or ("train", "eval")}
            return signature
    scenes = _scene_paths(cfg, module)
    documents = _configuration_files(scenes)
    root = _project_root(module)
    model = _plain(cfg.actor.model)
    model = {key: value for key, value in model.items()
             if key not in {"initial_checkpoint", "model_path", "precision"}}
    # The simulator is where the policy runs, not part of its interface: it is recorded and
    # reported, while the live joint/action order is still checked on the backend in use.
    backends = {name: cfg.env[name].init_params.get("backend", "isaac") for name in scenes}
    interface = {
        "model": model,
        "scenes": {name: {"observations": documents[path].get("observations"),
                          "actions": documents[path].get("actions"),
                          "control": documents[path].get("control")}
                   for name, path in scenes.items()},
        "code": {name: _hash(root / name) for name in ("policy.py", "terms.py") if (root / name).is_file()},
    }
    from shared.assembly.layout import load_world_spec
    from shared.assembly.project import project_registry
    interface["robots"] = {
        name: [{"path": node_path, "factory": node.factory, "parameters": node.parameters}
               for node_path, (node, _) in load_world_spec(path, factories=project_registry(
                   module.project.package, backend=backends[name])[0]).index().items()
               if node.kind == "robot"]
        for name, path in scenes.items()
    }
    # File labels remain stable when the same project moves into its snapshot.
    scene_documents = {str(path.relative_to(root)) if path.is_relative_to(root) else str(path): value
                       for path, value in documents.items()}
    sources = {str(path.relative_to(root)): _hash(path) for path in sorted(root.rglob("*.py"))
               if not any(part in _SKIP for part in path.relative_to(root).parts)}
    return {"interface": interface, "interface_sha256": _digest(interface), "backends": backends,
            "scene": scene_documents, "source_sha256": sources,
            "resolved_interface": getattr(module, "interface_metadata", None)}


def _versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "rlinf", "isaacsim", "isaaclab", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "numpy", "omegaconf", "hydra-core"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _runtime_sources():
    root = Path(__file__).resolve().parent
    files = [root / name for name in ("policies.py", "isaac_state.py", "terrains.py", "rewards.py", "observations.py")]
    files.extend(root / "assembly" / name for name in ("__init__.py", "project.py", "spec.py", "layout.py", "abstract.py", "signals.py", "control.py"))
    for directory in ("backends", "integrations"):
        files.extend((root / directory).rglob("*.py"))
    rlinf = Path(os.environ.get("RLINF_ROOT", root.parent / "third_party/RLinf-upstream")).expanduser().resolve()
    files.append(rlinf / "rlinf/workers/rollout/hf/huggingface_worker.py")
    return {str(path): _hash(path) for path in sorted(set(files)) if path.is_file()}


def create_snapshot(cfg, module, run_directory=None):
    """Save a new run and update ``cfg.runner.logger.log_path``; never overwrite.

    Call before registering the project, then use ``load_snapshot(path)`` to run
    the frozen source. ``run_directory`` must not already exist. Without it a
    unique directory is created below the configured log path's ``runs`` folder.
    """
    if not is_assembly_project(module):
        raise TypeError("Only assembly Project entries support project snapshots")
    project_root = _project_root(module)
    previous_snapshot = getattr(module, "experiment_snapshot", None)
    previous_manifest = read_manifest(previous_snapshot)[1] if previous_snapshot else None
    project_name = previous_manifest["project"] if previous_manifest else project_root.name
    original_package = module.project.package
    identifier = uuid.uuid4().hex[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = (Path(run_directory).expanduser().resolve() if run_directory else
                   Path(cfg.runner.logger.log_path).expanduser().resolve() / "runs" / f"{stamp}_{identifier}")
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / "snapshot"
    package = f"_experiment_{identifier}"
    frozen_root = snapshot / "source" / package
    frozen_root.mkdir(parents=True)
    signatures = project_signature(cfg, module)
    scenes = _scene_paths(cfg, module)
    documents = _configuration_files(scenes)
    source_files, assets, remote_assets, paths = {}, {}, set(), {}

    def target(path):
        path = path.resolve()
        if path not in paths:
            paths[path] = (frozen_root / path.relative_to(project_root) if path.is_relative_to(project_root)
                           else snapshot / "external" / hashlib.sha256(str(path).encode()).hexdigest()[:12] / path.name)
        return paths[path]

    def asset(path, reference=None):
        path = path.resolve()
        if path not in assets:
            assets[path] = {"path": str(path), "sha256": _hash(path), "size": path.stat().st_size,
                            "references": []}
        if reference and reference not in assets[path]["references"]:
            assets[path]["references"].append(reference)

    def resolve_reference(value, owner):
        if Path(value).suffix.lower() not in _ASSET_SUFFIXES | {".yaml", ".yml"}:
            return value
        if "://" in value:
            if Path(value).suffix.lower() in _ASSET_SUFFIXES:
                remote_assets.add(value)
            return value
        expanded = Path(os.path.expandvars(value)).expanduser()
        candidates = [expanded] if expanded.is_absolute() else [owner.parent / expanded, Path.cwd() / expanded]
        found = next((item.resolve() for item in candidates if item.is_file()), None)
        if found is None:
            return value
        if found in documents:
            return os.path.relpath(target(found), target(owner).parent)
        if found.suffix.lower() in _ASSET_SUFFIXES:
            asset(found, str(owner))
            return str(found)
        return value

    for path in sorted(project_root.rglob("*")):
        relative = path.relative_to(project_root)
        if any(part in _SKIP or part.startswith(".venv") for part in relative.parts):
            continue
        if not path.is_file() or path.suffix in {".pyc", ".pyo"} or _derived_cache(relative):
            continue
        output = frozen_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in _SOURCE_SUFFIXES or path.suffix == ".json" and path.stat().st_size < 1024 * 1024:
            source_files[str(relative)] = _hash(path)
            paths[path.resolve()] = output
            text = path.read_text()
            if path.suffix == ".py":
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        resolve_reference(node.value, path)
                text = re.sub(r"\b" + re.escape(original_package) + r"\b", package, text)
            output.write_text(text)
        else:
            asset(path, str(relative))
            output.symlink_to(path.resolve())
    for path, data in documents.items():
        output = target(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        source_files[str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path)] = _hash(path)
        rewritten = _map_strings(data, lambda value: resolve_reference(value, path))
        OmegaConf.save(OmegaConf.create(rewritten), output)
    config = _plain(cfg)
    config = _map_strings(config, lambda value: resolve_reference(
        value.replace(original_package + ".", package + "."), project_root / "train.yaml"))
    for name, path in scenes.items():
        config["env"][name]["init_params"]["scene_config"] = str(target(path))
    config["runner"]["logger"]["log_path"] = str(destination)
    OmegaConf.save(OmegaConf.create(config), snapshot / "train.yaml")
    frozen_files = {str(path.relative_to(snapshot)): _hash(path) for path in sorted(snapshot.rglob("*"))
                    if path.is_file() and not path.is_symlink()}
    manifest = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "project": project_name, "original_project": str(project_root), "original_package": original_package,
                "package": package, "snapshot_path": str(snapshot), "source_files": source_files,
                "frozen_files": frozen_files, "signature": signatures, "assets": list(assets.values()),
                "remote_assets": sorted(remote_assets), "dependencies": _versions(),
                "shared_sources": _runtime_sources(),
                "checkpoint_initialization": {
                    "resume_dir": config["runner"].get("resume_dir"),
                    "initial_checkpoint": config["actor"]["model"].get("initial_checkpoint"),
                    "parent_snapshot": str(previous_snapshot) if previous_snapshot else None,
                },
                "checkpoint_compatibility": getattr(module, "experiment_report", None),
                "limitations": ["SDK-provided and dynamically computed asset references are not copied or resolved.",
                                "Shared backend code and installed SDKs must remain compatible; their installed versions are recorded."]}
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with open_dict(cfg):
        cfg.runner.logger.log_path = str(destination)
    module.experiment_snapshot = snapshot
    return snapshot


def find_snapshot(path):
    """Find a run's snapshot from its checkpoint, run directory, or manifest."""
    if not path:
        return None
    path = Path(path).expanduser().resolve()
    start = path.parent if path.is_file() else path
    for directory in (start, *start.parents):
        for candidate in (directory, directory / "snapshot"):
            if (candidate / "manifest.json").is_file() and (candidate / "train.yaml").is_file():
                return candidate
    return None


def read_manifest(path):
    snapshot = find_snapshot(path)
    if snapshot is None:
        raise ValueError(f"No project snapshot found at {path}")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported project snapshot format")
    return snapshot, manifest


def verify_snapshot(path, *, allow_shared_change=False):
    snapshot, manifest = read_manifest(path)
    for relative, expected in manifest["frozen_files"].items():
        file = snapshot / relative
        if not file.is_file() or _hash(file) != expected:
            raise ValueError(f"Frozen project file changed or is missing: {file}")
    for asset in manifest["assets"]:
        file = Path(asset["path"])
        workspace = Path(__file__).resolve().parents[1]
        if file.is_relative_to(workspace) and _derived_cache(file.relative_to(workspace)):
            # Older snapshots recorded these; they are not inputs to the run.
            continue
        if not file.is_file() or _hash(file) != asset["sha256"]:
            raise ValueError(f"Referenced asset changed or is missing: {file}")
    changed = [name for name, expected in manifest.get("shared_sources", {}).items()
               if not Path(name).is_file() or _hash(name) != expected]
    if changed and not allow_shared_change:
        raise ValueError("Shared framework source changed: " + ", ".join(changed) +
                         ". Use --allow-shared-change only for an intentional compatibility evaluation.")
    manifest["shared_changes"] = changed
    return manifest


def _policy_interface(signature):
    # Snapshots before 2026-10-22 kept the backend inside the interface.
    return {key: value for key, value in signature["interface"].items() if key != "backends"}


def _backends(signature):
    return signature.get("backends", signature["interface"].get("backends"))


def compare_project(path, cfg, module, *, allow_scene_change=False, allow_shared_change=False):
    """Reject changed policy semantics; scene changes require an explicit opt-in."""
    manifest = verify_snapshot(path, allow_shared_change=allow_shared_change)
    current = project_signature(cfg, module)
    old = manifest["signature"]
    if _digest(_policy_interface(current)) != _digest(_policy_interface(old)):
        changed = ", ".join(item["path"] for item in _differences(_policy_interface(old), _policy_interface(current))[:8])
        raise ValueError("Observation/action semantics or policy code changed, even if dimensions match: " + changed + ". "
                         "Use --snapshot to replay the frozen project; --allow-scene-change does not change policy semantics.")
    bodies = _same_backend(old.get("resolved_interface"), current.get("resolved_interface"))
    resolved_before = _resolved_interface(old.get("resolved_interface"), bodies=bodies)
    resolved_after = _resolved_interface(current.get("resolved_interface"), bodies=bodies)
    resolved_checked = resolved_before is not None and resolved_after is not None
    if resolved_checked and resolved_before != resolved_after:
        changed = ", ".join(item["path"] for item in _differences(resolved_before, resolved_after)[:8])
        raise ValueError("Resolved observation/action or robot joint order changed: " + changed)
    differences = []
    for key, label in (("scene", "scene configuration"), ("source_sha256", "project source")):
        if current[key] != old[key]:
            differences.append(label)
    if differences and not allow_scene_change:
        raise ValueError("Project differs from the checkpoint: " + ", ".join(differences) +
                         ". Use --snapshot for the saved project or --allow-scene-change to evaluate this scene.")
    frozen_module = module.__name__.startswith("_experiment_")
    measured_now = resolved_checked and (not frozen_module or getattr(module, "interface_metadata_verified", False))
    backends = {"checkpoint": _backends(old), "current": _backends(current)}
    return {"snapshot": str(find_snapshot(path)), "changed": differences, "backends": backends,
            "backend_changed": backends["checkpoint"] != backends["current"],
            "differences": _differences(old["scene"], current["scene"], "scene") +
                           _differences(old["source_sha256"], current["source_sha256"], "source"),
            "policy_interface_matches": True, "uses_current_scene": bool(allow_scene_change),
            "shared_changes": manifest["shared_changes"], "resolved_interface_verified": measured_now,
            "limitations": [] if measured_now else ["Live manager/joint order has not been checked against the saved run; declarations and any available recorded metadata were compared."]}


def effective_checkpoint(cfg, operation, explicit=None):
    """The weights that win after model initialization and checkpoint loading."""
    if explicit:
        return explicit
    if operation == "train" and cfg.runner.get("resume_dir"):
        return cfg.runner.resume_dir
    if operation in ("evaluate", "assess") and cfg.runner.get("ckpt_path"):
        return cfg.runner.ckpt_path
    return cfg.actor.model.get("initial_checkpoint")


def checkpoint_compatibility(cfg, module, checkpoint, *, allow_scene_change=False,
                             allow_shared_change=False, allow_legacy_checkpoint=False):
    """Validate the selected project against the checkpoint's own saved interface."""
    if not checkpoint or not is_assembly_project(module):
        return None
    source = find_snapshot(checkpoint)
    selected = getattr(module, "experiment_snapshot", None)
    if selected and module.__name__ != read_manifest(selected)[1]["package"]:
        selected = None
    if source is None:
        if not allow_legacy_checkpoint:
            raise ValueError("This checkpoint has no project snapshot, so observation/action semantics cannot be verified. "
                             "Use --allow-legacy-checkpoint only for weights whose interface you have independently checked.")
        return {"snapshot": str(selected) if selected else None, "checkpoint_snapshot": None,
                "checkpoint": str(checkpoint), "changed": ["unverifiable legacy checkpoint"],
                "policy_interface_matches": None, "resolved_interface_verified": False,
                "legacy_checkpoint_explicitly_accepted": True, "uses_current_scene": selected is None}
    report = compare_project(source, cfg, module, allow_scene_change=allow_scene_change,
                             allow_shared_change=allow_shared_change)
    report.update(checkpoint=str(checkpoint), checkpoint_snapshot=str(source),
                  selected_snapshot=str(selected) if selected else None)
    return report


def verify_resolved_interface(module, actual, *, snapshot=None):
    """Check real manager outputs once a consumer has instantiated the saved scene."""
    snapshot = snapshot or getattr(module, "experiment_snapshot", None)
    if not snapshot:
        return False
    _, manifest = read_manifest(snapshot)
    saved = manifest["signature"].get("resolved_interface")
    bodies = _same_backend(saved, actual)
    expected = _resolved_interface(saved, bodies=bodies)
    if expected is None:
        return False
    found = _resolved_interface(actual, bodies=bodies)
    if expected != found:
        changed = ", ".join(item["path"] for item in _differences(expected, found)[:8])
        raise ValueError("Resolved observation/action or robot joint order changed: " + changed)
    module.interface_metadata = actual
    module.interface_metadata_verified = True
    return True


def load_snapshot(path, overrides=(), *, register=True, allow_shared_change=False):
    """Load frozen project source and its resolved recipe, without starting Isaac."""
    snapshot, manifest = read_manifest(path)
    checked = verify_snapshot(snapshot, allow_shared_change=allow_shared_change)
    cfg = OmegaConf.load(snapshot / "train.yaml")
    previous = manifest["snapshot_path"]
    cfg = OmegaConf.create(_map_strings(_plain(cfg), lambda value: value.replace(previous, str(snapshot))))
    if overrides:
        changed = OmegaConf.merge(cfg, OmegaConf.from_dotlist([item.lstrip("+") for item in overrides]))
        model = lambda value: {key: item for key, item in _plain(value.actor.model).items()
                               if key not in {"initial_checkpoint", "model_path", "precision"}}
        if model(cfg) != model(changed):
            raise ValueError("A frozen replay cannot change policy/model semantics")
        for name in ("train", "eval"):
            # device and simulator backend choose where the frozen policy runs.
            before = {key: value for key, value in _plain(cfg.env[name].init_params).items() if key not in {"device", "backend"}}
            after = {key: value for key, value in _plain(changed.env[name].init_params).items() if key not in {"device", "backend"}}
            if before != after:
                raise ValueError("A frozen replay cannot change its scene or policy interface")
        from shared.assembly.project import follow_train_backend
        cfg = follow_train_backend(changed, overrides, previous=cfg.env.train.init_params.get("backend", "isaac"))
    source = str(snapshot / "source")
    if source not in sys.path:
        sys.path.insert(0, source)
    existing = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([source, *(item for item in existing if item and item != source)])
    os.environ["RLINF_EXT_MODULE"] = manifest["package"]
    imported = sys.modules.get(manifest["package"])
    if imported is not None and Path(imported.__file__).resolve().parent != snapshot / "source" / manifest["package"]:
        # A moved run has the same immutable package identity but a new location.
        for name in list(sys.modules):
            if name == manifest["package"] or name.startswith(manifest["package"] + "."):
                sys.modules.pop(name, None)
    module = import_module(manifest["package"])
    module.experiment_snapshot = snapshot
    module.interface_metadata = manifest["signature"].get("resolved_interface")
    module.interface_metadata_verified = False
    module.experiment_report = {"snapshot": str(snapshot), "changed": [], "uses_current_scene": False,
                                "policy_interface_matches": True, "resolved_interface_verified": False,
                                "shared_changes": checked["shared_changes"]}
    if register:
        module.register()
    return cfg, module


def prepare_replay(cfg, module, *, checkpoint=None, allow_scene_change=False, allow_shared_change=False,
                   allow_legacy_checkpoint=False, overrides=()):
    """Use the saved project after validating the user's current selection."""
    if not is_assembly_project(module) or not checkpoint:
        return cfg, module, None
    report = checkpoint_compatibility(cfg, module, checkpoint, allow_scene_change=allow_scene_change,
                                      allow_shared_change=allow_shared_change,
                                      allow_legacy_checkpoint=allow_legacy_checkpoint)
    snapshot = find_snapshot(checkpoint)
    if snapshot is None:
        return cfg, module, report
    if not allow_scene_change:
        # Keep the requested runtime settings (including a resume path authored
        # in train.yaml), while the source and scene come from the saved run.
        saved_cfg, saved_module = load_snapshot(snapshot, register=False, allow_shared_change=allow_shared_change)
        config = _map_strings(_plain(cfg), lambda value: value.replace(
            module.project.package + ".", saved_module.project.package + "."))
        for name in ("train", "eval"):
            config["env"][name]["init_params"]["scene_config"] = saved_cfg.env[name].init_params.scene_config
        cfg, module = OmegaConf.create(config), saved_module
    return cfg, module, report


def evaluation_directory(cfg, report):
    """Keep independent evaluation logs and the recorded compatibility decision."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = Path(cfg.runner.logger.log_path) / "evaluations" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "project_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with open_dict(cfg):
        cfg.runner.logger.log_path = str(destination)
    return destination
