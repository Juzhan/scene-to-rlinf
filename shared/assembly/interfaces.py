"""Optional measured asset metadata for editors, invalidated by source changes."""

import hashlib
import json
import os
from pathlib import Path

from .assets import DEFAULT_BACKEND, selected_backend


def source_revision(scene):
    from .editing import SceneDocument
    document = SceneDocument(scene)
    revision = dict(document.revision)
    asset_suffixes = {".usd", ".usda", ".usdc", ".usdz", ".urdf", ".xml", ".mjcf", ".obj", ".stl", ".glb", ".gltf", ".dae"}
    for path in sorted(document.path.parent.rglob("*")):
        if not path.is_file() or path.suffix not in asset_suffixes | {".py"}:
            continue
        if not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(document.path.parent).parts):
            revision[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    def references(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)
        elif isinstance(value, str):
            yield value
    # Include explicitly referenced external assets, not just files in the project.
    for filename in document.revision:
        owner = Path(filename)
        for value in references(document.yaml.load(owner.read_text())):
            if "://" in value or Path(value).suffix.lower() not in asset_suffixes:
                continue
            path = Path(os.path.expandvars(value)).expanduser()
            path = path if path.is_absolute() else owner.parent / path
            if path.is_file():
                revision[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return revision


def _cache_path(scene, backend):
    # The default backend keeps the original cache name, so existing caches stay valid.
    suffix = '' if selected_backend({'backend': backend}) == DEFAULT_BACKEND else '_' + backend
    return scene.parent / ".cache" / (scene.stem + suffix + "_interface.json")


def save_interface(scene, report, *, expected_revision=None, backend=DEFAULT_BACKEND):
    scene = Path(scene).resolve()
    revision = source_revision(scene)
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("Scene or assets changed during interface inspection; inspect again")
    path = _cache_path(scene, backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"source_revision": revision, "interface": report}, indent=2) + "\n")
    if source_revision(scene) != revision:
        temporary.unlink(missing_ok=True)
        raise ValueError("Scene or assets changed while saving interface cache; inspect again")
    temporary.replace(path)
    return path


def load_interface(scene, *, backend=DEFAULT_BACKEND):
    scene = Path(scene).resolve()
    path = _cache_path(scene, backend)
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text())
        if saved["source_revision"] == source_revision(scene):
            return saved["interface"]
    except (OSError, ValueError, KeyError):
        pass
    return None
