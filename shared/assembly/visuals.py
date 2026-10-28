"""Disposable native geometry cache; it never replaces scene/robot assets."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import os

ROOT = Path(__file__).resolve().parents[2]
SHARED_CACHE = ROOT / '.cache' / 'assembly_visuals'


class SceneChanged(ValueError):
    """The scene file changed while its geometry was exported; export the new revision instead."""


def cache_path(scene):
    from .editing import edited_source
    scene = edited_source(Path(scene).resolve())  # The editor's working copy shares its scene's meshes.
    return scene.parent / '.cache' / (scene.stem + '_visuals.json')


ASSET_SUFFIXES = {'.usd', '.usda', '.usdc', '.usdz', '.urdf', '.xml', '.mjcf', '.obj', '.stl', '.glb', '.gltf', '.dae'}
# Model and Task code never feeds an asset factory; editing it must not re-export meshes.
NON_GEOMETRY = {'policy.py', 'task.py', 'observation.py'}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _referenced_files(value):
    """Asset files named in parameters, resolved like the factories (workspace-relative)."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _referenced_files(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _referenced_files(item)
    elif isinstance(value, str) and '://' not in value and Path(value).suffix.lower() in ASSET_SUFFIXES:
        path = Path(os.path.expandvars(value)).expanduser()
        path = path if path.is_absolute() else ROOT / path
        if path.is_file():
            yield path.resolve()


def geometry_backend(scene):
    """Backend that exports a scene's geometry: the training backend of the project's train.yaml.

    Other recipes and scene variants in the directory share the project's assets; a
    scene outside a project, or a recipe without a literal backend, uses the default.
    """
    from .assets import selected_backend
    recipe = Path(scene).resolve().parent / 'train.yaml'
    backend = None
    if recipe.is_file():
        try:
            from omegaconf import OmegaConf
            backend = OmegaConf.select(OmegaConf.load(recipe), 'env.train.init_params.backend')
        except Exception:  # an unresolvable recipe is reported by check, not by the editor
            backend = None
    return selected_backend({'backend': backend} if isinstance(backend, str) else {})


def exporter_available(backend):
    """Whether the backend provides the ``visuals`` hook; otherwise editors show proxies."""
    from importlib.util import find_spec
    return find_spec(f'shared.backends.{backend}.visuals') is not None


def exporter_python(backend):
    """Interpreter for a backend's exporter: ``RLINF_EXPORT_PYTHON``, its declared environment, else ``.venv``."""
    from importlib import import_module
    override = os.environ.get('RLINF_EXPORT_PYTHON')
    if override:
        return Path(override)
    package = import_module(f'shared.backends.{backend}')
    return ROOT / (getattr(package, 'INTERPRETER', None) or '.venv/bin/python')


def geometry_keys(scene, document=None):
    """Per-entity cache keys; equal assets share geometry across projects and the gallery.

    Pose edits reuse root-local geometry. A key changes with the exporting backend's
    factory, its parameters and initial state, the files those parameters name, the
    backend's asset code (its ``visuals.SOURCES``), and, for project-defined factories,
    the project's code and assets.
    """
    from .assets import backend_module, factories_for
    from .editing import SceneDocument, _plain
    document = document or SceneDocument(scene)
    backend = geometry_backend(scene)
    common = {'tools/export_scene_visuals.py': _digest(ROOT / 'tools/export_scene_visuals.py')}
    if exporter_available(backend):
        for pattern in backend_module(backend, 'visuals').SOURCES:
            for p in sorted(ROOT.glob(pattern)):
                common[str(p.relative_to(ROOT))] = _digest(p)
    directory = Path(scene).resolve().parent
    native, project = factories_for(backend), None
    keys = {}
    for path, node in document.entities.items():
        if not node.get('factory'):
            continue
        dependencies = dict(common)
        # A project assets.py may add factories or override a shared name.
        if node['factory'] not in native or (directory / 'assets.py').is_file():
            if project is None:
                project = {str(p): _digest(p) for p in sorted(directory.rglob('*'))
                           if p.is_file() and (p.suffix == '.py' and p.name not in NON_GEOMETRY or p.suffix in ASSET_SUFFIXES)
                           and not any(part.startswith('.') or part == '__pycache__'
                                       for part in p.relative_to(directory).parts)}
            dependencies.update(project)
        for file in _referenced_files(_plain(node.get('parameters', {}))):
            dependencies[str(file)] = _digest(file)
        keys[path] = hashlib.sha256(json.dumps({'factory': node.get('factory'),
                 'parameters': _plain({k: v for k, v in node.get('parameters', {}).items()
                     if not (k == 'fix_root_link' and node.get('factory') in ('piper', 'piper_x', 'm20', 'franka_panda'))}),
                 'initial_state': _plain(node.get('initial_state', {})), 'dependencies': dependencies,
                 'format': 1, 'seed': 42}, sort_keys=True).encode()).hexdigest()
    return keys


def load_visuals(scene, document=None):
    path = cache_path(scene)
    try:
        saved = json.loads(path.read_text()) if path.is_file() else {'entities': {}}
        keys = geometry_keys(scene, document)
        by_key = {item['key']: item for item in saved['entities'].values()}
        result = {}
        for name, key in keys.items():
            item = by_key.get(key)
            shared = SHARED_CACHE / (key + '.json')
            if item is None and shared.is_file():
                item = json.loads(shared.read_text())
            if item and (not item.get('path') or Path(item['path']).is_file()):
                result[name] = item
        return result
    except (ValueError, KeyError, OSError):
        return {}


def _ensure_lightweight(path):
    from .mesh_preview import merge_partials, pending
    if not pending(json.loads(path.read_text())):
        return
    blender = shutil.which(os.environ.get('BLENDER', 'blender'))
    if not blender or os.environ.get('RLINF_PREVIEW_SIMPLIFIER') == 'numpy':
        # The web editor and headless machines need no Blender installation.
        from .mesh_preview import simplify_manifest
        simplify_manifest(path)
        return
    # Parts are independent; a few background Blenders share them by triangle count.
    jobs = max(1, min(6, (os.cpu_count() or 2) // 2))
    partials = [path.with_suffix(f'.light-{job}.npz') for job in range(jobs)]
    logs = [path.with_suffix(f'.simplify-{job}.log') for job in range(jobs)]
    processes = []
    try:
        for job in range(jobs):
            with logs[job].open('w') as log:
                processes.append(subprocess.Popen(
                    [blender, '--background', '--factory-startup', '--threads', '2', '--python-exit-code', '1',
                     '--python', str(ROOT / 'tools/simplify_scene_visuals.py'), '--', '--manifest', str(path),
                     '--jobs', str(jobs), '--job', str(job), '--output', str(partials[job])],
                    stdout=log, stderr=subprocess.STDOUT))
        for job, process in enumerate(processes):
            if process.wait(timeout=240):
                raise RuntimeError(f'Lightweight mesh generation failed; see {logs[job]}\n{logs[job].read_text()[-1500:]}')
        merge_partials(path, partials, 'blender_decimate')
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for partial in partials:
            partial.unlink(missing_ok=True)


def ensure_visuals(scene, *, force=False, exporter=None):
    """``exporter(scene, force)`` runs the export in an already running SDK; default is a new process."""
    keys = geometry_keys(scene)
    path = cache_path(scene)
    if not keys:
        return {'cached': True, 'entities': 0, 'manifest': str(path), 'quality': 'lightweight'}
    cached = not force and keys.keys() <= load_visuals(scene).keys()
    if cached and not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'entities': load_visuals(scene)}))
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        if exporter is not None:
            exporter(Path(scene).resolve(), force)
            if not keys.keys() <= load_visuals(scene).keys():
                raise RuntimeError('Native geometry export did not produce every entity')
        else:
            logfile = path.with_suffix('.log')
            with logfile.open('w') as log:
                python = exporter_python(geometry_backend(scene))
                process = subprocess.run([str(python), str(ROOT / 'tools/export_scene_visuals.py'),
                                          '--scene', str(Path(scene).resolve()), *(['--force'] if force else [])], cwd=ROOT,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=240)
            if process.returncode or not keys.keys() <= load_visuals(scene).keys():
                raise RuntimeError(f'Native geometry export failed; see {logfile}\n{logfile.read_text()[-2500:]}')
    _ensure_lightweight(path)
    SHARED_CACHE.mkdir(parents=True, exist_ok=True)
    for record in json.loads(path.read_text())['entities'].values():
        record = dict(record)
        if record.get('path'):
            original = Path(record['path'])
            destination = SHARED_CACHE / original.name
            if not destination.exists():
                shutil.copy2(original, destination)
            record['path'] = str(destination)
        target = SHARED_CACHE / (record['key'] + '.json')
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(record))
        temporary.replace(target)
    return {'cached': cached, 'entities': len(keys), 'manifest': str(path), 'quality': 'lightweight'}


class ExportWorker:
    """Client for a long-lived ``export_scene_visuals.py --serve`` process.

    Started on the first uncached export; SDK start-up dominates a small export, so
    imports shortly after reuse it. The exporter exits by itself after ``idle``
    seconds, on errors, or when this process ends (stdin EOF); a request then
    simply starts a fresh one.
    """

    def __init__(self, python, log, idle=120):
        self.python, self.log, self.idle = Path(python), Path(log), idle
        self.process = self.replies = None

    def _start(self):
        read, write = os.pipe()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open('w') as log:
            self.process = subprocess.Popen(
                [str(self.python), str(ROOT / 'tools/export_scene_visuals.py'), '--serve', str(write),
                 '--idle', str(self.idle)], cwd=ROOT, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(write,), text=True)
        os.close(write)
        self.replies = os.fdopen(read)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.stdin.close()  # the exporter exits on EOF
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.replies.close()
        self.process = self.replies = None

    def request(self, scene, *, force=False, timeout=600):
        """Lightweight visuals of ``scene`` (``ensure_visuals``)."""
        import select
        message = {'scene': str(Path(scene).resolve()), 'force': force}
        for _ in range(2):
            if self.process is None or self.process.poll() is not None:
                self.close()
                self._start()
            try:
                self.process.stdin.write(json.dumps(message) + '\n')
                self.process.stdin.flush()
            except BrokenPipeError:  # exited while idle; start a new one
                self.close()
                continue
            if not select.select([self.replies], [], [], timeout)[0]:
                self.process.kill()
                self.close()
                raise RuntimeError(f'Native geometry export timed out after {timeout}s; see {self.log}')
            line = self.replies.readline()
            reply = json.loads(line) if line else {'restart': 'exited'}
            if 'restart' in reply:
                self.close()
                continue
            if reply.get('changed'):
                raise SceneChanged(reply['error'])  # Kit is fine; the caller re-exports the new file
            if 'error' in reply:
                self.close()
                raise RuntimeError(f'Native geometry export failed; see {self.log}\n{reply["error"]}')
            return reply['result']
        raise RuntimeError(f'Native geometry exporter could not start; see {self.log}')
