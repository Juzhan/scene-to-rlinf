"""Optional dependency groups and the interpreter that provides each of them.

One registry answers two questions: which group an entry point needs, and which
workspace environment has it. It imports only the standard library, so ``run.py``
can consult it before anything is installed, and it is the single place that
knows interpreter locations. ``RLINF_PYTHON`` overrides every choice.

Groups (docs/PACKAGING_PLAN.md): ``core`` is installed with the project itself;
``editor``, ``import``, ``usd``, ``isaaclab3``, ``train``, ``deploy`` and the frozen
``isaac``/``mjlab`` backends are extras. ``isaaclab3`` and ``train`` also need
source checkouts, which ``tools/setup_env.py`` installs (PyPI cannot express them).
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

GROUPS = ("core", "editor", "import", "usd", "isaaclab3", "train", "deploy", "isaac", "mjlab")
# A group includes its parents: installing it installs theirs too.
INCLUDES = {
    "editor": ("core",), "import": ("core",), "deploy": ("core",),
    "isaaclab3": ("core",), "mjlab": ("core",), "isaac": ("core",), "train": ("isaaclab3",),
}
# Modules that must be importable; ``ruamel.yaml`` is the distribution's import name.
MODULES = {
    "core": ("hydra", "omegaconf", "ruamel.yaml", "numpy"),
    "editor": ("viser", "trimesh", "PIL", "numpy"),
    "import": ("pxr", "trimesh", "PIL", "mujoco", "numpy"),
    "usd": ("pxr", "mujoco_usd_converter"),  # Kit-free MJCF -> USD; owns its ``mujoco`` (core 4.x)
    "isaaclab3": ("torch", "isaaclab", "isaaclab_newton"),
    "mjlab": ("torch", "mjlab"),
    "isaac": ("torch", "isaacsim"),
    "train": ("torch", "rlinf", "ray"),
    "deploy": ("torch",),
}
# What each ``run.py`` entry point needs in the process that dispatches it. Simulator work
# then restarts under the backend's interpreter (shared.assembly.project.backend_interpreter),
# so ``view``/``record``/``smoke``/``evaluate`` only need core here.
COMMAND_GROUPS = {
    "list": ("core",), "new": ("core",), "check": ("core",), "catalog": ("core",),
    "edit": ("editor",), "asset": ("import",), "probe": ("core",),
    "view": ("core",), "record": ("core",), "smoke": ("core",), "evaluate": ("core",),
    "train": ("train",), "assess": ("train",),  # assess runs a project evaluation through the trainer entry
}
# Workspace environments searched in this order; the backend's own declaration comes first.
ENVIRONMENTS = (".venv", ".venv-il3", ".venv-editor", ".venv-mjlab", ".venv-usd")
OVERRIDE = "RLINF_PYTHON"
_REEXEC_MARKER = "RLINF_GROUP_INTERPRETER"


def included(groups):
    """The groups plus everything they include, in dependency order."""
    result = []

    def add(group):
        if group not in GROUPS:
            raise ValueError(f"Unknown dependency group {group!r}; choose from {', '.join(GROUPS)}")
        for parent in INCLUDES.get(group, ()):
            add(parent)
        if group not in result:
            result.append(group)
    for group in groups:
        add(group)
    return result


def command_groups(arguments):
    """The groups a ``run.py`` invocation needs; ``--backend`` inspection and unknown arguments are core."""
    for value in arguments:
        if not value.startswith("-"):
            return COMMAND_GROUPS.get(value, ("core",))
    return ("core",)


def _probe(group, root=ROOT):
    """A one-line import check for one group: names the absent modules on stderr, exits 1."""
    return ("import importlib.util as u, sys; sys.path[:0] = {paths!r}; "
            "absent = [name for name in {modules!r} if not u.find_spec(name)]; "
            "print(' '.join(absent), file=sys.stderr); sys.exit(1 if absent else 0)").format(
                paths=_import_paths(root), modules=sorted(MODULES[group]))


def _import_paths(root):
    """The workspace itself, plus the RLinf checkout run.py puts on the path."""
    rlinf = Path(os.environ.get("RLINF_ROOT", Path(root) / "third_party/RLinf-upstream"))
    return [str(root), str(rlinf)]


def missing(groups, python=None, root=ROOT):
    """``[(group, [missing modules])]`` for what the interpreter lacks; ``python=None`` is this process."""
    if python is None or Path(python) == Path(sys.executable):
        import importlib.util
        absent, saved = [], list(sys.path)
        sys.path[:0] = [path for path in _import_paths(root) if path not in sys.path]
        try:
            for group in included(groups):
                names = []
                for module in MODULES[group]:
                    try:
                        found = importlib.util.find_spec(module) is not None
                    except (ImportError, ValueError):
                        found = False
                    if not found:
                        names.append(module)
                if names:
                    absent.append((group, names))
        finally:
            sys.path[:] = saved
        return absent
    absent = []
    for group in included(groups):
        result = subprocess.run([str(python), "-c", _probe(group, root)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            absent.append((group, result.stderr.split()))
    return absent


def install_hint(groups):
    """The one command that installs these groups."""
    names = ",".join(included(groups))
    return f"tools/setup_env.py --with {names}"


def message(groups):
    names = ", ".join(group for group in included(groups))
    return f"需要 [{names}]：{install_hint(groups)}（或设置 {OVERRIDE} 指向已安装这些组的解释器）"


def workspace_interpreters(root=ROOT):
    """Workspace interpreters in search order; ``None`` when an environment does not exist."""
    candidates = [(name, Path(root) / name / "bin/python") for name in ENVIRONMENTS]
    for name, path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            yield path


def find_interpreter(groups, root=ROOT):
    """First workspace interpreter that provides ``groups``, else None."""
    for python in workspace_interpreters(root):
        if not missing(groups, python, root):
            return python
    return None


def interpreter(groups=("core",), *, backend=None, root=ROOT):
    """Interpreter to run in: ``RLINF_PYTHON``, then the backend's declaration, then a workspace
    environment that has the groups, then the current interpreter (which reports what is missing)."""
    override = os.environ.get(OVERRIDE)
    if override:
        return Path(override)
    if backend is not None:
        declared = _backend_interpreter(backend, root)
        if declared is not None:
            return declared
    found = find_interpreter(groups, root)
    return found or Path(sys.executable)


def declared_python(backend, root=ROOT):
    """Interpreter a backend's exporter/converter runs in: the override, its ``INTERPRETER``, else .venv.

    Used to spawn a *new* process, so it does not depend on what this process can import.
    """
    override = os.environ.get(OVERRIDE)
    if override:
        return Path(override)
    from .assets import backend_module, selected_backend
    try:
        package = backend_module(selected_backend({"backend": backend}), "__init__")
    except ModuleNotFoundError:  # a removed backend (Isaac Lab 2): its converter environment may still be there
        package = None
    return Path(root) / (getattr(package, "INTERPRETER", None) or ".venv/bin/python")


def _backend_interpreter(backend, root):
    """The backend's ``INTERPRETER`` when this process cannot import its ``SDK``."""
    from .assets import backend_module, selected_backend
    try:
        package = backend_module(selected_backend({"backend": backend}), "__init__")
    except ModuleNotFoundError:
        return None
    sdk = getattr(package, "SDK", None)
    if sdk is None:
        return None
    import importlib.util
    try:
        if importlib.util.find_spec(sdk) is not None:
            return None
    except (ImportError, ValueError):
        pass
    declared = getattr(package, "INTERPRETER", None)
    candidate = Path(root) / declared if declared else None
    return candidate if candidate is not None and candidate.is_file() else None


def ensure(groups, *, arguments=None, root=ROOT):
    """Run this command in the interpreter that has ``groups``, or stop with the install hint.

    Restarts at most once: the marker keeps a misconfigured environment from looping.
    """
    absent = missing(groups, root=root)
    if not absent:
        return sys.executable
    if os.environ.get(OVERRIDE):
        raise SystemExit(message(groups))
    target = find_interpreter(groups, root)
    if target is None or os.environ.get(_REEXEC_MARKER):
        raise SystemExit(message(groups))
    os.environ[_REEXEC_MARKER] = "1"
    argv = [str(target), str(Path(root) / "run.py"), *(arguments if arguments is not None else sys.argv[1:])]
    print(f"{', '.join(included(groups))}: using {target}", file=sys.stderr, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(str(target), argv)
