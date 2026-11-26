"""Create a workspace environment with exactly the dependency groups a job needs.

    tools/setup_env.py --with core,editor --env .venv-editor
    tools/setup_env.py --with isaaclab3,train --env .venv-il3 --isaaclab-source ~/repos/IsaacLab-3.0
    tools/setup_env.py --with deploy --env .venv-deploy
    tools/setup_env.py --check --python .venv-il3/bin/python
    tools/setup_env.py --lock --from .venv-il3/bin/python --fallback .venv/bin/python

Groups are the packaging units of docs/PACKAGING_PLAN.md. What each one must be able
to import lives in shared/assembly/runtimes.py; its PyPI packages are the extras in
pyproject.toml, installed through the per-group lock file (``requirements/<group>.lock.txt``)
once one exists. Isaac Lab 3, RLinf and the frozen mjlab stack are source installs from
the lists below, because PyPI cannot pin a source checkout.

Installing needs network access; ``--check`` and ``--lock`` work offline against
interpreters that are already installed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements"

sys.path.insert(0, str(ROOT))
from shared.assembly import runtimes  # noqa: E402 - the registry is the single source of group names

# Where each group is installed by default; these are the environments runtimes.py searches.
ENV_DIRS = {
    "core": ".venv-core", "editor": ".venv-editor", "import": ".venv-import",
    "deploy": ".venv-deploy", "isaaclab3": ".venv-il3", "train": ".venv-il3",
    "isaac": ".venv", "mjlab": ".venv-mjlab",
}
# Source installs a group needs beyond its PyPI extras.
SOURCE_INSTALLS = {
    "train": ["third_party/RLinf-upstream"],  # pinned commit; see docs/RLINF_INTEGRATION.md
    "mjlab": ["-r shared/backends/mjlab/requirements.txt"],
    "isaaclab3": ["-r shared/backends/isaaclab3/requirements.txt"],
}
# Isaac Lab 3 packages installed editable from its source tree (release/3.0.0 @961760b).
ISAACLAB3_PACKAGES = ("isaaclab", "isaaclab_assets", "isaaclab_newton", "isaaclab_physx",
                      "isaaclab_tasks", "isaaclab_rl", "isaaclab_contrib")
# Groups that cannot be installed by this script; the message says what to do instead.
FROZEN_NOTE = {
    "isaac": "Isaac Sim is not on PyPI: keep the existing Isaac Sim environment and set RLINF_PYTHON to it.",
    "mjlab": "mjlab is frozen: install shared/backends/mjlab/requirements.txt into its own environment.",
}


def pyproject():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def direct_requirements(groups):
    """The PyPI requirements of ``groups``: the base dependencies plus their extras."""
    project = pyproject()["project"]
    lines = list(project["dependencies"])
    extras = project.get("optional-dependencies", {})
    for group in runtimes.included(groups):
        lines += extras.get(group, [])
    return list(dict.fromkeys(lines))


def lock_path(group):
    return REQUIREMENTS / f"{group}.lock.txt"


def install_plan(groups, python, *, isaaclab_source=None, use_lock=True):
    """The commands that install ``groups`` into ``python``, in dependency order.

    Uses ``uv pip`` so an environment without its own pip (a plain ``uv venv``) works.
    """
    install = ["uv", "pip", "install", "--python", str(python)]
    commands = []
    for group in runtimes.included(groups):
        if group in FROZEN_NOTE:
            print(f"[{group}] {FROZEN_NOTE[group]}", file=sys.stderr)
            continue
        locked = lock_path(group)
        if use_lock and locked.is_file():
            commands.append([*install, "-r", str(locked)])
        else:
            extras = ",".join(runtimes.included((group,)))
            target = f"{ROOT}[{extras}]" if group != "core" else str(ROOT)
            commands.append([*install, "-e", target])
        for entry in SOURCE_INSTALLS.get(group, []):
            if entry.startswith("-r "):
                commands.append([*install, "-r", str(ROOT / entry[3:])])
            else:
                commands.append([*install, "-e", str(ROOT / entry)])
        if group == "isaaclab3":
            source = Path(isaaclab_source or os.environ.get("ISAACLAB3_SOURCE", "")).expanduser()
            if not source.is_dir():
                raise SystemExit("isaaclab3 is a source install: pass --isaaclab-source PATH (the Isaac Lab 3 "
                                 "checkout, release/3.0.0) or set ISAACLAB3_SOURCE")
            for package in ISAACLAB3_PACKAGES:
                folder = source / "source" / package
                if folder.is_dir():
                    commands.append([*install, "-e", str(folder)])
    return commands


def create_environment(directory, python_version, *, create=True):
    python = directory / "bin/python"
    if python.is_file():
        return python
    if not create:
        return python
    print(f"creating {directory}", flush=True)
    subprocess.run(["uv", "venv", "--python", python_version, str(directory)], check=True)
    return python


# Runs inside an installed interpreter: walk Requires-Dist from the direct requirements.
FROZEN_SCRIPT = r"""
import importlib.metadata as metadata
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

installed = {}

def distribution(name):
    if name not in installed:
        try:
            installed[name] = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    return installed[name]

roots = [Requirement(line) for line in sys.stdin.read().splitlines() if line.strip()]
root_names = {canonicalize_name(requirement.name) for requirement in roots}
extras = {}
for requirement in roots:
    if requirement.extras:
        extras.setdefault(canonicalize_name(requirement.name), set()).update(requirement.extras)
seen, queue, missing = {}, [requirement.name for requirement in roots], []
while queue:
    name = canonicalize_name(queue.pop())
    if name in seen:
        continue
    found = distribution(name)
    if found is None or found.version is None:
        if name in root_names:
            missing.append(name)
        continue
    seen[name] = found
    wanted = extras.get(name) or set()
    for entry in found.requires or ():
        try:
            requirement = Requirement(entry)
        except Exception:
            continue
        if requirement.marker is not None:
            # Extras are conditional dependencies: evaluate with each requested extra and none.
            try:
                if not any(requirement.marker.evaluate({"extra": extra}) for extra in wanted | {""}):
                    continue
            except Exception:
                continue
        if requirement.extras:
            extras.setdefault(canonicalize_name(requirement.name), set()).update(requirement.extras)
        queue.append(requirement.name)
for name, found in sorted(seen.items()):
    print(f"{name}=={found.version}")  # canonical names: one line per package across sources
for name in sorted(missing):
    print(f"# not installed: {name}", file=sys.stderr)
"""


def freeze_roots(group, sources):
    """Exact versions of ``group``'s dependency closure, from the first source that has each package."""
    roots = direct_requirements((group,))
    resolved, origins = {}, {}
    for source in sources:
        result = subprocess.run([str(source), "-c", FROZEN_SCRIPT], input="\n".join(roots),
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"{source}: freezing [{group}] failed\n{result.stderr.strip()}")
        if result.stderr.strip():
            print(f"{source}: {result.stderr.strip()}", file=sys.stderr)
        for line in result.stdout.splitlines():
            name, _, version = line.partition("==")
            if name and name not in resolved:
                resolved[name] = version
                origins[name] = source
    return resolved, origins


def check(arguments):
    absent = runtimes.missing(tuple(arguments.groups), arguments.python, ROOT)
    if not absent:
        print(f"all groups present: {', '.join(runtimes.included(tuple(arguments.groups)))}")
        return 0
    for group, modules in absent:
        detail = f" (missing {', '.join(modules)})" if modules else ""
        print(f"missing [{group}]{detail}")
    print(runtimes.message(tuple(arguments.groups)))
    return 1


def write_lock(arguments):
    sources = [Path(source) for source in [arguments.source, *arguments.fallback] if source]
    REQUIREMENTS.mkdir(exist_ok=True)
    for group in runtimes.included(tuple(arguments.groups)):
        resolved, origins = freeze_roots(group, sources)
        if not resolved:
            print(f"[{group}] no package found in the source interpreters", file=sys.stderr)
            return 1
        header = [f"# Exact package set of [{group}]; regenerate with tools/setup_env.py --lock.",
                  "# Frozen from " + ", ".join(str(source) for source in sources)
                  + " so every group built from these\n# files keeps one version of each shared package."]
        # A local version tag (torch 2.11.0+cu130) comes from the wheel index, not from PyPI.
        tags = sorted({version.split("+", 1)[1] for version in resolved.values() if "+" in version})
        if len(tags) > 1:
            raise SystemExit(f"[{group}] several local version tags {tags}: freeze from one wheel index")
        if tags:
            header.append(f"--extra-index-url https://download.pytorch.org/whl/{tags[0]}")
        body = [f"{name}=={version}" for name, version in sorted(resolved.items(), key=lambda item: item[0].lower())]
        path = lock_path(group)
        path.write_text("\n".join(header + body) + "\n")
        print(f"{path}: {len(body)} packages")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Install the dependency groups of this workspace")
    parser.add_argument("--with", dest="group_list", default="",
                        help="comma-separated groups: " + ", ".join(runtimes.GROUPS) + ", or all")
    parser.add_argument("--env", help="workspace environment directory to create or reuse (e.g. .venv-editor)")
    parser.add_argument("--python", help="existing interpreter instead of --env")
    parser.add_argument("--python-version", default="3.11", help="interpreter version for a new environment")
    parser.add_argument("--isaaclab-source", help="Isaac Lab 3 checkout for the isaaclab3/train groups")
    parser.add_argument("--no-lock", action="store_true", help="install the pyproject extras instead of the lock files")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--check", action="store_true", help="report the groups an interpreter is missing")
    parser.add_argument("--lock", action="store_true", help="write requirements/<group>.lock.txt")
    parser.add_argument("--from", dest="source", help="validated interpreter to freeze (or check)")
    parser.add_argument("--fallback", action="append", default=[],
                        help="another validated interpreter for packages the first one lacks")
    arguments = parser.parse_args(argv)
    names = [part.strip() for part in arguments.group_list.split(",") if part.strip()]
    if "all" in names:
        names = [group for group in runtimes.GROUPS if group not in FROZEN_NOTE]
    if not names:
        parser.error("choose the groups to install: --with " + ", ".join(runtimes.GROUPS))
    try:
        arguments.groups = runtimes.included(tuple(names))
    except ValueError as error:
        parser.error(str(error))
    if arguments.check:
        return check(arguments)
    if arguments.lock:
        if not arguments.source:
            parser.error("--lock needs --from <validated interpreter>")
        return write_lock(arguments)
    python = Path(arguments.python) if arguments.python else None
    if python is None:
        directory = Path(arguments.env or ENV_DIRS[arguments.groups[-1]]).expanduser()
        directory = directory if directory.is_absolute() else ROOT / directory
        python = create_environment(directory, arguments.python_version, create=not arguments.dry_run)
    commands = install_plan(tuple(arguments.groups), python,
                            isaaclab_source=arguments.isaaclab_source, use_lock=not arguments.no_lock)
    for command in commands:
        print(" ".join(command), flush=True)
        if not arguments.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)
    if arguments.dry_run:
        return 0
    installed = runtimes.missing(tuple(arguments.groups), python, ROOT)
    if installed:
        for group, modules in installed:
            print(f"still missing [{group}]: {', '.join(modules) or 'see the install output'}")
        return 1
    print(f"[{', '.join(arguments.groups)}] ready in {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
