"""Dependency groups stay consistent: registry, pyproject extras, locks and test markers."""

from pathlib import Path
import re
import tomllib

import pytest

from shared.assembly import runtimes

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {"isaac", "mjlab"}  # not installable from PyPI; their requirements files pin them


def pyproject():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def lock_versions(path):
    versions = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "-")):
            continue
        name, _, version = line.partition("==")
        versions[name] = version
    return versions


def test_every_group_declares_where_it_comes_from():
    extras = pyproject()["project"].get("optional-dependencies", {})
    for group in runtimes.GROUPS:
        assert group in runtimes.MODULES, f"{group} has no importable requirement"
        if group == "core":
            assert pyproject()["project"]["dependencies"], "core is the base dependency set"
            continue
        assert group in extras or group in FROZEN, f"{group} has neither an extra nor a frozen requirements file"


def test_includes_are_known_groups():
    for group, parents in runtimes.INCLUDES.items():
        assert group in runtimes.GROUPS
        assert set(parents) <= set(runtimes.GROUPS)
    for command, groups in runtimes.COMMAND_GROUPS.items():
        assert set(groups) <= set(runtimes.GROUPS), command


def test_test_modules_declare_one_known_group():
    declared = {marker.split(":", 1)[0] for marker in pyproject()["tool"]["pytest"]["ini_options"]["markers"]}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        found = re.findall(r"^pytestmark = pytest\.mark\.(group_\w+)", path.read_text(), re.M)
        assert len(found) == 1, f"{path.name} must declare exactly one group marker"
        assert found[0] in declared, f"{path.name}: {found[0]} is not a declared marker"


def test_locks_pin_one_version_per_group_and_agree_across_groups():
    files = sorted((ROOT / "requirements").glob("*.lock.txt"))
    assert files, "no per-group lock files"
    versions = {}
    for path in files:
        group = path.name.removesuffix(".lock.txt")
        assert group in runtimes.GROUPS, path.name
        lines = [line for line in path.read_text().splitlines() if line and not line.startswith(("#", "-"))]
        names = [line.partition("==")[0] for line in lines]
        assert len(names) == len(set(names)), f"{path.name} pins a package twice: {names}"
        for name, version in lock_versions(path).items():
            versions.setdefault(name, {}).setdefault(version, []).append(group)
    conflicts = {name: found for name, found in versions.items() if len(found) > 1}
    assert not conflicts, f"one package has different versions across groups: {conflicts}"


pytestmark = pytest.mark.group_core  # runs in the [core] dependency group
