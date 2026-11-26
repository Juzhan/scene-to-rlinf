"""Collection filter for dependency groups (docs/PACKAGING_PLAN.md).

Every test module ends with ``pytestmark = pytest.mark.group_<name>`` naming the smallest
environment that can run it. Selecting one group normally still imports every module, which
fails in a small environment; set ``WORKSPACE_TEST_GROUP`` to collect only that group:

    WORKSPACE_TEST_GROUP=core .venv-core/bin/python -m pytest tests
    .venv-il3/bin/python -m pytest tests -m group_train

Without the variable all files are collected, as before.
"""
import os
from pathlib import Path
import re

GROUP = os.environ.get("WORKSPACE_TEST_GROUP", "").strip()
MARKER = re.compile(r"^pytestmark = pytest\.mark\.(group_\w+)", re.M)


def module_group(path):
    match = MARKER.search(path.read_text())
    return match.group(1) if match else ""


if GROUP:
    collect_ignore = [path.name for path in Path(__file__).parent.glob("test_*.py")
                      if module_group(path) != "group_" + GROUP]
