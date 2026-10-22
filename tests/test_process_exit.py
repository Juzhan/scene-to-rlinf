"""Real subprocess exit status and output; no Isaac application is started."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("location", ["normal", "except", "finally"])
def test_process_exit_preserves_failure_and_flushes_output(location):
    prefix = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path('third_party/RLinf').resolve()))
from shared.isaac_state import ProcessExit
assert not any(name == 'isaaclab' or name == 'isaacsim' for name in sys.modules)
owner = ProcessExit(None)
print('pending stdout', end='')
print('pending stderr', end='', file=sys.stderr)
"""
    body = {
        "normal": "owner.close()",
        "except": """
try:
    raise ValueError('factory validation failed')
except ValueError:
    owner.close()
""",
        "finally": """
try:
    raise ValueError('factory validation failed')
finally:
    owner.close()
""",
    }[location]
    result = subprocess.run([sys.executable, "-c", prefix + body],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if location == "normal" else 1)
    assert result.stdout == "pending stdout"
    assert "pending stderr" in result.stderr
    if location == "normal":
        assert "Traceback" not in result.stderr
    else:
        assert "Traceback (most recent call last)" in result.stderr
        assert "ValueError: factory validation failed" in result.stderr
