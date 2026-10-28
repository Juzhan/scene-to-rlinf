"""A failing simulator must end training, not leave RLinf waiting on its subprocess."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/RLinf-upstream"))
pytest.importorskip("rlinf")


def _transport(mode):
    from types import SimpleNamespace
    from functools import partial
    from rlinf.envs.sim.isaaclab.venv import SubProcIsaacLabEnv
    from shared.integrations.rlinf.isaac import _CheckedTransport, _make_reporting_environment

    def factory(parameters, num_envs, seed, max_steps):
        import os
        import torch

        def step(action):
            if mode == "raise":
                raise FloatingPointError("simulated physics failure")
            os._exit(3)

        environment = SimpleNamespace(num_envs=2, device="cpu", action_dim=1, close=lambda: None, step=step,
                                      reset=lambda **_: ({"x": torch.zeros(2, 1)}, {}))
        return environment, SimpleNamespace(close=lambda: None)

    return _CheckedTransport(SubProcIsaacLabEnv(partial(
        _make_reporting_environment, SimpleNamespace(action_dim=1), 2, 0, 10, factory=factory)))


@pytest.mark.parametrize("mode,message", [("raise", "simulated physics failure"), ("exit", "exited with code 3")])
def test_subprocess_failure_reaches_the_env_worker(mode, message):
    import torch
    transport = _transport(mode)
    try:
        transport.reset(seed=0)
        with pytest.raises(RuntimeError, match=message):
            transport.step(torch.zeros(2, 1))
    finally:
        transport.isaac_lab_process.kill()
