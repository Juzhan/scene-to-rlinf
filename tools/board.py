"""Serve the RLinf TensorBoard records of a training run while it trains (``run.py train --board``)."""
import socket
import subprocess
import sys
import time
from pathlib import Path


def board_directory(cfg):
    """Directory TensorBoard reads: all runs of the experiment, so earlier runs stay comparable."""
    backends = cfg.runner.logger.get("logger_backends", ["tensorboard"])
    if isinstance(backends, str):
        backends = [backends]
    if "tensorboard" not in (backends or []):
        raise ValueError("--board needs runner.logger.logger_backends to include tensorboard")
    path = Path(cfg.runner.logger.log_path).expanduser().resolve()
    # Frozen assembly runs write to <experiment>/runs/<run>; show the sibling runs as well.
    return path.parent if path.parent.name == "runs" else path


def free_port(host, port, attempts=20):
    for candidate in range(port, port + attempts):
        with socket.socket() as probe:
            if probe.connect_ex((host, candidate)) != 0:
                return candidate
    raise RuntimeError(f"No free TensorBoard port in {port}-{port + attempts - 1}")


def start_board(cfg, port=6006, host="127.0.0.1"):
    logdir = board_directory(cfg)
    logdir.mkdir(parents=True, exist_ok=True)
    port = free_port(host, port)
    log = Path(cfg.runner.logger.log_path) / "tensorboard_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--host", host, "--port", str(port)],
        stdout=log.open("w"), stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and process.poll() is None:
        with socket.socket() as probe:
            if probe.connect_ex((host, port)) == 0:
                break
        time.sleep(.5)
    if process.poll() is not None:
        print(f"TensorBoard failed to start; see {log}. Training continues.", flush=True)
        return None
    run = Path(cfg.runner.logger.log_path).resolve()
    shown = run.relative_to(logdir).as_posix() if run != logdir else "."
    print(f"TensorBoard: http://{host}:{port}  (run: {shown}; stops when training exits)", flush=True)
    return process


def stop_board(process, cfg):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    print(f"TensorBoard stopped. Reopen: {sys.executable} -m tensorboard.main --logdir {board_directory(cfg)}", flush=True)
