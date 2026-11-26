"""Verify a browser download and invoke the native RLinf single-task evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "45ccfcc4e28634f1576ebf78cab0fbe2fd82432d"
MODEL_BYTES = 7473091464
MODEL_SHA256 = "4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7"


def completed_checkpoint(directory):
    candidates = sorted(directory.glob("model*.safetensors"))
    return next((path for path in candidates if path.stat().st_size == MODEL_BYTES), None)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pi05_libero/browser_probe")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    status = {
        "state": "checking_download", "model": "RLinf/RLinf-Pi05-LIBERO-SFT",
        "revision": MODEL_REVISION, "task_id": 2, "suite": "libero_spatial",
        "task": "pick up the black bowl from table center and place it on the plate",
        "trained": False, "error": None,
    }

    def save_status():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(status), flush=True)

    save_status()
    try:
        source = completed_checkpoint(args.downloads)
        if source is None:
            status["state"] = "awaiting_download"
            save_status()
            return
        status["download"] = str(source)
        if args.inspect:
            status["state"] = "download_present_not_verified"
            save_status()
            return
        status["state"] = "verifying_checksum"
        save_status()
        if file_sha256(source) != MODEL_SHA256:
            raise ValueError("Downloaded checkpoint does not match the pinned official SHA-256")
        from safetensors import safe_open

        with safe_open(source, framework="pt", device="cpu") as checkpoint:
            status["tensor_count"] = len(checkpoint.keys())
        status["sha256"] = MODEL_SHA256
        model_dir = ROOT / "outputs/pi05_libero/model"
        temporary = model_dir / "verified.safetensors"
        if temporary.exists():
            temporary.unlink()
        os.link(source, temporary)
        temporary.replace(model_dir / "model.safetensors")
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        sys.path.insert(0, str(ROOT / "third_party/RLinf"))
        import torch
        from toolkits.standalone_eval_scripts.openpi.libero_eval import main as evaluate

        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        status["free_gpu_mib_before"] = torch.cuda.mem_get_info()[0] / 1024**2
        status["state"] = "evaluating"
        save_status()
        started = time.perf_counter()
        evaluate(SimpleNamespace(
            exp_name="task2", log_dir=str(args.output), seed=42,
            task_suite_name="libero_spatial", task_id=2,
            config_name="pi05_libero", pretrained_path=str(model_dir),
            num_trials_per_task=1, action_chunk=5, num_steps=5,
            num_steps_wait=10, num_save_videos=1, video_temp_subsample=1,
        ))
        log = (args.output / "task2.log").read_text()
        for name in ("total_episodes", "total_successes"):
            match = re.search(rf"results/{name}: (\d+)", log)
            if match is None:
                raise ValueError(f"Native evaluation omitted {name}")
            status[name] = int(match.group(1))
        status.update(
            state="completed", elapsed_seconds=time.perf_counter() - started,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        )
        save_status()
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        save_status()
        raise


if __name__ == "__main__":
    main()