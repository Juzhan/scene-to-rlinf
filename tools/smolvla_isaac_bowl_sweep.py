"""Run the frozen Smol prior, the pre-PPO learned residual, and PI0.5 in the Isaac bowl scene."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"policy": ".venv-smolvla", "residual": ".venv-smolvla", "pi05": ".venv-pi05"}


def reference_for(arguments, seed):
    if arguments.reference_pattern:
        path = Path(arguments.reference_pattern.format(seed=seed))
        if not path.exists():
            raise SystemExit(f"Missing scene reference for seed {seed}: {path}")
        return path.resolve()
    return Path(arguments.reference).resolve()


def run_one(policy, seed, arguments, out_root):
    directory = out_root / f"{policy}_seed{seed}"
    if directory.exists():
        if (directory / "report.json").exists():
            raise SystemExit(f"Refusing to overwrite {directory}")
        shutil.rmtree(directory)
    log = out_root / "logs" / f"{policy}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / POLICIES[policy] / "bin/python"), "-m", "projects.franka_stack_vla.bowl_plate_transfer",
               "--reference", str(reference_for(arguments, seed)), "--output", str(directory), "--mode", policy,
               "--steps", str(arguments.steps), "--seed", str(seed),
               "--light-intensity", str(arguments.light_intensity)]
    if policy == "residual":
        command += ["--residual-checkpoint", str(arguments.residual_checkpoint)]
    environment = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'third_party/RLinf'}",
                   "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8", "MUJOCO_GL": "egl"}
    started = time.perf_counter()
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    report = directory / "report.json"
    if result.returncode or not report.exists():
        return {"policy": policy, "seed": seed, "failed": True, "returncode": result.returncode,
                "elapsed_seconds": elapsed, "log": str(log)}
    saved = json.loads(report.read_text())
    saved.update(policy=policy, elapsed_seconds=elapsed, log=str(log))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-pattern", help="Per-seed scene reference with a {seed} placeholder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--light-intensity", type=float, default=1000.0)
    parser.add_argument("--policies", nargs="+", default=["policy", "residual", "pi05"], choices=sorted(POLICIES))
    arguments = parser.parse_args()
    if bool(arguments.reference) == bool(arguments.reference_pattern):
        parser.error("Provide exactly one of --reference or --reference-pattern")
    if arguments.reference:
        arguments.reference = arguments.reference.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for policy in arguments.policies:
        for seed in arguments.seeds:
            summary = run_one(policy, seed, arguments, output)
            summaries.append(summary)
            merged = {}
            for entry in (json.loads((output / "sweep_report.json").read_text())
                          if (output / "sweep_report.json").exists() else []):
                if not entry.get("policy"):
                    entry["policy"] = entry.get("mode")
                merged[(entry.get("policy"), entry.get("seed"))] = entry
            merged.update({(entry.get("policy"), entry.get("seed")): entry for entry in summaries})
            summaries = [merged[key] for key in sorted(merged, key=lambda item: (str(item[0]), item[1]))]
            (output / "sweep_report.json").write_text(json.dumps(summaries, indent=2))
            print(json.dumps({key: summary.get(key) for key in
                              ("policy", "seed", "failed", "success", "settled_success", "strict_success",
                               "max_lift_m", "min_xy_error_m", "elapsed_seconds")}), flush=True)
    rows = {}
    for policy in arguments.policies:
        entries = [summary for summary in summaries if summary.get("policy") == policy]
        if not entries:
            continue
        rows[policy] = {"episodes": len(entries),
                        "settled": sum(bool(entry.get("settled_success")) for entry in entries),
                        "strict": sum(bool(entry.get("strict_success")) for entry in entries),
                        "legacy": sum(bool(entry.get("success")) for entry in entries),
                        "max_lift_m": max(entry.get("max_lift_m", 0.0) for entry in entries),
                        "min_xy_cm": min(entry.get("min_xy_error_m", float("nan")) for entry in entries) * 100}
    (output / "sweep_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
