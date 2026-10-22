"""Train centered corrections from scratch or continue with ten additional episodes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from projects.franka_stack_vla.supervised_bowl_residual import SupervisedBowlResidual, fit_residual, load_episodes, plot_report


def assemble_dataset(root, old_source):
    sources = [root / "pilot_step10", root / "collection"]
    records, attempts = [], []
    for source in sources:
        report = json.loads((source / "teacher_report.json").read_text())
        if report["residual_start_step"] != 10 or report["teacher_from_start"] or report["learner_checkpoint"]:
            raise ValueError("Expected step-10 corrections of frozen Smol, without a learner")
        audits = {row["seed"]: row for row in json.loads((source / "correction_audit.json").read_text())}
        for result in report["corrections"]:
            attempts.append(result)
            if not result["accepted"]:
                continue
            seed, layout = result["seed"], result["layout"]
            audit = audits[seed]
            if result["takeover_step"] != 10 or not audit["native_settled_success"] or audit["matched_prefix_steps"] != 10:
                raise ValueError("Accepted correction did not pass the takeover/success audit")
            if layout["sampling_mode"] != "inward_center" or layout["reference_closer_m"] != 0.1:
                raise ValueError("Unexpected correction scene")
            for name, position in layout["object_positions"].items():
                reference = layout["reference_object_positions"][name]
                np.testing.assert_allclose(np.array(position)[[0, 2]], np.array(reference)[[0, 2]], atol=1e-7)
                base_y = layout["robot_base_position"][1]
                if abs(position[1] - base_y) >= abs(reference[1] - base_y):
                    raise ValueError("Object is not inward of the reference lateral boundary")
            path = source / f"placement_candidate_{seed}.npz"
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["step"], np.arange(10, audit["first_native_settled_step"]))
                np.testing.assert_allclose(data["teacher_action"].clip(-1, 1), data["applied_action"], atol=1e-7)
            records.append({"seed": seed, "source": str(path.resolve()), "kind": "centered", "layout": layout})
    if len(records) != 28 or len({record["layout"]["layout_seed"] for record in records}) != 28:
        raise ValueError(f"Expected 28 successful distinct new layouts, found {len(records)}")
    candidates = sorted(record["seed"] for record in records if record["seed"] != 60001)
    validation_seeds = set(np.random.default_rng(42).choice(candidates, size=6, replace=False).tolist())
    old_report = json.loads((old_source / "teacher_report.json").read_text())
    old_results = {row["seed"]: row for row in old_report["corrections"]}
    for seed in (4, 5):
        if not old_results[seed]["accepted"]:
            raise ValueError("Selected legacy correction was not accepted")
        path = old_source / f"placement_candidate_{seed}.npz"
        records.append({"seed": seed, "source": str(path.resolve()), "kind": "legacy", "layout": None})
    data_root = root / "dataset"
    data_root.mkdir()
    for record in records:
        path = Path(record["source"])
        destination = data_root / path.name
        shutil.copy2(path, destination)
        with destination.open("rb") as stream:
            record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        record["split"] = "validation" if record["seed"] in validation_seeds else "train"
    episodes = load_episodes(data_root, 30)
    manifest = {
        "residual_start_step": 10, "old_episodes": 2, "new_episodes": 28,
        "new_step10_attempts": len(attempts), "failed_step40_pilot_attempts": 1,
        "training_episodes": 24, "validation_episodes": 6,
        "split_rule": "seed42 sample of six new layouts, excluding timing-selection pilot; legacy episodes train only",
        "adaptation_policy_seeds": sorted(set(old_report["seeds"] + [51001, 51002] + [row["seed"] for row in attempts])),
        "adaptation_layout_seeds": sorted({6201, 6202} | {row["layout"]["layout_seed"] for row in attempts}),
        "episodes": sorted(records, key=lambda record: record["seed"]),
        "samples": sum(len(episode["targets"]) for episode in episodes),
        "scope": "2 fixed 0.05m legacy episodes + 28 inward-centered layouts at 0.10m X translation",
        "success_criterion": "native On, released, supported, stationary for 0.5s; XY precision reported separately",
    }
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Verified dataset: 2 legacy + 28 centered episodes, {manifest['samples']} samples", flush=True)
    return data_root, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/smolvla_libero/centered_corrections30_20261011"))
    parser.add_argument("--old-source", type=Path, default=Path("outputs/smolvla_libero/teacher30_near005_20261011"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--resume-root", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    root = args.root.resolve()
    if any((root / name).exists() for name in ("dataset", "model", "evaluation")):
        raise ValueError("Refusing to overwrite an existing dataset, training run, or evaluation")
    model_root = root / "model"
    if args.resume_root:
        previous_root = args.resume_root.resolve()
        collection = root / "collection"
        collection_report = json.loads((collection / "teacher_report.json").read_text())
        if collection_report["accepted_episodes"] != 10 or collection_report["teacher_from_start"]:
            raise ValueError("Continuation requires ten successful correction episodes, not standalone demonstrations")
        load_episodes(collection, 10)
        previous = SupervisedBowlResidual.from_checkpoint(previous_root / "model")
        model, metadata = fit_residual(previous_root / "dataset", model_root, epochs=args.epochs,
                                       resume_from=previous_root / "model", additional_sources=[collection])
        for name in ("mean", "std", "scale"):
            np.testing.assert_array_equal(getattr(model, name).cpu().numpy(), getattr(previous, name).cpu().numpy())
        previous_metadata = json.loads((previous_root / "model" / "training_report.json").read_text())
        if metadata["validation_episodes"] != previous_metadata["validation_episodes"] or len(metadata["training_episodes"]) != 34:
            raise ValueError("Continuation did not preserve the original validation split")
        (root / "manifest.json").write_text(json.dumps(metadata["dataset_manifest"], indent=2))
        comparison_arguments = ["--compare-residual", str(previous_root / "model")]
        comparison_label = "previous_student"
        policy_seed, layout_seed = 90001, 101001
    else:
        data_root, manifest = assemble_dataset(root, args.old_source)
        model, metadata = fit_residual(data_root, model_root, epochs=args.epochs, dataset_manifest=manifest)
        comparison_arguments = ["--compare-baseline"]
        comparison_label = "baseline"
        policy_seed, layout_seed = 70001, 81001
    if metadata["residual_start_step"] != 10:
        raise ValueError("Unexpected training initialization or residual activation")
    del model
    command = [sys.executable, "tools/smolvla_libero_probe.py", "--suite", "libero_spatial", "--task-id", "8",
               "--bowl-on-plate", "--clear-table", "--closer-m", "0.10", "--center-layout", "--vary-layout",
               "--layout-seed", str(layout_seed), "--seed", str(policy_seed), "--episodes", "10",
               "--evaluate-residual", str(model_root), "--output", str(root / "evaluation"), *comparison_arguments]
    with (root / "evaluation.log").open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    evaluation = json.loads((root / "evaluation" / "evaluation_report.json").read_text())
    if [row["layout"] for row in evaluation[comparison_label]] != [row["layout"] for row in evaluation["student"]]:
        raise ValueError("Comparison policies used different initial layouts")
    for seed in evaluation["evaluation_seeds"]:
        baseline = json.loads((root / "evaluation" / f"{comparison_label}_{seed}_trace.json").read_text())
        student = json.loads((root / "evaluation" / f"student_{seed}_trace.json").read_text())
        np.testing.assert_array_equal([row["applied_action"] for row in baseline[:10]],
                                      [row["applied_action"] for row in student[:10]])
        if comparison_label == "baseline":
            for row in baseline:
                np.testing.assert_array_equal(row["residual"], np.zeros(7))
                np.testing.assert_array_equal(row["applied_action"], np.clip(row["base_action"], -1, 1))
    for label, directory in evaluation["checkpoints"].items():
        with (Path(directory) / "residual_best.pt").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != evaluation["checkpoint_sha256"][label]:
                raise ValueError("Checkpoint changed during evaluation")
    report = {**metadata, **evaluation, "trained": True, "evaluation_steps_before_residual": 10,
              "evaluation_audit": f"matched layouts, identical first 10 commands, comparison={comparison_label}, unchanged checkpoints",
              "reward_used_for_training": False, "frozen_prior": "lerobot/smolvla_libero"}
    (model_root / "supervised_report.json").write_text(json.dumps(report, indent=2))
    plot_report(model_root)
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in (comparison_label + "_successes", "student_successes", "best_epoch",
                                                  "training_samples", "validation_samples")}), flush=True)


if __name__ == "__main__":
    main()