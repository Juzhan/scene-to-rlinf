"""Run the original standalone residual trainer on the fixed 40-episode split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import torch
from omegaconf import OmegaConf

from projects.franka_stack_vla.supervised_bowl_residual import SupervisedBowlResidual, fit_residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-from", type=Path, default=Path("outputs/smolvla_libero/centered_plus10_20261011/model"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/smolvla_libero/centered_rlinf_scratch400_20261012"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    source_report = json.loads((args.data_from / "training_report.json").read_text())
    reference_report = json.loads((args.reference / "model/training_report.json").read_text())
    for key in ("training_episodes", "validation_episodes"):
        if source_report[key] != reference_report[key]:
            raise ValueError("Source episode order differs from the RLinf reference")
    manifest = source_report["dataset_manifest"]
    paths = source_report["training_episodes"] + source_report["validation_episodes"]
    if len(paths) != 40 or len(source_report["training_episodes"]) != 34:
        raise ValueError("This experiment requires the original 34/6 split of 40 episodes")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "dataset"
    dataset.mkdir()
    records = {row["seed"]: row for row in manifest["episodes"]}
    ordered_records = []
    for original in paths:
        source = Path(original)
        seed = int(source.stem.rsplit("_", 1)[1])
        record = dict(records[seed])
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError("Source correction hash mismatch")
        destination = dataset / source.name
        shutil.copyfile(source, destination)
        record.update(original_source=original, source=str(destination))
        ordered_records.append(record)
    manifest.update(episodes=ordered_records, training_episodes=34, validation_episodes=6)
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    config = {"actor": {"model": {"model_path": None}}, "backend": "original_standalone_pytorch",
              "data_reference": str(args.data_from.resolve()), "reference_run": str(args.reference.resolve()),
              "epochs": args.epochs, "early_stopping": False, "learning_rate": 3e-4,
              "optimizer": "Adam", "batch_size": 128, "gradient_clip": 1.0, "seed": 42, "device": "cpu"}
    OmegaConf.save(OmegaConf.create(config), output / "config.yaml")
    _, report = fit_residual(dataset, output / "model", epochs=args.epochs, expected_episodes=40,
                             dataset_manifest=manifest_path, early_stop_patience=None)
    for key in ("training_episodes", "validation_episodes"):
        if report[key] != [str(dataset / Path(path).name) for path in source_report[key]]:
            raise ValueError("Standalone trainer changed the source episode order")
    initial = torch.load(output / "model/residual_initial.pt", map_location="cpu", weights_only=True)
    reference_initial = torch.load(args.reference / "model/residual_initial.pt", map_location="cpu", weights_only=True)
    if initial.keys() != reference_initial.keys() or not all(torch.equal(initial[key], value) for key, value in reference_initial.items()):
        raise ValueError("Initial weights or transforms differ from the RLinf from-scratch run")
    model = SupervisedBowlResidual(initial["mean"], initial["std"], initial["scale"])
    model.load_state_dict(initial)
    with torch.no_grad():
        if torch.count_nonzero(model(torch.zeros(3, 39))):
            raise ValueError("Initial residual must output zero")
    with (output / "model/training.csv").open() as stream:
        history = list(csv.DictReader(stream))
    batch_rng = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(torch.arange(report["training_samples"]), batch_size=128,
                                       shuffle=True, generator=batch_rng)
    rlinf_order = torch.cat(list(loader)).numpy()
    rlinf_order_hash = hashlib.sha256(rlinf_order.tobytes()).hexdigest()
    first_order_hash = history[0]["batch_order_sha256"]
    observations = torch.arange(8 * 39, dtype=torch.float32).reshape(8, 39) / 100
    targets = torch.ones(8, 7) * .1
    first = SupervisedBowlResidual(initial["mean"], initial["std"], initial["scale"])
    second = SupervisedBowlResidual(initial["mean"], initial["std"], initial["scale"])
    first.load_state_dict(initial)
    second.load_state_dict(initial)
    for candidate, optimizer in ((first, torch.optim.Adam(first.parameters(), lr=3e-4)),
                                 (second, torch.optim.AdamW(second.parameters(), lr=3e-4, weight_decay=0))):
        loss = torch.nn.functional.mse_loss(candidate(observations), targets / candidate.scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
    optimizer_difference = max(float((left - right).detach().abs().max()) for left, right in zip(first.parameters(), second.parameters()))
    assert report["epochs"] == args.epochs and report["optimizer_steps"] == args.epochs * 30
    audit = {"same_initial_weights_and_transforms_as_rlinf": True, "same_data_split_and_sample_order": True,
             "zero_initial_residual": True, "pretrained_residual_loaded": False,
             "standalone_first_epoch_order_sha256": first_order_hash,
             "rlinf_first_epoch_order_sha256": rlinf_order_hash,
             "minibatch_orders_differ": first_order_hash != rlinf_order_hash,
             "cpu_adam_vs_zero_decay_adamw_one_step_max_difference": optimizer_difference,
             "remaining_differences": ["global-RNG randperm versus independently seeded DataLoader shuffling", "CPU versus GPU/FSDP numerical path"],
             "causal_limit": "Different minibatch order is verified, not isolated as the cause of success-rate differences"}
    (output / "implementation_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({"training": {key: report[key] for key in ("epochs", "best_epoch", "initial_validation_mse", "best_validation_mse", "last_validation_mse")}, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()