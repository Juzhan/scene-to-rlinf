"""Train or fine-tune the bowl residual through RLinf's official SFT loop."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf

from projects.franka_stack_vla.rlinf_residual_sft import load_correction_splits


def summarize(output, training_run=None):
    from projects.franka_stack_vla.supervised_bowl_residual import SupervisedBowlResidual, plot_report

    source_run = Path(training_run).resolve() if training_run is not None else output
    model_root = source_run / "model"
    report_model_root = output / "model"
    if training_run is not None:
        if source_run == output.resolve():
            raise ValueError("Evaluation-only report must not overwrite the training run")
        report_model_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_root / "training.csv", report_model_root / "training.csv")
        shutil.copyfile(source_run / "config.yaml", output / "config.yaml")
    training = json.loads((model_root / "training_report.json").read_text())
    standalone = training.get("framework", {}).get("training_loop", "").endswith(".fit_residual")
    official_parameters = training.get("official_parameters")
    resume_training = training.get("resume_training_audit")
    cosine_schedule = bool(official_parameters and official_parameters.get("scheduler_override") == "cosine")
    engine = "Standalone" if standalone else "RLinf cosine" if cosine_schedule else "RLinf official" if official_parameters else "RLinf"
    previous_name = "Warmup constant best" if cosine_schedule else "RLinf custom best" if official_parameters else "RLinf validation best" if standalone else "Previous residual"
    if resume_training:
        engine = "RLinf resumed"
        previous_name = "Preserved source"
    policy_names = {"baseline": "No residual", "previous_student": previous_name,
                    "student": f"{engine} best", "last_student": f"{engine} epoch {training['epochs']:g}"}
    if resume_training:
        policy_names.update(student="Resumed best", last_student=f"After +{training['epochs']:g}")
    evaluation = json.loads((output / "evaluation/evaluation_report.json").read_text())
    baseline = json.loads((output / "baseline_evaluation/evaluation_report.json").read_text())
    if evaluation["evaluation_seeds"] != baseline["evaluation_seeds"]:
        raise ValueError("Baseline and residual evaluation seeds differ")
    labels = ("baseline", "previous_student", "student")
    results = {"baseline": baseline["baseline"], **{key: evaluation[key] for key in labels[1:]}}
    last_evaluation = None
    last_path = output / "last_evaluation/evaluation_report.json"
    if last_path.exists():
        last_evaluation = json.loads(last_path.read_text())
        if last_evaluation["evaluation_seeds"] != evaluation["evaluation_seeds"] or last_evaluation["layout_seeds"] != evaluation["layout_seeds"]:
            raise ValueError("Last-epoch evaluation is not paired")
        if last_evaluation["checkpoint_sha256"]["previous_student"] != evaluation["checkpoint_sha256"]["student"]:
            raise ValueError("Last-epoch comparison used a different validation-selected model")
        results["last_student"] = last_evaluation["student"]
        labels += ("last_student",)
    layouts = [row["layout"] for row in results["baseline"]]
    for label in labels:
        if [row["layout"] for row in results[label]] != layouts:
            raise ValueError("Actual initial layouts differ")
    for seed in evaluation["evaluation_seeds"]:
        traces = {}
        for label in labels:
            directory = "last_evaluation" if label == "last_student" else "baseline_evaluation" if label == "baseline" else "evaluation"
            prefix = "student" if label == "last_student" else label
            traces[label] = json.loads((output / directory / f"{prefix}_{seed}_trace.json").read_text())
        if last_evaluation is not None:
            repeated_best = json.loads((output / "last_evaluation" / f"previous_student_{seed}_trace.json").read_text())
            if repeated_best != traces["student"]:
                raise ValueError("Repeated best-model trajectories differ")
        for label in labels[1:]:
            np.testing.assert_array_equal([row["applied_action"] for row in traces["baseline"][:10]],
                                          [row["applied_action"] for row in traces[label][:10]])
        for row in traces["baseline"]:
            np.testing.assert_array_equal(row["residual"], np.zeros(7))
            np.testing.assert_array_equal(row["applied_action"], np.clip(row["base_action"], -1, 1))
    states = {}
    for label, path in evaluation["checkpoints"].items():
        with (Path(path) / "residual_best.pt").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != evaluation["checkpoint_sha256"][label]:
                raise ValueError("Checkpoint changed during evaluation")
        states[label] = SupervisedBowlResidual.from_checkpoint(path).state_dict()
    if last_evaluation is not None:
        last_model_path = Path(last_evaluation["checkpoints"]["student"])
        with (last_model_path / "residual_best.pt").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != last_evaluation["checkpoint_sha256"]["student"]:
                raise ValueError("Last-epoch checkpoint changed")
        last_model = SupervisedBowlResidual.from_checkpoint(last_model_path)
        _, last_validation, last_report = load_correction_splits(last_model_path)
        with torch.no_grad():
            last_loss = float(torch.nn.functional.mse_loss(last_model(last_validation.observations), last_validation.targets / last_model.scale))
        if last_report.get("selection") != "last_epoch" or last_report["epochs"] != training["epochs"] or abs(last_loss - training["last_validation_mse"]) > 1e-7:
            raise ValueError("Last-epoch export does not match training")
    identical = states["student"].keys() == states["previous_student"].keys() and all(
        torch.equal(value, states["student"][key]) for key, value in states["previous_student"].items())
    initialization_audit = None
    from_scratch = training.get("from_scratch", False)
    if from_scratch:
        from projects.franka_stack_vla.rlinf_residual_sft import ResidualSftPolicy

        with (model_root / "residual_initial.pt").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != training["initialization_sha256"]:
                raise ValueError("Initial model hash changed")
        initial = torch.load(model_root / "residual_initial.pt", map_location="cpu", weights_only=True)
        initial_model = ResidualSftPolicy(initial["mean"], initial["std"], initial["scale"])
        initial_model.load_state_dict(initial)
        train_data, val_data, _ = load_correction_splits(model_root)
        reference = ResidualSftPolicy.from_training_data(train_data)
        for key in ("mean", "std", "scale"):
            torch.testing.assert_close(initial[key], reference.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(initial[key], states["student"][key], rtol=0, atol=0)
        with torch.no_grad():
            if torch.count_nonzero(initial_model(train_data.observations)) or torch.count_nonzero(initial_model(val_data.observations)):
                raise ValueError("Fresh residual was not zero at initialization")
            selected_model = SupervisedBowlResidual.from_checkpoint(model_root)
            selected_loss = float(torch.nn.functional.mse_loss(selected_model(val_data.observations), val_data.targets / selected_model.scale))
        if abs(selected_loss - training["best_validation_mse"]) > 1e-7:
            raise ValueError("Saved model does not reproduce selected validation loss")
        changed = any(not torch.equal(value, states["student"][key]) for key, value in initial.items() if key.startswith("network."))
        cfg = OmegaConf.load(source_run / "config.yaml")
        if training["resume_from"] is not None or training["pretrained_residual_loaded"] or cfg.actor.model.model_path is not None:
            raise ValueError("From-scratch run contains a pretrained model source")
        initialization_audit = {"zero_initial_output": True, "train_only_normalization_and_scale": True,
                                "pretrained_residual_loaded": False, "network_changed_from_initial": changed,
                                "selected_validation_mse_reloaded": selected_loss,
                                "initialization_sha256": training["initialization_sha256"]}
    if identical:
        for seed in evaluation["evaluation_seeds"]:
            previous = json.loads((output / "evaluation" / f"previous_student_{seed}_trace.json").read_text())
            selected = json.loads((output / "evaluation" / f"student_{seed}_trace.json").read_text())
            if previous != selected:
                raise ValueError("Identical models did not reproduce identical trajectories")
    with (model_root / "training.csv").open() as stream:
        history = list(csv.DictReader(stream))
    if resume_training:
        preserved = source_run / "preserved_source"
        saved_hashes = json.loads((preserved / "sha256_manifest.json").read_text())
        for relative, expected_hash in saved_hashes.items():
            with (preserved / relative).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                    raise ValueError("Preserved source checkpoint changed")
        initial = torch.load(model_root / "residual_initial.pt", map_location="cpu", weights_only=True)
        if any(not torch.equal(value, states["previous_student"][key]) for key, value in initial.items()):
            raise ValueError("Resumed initial model differs from the preserved comparison policy")
        selected_model = SupervisedBowlResidual.from_checkpoint(model_root)
        _, validation_data, _ = load_correction_splits(model_root)
        with torch.no_grad():
            selected_loss = float(torch.nn.functional.mse_loss(selected_model(validation_data.observations), validation_data.targets / selected_model.scale))
        if abs(selected_loss - training["best_validation_mse"]) > 1e-7:
            raise ValueError("Continued validation-best export does not reproduce its loss")
    schedule_comparison_audit = None
    if cosine_schedule:
        reference_model = Path(evaluation["checkpoints"]["previous_student"])
        reference_cfg = OmegaConf.load(reference_model.parent / "config.yaml")
        current_cfg = OmegaConf.load(source_run / "config.yaml")
        current_actor = OmegaConf.to_container(current_cfg.actor, resolve=True)
        current_actor["optim"].pop("lr_scheduler")
        if current_actor != OmegaConf.to_container(reference_cfg.actor, resolve=True):
            raise ValueError("Cosine comparison changed actor settings beyond the scheduler")
        reference_report = json.loads((reference_model / "training_report.json").read_text())
        for key in ("training_episodes", "validation_episodes", "optimizer_steps", "epochs"):
            if training[key] != reference_report[key]:
                raise ValueError("Cosine comparison changed the ordered dataset or training budget")
        reference_initial = torch.load(reference_model / "residual_initial.pt", map_location="cpu", weights_only=True)
        if initial.keys() != reference_initial.keys() or any(not torch.equal(value, reference_initial[key]) for key, value in initial.items()):
            raise ValueError("Cosine comparison changed initial weights or transforms")
        with (reference_model / "training.csv").open() as stream:
            reference_history = list(csv.DictReader(stream))
        warmup = current_cfg.actor.optim.lr_warmup_steps
        prefix = [row for row in history if float(row["step"]) <= warmup]
        reference_prefix = [row for row in reference_history if float(row["step"]) <= warmup]
        if not prefix or prefix != reference_prefix:
            raise ValueError("Recorded warmup trajectories do not match the constant-schedule run")
        schedule_comparison_audit = {"only_actor_change": "optim.lr_scheduler=cosine",
            "identical_initial_tensors_and_transforms": True, "identical_ordered_data_and_budget": True,
            "identical_warmup_history_rows": len(prefix), "last_matching_warmup_step": float(prefix[-1]["step"]),
            "reference_run": str(reference_model.parent.resolve())}
    provenance = json.loads((output / "evaluation/provenance.json").read_text())
    prior_weights = Path("outputs/smolvla_libero/model/model.safetensors")
    with prior_weights.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != provenance["sha256"]:
            raise ValueError("Frozen Smol checkpoint changed")
    report = {**training, **evaluation, **results, "trained": training_run is None,
              "policy_labels": policy_names, "training_label": engine,
              "reference_training_history": str(Path(evaluation["checkpoints"]["previous_student"]) / "training.csv") if standalone or official_parameters else None,
              "reference_training_label": "Source held-out" if resume_training else "Warmup constant held-out" if cosine_schedule else "RLinf custom held-out" if official_parameters else "RLinf held-out",
              "evaluation_is_retest": bool(official_parameters and not resume_training),
              "training_performed_this_run": training_run is None,
              "training_run": str(source_run.resolve()),
              "baseline_successes": baseline["baseline_successes"],
              "selected_weights_equal_previous": identical,
              "initialization_audit": initialization_audit,
              "schedule_comparison_audit": schedule_comparison_audit,
              "last_epoch_evaluation": last_evaluation,
              "last_student_successes": sum(row["is_success"] for row in results.get("last_student", [])) if last_evaluation else None,
              "last_validation_mse": float(history[-1]["validation_mse"]),
              "reward_used_for_training": False, "frozen_prior": provenance,
              "actions": "clip(Smol + gate(step>=10) * saved_scale * tanh(MLP), -1, 1); XYZ 0.05m, rotation 0.5rad, gripper +close/-open; 20Hz",
              "audit": "40 episode hashes and preserved 34/6 split; each policy uses its own saved transforms and scale; matched actual layouts; identical first 10 commands; zero raw baseline residual; checkpoint hashes unchanged",
              "summary": {key: {"successes": sum(row["is_success"] for row in results[key]),
                                "episodes": len(results[key]),
                                "mean_xy_cm": float(np.mean([row["xy_error_m"] for row in results[key]]) * 100)} for key in labels}}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (report_model_root / "supervised_report.json").write_text(json.dumps(report, indent=2))
    plot_report(report_model_root)
    cards = []
    displayed = results["last_student"] if last_evaluation else results["student"]
    displayed_directory = "last_evaluation" if last_evaluation else "evaluation"
    for raw, selected in zip(results["baseline"], displayed):
        seed = selected["seed"]
        destination = output / f"no_residual_vs_{engine.lower().replace(' ', '_')}_{seed}.mp4"
        longest_steps = max(raw["steps"], selected["steps"])
        expected_duration = (longest_steps + 1) / 20
        needs_render = not destination.exists()
        if not needs_render:
            duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(destination)], text=True))
            needs_render = abs(duration - expected_duration) > .02
        if needs_render:
            raw_status = "SUCCESS" if raw["is_success"] else "FAILED"
            raw_chain = "[0:v]setpts=PTS-STARTPTS"
            if raw["steps"] < longest_steps:
                raw_chain += f",tpad=stop_mode=clone:stop_duration={(longest_steps - raw['steps']) / 20:.2f}"
            raw_chain += f",drawtext=text='NO RESIDUAL - {seed} - {raw_status}':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.65"
            if raw["steps"] < longest_steps:
                raw_chain += f",drawtext=text='Episode ended - final frame held':x=8:y=230:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.65:enable='gte(t,{(raw['steps'] + 1) / 20:.2f})'"
            filters = [raw_chain + "[top]"]
            chain = "[1:v]setpts=PTS-STARTPTS"
            if selected["steps"] < longest_steps:
                chain += f",tpad=stop_mode=clone:stop_duration={(longest_steps - selected['steps']) / 20:.2f}"
            status = "SUCCESS" if selected["is_success"] else "FAILED"
            model_label = f"{engine} EPOCH {training['epochs']:g}" if last_evaluation else f"{engine} FROM SCRATCH" if from_scratch else f"{engine} SELECTED"
            chain += f",drawtext=text='{model_label} - {seed} - {status}':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.65"
            if selected["steps"] < longest_steps:
                chain += f",drawtext=text='Successful end frame held':x=8:y=230:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.65:enable='gte(t,{(selected['steps'] + 1) / 20:.2f})'"
            filters.extend([chain + "[bottom]", "[top][bottom]vstack=inputs=2:shortest=1[video]"])
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(output / "baseline_evaluation" / f"baseline_{seed}.mp4"),
                            "-i", str(output / displayed_directory / f"student_{seed}.mp4"), "-filter_complex", ";".join(filters),
                            "-map", "[video]", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], check=True)
        media = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-show_entries", "format=duration", "-of", "json", str(destination)], text=True))
        if (media["streams"][0]["width"], media["streams"][0]["height"]) != (512, 512) or abs(float(media["format"]["duration"]) - expected_duration) > .02:
            raise ValueError("Unexpected comparison video dimensions or duration")
        status = "Success" if selected["is_success"] else "Failed"
        cards.append(f'<article><h3>Seed {seed} <span class="{status.lower()}">{status}</span></h3>'
                     f'<video controls preload="metadata" playsinline src="{destination.name}"></video>'
                     f'<p>Final XY: {raw["xy_error_m"] * 100:.2f} cm &rarr; {selected["xy_error_m"] * 100:.2f} cm</p>'
                     f'<a href="evaluation/previous_student_{seed}.mp4">{previous_name} recording</a> | '
                     f'<a href="evaluation/student_{seed}.mp4">Validation-selected recording</a></article>')
    score_names = ("No residual", "Previous residual", "RLinf from scratch" if from_scratch else "RLinf selected")
    if last_evaluation:
        score_names = ("No residual", "Previous residual", f"Best epoch {training['best_epoch']:g}", f"Last epoch {training['epochs']:g}")
    if standalone or official_parameters:
        score_names = tuple(policy_names[key] for key in labels)
    scores = "".join(f'<div><strong>{report["summary"][key]["successes"]} / {report["summary"][key]["episodes"]}</strong><span>{name}</span></div>'
                     for key, name in zip(labels, score_names))
    conclusion = ("Validation selected the original weights (epoch 0). Continued training did not improve the model."
                  if identical else "Validation selected updated residual weights; see the paired evaluation below.")
    if from_scratch:
        conclusion = (f"Residual trained from zero-output initialization with {engine}, without loading previous residual weights. "
                      f"Selected epoch {training['best_epoch']:g}; stable placement {report['student_successes']}/{len(results['student'])}.")
    if resume_training:
        conclusion = (f"Resumed the preserved epoch {training['source_best_epoch']:g} model with optimizer, scheduler and RNG state restored. "
            f"Completed {training['epochs']:g} additional epochs, reaching total epoch {training['total_epochs']:g}. "
            f"Validation selected additional epoch {training['best_epoch']:g} (total {training['source_best_epoch'] + training['best_epoch']:g}). "
            + ("The selected weights remain identical to the source." if identical else "The selected weights changed during continuation."))
    comparison_note = ("trained from zero-output initialization, with no previous residual weights loaded" if from_scratch
                       else "identical to the previous model in this run" if identical else "updated by supervised training")
    displayed_title = f"{engine}-selected residual"
    if last_evaluation:
        conclusion += f" Completed all {training['epochs']:g} epochs without early stopping. Last epoch: {report['last_student_successes']}/{len(displayed)}. Last-epoch evaluation is diagnostic; model selection remains validation-only."
        displayed_title = f"{engine} epoch {training['epochs']:g} residual (last epoch, not validation-selected)"
        if resume_training:
            displayed_title = f"residual after {training['epochs']:g} additional epochs (total {training['total_epochs']:g}, not validation-selected)"
            comparison_note = "last-epoch continuation weights"
    if training_run is not None:
        conclusion = "Evaluation-only rerun with new seeds and layouts. No training or checkpoint updates. Source model: " + conclusion
    policy_seeds = evaluation["evaluation_seeds"]
    layout_seeds = evaluation["layout_seeds"]
    evaluation_note = ("These ten layouts are a matched retest of the previous comparison, not a new untouched test set."
                       if official_parameters and not resume_training else "The 10 new paired layouts are a small independent check, not a broad generalization estimate.")
    implementation_note = ""
    schedule_figure = ""
    if official_parameters:
        cfg = OmegaConf.load(source_run / "config.yaml")
        official_cfg = OmegaConf.load(source_run / "official_source.yaml")
        expected_optim = OmegaConf.to_container(official_cfg.actor.optim)
        if cosine_schedule:
            expected_optim["lr_scheduler"] = "cosine"
        if OmegaConf.to_container(cfg.actor.optim) != expected_optim:
            raise ValueError("Official optimizer parameters were overridden")
        with (source_run / "official_source.yaml").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != official_parameters["source_sha256"]:
                raise ValueError("Official source configuration changed")
        for key in ("micro_batch_size", "global_batch_size", "seed"):
            if cfg.actor[key] != official_cfg.actor[key] or training[key] != official_cfg.actor[key]:
                raise ValueError("Official batch or seed parameters were overridden")
        actual = training["optimizer_settings"]
        if actual["betas"] != [cfg.actor.optim.adam_beta1, cfg.actor.optim.adam_beta2] or actual["weight_decay"] != cfg.actor.optim.weight_decay:
            raise ValueError("Actual optimizer parameter groups differ from official parameters")
        for row in history:
            scheduler_step = float(row.get("scheduler_step", row["step"]))
            expected_lr = cfg.actor.optim.lr * min(scheduler_step / cfg.actor.optim.lr_warmup_steps, 1.0)
            if cosine_schedule and scheduler_step >= cfg.actor.optim.lr_warmup_steps:
                progress = (scheduler_step - cfg.actor.optim.lr_warmup_steps) / (cfg.actor.optim.total_training_steps - cfg.actor.optim.lr_warmup_steps)
                expected_lr = cfg.actor.optim.lr * .5 * (1 + math.cos(math.pi * progress))
            if abs(float(row["learning_rate"]) - expected_lr) > 1e-12:
                raise ValueError("Recorded learning rate differs from the declared schedule")
        schedule_note = ("The source warmup_style=cosine is explicitly mapped to RLinf's supported lr_scheduler=cosine: linear warmup for 1000 updates, "
            "then cosine decay over the retained 30000-step horizon. It is not cosine-shaped warmup. This mapping is a declared adaptation, not an unmodified official recipe. "
            "At 11600 updates the decay has not reached zero. Only this scheduler setting differs from the preceding official-parameter run."
            if cosine_schedule else "The source YAML says warmup_style=cosine but does not set lr_scheduler. The current generic FSDP code defaults to linear warmup followed by a constant rate; "
            "the recorded learning rates verify that behavior. No cosine decay is claimed. total_training_steps=30000 is retained exactly from the source.")
        if cosine_schedule:
            import matplotlib.pyplot as plt

            with Path(report["reference_training_history"]).open() as stream:
                reference_history = list(csv.DictReader(stream))
            figure, axis = plt.subplots(figsize=(9, 3))
            for rows, name, style in ((reference_history, "Warmup then constant", "--"), (history, "Warmup then cosine", "-")):
                axis.plot([float(row["step"]) for row in rows], [float(row["learning_rate"]) for row in rows], style, label=name)
            axis.axvline(cfg.actor.optim.lr_warmup_steps, color="#777777", linewidth=1, label="Warmup end")
            axis.set(xlabel="Optimizer updates", ylabel="Recorded learning rate")
            axis.legend()
            figure.tight_layout()
            figure.savefig(report_model_root / "learning_rate.png", dpi=150)
            plt.close(figure)
            schedule_figure = '<h2>Recorded Learning Rates</h2><img src="model/learning_rate.png" alt="Actual learning rate: shared linear warmup, then constant versus cosine decay">'
        implementation_note = ("<h2>Official Parameters and Adaptations</h2>"
            "<p>Parameters are copied directly from the repository's LIBERO OpenPI SFT YAML: AdamW learning rate 0.000025, betas (0.9, 0.95), "
            "epsilon 1e-8, weight decay 1e-10, gradient clip 1.0, microbatch 4, global batch 128, accumulation 32, seed 0, and 1000 warmup steps. "
            + ("Only the post-warmup scheduler differs from the constant-rate comparator.</p>" if cosine_schedule else "This changes several settings together, not only weight decay.</p>")
            + f"<p>{schedule_note}</p>"
            +
            "<p>Adaptations: a fresh FP32 additive residual MLP with frozen Smol, the existing 34/6 correction split, one actor rank, validation-only model selection, "
            f"and {training['epochs']:g} runner epochs / {training['optimizer_steps']} updates rather than the official 30000-update budget. "
            "A runner epoch is 29 updates here (floor of 941 microbatches / 32); the loader continues across runner epoch boundaries. "
            "This is an official-parameter adaptation, not the unmodified OpenPI training recipe. These ten layouts were already used in the prior comparison, so this is a matched retest, not a new untouched test set.</p>"
            '<p><a href="official_source.yaml">Unmodified official source YAML</a></p>')
        if resume_training:
            implementation_note = ("<h2>Preserved Checkpoint and Continuation</h2>"
                f"<p>The source checkpoint is preserved with a SHA256 manifest. Model, Adam moments and step counters, scheduler step {resume_training['restored_scheduler_step']}, "
                f"and RNG state were restored. The deterministic DataLoader position was reconstructed at data epoch {resume_training['data_epoch']}, microbatch offset {resume_training['data_microbatch_offset']}. "
                "Learning rate remains 0.000025; warmup is not restarted. No model, optimizer, data or action-scale changes were requested or applied.</p>"
                f"<p>{training['epochs']:g} additional runner epochs / {training['optimizer_steps']} new updates; {training['total_optimizer_steps']} cumulative updates. "
                "Original 40 correction episodes, preserved 34/6 split, AdamW betas (0.9, 0.95), weight decay 1e-10, microbatch 4 / global batch 128, clip 1.0 and seed 0. "
                "The original 8/10 was measured on different layouts; compare the two models on this new matched batch, not across those test sets.</p>"
                '<p><a href="preserved_source/sha256_manifest.json">Preserved-source hashes</a> | <a href="official_source.yaml">Parameter source</a></p>')
    if standalone:
        audit = json.loads((source_run / "implementation_audit.json").read_text())
        implementation_note = ("<h2>Implementation Comparison</h2><p>The original CPU Adam training loop and RLinf started with identical weights, "
                               "training-only transforms, data order, and 34/6 split. Both ran 400 epochs with batch size 128, learning rate 0.0003, "
                               "and gradient clipping 1.0. Their minibatch shuffle sequences differ; RLinf runs GPU/FSDP while the original runs CPU.</p>"
                               f"<p>On the same CPU input, Adam and zero-decay AdamW one-step maximum parameter difference was {audit['cpu_adam_vs_zero_decay_adamw_one_step_max_difference']:g}. "
                               "These observations do not isolate a cause. Held-out action MSE is not a closed-loop placement score; small action differences can accumulate over 280 steps. "
                               "Ten layouts are not enough to claim a general framework ranking. The prior 30+10 staged model is not one of the policies in this comparison.</p>"
                               '<p><a href="implementation_audit.json">Implementation audit</a></p>')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{engine} Residual SFT | Results</title><style>
:root{{font-family:Georgia,serif;color:#202724;background:#f4f6f5;letter-spacing:0}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1120px;margin:auto;padding:32px 20px}}h1{{font-size:32px;margin:0 0 12px}}h2{{font-size:24px}}h3{{font-size:18px}}p{{line-height:1.6}}a{{color:#166653}}.scores{{display:flex;gap:40px;border-block:1px solid #bcc9c2;padding:22px 0;margin:24px 0;flex-wrap:wrap}}.scores strong{{display:block;font-size:32px}}.scores span{{display:block;margin-top:6px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}article{{min-width:0;border-top:1px solid #bbc7c1;padding-top:6px}}article h3{{display:flex;justify-content:space-between;gap:12px}}video{{width:100%;aspect-ratio:1;background:#111;display:block}}img{{width:100%;height:auto}}.success{{color:#17694a}}.failed{{color:#a33743}}nav{{display:flex;gap:24px;flex-wrap:wrap;margin:20px 0}}footer{{margin-top:32px;border-top:1px solid #bbc7c1;padding-top:16px}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}main{{padding:24px 14px}}h1{{font-size:28px}}.scores{{gap:22px}}.scores strong{{font-size:26px}}}}
</style><main><h1>{engine} Residual SFT</h1><p>{conclusion}</p><div class="scores">{scores}</div>
<p>{training['epochs']:g} epochs / {training['optimizer_steps']} optimizer updates. Existing 40 correction episodes: 34 training, 6 validation. SmolVLA frozen; 22,535 trainable residual parameters. No new collection or reward optimization.</p>
<p>Validation MSE: initial {training['initial_validation_mse']:.8f}; selected {training['best_validation_mse']:.8f}; last {report['last_validation_mse']:.8f}. {evaluation_note}</p>
<p>Objective: mean squared error of the teacher-minus-Smol action residual divided by the saved per-axis scale. Actions: clip(Smol + gate * scale * tanh(MLP), -1, 1); translation 0.05 m, rotation 0.5 rad, gripper positive closes and negative opens, at 20 Hz.</p>
<nav><a href="report.json">Audited report</a><a href="config.yaml">Training configuration</a><a href="model/training.csv">Training history</a></nav>
{implementation_note}
<h2>Training and Evaluation</h2><img src="model/training_curves.png" alt="Actual training and held-out loss, stable successes, and final XY errors">
{schedule_figure}
<h2>All {len(results['student'])} Episodes</h2><p>Top: no residual. Bottom: {displayed_title}, {comparison_note}. Success requires released, supported, stationary placement for 0.5 seconds; XY precision is reported separately.</p>
<div class="grid">{''.join(cards)}</div><footer>Centered randomized bowl-on-plate scenes, policy seeds {policy_seeds[0]}-{policy_seeds[-1]} / layout seeds {layout_seeds[0]}-{layout_seeds[-1]}. Residual starts after 10 Smol actions. Local experiment, not an official LIBERO benchmark score.</footer>
<script>document.querySelectorAll('video').forEach(video=>video.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{{if(other!==video)other.pause()}})))</script></main></html>'''
    (output / "report.html").write_text(html)
    print(json.dumps({"selected_weights_equal_previous": identical, "summary": report["summary"]}, indent=2))


from projects.franka_stack_vla.sft import apply_official_sft_parameters, main


if __name__ == "__main__":
    main()