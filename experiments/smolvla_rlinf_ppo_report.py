"""Audit and report a fixed-budget RLinf bowl PPO run and paired evaluation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


def summarize(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training = json.loads((output / "training_report.json").read_text())
    evaluation = json.loads((output / "evaluation/evaluation_report.json").read_text())
    baseline = json.loads((output / "baseline_evaluation/evaluation_report.json").read_text())
    assert evaluation["evaluation_seeds"] == baseline["evaluation_seeds"]
    assert evaluation["layout_seeds"] == baseline["layout_seeds"]
    seeds = evaluation["evaluation_seeds"]
    with (output / "training.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == training["iterations"] == training["environment_steps"] // 256
    assert int(float(history[-1]["env_steps"])) == training["environment_steps"]
    assert training["optimizer_steps"] == 2 * len(history)
    assert training["framework"]["run_training_inherited"]
    assert training["prior_frozen"] and not training["teacher_used"]
    for directory in (Path(training["source"]), output / "preserved_source"):
        assert hashlib.sha256((directory / "residual_best.pt").read_bytes()).hexdigest() == training["source_sha256"]
    original = torch.load(output / "preserved_source/residual_best.pt", map_location="cpu", weights_only=True)
    initial = torch.load(output / "residual_initial.pt", map_location="cpu", weights_only=True)
    final = torch.load(output / "model/residual_best.pt", map_location="cpu", weights_only=True)
    for key, value in original.items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    for key in ("mean", "std", "scale"):
        torch.testing.assert_close(original[key], final[key], rtol=0, atol=0)
    assert any(not torch.equal(value, final[key]) for key, value in original.items() if key.startswith("network."))
    for label, directory in evaluation["checkpoints"].items():
        assert hashlib.sha256((Path(directory) / "residual_best.pt").read_bytes()).hexdigest() == evaluation["checkpoint_sha256"][label]
    assert evaluation["checkpoint_sha256"]["previous_student"] == training["source_sha256"]
    policies = {"baseline": baseline["baseline"], "previous_student": evaluation["previous_student"],
                "student": evaluation["student"]}
    names = {"baseline": "No residual", "previous_student": "Before PPO", "student": "After PPO"}
    layouts = [row["layout"] for row in policies["baseline"]]
    for rows in policies.values():
        assert [row["layout"] for row in rows] == layouts
    train_seeds = {row["seed"] for row in training["episodes"]}
    train_layouts = {row["layout_seed"] for row in training["episodes"]}
    assert not set(seeds) & train_seeds and not set(evaluation["layout_seeds"]) & train_layouts
    for seed in seeds:
        traces = {}
        for label in policies:
            directory = "baseline_evaluation" if label == "baseline" else "evaluation"
            traces[label] = json.loads((output / directory / f"{label}_{seed}_trace.json").read_text())
        for label in ("previous_student", "student"):
            np.testing.assert_array_equal([row["applied_action"] for row in traces["baseline"][:10]],
                                          [row["applied_action"] for row in traces[label][:10]])
        for row in traces["baseline"]:
            np.testing.assert_array_equal(row["residual"], np.zeros(7))
            np.testing.assert_array_equal(row["applied_action"], np.clip(row["base_action"], -1, 1))
    summary = {label: {"successes": sum(row["is_success"] for row in rows), "episodes": len(rows),
                       "mean_xy_cm": float(np.mean([row["xy_error_m"] * 100 for row in rows]))}
               for label, rows in policies.items()}
    gains = [seed for seed, before, after in zip(seeds, policies["previous_student"], policies["student"])
             if after["is_success"] and not before["is_success"]]
    losses = [seed for seed, before, after in zip(seeds, policies["previous_student"], policies["student"])
              if before["is_success"] and not after["is_success"]]
    report = {"training": training, "summary": summary, "gained_seeds": gains, "lost_seeds": losses,
              "evaluation_seeds": seeds, "layout_seeds": evaluation["layout_seeds"], "policies": policies,
              "audit": {"matched_layouts": True, "raw_baseline": True, "matched_first_ten_actions": True,
                        "source_preserved": True, "exact_initial_actor": True, "transforms_unchanged": True,
                        "full_budget": True}, "residual_scale": final["scale"].tolist()}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    episodes = training["episodes"]
    episode_steps = [row["env_steps"] for row in episodes]
    for axis, key, title in ((axes[0, 0], "return", "Training episode return"),
                             (axes[0, 1], "success", "Training success (10-episode mean)")):
        values = [row[key] for row in episodes]
        axis.plot(episode_steps, [np.mean(values[max(0, index - 9):index + 1]) for index in range(len(values))])
        axis.set(title=title, xlabel="Environment steps")
    steps = [float(row["env_steps"]) for row in history]
    axes[0, 2].plot(steps, [float(row["actor/approx_kl"]) for row in history])
    axes[0, 2].set(title="PPO approximate KL", xlabel="Environment steps")
    axes[1, 0].plot(steps, [float(row["critic/value_loss"]) for row in history])
    axes[1, 0].set(title="Critic loss", xlabel="Environment steps")
    axes[1, 1].bar(list(names.values()), [summary[label]["successes"] for label in policies],
                   color=["#777777", "#356a9c", "#168578"])
    axes[1, 1].set(title="Independent paired evaluation", ylabel=f"Successes / {len(seeds)}", ylim=(0, len(seeds)))
    for label, rows in policies.items():
        axes[1, 2].plot(range(1, len(seeds) + 1), [row["xy_error_m"] * 100 for row in rows], marker="o", label=names[label])
    axes[1, 2].set(title="Final XY error", xlabel="Evaluation episode", ylabel="cm")
    axes[1, 2].legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=150)
    plt.close(figure)
    cards = []
    for index, seed in enumerate(seeds):
        rows = [policies[label][index] for label in policies]
        durations = [(row["steps"] + 1) / 20 for row in rows]
        duration = max(durations)
        destination = output / f"comparison_{seed}.mp4"
        if not destination.exists():
            inputs, filters = [], []
            for number, label in enumerate(policies):
                directory = "baseline_evaluation" if label == "baseline" else "evaluation"
                inputs.extend(["-i", str(output / directory / f"{label}_{seed}.mp4")])
                text = names[label].upper() + (" - SUCCESS" if rows[number]["is_success"] else " - FAILED")
                filters.append(f"[{number}:v]tpad=stop_mode=clone:stop_duration={duration - durations[number]:.3f},"
                               f"trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
                               f"drawtext=text='{text}':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.7,"
                               f"drawtext=text='END FRAME HELD':x=8:y=h-24:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.7:"
                               f"enable='gte(t,{durations[number]:.3f})'[view{number}]")
            filters.append("[view0][view1][view2]vstack=inputs=3[stack]")
            subprocess.run(["ffmpeg", "-v", "error", *inputs, "-filter_complex", ";".join(filters),
                            "-map", "[stack]", "-an", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(destination)], check=True)
        status = " / ".join(f"{names[label]}: {'success' if policies[label][index]['is_success'] else 'failed'}" for label in policies)
        cards.append(f'<article><h3>Seed {seed}</h3><p>{status}</p><video controls preload="metadata" src="{destination.name}"></video></article>')
    table = "".join(f"<tr><td>{names[label]}</td><td>{data['successes']}/{data['episodes']}</td><td>{data['mean_xy_cm']:.2f} cm</td></tr>" for label, data in summary.items())
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLinf PPO Bowl Placement</title><style>body{{font:16px Georgia,serif;margin:24px auto;padding:0 18px;max-width:1160px;color:#222;background:#f5f7f8}}h1{{font-size:30px}}p{{line-height:1.5}}table{{border-collapse:collapse}}td,th{{padding:10px 24px 10px 0;text-align:left;border-bottom:1px solid #bbb}}img{{width:100%;height:auto}}.videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:24px}}video{{width:100%;max-width:512px;height:auto}}a{{color:#126b60}}</style>
<h1>RLinf PPO Bowl Placement</h1><p>Completed {training['environment_steps']:,} environment steps, {training['iterations']} rollout updates and {int(training['optimizer_steps'])} optimizer steps. The existing supervised actor was preserved exactly at initialization. Smol stayed frozen. No teacher was used.</p>
<table><tr><th>Policy</th><th>Stable placement</th><th>Mean final XY</th></tr>{table}</table>
<p>Gained seeds: {gains or 'none'}. Lost seeds: {losses or 'none'}. Ten paired unseen layouts; a small evaluation, not a broad reliability claim. The final fixed-budget checkpoint was evaluated without test-based checkpoint selection.</p>
<h2>Task, Reward And Actions</h2><p>Native LIBERO spatial task 8, distractors removed, objects shifted 0.10 m closer and randomized inward around the robot centerline. Success: native On, released, supported and stationary for 10 frames (0.5 s). XY is reported separately, not a success threshold.</p>
<p>Reward: 0.99 * potential(next) - potential(previous), step cost -0.002, applied residual squared cost -0.005; valid released supported placement adds 1 + 3 exp(-100 * XY error); success +10, fall -10. Potential combines reach, lift, alignment and lowering. Finite horizon 280 steps; no bootstrap across episode end.</p>
<p>Action: clip(Smol + gate * saved_scale * tanh(latent), -1, 1). The first 10 commands are pure Smol and excluded from PPO loss. XYZ controller scale 0.05 m, rotation 0.5 rad, positive gripper closes. Residual scales: {final['scale'].tolist()}.</p>
<p>RLinf native EmbodiedFSDPActor PPO/GAE/FSDP updates, with a project-specific single-environment sampler (not EmbodiedRunner or Stable-Baselines3). Actor learning rate 1e-5, critic 1e-4, Gaussian initial std exp(-3.5), PPO clip 0.1, two epochs per 256-step rollout, gamma 0.99, lambda 0.95. New PPO optimizer; no SFT optimizer resume, teacher loss or reference-policy KL penalty.</p>
<h2>Actual Curves</h2><img src="training_curves.png" alt="Training return, success, approximate KL, critic loss and paired evaluation">
<p><a href="report.json">Results and audits</a> | <a href="training.csv">Training metrics</a> | <a href="episodes.json">Training episodes</a> | <a href="config.yaml">Configuration</a> | <a href="preserved_source/residual_best.pt">Preserved source</a> | <a href="model/residual_best.pt">Final PPO actor</a></p>
<h2>All Evaluation Episodes</h2><p>Top: no residual. Middle: before PPO. Bottom: after PPO. Earlier endings hold their final frame with a label.</p><section class="videos">{''.join(cards)}</section></html>'''
    (output / "report.html").write_text(html)
    print(json.dumps({"summary": summary, "gained_seeds": gains, "lost_seeds": losses}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args().output.resolve())