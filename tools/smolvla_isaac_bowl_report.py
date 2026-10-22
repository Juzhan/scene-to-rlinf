"""Audit and report the Isaac bowl sweep: frozen Smol, pre-PPO residual and PI0.5."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

NAMES = {"policy": "Frozen Smol, no residual", "residual": "Pre-PPO learned residual", "pi05": "PI0.5 teacher"}
ORDER = ("policy", "residual", "pi05")


def load(out_root, policy, seed):
    directory = out_root / f"{policy}_seed{seed}"
    report = json.loads((directory / "report.json").read_text())
    trace = json.loads((directory / "trace.json").read_text())
    assert report["mode"] == policy and report["seed"] == seed
    assert len(trace) == report["steps"]
    assert report["success"] == any(row["success"] for row in trace)
    assert report["settled_success"] == any(row["settled_success"] for row in trace)
    assert report["strict_success"] == any(row["strict_success"] for row in trace)
    assert report["max_lift_m"] == max(row["lift_m"] for row in trace)
    errors = report.get("reset_object_error_mm")
    if errors:
        assert max(errors.values()) <= report["reset_tolerance_m"] * 1000 + 1e-6, \
            f"Isaac object reset exceeded the layout tolerance for seed {seed}: {errors}"
    if policy == "residual":
        assert report["residual_checkpoint"]["residual_start_step"] == report["residual_gate"] == 10
        pure = [row for row in trace if not row["active"]]
        assert len(pure) == 10 and all(row["residual"] == [0.0] * 7 for row in pure)
        assert all(np.allclose(row["action"], np.clip(row["base_action"], -1, 1), atol=1e-6) for row in pure)
        applied = [row for row in trace if row["active"]]
        assert applied and any(any(value != 0 for value in row["residual"]) for row in applied)
        assert report["residual_steps"] == len(applied)
        for row in applied:
            np.testing.assert_allclose(row["action"],
                                       np.clip(np.asarray(row["base_action"]) + np.asarray(row["residual"]), -1, 1),
                                       atol=1e-5)
        assert len(Path(report["residual_checkpoint"]["path"]).joinpath("residual_best.pt").read_bytes()) > 0
    else:
        # base_action is a copy of action for these modes, so this equality is true by construction
        # and only checks that the trace recorded both fields consistently.
        assert all(row["residual"] == [0.0] * 7 for row in trace)
        np.testing.assert_allclose([row["action"] for row in trace], [row["base_action"] for row in trace], atol=1e-6)
    return report, trace


def comparison(out_root, seed, reports, policies):
    destination = out_root / f"comparison_seed{seed}.mp4"
    if destination.exists():
        return destination
    durations = [(len(json.loads((out_root / f"{policy}_seed{seed}" / "trace.json").read_text())) + 1) / 20
                 for policy in policies]
    duration = max(durations)
    inputs, filters = [], []
    for index, policy in enumerate(policies):
        inputs += ["-i", str(out_root / f"{policy}_seed{seed}" / "rollout.mp4")]
        text = NAMES[policy].upper() + (" - PASS" if reports[policy]["settled_success"] else " - FAIL")
        filters.append(f"[{index}:v]tpad=stop_mode=clone:stop_duration={duration - durations[index]:.3f},"
                       f"trim=duration={duration:.3f},setpts=PTS-STARTPTS,"
                       f"drawtext=text='{text}':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.7[view{index}]")
    filters.append("".join(f"[view{index}]" for index in range(len(policies)))
                   + f"vstack=inputs={len(policies)}[stack]")
    subprocess.run(["ffmpeg", "-v", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "[stack]",
                    "-an", "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(destination)], check=True)
    return destination


def summarize(out_root):
    sweep = json.loads((out_root / "sweep_report.json").read_text())
    assert sweep, "Sweep produced no results"
    for entry in sweep:
        if not entry.get("policy"):
            entry["policy"] = entry.get("mode")

    def complete(entry):
        return bool(entry.get("policy")) and not entry.get("failed") and (
            out_root / f"{entry['policy']}_seed{entry['seed']}" / "report.json").exists()

    keys = {(entry["policy"], entry["seed"]) for entry in sweep if complete(entry)}
    failed = [entry for entry in sweep if (entry.get("policy"), entry.get("seed")) not in keys]
    present = [name for name in ORDER if any(key[0] == name for key in keys)]
    seeds = sorted({key[1] for key in keys})
    reports = {}
    for policy in present:
        for seed in seeds:
            if (policy, seed) in keys:
                reports[(policy, seed)] = load(out_root, policy, seed)[0]
    table = {}
    for policy in present:
        entries = [reports[(policy, seed)] for seed in seeds if (policy, seed) in reports]
        table[policy] = {
            "name": NAMES[policy], "episodes": len(entries),
            "settled": sum(entry["settled_success"] for entry in entries),
            "strict": sum(entry["strict_success"] for entry in entries),
            "legacy": sum(entry["success"] for entry in entries),
            "max_lift_cm": max(entry["max_lift_m"] for entry in entries) * 100,
            "mean_min_xy_cm": float(np.mean([entry["min_xy_error_m"] * 100 for entry in entries]))}
    residual_sha = next((reports[key]["residual_checkpoint"]["sha256"] for key in reports
                         if key[0] == "residual"), None)
    scenes, layouts = {}, {}
    for key in sorted(keys):
        source = Path(reports[key]["source"]).resolve()
        if key[1] in scenes and scenes[key[1]] != source:
            raise AssertionError(f"Seed {key[1]} ran in different scenes for different policies")
        scenes[key[1]] = source
    for seed, source in scenes.items():
        layout_path = Path(source).parent / f"layout_{seed}.json"
        if not layout_path.exists():
            raise AssertionError(f"Missing {layout_path.name} for seed {seed}; the layout rule cannot be audited")
        layout = json.loads(layout_path.read_text())
        bowl = np.array(layout["object_positions"]["akita_black_bowl_1"])
        plate = np.array(layout["object_positions"]["plate_1"])
        separation = float(np.linalg.norm(bowl[:2] - plate[:2]))
        if layout["sampling_mode"] != "inward_center" or abs(layout["reference_closer_m"] - 0.10) > 1e-9:
            raise AssertionError(f"Seed {seed} scene does not use the centered 0.10 m training rule")
        if not 0.11 <= separation <= 0.20:
            raise AssertionError(f"Seed {seed} bowl-plate separation {separation} is outside the sampling rule")
        layouts[seed] = {"layout_seed": layout["layout_seed"], "sampling_mode": layout["sampling_mode"],
                         "closer_m": layout["reference_closer_m"], "separation_cm": separation * 100,
                         "bowl_position": bowl.tolist(), "plate_position": plate.tolist()}
    report = {"scenes": {seed: str(path) for seed, path in scenes.items()}, "layouts": layouts,
              "seeds": seeds, "table": table, "failed": failed,
              "residual_sha256": residual_sha, "policies": present,
              "primary_criterion": "bowl-plate contact, XY <3cm, bowl above plate, plate on table without bowl-table contact, released and stationary for 0.5s",
              "strict_criterion": "primary plus XY <=1.5cm held for 0.5s",
              "audits": {"paired_same_scene_per_seed": True, "centered_inward_rule": True, "closer_m": 0.10,
                         "gate_ten_pure_smol_prefix": True, "residual_zero_before_gate": True,
                         "action_equals_clipped_base_plus_residual": True, "reported_success_from_trace": True,
                         "executable_environment": "isaaclab PhysX + RTX rendering, 20 Hz, native relative DLS IK"}}
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    rows = "".join(
        f"<tr><td>{data['name']}</td><td>{data['settled']}/{data['episodes']}</td><td>{data['strict']}/{data['episodes']}</td>"
        f"<td>{data['max_lift_cm']:.2f} cm</td><td>{data['mean_min_xy_cm']:.2f} cm</td></tr>"
        for _, data in table.items())
    layout_rows = "".join(
        f"<tr><td>{seed}</td><td>{data['layout_seed']}</td><td>{data['sampling_mode']}</td>"
        f"<td>{data['closer_m']:.2f} m</td><td>{data['separation_cm']:.2f} cm</td>"
        f"<td>{data['bowl_position'][1]:+.4f}</td><td>{data['plate_position'][1]:+.4f}</td></tr>"
        for seed, data in sorted(layouts.items()))
    cards = []
    legend = ", ".join(f"{index + 1}. {NAMES[policy]}" for index, policy in enumerate(present))
    for seed in seeds:
        available = [policy for policy in present if (policy, seed) in reports]
        if len(available) != len(present):
            continue
        status = " / ".join(f"{NAMES[policy]}: {'pass' if reports[(policy, seed)]['settled_success'] else 'fail'}"
                            for policy in present)
        video = comparison(out_root, seed, {policy: reports[(policy, seed)] for policy in present}, present) \
            if len(present) > 1 else None
        media = (f'<video controls preload="metadata" src="{video.name}"></video>' if video else
                 "".join(f'<video controls preload="metadata" src="{policy}_seed{seed}/rollout.mp4"></video>'
                         for policy in present))
        detail = "".join(
            f"<li>{NAMES[policy]}: settled {'yes' if reports[(policy, seed)]['settled_success'] else 'no'}, "
            f"lift {reports[(policy, seed)]['max_lift_m'] * 100:.2f} cm, "
            f"closest XY {reports[(policy, seed)]['min_xy_error_m'] * 100:.2f} cm, "
            f"<a href='{policy}_seed{seed}/report.json'>report</a></li>" for policy in present)
        cards.append(f'<article><h3>Seed {seed}</h3><p>{status}</p>{media}'
                     f'<ul>{detail}</ul></article>')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Isaac Bowl Transfer: Residual vs PI0.5</title><style>body{{font:16px Georgia,serif;margin:24px auto;padding:0 18px;max-width:1160px;color:#222;background:#f5f7f8}}h1{{font-size:29px}}table{{border-collapse:collapse}}td,th{{padding:9px 22px 9px 0;text-align:left;border-bottom:1px solid #bbb}}video{{width:100%;max-width:512px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:24px}}a{{color:#126b60}}</style>
<h1>Isaac Bowl Transfer: Learned Residual vs PI0.5</h1>
<p>{len(seeds)} seeds, {len(table)} policies, {len(seeds) * len(table)} Isaac episodes of 280 control steps at 20 Hz. The primary criterion is stable placement under the transfer conditions. This is a simulator-transfer check, not an official LIBERO score.</p>
<table><tr><th>Policy</th><th>Stable placement</th><th>Strict XY &le;1.5cm</th><th>Best lift</th><th>Mean closest XY</th></tr>{rows}</table>
<p>Residual checkpoint SHA256 <code>{residual_sha}</code>. The residual is the pre-PPO validation-selected model; its first ten commands are pure frozen Smol and it never sees PI0.5 at evaluation time.</p>
<h2>Reading The Comparison Honestly</h2>
<p>Object geometry now matches the residual's training and validation rule: only the bowl and plate are kept, translated 0.10 m toward the robot, and placed with the same centered inward sampling (midpoint within 2.5 cm of the robot centerline, individual jitter within 0.5 cm, separation 0.11-0.20 m). Each seed uses its own scene, and both policies ran in the <em>same</em> scene for that seed, so the comparison is paired. The layout file is re-checked before this page is written.</p>
<table><tr><th>Seed</th><th>Layout seed</th><th>Sampling</th><th>Closer</th><th>Separation</th><th>Bowl y</th><th>Plate y</th></tr>{layout_rows}</table>
<p>Remaining differences from the native environment are still large: RTX rendering instead of MuJoCo's camera, PhysX with a high-PD Panda and relative DLS IK instead of MuJoCo OSC_POSE, and per-engine object velocity and contact semantics. Isaac also has no native LIBERO <code>On</code> predicate, so the criterion uses bowl-plate contact force, root XY under 3 cm, the bowl above the plate, plate-table support without bowl-table contact, an open gripper without finger contact, and a 0.5 s hold. No policy was retrained and no teacher data was collected for this sweep.</p>
<p>PI0.5 sees only two 256px cameras, the eight-value state and the instruction, and executes five actions per replan, so it is unaffected by the residual's privileged-input assumptions but still faces the same renderer and controller gaps.</p>
<h2>All Episodes</h2><p>Stacked top to bottom: {legend}. Shorter rollouts hold their final frame.</p>
<section class="cards">{''.join(cards)}</section>
<p><a href="sweep_report.json">Raw per-run results</a> | <a href="report.json">Audited summary</a></p></html>'''
    (out_root / "report.html").write_text(html)
    print(json.dumps({"table": table, "failed": [entry.get("policy") for entry in failed]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    summarize(parser.parse_args().output.resolve())
