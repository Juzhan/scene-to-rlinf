"""Inspect a configured native environment in a short-lived simulator process.

The returned names and dimensions come from the loaded assets and managers,
not a table of guessed dimensions. No policy or trainer is constructed.
"""

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from shared.assembly.assets import DEFAULT_BACKEND, selected_backend


def native_interface(native):
    observations = native.observation_manager
    actions = native.action_manager
    result = {
        "backend": getattr(native, 'backend_name', DEFAULT_BACKEND),
        "observations": {}, "actions": [], "articulations": {},
        "action_dim": actions.total_action_dim, "step_dt": native.step_dt,
    }
    samples = observations.compute(update_history=False)
    for group, names in observations.active_terms.items():
        result["observations"][group] = {
            "shape": list(observations.group_obs_dim[group]),
            "dtype": str(samples[group].dtype).removeprefix("torch.") if hasattr(samples[group], "dtype") else None,
            "terms": [{"name": name, "shape": list(shape)} for name, shape in
                      zip(names, observations.group_obs_term_dim[group])],
        }
    for name in actions.active_terms:
        term = actions.get_term(name)
        result["actions"].append({
            "name": name, "dimension": term.action_dim,
            "asset": getattr(term.cfg, 'asset_name', getattr(term.cfg, 'entity_name', None)),
            "joint_names": list(getattr(term, "_joint_names", [])),
            "implementation": type(term).__module__ + "." + type(term).__name__,
        })
    for name in actions.active_terms:
        term = actions.get_term(name)
        if hasattr(term.cfg, "control"):
            result["control"] = {
                "train": term.cfg.control["train"],
                "components": [{"name": key, "dimension": term.chain.outputs[key].shape[1],
                                "period_s": value["period"],
                                "trained": key == term.cfg.control["train"]}
                               for key, value in term.cfg.control["components"].items()],
                "output": {key: {**term.cfg.control["output"][key], "dimension": endpoint.action_dim,
                                  "resolved_joint_names": list(endpoint._joint_names)}
                           for key, endpoint in term.endpoints.items()},
            }
    articulations = (native.scene.articulations if hasattr(native.scene, 'articulations') else
                     {name: asset for name, asset in native.scene.entities.items() if asset.is_articulated})
    for name, asset in articulations.items():
        result["articulations"][name] = {
            "joint_names": list(asset.joint_names), "body_names": list(asset.body_names),
            "default_joint_position": asset.data.default_joint_pos[0].cpu().tolist(),
            "joint_position_limits": asset.data.soft_joint_pos_limits[0].cpu().tolist(),
        }
    return result


def apply_dimensions(cfg, report):
    """Resolve the standard flat-state policy interface; custom models stay explicit."""
    model = cfg.actor.model
    if model.get("obs_dim") == "auto":
        state = report["observations"].get("states")
        if not state or len(state["shape"]) != 1:
            raise ValueError("obs_dim=auto requires a flat 'states' observation group; use a custom model for images/dicts")
        model.obs_dim = int(state["shape"][0])
    if model.get("action_dim") == "auto":
        model.action_dim = int(report["action_dim"])
    from omegaconf import open_dict
    with open_dict(model):
        model.observation_shapes = {name: value['shape'] for name, value in report['observations'].items()}
        model.observation_dtypes = {name: value.get('dtype') for name,value in report['observations'].items()}
    if type(model.get('action_dim')) is not int or model.action_dim < 1:
        raise ValueError('actor.model.action_dim must resolve to a positive integer')
    if model.get('obs_dim') is not None and (type(model.obs_dim) is not int or model.obs_dim < 1):
        raise ValueError('actor.model.obs_dim must be null for object inputs or a positive integer')
    return report


def inspect_configuration(cfg, module, *, timeout=180):
    from omegaconf import OmegaConf
    from shared.assembly.interfaces import source_revision, save_interface
    if not hasattr(module, "project") or not hasattr(module.project, "make_environment"):
        raise ValueError("Automatic interface discovery requires a configured Assembly Project")
    split = "eval" if cfg.runner.get("only_eval", False) else "train"
    scene = cfg.env[split].init_params.get("scene_config", module.project.scene_config)
    if (Path(scene).parent / "task.py").is_file():
        # The worker regenerates observation.py; do it first so the revision
        # describes the files the inspected environment actually uses.
        from shared.assembly.object_schema import generate
        from shared.assembly.project import project_registry
        factories, terms, _ = project_registry(module.project.package,
                                               backend=selected_backend(cfg.env[split].init_params))
        generate(scene, factories=factories, terms=terms)
    revision = source_revision(scene)
    with tempfile.TemporaryDirectory(prefix="rlinf-interface-") as directory:
        directory = Path(directory)
        recipe, output, logfile = (directory / value for value in ("recipe.yaml", "interface.json", "probe.log"))
        OmegaConf.save(cfg, recipe, resolve=True)
        from shared.assembly.project import backend_interpreter
        interpreter = backend_interpreter(selected_backend(cfg.env[split].init_params))
        command = [interpreter, str(Path(__file__).resolve()), "--worker", "--module", module.__name__,
                   "--recipe", str(recipe), "--output", str(output)]
        environment = dict(os.environ)
        paths = [str(Path(__file__).resolve().parents[1]), *sys.path]
        if interpreter != sys.executable:
            # The other interpreter brings its own standard library and site-packages.
            installed = tuple(str(Path(prefix).resolve()) for prefix in (sys.prefix, sys.base_prefix))
            paths = [p for p in paths if p and not str(Path(p).resolve()).startswith(installed)]
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(str(p) for p in paths if p))
        with logfile.open("w") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=environment, timeout=timeout, check=False)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"Scene interface inspection exceeded {timeout}s\n{logfile.read_text()[-5000:]}") from error
        if process.returncode or not output.is_file():
            raise RuntimeError(f"Scene interface inspection failed ({process.returncode})\n{logfile.read_text()[-7000:]}")
        report = json.loads(output.read_text())
    save_interface(scene, report, expected_revision=revision,
                   backend=selected_backend(cfg.env[split].init_params))
    module.interface_metadata = report
    return report


def resolve_model_dimensions(cfg, module):
    if any(cfg.actor.model.get(name) == "auto" for name in ("obs_dim", "action_dim")):
        print("Inspecting the scene to resolve policy inputs, actions and robot joints...", flush=True)
        report = inspect_configuration(cfg, module)
        apply_dimensions(cfg, report)
        print(f"Resolved policy: {len(report['observations'])} observation fields, {cfg.actor.model.action_dim} actions", flush=True)
        return report
    return None


def worker(options):
    from omegaconf import OmegaConf
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    module = import_module(options.module)
    cfg = OmegaConf.load(options.recipe)
    selected = cfg.env.eval if cfg.runner.get("only_eval", False) else cfg.env.train
    parameters = selected.init_params
    environment, owner = module.project.make_environment(
        parameters, 1, selected.seed, selected.max_episode_steps, headless=True,
    )
    try:
        environment.reset(seed=selected.seed)
        report = native_interface(environment.native)
        options.output.write_text(json.dumps(report, indent=2) + "\n")
        environment.close()
    finally:
        owner.close()


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?")
    parser.add_argument("--config", default="train")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module", help=argparse.SUPPRESS)
    parser.add_argument("--recipe", type=Path, help=argparse.SUPPRESS)
    options, overrides = parser.parse_known_args(arguments)
    # Import workspace path setup before importing the project or SDK.
    from run import compose_project
    if options.worker:
        return worker(options)
    if not options.project:
        parser.error("a project name is required")
    cfg, module = compose_project(options.project, options.config, overrides)
    cfg.runner.only_eval = options.split == "eval"
    report = inspect_configuration(cfg, module)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(text)
    print(text)
    return report


if __name__ == "__main__":
    # Direct execution from tools/ needs workspace imports too.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
