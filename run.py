"""Resolve a project and invoke RLinf; no training or worker implementation lives here."""

import argparse
from importlib import import_module
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RLINF = Path(os.environ.get("RLINF_ROOT", ROOT / "third_party/RLinf-upstream")).resolve()
os.environ['RLINF_ROOT'] = str(RLINF)
os.environ["WORKSPACE_ROOT"] = str(ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(RLINF)))
sys.path[:0] = [str(ROOT), str(RLINF)]


def project_directories():
    for directory in sorted((ROOT / "projects").iterdir()):
        if (directory / "project.py").is_file() or (
            (directory / "__init__.py").is_file()
            and ((directory / "train.yaml").is_file() or (directory / "recipes/train.yaml").is_file())
        ):
            yield directory


def recipe_paths(directory):
    """Discover explicit recipes and legacy configs that define/inherit a training config.

    Scene files may use any name. A legacy recipe has runner/env/actor sections,
    or inherits another such recipe through its local Hydra defaults.
    """
    from omegaconf import DictConfig, OmegaConf

    directory = Path(directory)
    candidates = {path.stem: path for path in sorted(directory.glob("*.yaml"))}
    explicit = {path.stem: path for path in sorted((directory / "recipes").glob("*.yaml"))}
    documents = {name: OmegaConf.load(path) for name, path in candidates.items()}

    def is_recipe(name, visiting=()):
        if name in visiting or name not in documents:
            return False
        cfg = documents[name]
        if not isinstance(cfg, DictConfig):
            return False
        if all(key in cfg for key in ("runner", "env", "actor")):
            return True
        if isinstance(cfg.get("runner"), DictConfig) and cfg.runner.get("task_type") == "sft":
            return True
        return any(
            isinstance(default, str) and is_recipe(default, (*visiting, name))
            for default in cfg.get("defaults", [])
        )

    recipes = {name: path for name, path in candidates.items() if is_recipe(name)}
    duplicates = recipes.keys() & explicit.keys()
    if duplicates:
        raise ValueError(f"Duplicate recipe names in {directory}: {', '.join(sorted(duplicates))}")
    recipes.update(explicit)
    return tuple(recipes[name] for name in sorted(recipes))


def _configure_checkpoint_interval(cfg):
    from omegaconf import OmegaConf

    if OmegaConf.to_container(cfg.runner, resolve=False).get("save_interval") != "${runner.max_epochs}":
        return
    budget = max(cfg.runner.max_epochs, cfg.runner.get("max_steps", -1))
    if budget < 1:
        raise ValueError("Final-only checkpoint saving requires a finite training budget")
    cadence = max(1, cfg.runner.val_check_interval)
    cfg.runner.save_interval = ((budget + cadence - 1) // cadence) * cadence


def compose_project(name, config="train", overrides=(), *, register=True):
    from hydra import compose, initialize_config_dir

    if not name.isidentifier() or not config.isidentifier():
        raise ValueError("Project and configuration names must be Python identifiers")
    directory = ROOT / "projects" / name
    has_project_module = (directory / "project.py").is_file()
    if not has_project_module and not (directory / "__init__.py").is_file():
        raise ValueError(f"Unknown project: {name}")
    recipes = {path.stem: path for path in recipe_paths(directory)}
    if config not in recipes:
        available = ", ".join(recipes) or "none"
        raise ValueError(f"Not a training recipe: {name}/{config}; available recipes: {available}")
    with initialize_config_dir(version_base="1.3", config_dir=str(recipes[config].parent)):
        cfg = compose(config_name=config, overrides=[
            f"hydra.searchpath=[file://{ROOT}/configs,file://{RLINF}/examples/embodiment/config]",
            *overrides,
        ])
    from shared.assembly.project import follow_train_backend
    follow_train_backend(cfg, overrides)
    _configure_checkpoint_interval(cfg)
    extension = f"projects.{name}.project" if has_project_module else f"projects.{name}"
    os.environ["RLINF_EXT_MODULE"] = extension
    module = import_module(extension)
    if register and cfg.runner.get("task_type") != "sft":
        module.register()
    return cfg, module


def execute_project_recipe(cfg, module, operation, options=None):
    """Dispatch data-trained projects without forcing the Isaac/RLinf env path."""
    handler = getattr(module, "run_recipe", None)
    if handler is None:
        raise ValueError(f"{module.__name__} does not provide an offline training recipe handler")
    if operation in ("check", "train"):
        return handler(cfg, operation)
    if not getattr(getattr(module, "project", None), "online", False):
        raise ValueError("This offline SFT project has no online environment/policy")
    return handler(cfg, operation, options=options)


def execute(cfg, operation):
    import ray
    from omegaconf import OmegaConf, open_dict
    from shared.integrations.rlinf.upstream import validate_runtime
    validate_runtime(cfg)

    if cfg.env.train.env_type == "isaaclab":
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    original_runner = None
    try:
        if operation == "evaluate":
            with open_dict(cfg):
                cfg.runner.only_eval = True
                cfg.runner.task_type = "embodied_eval"
                cfg.rollout.model = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
            from evaluations.eval_embodied_agent import main
        elif cfg.runner.get("async_training", False):
            from examples.embodiment.train_async import main
        else:
            from examples.embodiment import train_embodied_agent
            if cfg.runner.get("keep_last_checkpoints", -1) >= 0:
                from shared.integrations.rlinf.checkpoints import RetainingEmbodiedRunner
                original_runner = train_embodied_agent.EmbodiedRunner
                train_embodied_agent.EmbodiedRunner = RetainingEmbodiedRunner
            main = train_embodied_agent.main
        return main(cfg)
    finally:
        if original_runner is not None:
            train_embodied_agent.EmbodiedRunner = original_runner
        ray.shutdown()



def use_backend_interpreter(environment):
    """Restart under the interpreter the configured simulator backend declares."""
    from shared.assembly.assets import selected_backend
    from shared.assembly.project import backend_interpreter
    try:
        interpreter = backend_interpreter(selected_backend(environment.init_params))
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    if interpreter != sys.executable:
        print(f"backend {environment.init_params.backend}: using {interpreter}", flush=True)
        os.execv(interpreter, [interpreter, str(ROOT / "run.py"), *sys.argv[1:]])
def main():
    if any(value == '--backend' or value.startswith('--backend=') for value in sys.argv[1:]):
        from tools.scene_backend import main as scene_backend
        return scene_backend(sys.argv[1:])
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        parser = argparse.ArgumentParser(
            description="Run a project's bounded behavior probe without training",
            add_help=len(sys.argv) <= 2 or sys.argv[2] in ("-h", "--help"))
        parser.add_argument("project")
        options, remaining = parser.parse_known_args(sys.argv[2:])
        if not options.project.isidentifier() or not (ROOT / "projects" / options.project / "project.py").is_file():
            parser.error("Unknown project")
        module = import_module(f"projects.{options.project}.project")
        probe = getattr(module, "probe", None)
        if probe is None:
            parser.error(f"Project {options.project} does not provide a behavior probe")
        return probe(remaining)
    if len(sys.argv) > 1 and sys.argv[1] == "asset":
        from shared.asset_formats import main as asset
        raise SystemExit(asset(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "assess":
        from tools.go2_evaluation import main as assess
        return assess(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        from tools.inspect_scene import main as inspect_scene
        return inspect_scene(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] in ("edit", "catalog"):
        editor = argparse.ArgumentParser(description="Edit a project in the browser (--blender: Blender) or list its available components")
        editor.add_argument("project", nargs='?')
        editor.add_argument("--scene", type=Path)
        editor.add_argument("--blender", action="store_true", help="Open Blender instead of the browser editor (executable: $BLENDER)")
        editor.add_argument("--web", action="store_true", help=argparse.SUPPRESS)  # former opt-in; now the default
        editor.add_argument("--host", help="Browser editor address (default 127.0.0.1)")
        editor.add_argument("--port", help="Browser editor port (default 8080)")
        options, remaining = editor.parse_known_args(sys.argv[2:])
        scene = options.scene or (ROOT / "projects" / options.project / "scene.yaml" if options.project else None)
        if sys.argv[1] == "edit":
            from tools.blender_editor import main as open_editor
            if options.blender and options.web:
                editor.error("Choose either --blender or the browser editor")
            if not options.blender and scene is None:
                editor.error("The browser editor needs a project or --scene; use --blender for an empty environment")
            command = "open" if options.blender else "web"
            remaining += [value for name in ("host", "port") if getattr(options, name)
                          for value in (f"--{name}", getattr(options, name))]
            raise SystemExit(open_editor([command, *(["--scene", str(scene)] if scene else []), *remaining]))
        if scene is None:
            editor.error('catalog requires a project or --scene')
        if remaining:
            editor.error("Unknown catalog options: " + " ".join(remaining))
        import json
        from shared.assembly.catalog import describe_scene
        print(json.dumps(describe_scene(scene), ensure_ascii=False, indent=2))
        return
    if len(sys.argv) > 1 and sys.argv[1] in ("view", "record") and any(
        value in ("-h", "--help") for value in sys.argv[2:]
    ):
        from tools.preview import parse_options
        return parse_options(["--help"], sys.argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("list", "check", "smoke", "probe", "view", "record", "train", "evaluate", "test", "new", "edit", "catalog", "inspect"))
    parser.add_argument("project", nargs="?")
    parser.add_argument("--config", default="train")
    parser.add_argument("--snapshot", type=Path, help="Use the frozen project and recipe from a saved run")
    parser.add_argument("--allow-scene-change", action="store_true", help="Evaluate an intentionally changed scene with the same policy interface")
    parser.add_argument("--allow-shared-change", action="store_true", help="Explicitly accept and record changed shared framework code")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true", help="Explicitly accept weights without a saved project/interface")
    parser.add_argument("--board", action="store_true", help="train: serve this experiment's TensorBoard records while training")
    parser.add_argument("--board-port", type=int, default=6006, help="first TensorBoard port to try (default 6006)")
    args, overrides = parser.parse_known_args()
    if args.board and args.operation != "train":
        parser.error("--board is only available for train")
    if args.operation == "list":
        for directory in project_directories():
            recipes = ", ".join(item.stem for item in recipe_paths(directory)) or "no RLinf recipe"
            print(f"{directory.name}: {recipes}")
        return
    if args.operation == "new":
        from tools.create import main as create
        return create(([args.project] if args.project else []) + overrides)
    if args.operation == "test":
        import pytest
        directory = ROOT / "projects" / args.project if args.project else ROOT / "tests"
        if args.project and (not args.project.isidentifier() or not directory.is_dir()):
            parser.error("Unknown project test directory")
        raise SystemExit(pytest.main([str(directory), "--import-mode=importlib", *overrides]))
    if not args.project:
        parser.error("A project is required")
    preview_options = None
    if args.operation in ("view", "record"):
        from tools.preview import parse_options
        actions_explicit = any(value == "--actions" or value.startswith("--actions=") for value in overrides)
        preview_options, overrides = parse_options(overrides, args.operation)
        preview_options.actions_explicit = actions_explicit
    snapshot_operation = args.operation in ("train", "evaluate", "view", "record", "smoke")
    replay_report = None
    if args.snapshot:
        from shared.experiments import load_snapshot, read_manifest
        _, manifest = read_manifest(args.snapshot)
        if args.project != manifest["project"]:
            parser.error(f"Snapshot belongs to project {manifest['project']}, not {args.project}")
        cfg, module = load_snapshot(args.snapshot, overrides, register=False, allow_shared_change=args.allow_shared_change)
        if snapshot_operation:
            use_backend_interpreter(cfg.env.eval if args.operation == "evaluate" else cfg.env.train)
        replay_report = module.experiment_report
        from shared.experiments import checkpoint_compatibility, effective_checkpoint
        checkpoint = effective_checkpoint(cfg, args.operation, preview_options.checkpoint if preview_options else None)
        if checkpoint:
            replay_report = checkpoint_compatibility(
                cfg, module, checkpoint, allow_scene_change=args.allow_scene_change,
                allow_shared_change=args.allow_shared_change, allow_legacy_checkpoint=args.allow_legacy_checkpoint)
        if not snapshot_operation:
            module.register()
    elif snapshot_operation:
        cfg, module = compose_project(args.project, args.config, overrides, register=False)
        if cfg.runner.get("task_type") == "sft":
            if args.board:
                parser.error("--board is not wired into project SFT recipes yet")
            return execute_project_recipe(cfg, module, args.operation, preview_options)
        from shared.integrations.rlinf.upstream import validate_runtime
        validate_runtime(cfg)
        if args.operation == "evaluate":
            cfg.runner.only_eval = True
        from shared.assembly.project import Project, require_online_backend
        selected_environment = cfg.env.eval if cfg.runner.only_eval else cfg.env.train
        if isinstance(getattr(module, 'project', None), Project):
            require_online_backend(selected_environment.init_params)
            use_backend_interpreter(selected_environment)
        from tools.inspect_scene import resolve_model_dimensions
        resolve_model_dimensions(cfg, module)
        from shared.experiments import checkpoint_compatibility, effective_checkpoint, prepare_replay
        checkpoint = effective_checkpoint(cfg, args.operation, preview_options.checkpoint if preview_options else None)
        if args.operation == "train" and not cfg.runner.get("resume_dir"):
            replay_report = checkpoint_compatibility(
                cfg, module, checkpoint, allow_scene_change=args.allow_scene_change,
                allow_shared_change=args.allow_shared_change, allow_legacy_checkpoint=args.allow_legacy_checkpoint)
        else:
            cfg, module, replay_report = prepare_replay(
                cfg, module, checkpoint=checkpoint, allow_scene_change=args.allow_scene_change,
                allow_shared_change=args.allow_shared_change, allow_legacy_checkpoint=args.allow_legacy_checkpoint,
                overrides=overrides)
    else:
        cfg, module = compose_project(args.project, args.config, overrides)
        if cfg.runner.get("task_type") == "sft":
            return execute_project_recipe(cfg, module, args.operation, preview_options)
    if snapshot_operation:
        import json
        from shared.experiments import create_snapshot, evaluation_directory, is_assembly_project, load_snapshot
        if replay_report:
            module.experiment_report = replay_report
        if args.operation == "train" and is_assembly_project(module):
            path = create_snapshot(cfg, module)
            print(f"Frozen project: {path}", flush=True)
            cfg, module = load_snapshot(path, register=False)
        elif args.operation == "evaluate" and is_assembly_project(module):
            evaluation_directory(cfg, replay_report or {"checkpoint": None, "untrained_policy": True})
        if replay_report:
            print(json.dumps(replay_report, indent=2, ensure_ascii=False), flush=True)
        if preview_options is not None:
            preview_options.experiment_report = replay_report
            preview_options.experiment_module = module.__name__
        module.register()
    if args.operation == "check":
        import json
        from omegaconf import OmegaConf
        from rlinf.envs import get_env_cls
        from shared.integrations.rlinf.upstream import validate_runtime
        validate_runtime(cfg)

        environment = cfg.env.eval if cfg.runner.only_eval else cfg.env.train
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(f"Environment: {get_env_cls(environment.env_type, environment).__name__}")
        check = getattr(module, "check", None)
        if check is not None:
            print(json.dumps(check(cfg), indent=2, ensure_ascii=False))
        print("PASS static project configuration and registration; no simulator or model instantiated")
        return
    preflight = getattr(module, "preflight", None)
    from shared.integrations.rlinf.upstream import validate_runtime
    validate_runtime(cfg)
    if preflight is not None:
        preflight(cfg)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if preview_options is not None:
        from tools.preview import main as preview
        return preview(cfg, preview_options, recording=args.operation == "record")
    if args.operation == "smoke":
        from tools.smoke import smoke
        return smoke(cfg)
    if not args.board:
        return execute(cfg, args.operation)
    from tools.board import start_board, stop_board
    board = start_board(cfg, args.board_port)
    try:
        return execute(cfg, args.operation)
    finally:
        stop_board(board, cfg)


if __name__ == "__main__":
    main()
