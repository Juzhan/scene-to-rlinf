"""Create a project by copying a task template; no central registry is modified."""

import argparse
import ast
import io
import keyword
from pathlib import Path
import re
import shutil
import tempfile
import tokenize

from omegaconf import OmegaConf, open_dict

from run import declares_backend, project_directories

PROJECTS = Path(__file__).resolve().parents[1] / "projects"

# A template must name its backend, so a new project never runs on an implicit default.
MAINTAINED_TEMPLATES = ("planner_executor",)


def _assembly_registration_names(source, name):
    """Namespace only registrations owned by an assembly template.

    Native task IDs can identify upstream Gym environments, so never infer their
    ownership from a name prefix or rewrite their text globally.
    """
    entry = source / "project.py"
    if not entry.is_file():
        entry = source / "__init__.py"
    registrations = {"register_isaac_assembly_env": [], "register_model": []}
    unresolved = set()
    for node in ast.walk(ast.parse(entry.read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "Project":
            values = {item.arg: item.value for item in node.keywords}
            for key, registration in (("task_id", "register_isaac_assembly_env"),
                                      ("model_type", "register_model")):
                value = values.get(key)
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    unresolved.add(registration)
                else:
                    registrations[registration].append(value.value)
            continue
        if node.func.id not in registrations:
            continue
        argument = node.args[0] if node.args else next(
            (item.value for item in node.keywords if item.arg in ("task_id", "model_type")), None
        )
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            unresolved.add(node.func.id)
            continue
        registrations[node.func.id].append(argument.value)
    if not registrations["register_isaac_assembly_env"] and "register_isaac_assembly_env" not in unresolved:
        return {}
    if unresolved:
        raise ValueError("Assembly template registration must use explicit string task IDs and model names")
    return {
        **{value: f"{name}/{value}" for value in registrations["register_isaac_assembly_env"]},
        **{value: f"{name}__{value}" for value in registrations["register_model"]},
    }


def _rewrite_python_strings(text, names):
    if not names:
        return text
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                value = None  # An f-string is executable code, not a registration literal.
            if isinstance(value, str) and value in names:
                token = token._replace(string=repr(names[value]))
        tokens.append(token)
    return tokenize.untokenize(tokens)


def _rewrite_config_values(value, names):
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, list):
        return [_rewrite_config_values(item, names) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_config_values(item, names) for key, item in value.items()}
    return value


def create_project(name, template="planner_executor", policy=None, dry_run=False, destination=None):
    if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
        raise ValueError("Use a Python package name, for example my_robot")
    source = PROJECTS / template
    if not template.isidentifier() or source not in project_directories():
        raise ValueError(f"Unknown template: {template}")
    target = (Path(destination) if destination else PROJECTS) / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    if policy is not None and (source / "scene.yaml").is_file() and "control" in OmegaConf.load(source / "scene.yaml"):
        raise ValueError("Planner/executor projects define models in policy.py; --policy overrides are not supported")
    if policy is not None and template == "franka_stack_vla":
        raise ValueError("MLP policy options do not apply to the GR00T template")
    # A scene project that inherits its backend from configs/state_ppo.yaml keeps running there,
    # but the copy names it, so a new project never runs on an implicit default.
    inherited = None
    if (source / "scene.yaml").is_file() and not declares_backend(source):
        inherited = OmegaConf.select(OmegaConf.load(PROJECTS.parent / "configs/state_ppo.yaml"), "env.train.init_params.backend")
    registration_names = _assembly_registration_names(source, name)
    files = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".yaml", ".md", ".rst", ".txt"):
            continue
        text = re.sub(rf"\bprojects\.{re.escape(template)}(?=\b)", f"projects.{name}", path.read_text())
        relative = path.relative_to(source)
        if path.suffix in ('.md', '.rst'):
            text = re.sub(rf'\b{re.escape(template)}\b', name, text)
        if path.suffix == ".yaml":
            cfg = OmegaConf.create(_rewrite_config_values(
                OmegaConf.to_container(OmegaConf.create(text), resolve=False), registration_names
            ))
            if OmegaConf.select(cfg, "runner.logger.experiment_name") is not None:
                cfg.runner.logger.experiment_name = name if path.stem == "train" else f"{name}_{path.stem}"
            if policy is not None and path.stem == "train":
                with open_dict(cfg):
                    if "actor" not in cfg:
                        cfg.actor = {}
                    if "model" not in cfg.actor:
                        cfg.actor.model = {}
                    cfg.actor.model.network = policy
            if inherited and path.stem == "train":
                OmegaConf.update(cfg, "env.train.init_params.backend", inherited, force_add=True)
            text = OmegaConf.to_yaml(cfg)
        if path.suffix == ".py":
            text = _rewrite_python_strings(text, registration_names)
            ast.parse(text)
        files[relative] = text
    if policy == "scratch":
        files[Path("policy.py")] = (PROJECTS / "cartpole/policy.py").read_text()
    if dry_run:
        return target, tuple(map(str, files))
    if inherited:
        print(f"{template} inherits its backend; {name}/train.yaml now names it: "
              f"env.train.init_params.backend: {inherited} (new projects use isaaclab3)")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=target.parent))
    try:
        for relative, text in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        if (staging / 'task.py').is_file():
            from shared.assembly.object_schema import generate
            generate(staging / 'scene.yaml', catalog_directory=source)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return target, tuple(map(str, files))


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?")
    parser.add_argument("--template", default="planner_executor",
                        help="Maintained: " + ", ".join(MAINTAINED_TEMPLATES) + "; other projects can be copied explicitly")
    parser.add_argument("--policy", choices=("scratch", "upstream"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(arguments)
    if args.interactive:
        args.name = args.name or input("Project name: ").strip()
        choices = ", ".join(MAINTAINED_TEMPLATES)
        args.template = input(f"Template ({choices}) [planner_executor]: ").strip() or "planner_executor"
        scene = PROJECTS / args.template / "scene.yaml"
        has_components = scene.is_file() and "control" in OmegaConf.load(scene)
        if args.template != "franka_stack_vla" and not has_components:
            args.policy = input("Policy (upstream/scratch) [upstream]: ").strip() or "upstream"
            if args.policy not in ("upstream", "scratch"):
                parser.error("Policy must be upstream or scratch")
    if not args.name:
        parser.error("Provide a name or use --interactive")
    try:
        target, files = create_project(args.name, args.template, args.policy, args.dry_run)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"{'Would create' if args.dry_run else 'Created'}: {target}")
    print("\n".join(f"  {filename}" for filename in files))
    print(f"bash run.sh check {args.name}")
    print(f"bash run.sh train {args.name}")
    print("No simulator, training, installation or download was started.")
