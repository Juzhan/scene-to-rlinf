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
# planner_executor: a Go2 walking to a goal (train the executor); piper_pick: a Piper X arm picking a cube
# (train the planner over a pretrained executor in weights/).
MAINTAINED_TEMPLATES = ("blank", "go2_goal", "piper_pick", "planner_executor")
TEMPLATE_LABELS = {"blank": "空白：只有地面和灯光，自己添加机器人与物体",
                   "go2_goal": "四足机器狗走到目标点（一层策略，从头训练）",
                   "piper_pick": "Piper X 机械臂抓方块（一层策略，带已训练的权重）",
                   "planner_executor": "四足机器狗走到目标点（两层：planner 定目标、executor 行走）"}


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
    # A template's own namespace (piper_pick/..., piper_pick__...) becomes the new project's, not nested in it.
    own = lambda value, separator: value.removeprefix(source.name + separator)
    return {
        **{value: f"{name}/{own(value, '/')}" for value in registrations["register_isaac_assembly_env"]},
        **{value: f"{name}__{own(value, '__')}" for value in registrations["register_model"]},
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


# "blank": go2_goal's recipe and entry files with an empty scene (ground and light) and a policy left to write.
BLANK_SCENE = """schema_version: 1
name: world
simulation:
  dt: 0.005
  render_interval: 4
entities:
- name: ground
  scope: global
  factory: ground_plane
  prim_name: GroundPlane
- name: sky_light
  scope: global
  factory: clear_sky
  prim_name: skyLight
policy:
  layers:
    policy:
      model: .policy:Policy
      period: 0.02
      inputs: []
  decode: .policy:decode_action
"""
BLANK_POLICY = '''"""The policy: a PyTorch module that reads ``obs`` and returns actions (scene.yaml ``policy``).

Add a robot to the scene and choose what the policy reads (editor: 策略 › 添加输入); saving refreshes observation.py
with the types. Then write:

- ``Policy(config, actions)``: ``config`` is train.yaml ``actor.model`` (put sizes and your own settings there,
  read them as ``config.<name>``); ``actions`` is the number of joints the policy drives.
- ``Policy.forward(obs)``: the actions [B, actions] from the inputs (``obs.robot.joint_position`` ...); training adds
  the action noise and a value network itself.
- ``decode_action(action, obs)``: the joint targets of every driven robot, in the order of its joint_position, in real
  units (rad, m, N·m).
"""
from torch import nn
from shared.policy import flat, mlp
from .observation import Action, PolicyObs, Targets


class Policy(nn.Module):
    def __init__(self, config, actions):
        """``config``: train.yaml ``actor.model`` (hidden_dims, and any setting you add there)."""
        super().__init__()
        self.net = mlp(*config.hidden_dims, actions)

    def forward(self, obs: PolicyObs) -> Action:
        # For example: return self.net(flat(obs.robot.joint_position, obs.robot.joint_velocity))
        raise NotImplementedError('policy.py: write what the network reads (obs.<entity>.<quantity>)')


def decode_action(action: Action, obs: PolicyObs) -> Targets:
    # For example, offsets from a standing pose you choose (a constant of the policy, the same on the real robot):
    # return {'robot': action.new_tensor(STANCE) + .25 * action}
    raise NotImplementedError("policy.py: turn the actions into each robot's joint targets")
'''
BLANK_TASK = '''"""The task: reward, success, failure and metrics, in plain Torch (see observation.py SceneInfo for what it reads).

``evaluate(obs)`` runs once per decision step for all environments and returns [B] tensors: ``reward`` (default 0),
``success`` / ``failure`` (default False; either ends the episode) and ``metrics``. Relations such as inside / on_top /
is_open are in shared.predicates.
"""
from .observation import SceneInfo, TaskResult


def evaluate(obs: SceneInfo) -> TaskResult:
    return {}
'''
BLANK_README = """# {name}

从空白模板新建：场景里只有地面和灯光。

1. 用编辑器（`bash run.sh edit {name}`）添加机器人和物体，在“策略”里选择策略读哪些量。
2. 在 `policy.py` 写网络（`Policy.forward`）和 `decode_action`，在 `task.py` 写奖励、成功与失败。
3. `bash run.sh check {name}`、`bash run.sh smoke {name}`，再 `bash run.sh train {name}`。

## 当前内容

（在这里记录本项目的选择：机器人、输入、动作含义、奖励、已验证到哪一步。）
"""


def create_project(name, template="go2_goal", policy=None, dry_run=False, destination=None, overrides=None):
    if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
        raise ValueError("Use a Python package name, for example my_robot")
    if template == "blank":
        return create_project(name, "go2_goal", dry_run=dry_run, destination=destination, overrides={
            "scene.yaml": BLANK_SCENE, "policy.py": BLANK_POLICY, "task.py": BLANK_TASK,
            "README.md": BLANK_README.format(name=name)})
    source = PROJECTS / template
    if not template.isidentifier() or source not in project_directories():
        raise ValueError(f"Unknown template: {template}")
    target = (Path(destination) if destination else PROJECTS) / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    if policy is not None and (source / "scene.yaml").is_file() and OmegaConf.load(source / "scene.yaml").keys() & {"control", "policy"}:
        raise ValueError("Projects with a policy define their models in policy.py; --policy overrides are not supported")
    if policy is not None and template == "franka_stack_vla":
        raise ValueError("MLP policy options do not apply to the GR00T template")
    # A scene project that inherits its backend from configs/state_ppo.yaml keeps running there,
    # but the copy names it, so a new project never runs on an implicit default.
    inherited = None
    if (source / "scene.yaml").is_file() and not declares_backend(source):
        inherited = OmegaConf.select(OmegaConf.load(PROJECTS.parent / "configs/state_ppo.yaml"), "env.train.init_params.backend")
    registration_names = _assembly_registration_names(source, name)
    files = {}
    # Weights a template's fixed components load (piper_pick's pretrained executor) and the project's own imported
    # assets (projects/<name>/assets/) come along unchanged.
    binaries = {path.relative_to(source): path for folder in ("weights", "assets")
                for path in sorted((source / folder).rglob("*")) if path.is_file()}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".yaml", ".md", ".rst", ".txt"):
            continue
        text = re.sub(rf"\bprojects\.{re.escape(template)}(?=\b)", f"projects.{name}", path.read_text())
        relative = path.relative_to(source)
        if path.suffix in ('.md', '.rst'):
            text = re.sub(rf'\b{re.escape(template)}\b', name, text)
        if path.suffix == ".yaml":
            # The project's own assets are referenced from the workspace: projects/<template>/assets/...
            text = text.replace(f"projects/{template}/", f"projects/{name}/")
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
    for relative, text in (overrides or {}).items():
        if relative.endswith(".py"):
            ast.parse(text)
        files[Path(relative)] = text
    if dry_run:
        return target, tuple(map(str, [*files, *binaries]))
    if inherited:
        print(f"{template} inherits its backend; {name}/train.yaml now names it: "
              f"env.train.init_params.backend: {inherited} (new projects use isaaclab3)")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=target.parent))
    try:
        for relative, text in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for relative, original in binaries.items():
            if relative not in files:
                (staging / relative).parent.mkdir(parents=True, exist_ok=True)
                if original.suffix == ".json":  # the shelf library and import records name their files
                    (staging / relative).write_text(original.read_text().replace(f"projects/{template}/", f"projects/{name}/"))
                else:
                    shutil.copyfile(original, staging / relative)
        if (staging / 'task.py').is_file():
            from shared.assembly.object_schema import generate
            from shared.assembly.project import project_registry
            # The same static view as check: asset names of every backend, not just the default one.
            factories, terms, _ = project_registry(backend=None)
            generate(staging / 'scene.yaml', catalog_directory=source, factories=factories, terms=terms)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return target, tuple(map(str, [*files, *binaries]))


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?")
    parser.add_argument("--template", default="go2_goal",
                        help="Maintained: " + ", ".join(MAINTAINED_TEMPLATES) + "; other projects can be copied explicitly")
    parser.add_argument("--policy", choices=("scratch", "upstream"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(arguments)
    if args.interactive:
        args.name = args.name or input("Project name: ").strip()
        choices = ", ".join(MAINTAINED_TEMPLATES)
        args.template = input(f"Template ({choices}) [go2_goal]: ").strip() or "go2_goal"
        scene = PROJECTS / args.template / "scene.yaml"
        has_components = scene.is_file() and bool(OmegaConf.load(scene).keys() & {"control", "policy"})
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
