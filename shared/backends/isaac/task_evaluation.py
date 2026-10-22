"""Bridge task expressions to native Lab terms, using the scene signal catalog."""
from shared.assembly.signals import Signal, Signals
from shared.assembly.legacy.task import TaskEvaluator, validate_evaluation, reward_entries
from .entity_observations import build_signal, parse_signal, signal_information, command_information
from shared.assembly.information import Information
from shared.assembly.legacy.task_types import infer_evaluation
from .targets import target_catalog, read_target


ACTION_SIGNALS = {"policy.action": "action", "policy.previous_action": "prev_action"}


def validate_sources(config, signal_factories=None, settings=None):
    report = validate_evaluation(config)
    sources = {}
    settings = settings or {}
    targets = target_catalog(settings)
    task = settings.get("task", {})
    native_defaults = {
        "lift": {"object_pose": {"type": "UniformPoseCommandCfg", "asset_name": "robot"}},
        "velocity": {"base_velocity": {"type": "UniformVelocityCommandCfg", "asset_name": "robot"}},
    }
    for path in report["signals"]:
        if path in (signal_factories or {}):
            sources[path] = getattr(signal_factories[path], "information", Information())
            continue
        if path in targets:
            sources[path] = targets[path][1]
        elif path.split('.')[0] in settings.get('control', {}).get('components', {}) and path.split('.')[-1] in ('input', 'output', 'previous_output'):
            sources[path] = Information('number', (None,), 'float', axes=('component',))
        elif path in ACTION_SIGNALS:
            sources[path] = Information("number", (None,), "float", axes=("action",))
        elif path == "simulation.gravity" or parse_signal(path):
            sources[path] = signal_information(path)
        elif path.startswith("command__") and len(path) > 9:
            name = path[9:]
            default = native_defaults.get(task.get("preset"), {}).get(name, {})
            changes = (task.get("commands") or {}).get(name, {}) or {}
            sources[path] = command_information({**default, **changes})
        else:
            raise ValueError(f"Unknown task signal {path!r}: select a scene property or abstract entity field")
    types = infer_evaluation(config, sources)
    report["information"] = {name: info.describe() for name, info in types["nodes"].items()}
    report["sources"] = {name: info.describe() for name, info in sources.items()}
    report["untyped_signals"] = types["untyped_signals"]
    report["untyped_variables"] = types["untyped_variables"]
    return report


def _runtime(env, config, signal_factories, targets=None):
    runtime = getattr(env, "_assembly_task_evaluator", None)
    if runtime is not None:
        return runtime
    signals = Signals()
    for path in validate_evaluation(config)["signals"]:
        if path in signal_factories:
            signal = signal_factories[path](env)
            if not isinstance(signal, Signal):
                raise TypeError(f"Task signal factory {path!r} must return Signal")
            signals.register(path, signal)
        elif path in (targets or {}):
            value, info = targets[path]
            signals.register(path, Signal(lambda value=value: read_target(env, value), info=info, source="authored target"))
        elif path.split('.')[0] in ('planner', 'executor') and path.split('.')[-1] in ('input', 'output', 'previous_output'):
            role, field = path.split('.')
            signals.register(path, Signal(lambda role=role, field=field: env._assembly_control_chain.read(role, field),
                category='policy', info=Information('number', (None,), 'float'), source='component'))
        elif path in ACTION_SIGNALS:
            signals.register(path, Signal(lambda path=path: getattr(env.action_manager, ACTION_SIGNALS[path]),
                info=Information("number", (env.action_manager.total_action_dim,), "float", axes=("action",)), source="policy"))
        elif path.startswith("command__"):
            name = path[len("command__"):]
            info = command_information(env.command_manager.get_term(name).cfg)
            signals.register(path, Signal(lambda name=name: env.command_manager.get_command(name), category="command", info=info, source="command"))
        else:
            term = build_signal(path)
            for parameter in term.params.values():
                if hasattr(parameter, "resolve"):
                    parameter.resolve(env.scene)
            info = signal_information(path)
            key, field = parse_signal(path) or (None, None)
            if field in ("joint_position", "joint_velocity", "joint_effort", "default_joint_position", "default_joint_velocity"):
                from dataclasses import replace
                names = tuple(env.scene[key].joint_names)
                info = replace(info, shape=(len(names),), labels=names)
            signals.register(path, Signal(lambda term=term: term.func(env, **term.params),
                info=info, entity=key, source="simulation"))
    runtime = TaskEvaluator(config, signals, num_envs=env.num_envs, device=env.device)
    from shared.assembly.legacy.task import physics_variables
    fast = physics_variables(config)
    if fast:
        runtime.physics_runtime = TaskEvaluator({'variables': fast}, signals, num_envs=env.num_envs, device=env.device)
        for name in fast:
            runtime.variable_readers[name] = lambda name=name: runtime.physics_runtime.read_variable(
                name, dt=env.physics_dt, step=env._sim_step_counter)
    setattr(env, "_assembly_task_evaluator", runtime)
    return runtime


def evaluate_term(env, config, kind, name="", signal_factories=None, targets=None):
    runtime = _runtime(env, config, signal_factories or {}, targets)
    result = runtime.evaluate(dt=env.step_dt, step=env.common_step_counter)
    # Values and validity remain per environment, including terminal samples.
    # They are not observations and never get appended to policy input.
    env.extras["task_metrics"] = {
        key: {"value": reading.value.clone(), "valid": reading.valid.clone()}
        for key, reading in result.metrics.items()
    }
    env.extras.setdefault("metrics", {}).update(
        success=result.success.clone(), failure=result.failure.clone())
    if kind == "reward":
        return result.rewards[name]
    return getattr(result, kind)


def reset_evaluation(env, env_ids):
    runtime = getattr(env, "_assembly_task_evaluator", None)
    if runtime is not None:
        runtime.reset(env_ids)


def attach_evaluation(output, config, signal_factories=None, settings=None):
    from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
    validate_sources(config, signal_factories, settings)
    parameters = {"config": config, "signal_factories": signal_factories or {},
                  "targets": target_catalog(settings or {})}
    additions = {
        "terminations": {
            "task_success": TerminationTermCfg(func=evaluate_term, params={**parameters, "kind": "success"}),
            "task_failure": TerminationTermCfg(func=evaluate_term, params={**parameters, "kind": "failure"}),
        },
        "events": {"task_evaluation_reset": EventTermCfg(func=reset_evaluation, mode="reset")},
        "rewards": {"task_" + name: RewardTermCfg(func=evaluate_term, weight=1.0,
                    params={**parameters, "kind": "reward", "name": name})
                    for name in reward_entries(config)},
    }
    for category, terms in additions.items():
        overlap = output[category].keys() & terms.keys()
        if overlap:
            raise ValueError(f"Reserved task evaluation term names: {sorted(overlap)}")
        output[category].update(terms)
