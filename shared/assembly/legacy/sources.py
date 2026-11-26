"""Validate a legacy task/variables expression and type its signals.

Signal names come from the public information catalog, authored targets and the
project's own factories; nothing here imports a simulator SDK, so ``check`` and
the editor type the same expressions the native backends later evaluate.
"""
from shared.assembly.entity_information import command_information, parse_signal, signal_information
from shared.assembly.information import Information
from shared.assembly.legacy.task import validate_evaluation
from shared.assembly.legacy.task_types import infer_evaluation
from shared.assembly.targets import target_catalog

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
