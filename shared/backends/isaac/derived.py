"""Shared variable discovery and reads; observations and Task share one runtime."""
from shared.assembly.information import Information
from shared.assembly.legacy.task import evaluation_config


def validate_variables(settings, signal_factories=None):
    from .task_evaluation import validate_sources
    config = evaluation_config(settings)
    return config, validate_sources(config, signal_factories, settings=settings)


def variable_information(settings, name, signal_factories=None):
    _, report = validate_variables(settings, signal_factories)
    values = dict(report['information'][name])
    values.pop('label', None)
    values.pop('operations', None)
    for key in ('shape', 'axes', 'labels'):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return Information(**values)


def read_variable(env, settings, name, signal_factories=None):
    from .task_evaluation import _runtime
    from .targets import target_catalog
    runtime = getattr(env, '_assembly_task_evaluator', None)
    if runtime is None:
        runtime = _runtime(env, evaluation_config(settings), signal_factories or {}, target_catalog(settings))
    reading = runtime.read_variable(name, dt=env.step_dt, step=env.common_step_counter)
    if not reading.valid.all():
        raise ValueError(f'Computed observation {name!r} is invalid')
    return reading.value


def build_variable(settings, name, signal_factories=None):
    from isaaclab.managers import ObservationTermCfg
    def read(env):
        value = read_variable(env, settings, name, signal_factories)
        return value[:, None] if value.ndim == 1 else value
    return ObservationTermCfg(func=read)
