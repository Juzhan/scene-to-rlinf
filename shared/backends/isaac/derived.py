"""Isaac readers for shared ``variables:`` expressions; discovery lives in shared.assembly.legacy.derived."""
from shared.assembly.legacy.derived import variable_information, validate_variables  # noqa: F401 - re-exported
from shared.assembly.legacy.task import evaluation_config


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
