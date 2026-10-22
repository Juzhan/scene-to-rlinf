"""Native mjlab manager bindings for the same project task.py."""
from shared.assembly.python_task import Runtime as TaskRuntime


class Runtime(TaskRuntime):
    def __init__(self, env, config, settings):
        from .readers import bind_reader
        super().__init__(env, config, settings,
                         bind_reader=lambda path: bind_reader(env, settings, path))


def evaluate(env, config, settings, field):
    runtime = getattr(env, '_assembly_python_task', None)
    if runtime is None:
        runtime = env._assembly_python_task = Runtime(env, config, settings)
    return runtime.compute()[field]


def reset(env, env_ids):
    runtime = getattr(env, '_assembly_python_task', None)
    if runtime is not None:
        runtime.reset(env_ids)


def attach_task(output, config, settings):
    from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
    params = {'config': config, 'settings': settings}
    output['rewards']['project_reward'] = RewardTermCfg(func=evaluate, weight=1., params={**params, 'field': 'reward'})
    for field in ('success', 'failure'):
        output['terminations']['project_' + field] = TerminationTermCfg(func=evaluate, params={**params, 'field': field})
    output['events']['project_task_reset'] = EventTermCfg(func=reset, mode='reset')
