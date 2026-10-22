"""Isaac-specific resolution of native scene tensors."""
from shared.assembly.readers import bind_common


def bind_reader(env, settings, path):
    from .entity_observations import build_signal
    from .targets import read_target
    def physical(source):
        term = build_signal(source)
        for parameter in term.params.values():
            if hasattr(parameter, 'resolve'):
                parameter.resolve(env.scene)
        return lambda: term.func(env, **term.params)
    return bind_common(env, settings, path, bind_physical=physical,
                       bind_sampled_target=lambda value: lambda: read_target(env, value))
