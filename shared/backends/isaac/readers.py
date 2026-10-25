"""Isaac-specific resolution of native scene tensors."""
from shared.assembly.entity_information import parse_signal
from shared.assembly.joint_order import JOINT_FIELDS
from shared.assembly.readers import bind_common


def bind_reader(env, settings, path):
    from .entity_observations import build_signal
    from .robot import joint_index, robots
    from .targets import read_target
    def physical(source):
        term = build_signal(source)
        for parameter in term.params.values():
            if hasattr(parameter, 'resolve'):
                parameter.resolve(env.scene)
        read = lambda: term.func(env, **term.params)
        parsed = parse_signal(source)
        if parsed and parsed[1] in JOINT_FIELDS:
            index = joint_index(env, parsed[0])
            if index is not None:
                return lambda: read()[:, index]
        return read
    return bind_common(env, settings, path, bind_physical=physical,
                       bind_sampled_target=lambda value: lambda: read_target(env, value),
                       robots=robots(env, settings))
