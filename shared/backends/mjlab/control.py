"""Native mjlab action terms around the shared policy runtime."""
from functools import lru_cache
import torch
from shared.assembly.policy_runtime import ComponentInput, FeedbackChain, validate_targets
from .readers import bind_reader
from .robot import robots


def component_input(config, env, links, settings):
    return ComponentInput(config, env, links, settings,
                          bind_reader=lambda path: bind_reader(env, settings, path), robots=robots(env, settings))


def _actuator_types(module, *names):
    """Actuator classes present in this mjlab version (names changed after 1.1)."""
    found = tuple(getattr(module, name) for name in names if hasattr(module, name))
    if not found:
        raise ValueError(f'mjlab.actuator has none of {names}')
    return found


class Endpoint:
    def __init__(self, binding, env):
        from mjlab import actuator
        self.entity = env.scene[binding['asset_name']]
        indices, names = self.entity.find_joints(binding['joint_names'], preserve_order=True)
        if not indices:
            raise ValueError(f'Control target selected no joints: {binding}')
        mode = binding['type']
        supported = {
            'joint_position': _actuator_types(
                actuator, 'BuiltinPositionActuator', 'XmlPositionActuator',
                'DcMotorActuator', 'IdealPdActuator'),
            'joint_velocity': _actuator_types(
                actuator, 'BuiltinVelocityActuator', 'XmlVelocityActuator', 'IdealPdActuator'),
            'joint_effort': _actuator_types(
                actuator, 'BuiltinMotorActuator', 'XmlMotorActuator', 'IdealPdActuator'),
        }
        # A DelayedActuator (mjlab's delay wrapper) controls what its base actuator does.
        covered = {name for act in self.entity.actuators
                   if isinstance(getattr(act, 'base_actuator', act), supported[mode])
                   and act.cfg.transmission_type == 'joint' for name in act.target_names}
        if not set(names) <= covered:
            raise ValueError(f'{binding["asset_name"]}: native actuators do not support {mode} for {sorted(set(names)-covered)}')
        self.indices = indices
        self._joint_names = names
        self.ids = torch.tensor(indices, dtype=torch.long, device=env.device)
        self.action_dim = len(indices)
        self.write = getattr(self.entity, 'set_' + mode + '_target')
        self.clamp = mode == 'joint_position'
        # Isaac PD actuators under effort control add the effort to a PD hold of the default pose
        # (reset writes it as their position target); mjlab reset clears the targets to zero.
        ideal_pd = getattr(actuator, 'IdealPdActuator', ())
        self.hold = mode == 'joint_effort' and bool(ideal_pd) and any(
            isinstance(getattr(act, 'base_actuator', act), ideal_pd) and set(act.target_names) & set(names)
            for act in self.entity.actuators)

    def apply(self, value):
        if self.clamp:
            # MuJoCo joint limits are soft: a position target beyond the range keeps the actuator
            # pushing through the limit (a 25 g finger sinks 1-2 cm at 10 N). PhysX limits are hard,
            # so clamp the target to the range; unlimited joints report an empty [0, 0] range.
            lower, upper = self.entity.data.joint_pos_limits[:, self.ids].unbind(-1)
            limited = lower < upper
            value = torch.where(limited, torch.minimum(torch.maximum(value, lower), upper), value)
        if self.hold:
            self.entity.set_joint_position_target(self.entity.data.default_joint_pos[:, self.ids], joint_ids=self.ids)
            self.entity.set_joint_velocity_target(torch.zeros_like(value), joint_ids=self.ids)
        self.write(value, joint_ids=self.ids)


@lru_cache(maxsize=1)
def _config_type():
    from dataclasses import dataclass
    from mjlab.managers import ActionTerm, ActionTermCfg
    class ChainAction(ActionTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.endpoints = {name: Endpoint(binding, env) for name, binding in cfg.control['output'].items()}
            occupied = set()
            for name, endpoint in self.endpoints.items():
                entity = cfg.control['output'][name]['asset_name']
                for index in endpoint.indices:
                    if (entity, index) in occupied:
                        raise ValueError('Control targets overlap on the same joint')
                    occupied.add((entity, index))
            self.chain = FeedbackChain(cfg.control, cfg.dt, env,
                sum(endpoint.action_dim for endpoint in self.endpoints.values()), cfg.settings,
                input_factory=component_input)
        @property
        def action_dim(self): return self.chain.dimension
        @property
        def raw_action(self): return self.chain.outputs[self.chain.selected]
        def process_actions(self, actions): self.chain.process(actions)
        def apply_actions(self):
            action = self.chain.advance()
            if (self.chain.tick - 1) % self.chain.periods['executor'] == 0:
                targets = self.chain.decode(action, self.chain.readers['executor']())
                validate_targets(targets, self.endpoints, self.num_envs, action.device)
                for name, endpoint in self.endpoints.items():
                    endpoint.apply(targets[name])
        def reset(self, env_ids=None): self.chain.reset(env_ids)
    @dataclass(kw_only=True)
    class ChainCfg(ActionTermCfg):
        control: dict
        dt: float
        settings: dict
        def build(self, env): return ChainAction(self, env)
    return ChainCfg


def training_observation(env, source):
    return env._assembly_control_chain.observation(source)


def install_terms(control, dt, settings, registry):
    def native_config():
        return _config_type()(entity_name=next(iter(control['output'].values()))['asset_name'],
                              control=control, dt=dt, settings=settings)
    def input_config(source):
        from mjlab.managers import ObservationTermCfg
        return ObservationTermCfg(func=training_observation, params={'source': source})
    registry.setdefault('actions', {})['control_chain'] = native_config
    registry.setdefault('observations', {}).update({source: (lambda source=source: input_config(source))
        for source in control['components'][control['train']]['observations']})
