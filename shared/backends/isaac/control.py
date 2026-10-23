"""Isaac action-manager binding for the shared planner/executor runtime."""
from functools import lru_cache
from shared.assembly.control import load_factory
from shared.assembly.policy_runtime import ComponentInput as BaseInput, FeedbackChain as BaseChain, validate_targets


class ComponentInput(BaseInput):
    def __init__(self, config, env, links, settings):
        from .readers import bind_reader
        from .robot import robots
        super().__init__(config, env, links, settings,
                         bind_reader=lambda path: bind_reader(env, settings, path), robots=robots(env, settings))


class FeedbackChain(BaseChain):
    def __init__(self, control, dt, env, endpoint_dimension, settings, initial_output=None):
        super().__init__(control, dt, env, endpoint_dimension, settings, initial_output,
                         input_factory=ComponentInput, model_factory=load_factory)


def training_observation(env, source):
    return env.action_manager.get_term('control_chain').chain.observation(source)


@lru_cache(maxsize=1)
def _config_type():
    from isaaclab.managers import ActionTerm, ActionTermCfg
    from isaaclab.utils import configclass

    class ChainAction(ActionTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            env._assembly_signal_factories = cfg.signal_factories
            self.endpoints = {name: endpoint.class_type(endpoint, env) for name, endpoint in cfg.endpoints.items()}
            # Resolve overlaps once, before simulation stepping.
            occupied = set()
            for endpoint in self.endpoints.values():
                ids = endpoint._joint_ids
                indices = range(endpoint._asset.num_joints) if isinstance(ids, slice) else ids
                for index in indices:
                    key = (endpoint.cfg.asset_name, int(index))
                    if key in occupied:
                        raise ValueError('Control targets overlap on the same joint')
                    occupied.add(key)
            dimension = sum(endpoint.action_dim for endpoint in self.endpoints.values())
            self.chain = FeedbackChain(cfg.control, cfg.dt, env, dimension, cfg.settings)

        @property
        def action_dim(self): return self.chain.dimension
        @property
        def raw_actions(self): return self.chain.outputs[self.chain.selected]
        @property
        def processed_actions(self): return self.raw_actions
        def process_actions(self, actions): self.chain.process(actions)
        def apply_actions(self):
            action = self.chain.advance()
            if (self.chain.tick - 1) % self.chain.periods['executor'] == 0:
                targets = self.chain.decode(action, self.chain.readers['executor']())
                validate_targets(targets, self.endpoints, self.chain.env.num_envs, action.device)
                for name, endpoint in self.endpoints.items():
                    endpoint.process_actions(targets[name])
            for endpoint in self.endpoints.values():
                endpoint.apply_actions()
        def reset(self, env_ids=None):
            self.chain.reset(env_ids)
            for endpoint in self.endpoints.values():
                endpoint.reset(env_ids)

    @configclass
    class ChainCfg(ActionTermCfg):
        class_type: type = ChainAction
        control: dict = None
        dt: float = .005
        endpoints: dict = None
        settings: dict = None
        signal_factories: dict = None
    return ChainCfg


def chain_config(control, dt, settings, signal_factories=None):
    from isaaclab.envs import mdp
    from .task_config import _ACTION_CLASSES
    endpoints = {}
    for name, binding in control['output'].items():
        parameters = dict(binding)
        mode = parameters.pop('type')
        parameters.update(scale=1.0, offset=0.0, preserve_order=True)
        if mode in ('joint_position', 'joint_velocity'):
            parameters['use_default_offset'] = False
        endpoints[name] = getattr(mdp, _ACTION_CLASSES[mode])(**parameters)
    return _config_type()(asset_name=next(iter(endpoints.values())).asset_name, control=control,
                          dt=dt, endpoints=endpoints, settings=settings, signal_factories=signal_factories or {})




def install_terms(control, dt, settings, registry):
    def native_config():
        from shared.backends.isaac.control import chain_config
        return chain_config(control, dt, settings, registry.get("signals", {}))
    def input_config(source):
        from isaaclab.managers import ObservationTermCfg
        from shared.backends.isaac.control import training_observation
        return ObservationTermCfg(func=training_observation, params={'source': source})
    registry.setdefault('actions', {})['control_chain'] = native_config
    registry.setdefault('observations', {}).update({source: (lambda source=source: input_config(source))
                                                  for source in control['components'][control['train']]['observations']})
