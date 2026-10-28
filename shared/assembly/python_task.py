"""Backend-independent lazy Task evaluation, result validation and per-environment reset.

Besides reward/success/failure/metrics a Task may return ``set``: abstract entity name ->
next (B, 3) position or (B, 7) xyz + wxyz pose. It is the Task's per-step event; the backend's
command term applies it after resets, before observations (``take``).
"""
from importlib import import_module
import torch
from .signals import Signals, Signal


class Runtime:
    def __init__(self, env, config, settings, *, bind_reader, robots=None, module_loader=import_module):
        module = module_loader(config['module'] + '.task')
        schema = module_loader(config['module'] + '.observation')
        self.task = module.Task() if hasattr(module, 'Task') else module
        self.evaluate = self.task.evaluate
        self.info_type = schema.SceneInfo
        self.paths = schema.SCENE_FIELDS
        self.env, self.settings = env, settings
        self.bind_reader, self.robots = bind_reader, robots
        self.signals = Signals()
        self.step = None
        self.result = None
        self.pending = {}

    def read(self, path):
        if path not in self.signals:
            reader = self.bind_reader(path)
            self.signals.register(path, Signal(reader))
        return self.signals.read(path)

    def info(self):
        """The project's SceneInfo view of live scene information."""
        return self.info_type(reader=self.read, paths=self.paths, robots=self.robots)

    def compute(self):
        env = self.env
        if self.step == env.common_step_counter:
            return self.result
        info = self.info()
        with torch.no_grad():
            result = self.evaluate(info)
        if not isinstance(result, dict) or result.keys() - {'reward','success','failure','metrics','set'}:
            raise ValueError('Task.evaluate must return reward/success/failure/metrics/set')
        def tensor(value, name, boolean=False):
            value = torch.as_tensor(value, device=env.device)
            if value.ndim == 0: value = value.expand(env.num_envs)
            if value.shape != (env.num_envs,):
                raise ValueError(f'Task {name} must contain one value per environment, got shape {tuple(value.shape)}')
            if value.is_floating_point() and not torch.isfinite(value).all():
                envs = torch.nonzero(~torch.isfinite(value)).flatten()
                # Name the metric terms that are also non-finite, e.g. one reward_* component.
                terms = [key for key, item in result.get('metrics', {}).items()
                         if torch.is_tensor(item) and item.is_floating_point() and not torch.isfinite(item).all()]
                raise ValueError(f'Task {name} is non-finite in {envs.numel()} environments '
                                 f'(first {envs[:8].tolist()}, values {value[envs[:8]].tolist()})'
                                 + (f'; non-finite metrics: {terms}' if terms else ''))
            if boolean and value.dtype != torch.bool:
                raise TypeError(f'Task {name} must be boolean')
            return value.detach().clone()
        pending = self._settings(result.get('set') or {})
        result = {'reward': tensor(result.get('reward', 0.), 'reward'),
                  'success': tensor(result.get('success', False), 'success', True),
                  'failure': tensor(result.get('failure', False), 'failure', True),
                  'metrics': {key: tensor(value, 'metrics.' + key) for key, value in result.get('metrics', {}).items()}}
        self.pending = pending
        env.extras['task_metrics'] = {key: {'value': value.clone(), 'valid': torch.ones_like(value,dtype=torch.bool)} for key,value in result['metrics'].items()}
        env.extras.setdefault('metrics', {}).update(success=result['success'].clone(), failure=result['failure'].clone())
        self.result, self.step = result, env.common_step_counter
        return result

    def _settings(self, values):
        """Validate the ``set`` entries: commanded abstract entities, one pose row per environment."""
        from .abstract import abstract_entities, commanded
        env = self.env
        if not isinstance(values, dict):
            raise ValueError('Task set must map abstract entity names to poses')
        held = commanded(self.settings)
        result = {}
        for name, value in values.items():
            if name not in held:
                node = abstract_entities(self.settings).get(name)
                reason = ('it is not an abstract entity' if node is None else
                          'it follows its mounted entity' if node.mount is not None else
                          'it is a reset region, which stays static')
                raise ValueError(f'Task set {name!r}: cannot move it, {reason}')
            if (not torch.is_tensor(value) or not value.is_floating_point()
                    or value.shape not in ((env.num_envs, 3), (env.num_envs, 7))):
                raise ValueError(f'Task set {name!r} must be a float tensor of shape ({env.num_envs}, 3) position '
                                 f'or ({env.num_envs}, 7) xyz + wxyz, got {getattr(value, "shape", type(value).__name__)}')
            if value.device != torch.empty(0, device=env.device).device:
                raise ValueError(f'Task set {name!r} must be on {env.device}, got {value.device}')
            if not torch.isfinite(value).all():
                envs = torch.nonzero(~torch.isfinite(value).all(-1)).flatten()
                raise ValueError(f'Task set {name!r} is non-finite in environments {envs[:8].tolist()}')
            if value.shape[1] == 7 and (value[:, 3:7].norm(dim=-1) < 1e-6).any():
                raise ValueError(f'Task set {name!r} has a zero quaternion')
            result[name] = value.detach().clone()
        return result

    def take(self, name):
        """This step's Task-set pose of one abstract entity, once; None when the Task did not move it."""
        if self.step != self.env.common_step_counter:
            return None
        return self.pending.pop(name, None)

    def reset(self, env_ids):
        from shared.assembly.task_memory import TaskMemory
        for value in vars(self.task).values():
            if isinstance(value, TaskMemory): value.reset(env_ids)
        reset = getattr(self.task, 'reset', None)
        if reset is not None: reset(env_ids)
        # Keep this step's terminal result for other native managers. The next
        # decision step computes once from the new state after reset.


