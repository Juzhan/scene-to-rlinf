"""Backend-independent lazy Task evaluation, result validation and per-environment reset."""
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

    def read(self, path):
        if path not in self.signals:
            reader = self.bind_reader(path)
            self.signals.register(path, Signal(reader))
        return self.signals.read(path)

    def compute(self):
        env = self.env
        if self.step == env.common_step_counter:
            return self.result
        info = self.info_type(reader=self.read, paths=self.paths, robots=self.robots)
        with torch.no_grad():
            result = self.evaluate(info)
        if not isinstance(result, dict) or result.keys() - {'reward','success','failure','metrics'}:
            raise ValueError('Task.evaluate must return reward/success/failure/metrics')
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
        result = {'reward': tensor(result.get('reward', 0.), 'reward'),
                  'success': tensor(result.get('success', False), 'success', True),
                  'failure': tensor(result.get('failure', False), 'failure', True),
                  'metrics': {key: tensor(value, 'metrics.' + key) for key, value in result.get('metrics', {}).items()}}
        env.extras['task_metrics'] = {key: {'value': value.clone(), 'valid': torch.ones_like(value,dtype=torch.bool)} for key,value in result['metrics'].items()}
        env.extras.setdefault('metrics', {}).update(success=result['success'].clone(), failure=result['failure'].clone())
        self.result, self.step = result, env.common_step_counter
        return result

    def reset(self, env_ids):
        from shared.assembly.task_memory import TaskMemory
        for value in vars(self.task).values():
            if isinstance(value, TaskMemory): value.reset(env_ids)
        reset = getattr(self.task, 'reset', None)
        if reset is not None: reset(env_ids)
        # Keep this step's terminal result for other native managers. The next
        # decision step computes once from the new state after reset.


