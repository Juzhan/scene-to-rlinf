"""Backend-independent lazy Task evaluation, result validation and per-environment reset."""
from importlib import import_module
import torch
from .signals import Signals, Signal


class Runtime:
    def __init__(self, env, config, settings, *, bind_reader, module_loader=import_module):
        module = module_loader(config['module'] + '.task')
        schema = module_loader(config['module'] + '.observation')
        self.task = module.Task() if hasattr(module, 'Task') else module
        self.evaluate = self.task.evaluate
        self.info_type = schema.SceneInfo
        self.paths = schema.SCENE_FIELDS
        self.env, self.settings = env, settings
        self.bind_reader = bind_reader
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
        info = self.info_type(reader=self.read, paths=self.paths)
        with torch.no_grad():
            result = self.evaluate(info)
        if not isinstance(result, dict) or result.keys() - {'reward','success','failure','metrics'}:
            raise ValueError('Task.evaluate must return reward/success/failure/metrics')
        def tensor(value, name, boolean=False):
            value = torch.as_tensor(value, device=env.device)
            if value.ndim == 0: value = value.expand(env.num_envs)
            if value.shape != (env.num_envs,) or (value.is_floating_point() and not torch.isfinite(value).all()):
                raise ValueError(f'Task {name} must contain one finite value per environment')
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


