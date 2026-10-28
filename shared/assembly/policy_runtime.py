"""Shared planner/executor scheduling and tensor contracts; native managers own stepping."""
import torch
from .control import load_factory, component_observations


class ComponentInput:
    """Read named native tensors without flattening, encoding or adding noise.

    Observations listed in ``history`` are kept as per-environment windows [B, T, ...]
    (oldest -> newest, see shared/assembly/history.py). A window advances once per decision
    of this component; reads in between (decode, initial, terminal snapshots restored
    afterwards) return the stored window.
    """
    def __init__(self, config, env, links, settings, *, bind_reader, robots=None):
        from shared.assembly.signals import Signals, Signal, Observation
        from shared.assembly.history import HistoryWindow, validate_history
        # Generated <Role>Obs of the project, so fixed components and decode get the same typed
        # view (and robot methods) as written in policy.py; plain views without a project.
        self.view_type, self.robots = config.get('_view'), robots
        self.signals = Signals()
        self.selection = Observation({entry['source']: entry['source'] for entry in component_observations(config['observations'])})
        for source in self.selection.fields:
            if source in links:
                reader = links[source]
            else:
                reader = bind_reader(source)
            self.signals.register(source, Signal(reader))
        current = self.selection.read(self.signals)
        history = validate_history(config.get('history'), self.selection.fields)
        self.windows = {source: HistoryWindow(current[source], length) for source, length in history.items()}
        self.pending = {source: torch.ones_like(window.count, dtype=torch.bool) for source, window in self.windows.items()}
        self.last = {key: self.windows[key].buffer if key in self.windows else torch.zeros_like(value)
                     for key, value in current.items()}

    def read(self, source, rows=None):
        """Current value; for a windowed source push it into ``rows`` (None: only show the window)."""
        value = self.signals.read(source)
        window = self.windows.get(source)
        if window is None:
            self.last[source].copy_(value)
        elif rows is None:
            window.peek(value)
        else:
            window.push(value, rows)
        return self.last[source]

    def observe(self, source):
        """Native observation term read: windows advance once per decision, however often the
        manager computes (repeated or partial-reset computes push only rows still pending)."""
        if source not in self.windows:
            return self.read(source)
        rows = self.pending[source].clone()
        self.pending[source].zero_()
        return self.read(source, rows)

    def mark(self, env_ids=None):
        for pending in self.pending.values():
            pending[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, rows=None):
        if not isinstance(self.view_type, type):
            from shared.assembly.objects import ObjectView
            self.view_type = load_factory(self.view_type) if self.view_type else ObjectView
        return self.view_type({source: self.read(source, rows) for source in self.selection.fields}, robots=self.robots)

    def reset(self, env_ids):
        for key, value in self.last.items():
            if key in self.windows:
                self.windows[key].reset(env_ids)
            else:
                value[slice(None) if env_ids is None else env_ids] = 0
        self.mark(env_ids)

    def state(self):
        return ({k: v.clone() for k, v in self.last.items()},
                {k: (w.count.clone(), self.pending[k].clone()) for k, w in self.windows.items()})

    def restore(self, state):
        inputs, windows = state
        for field, value in inputs.items():
            self.last[field].copy_(value)
        for field, (count, pending) in windows.items():
            self.windows[field].count.copy_(count)
            self.pending[field].copy_(pending)


class FeedbackChain:
    def __init__(self, control, dt, env, endpoint_dimension, settings, initial_output=None, *, input_factory, model_factory=load_factory):
        self.env, self.control, self.dt = env, control, dt
        self.initial_output = initial_output
        self.selected = control['train']
        self.outputs, self.previous, self.readers, self.models, self.periods = {}, {}, {}, {}, {}
        env._assembly_control_chain = self
        self.tick = 0
        self.planner_due = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        for name, config in control['components'].items():
            dimension = endpoint_dimension if name == 'executor' and config['dimension'] == 'auto' else config['dimension']
            if dimension == 'auto':
                raise ValueError('Planner output dimension must be explicit for object inputs')
            self.periods[name] = round(config['period'] / dt)
            self.outputs[name] = torch.zeros(env.num_envs, 0 if dimension == 'auto' else dimension, device=env.device)
            if name == 'executor' and initial_output is not None:
                self.outputs[name].copy_(initial_output)
            self.previous[name] = self.outputs[name].clone()
            links = {name + '.output': lambda name=name: self.outputs[name],
                     name + '.previous_output': lambda name=name: self.previous[name]}
            if name == 'executor' and 'planner' in self.outputs:
                links['planner.output'] = lambda: self.outputs['planner']
            self.readers[name] = input_factory(config, env, links, settings)
            if name != self.selected:
                model = model_factory(config['model'])(**config.get('parameters', {}))
                if not isinstance(model, torch.nn.Module):
                    raise TypeError('Fixed model must be a torch.nn.Module')
                self.models[name] = model.to(env.device).requires_grad_(False).eval()
        self.dimension = self.outputs[self.selected].shape[1]
        executor = control['components']['executor']
        self.decode = model_factory(executor['decode'])
        if 'initial' in executor:
            self.initial_output = model_factory(executor['initial'])(self.readers['executor']()).clone()
            if self.initial_output.shape != self.outputs['executor'].shape:
                raise ValueError('executor.initial has incorrect shape')
            self.outputs['executor'].copy_(self.initial_output)
            self.previous['executor'].copy_(self.initial_output)


    def _infer(self, name, rows=None):
        # A fixed component's decision advances its input windows for the deciding rows.
        value = self.models[name](self.readers[name](True if rows is None else rows))
        if not isinstance(value, torch.Tensor) or value.shape != self.outputs[name].shape:
            raise ValueError(f'{name} returned an invalid output shape or non-finite values')
        torch._assert_async(torch.isfinite(value).all(), f'Non-finite {name} action')
        return value

    def observation(self, source):
        with torch.no_grad():
            if self.selected == 'executor' and 'planner' in self.models:
                step = self.env.common_step_counter
                update = self.planner_due <= step
                if update.any():
                    previous_input = {k: v.clone() for k, v in self.readers['planner'].last.items()}
                    value = self._infer('planner', update)
                    for key, cached_input in previous_input.items():
                        self.readers['planner'].last[key][~update] = cached_input[~update]
                    self.previous['planner'][update] = self.outputs['planner'][update]
                    self.outputs['planner'][update] = value[update]
                    self.planner_due[update] = step + self.periods['planner'] // self.periods['executor']
            return self.readers[self.selected].observe(source)

    def process(self, action):
        if action.shape != self.outputs[self.selected].shape:
            raise ValueError('Control input has wrong shape or non-finite values')
        torch._assert_async(torch.isfinite(action).all(), 'Non-finite control action')
        self.previous[self.selected].copy_(self.outputs[self.selected])
        self.outputs[self.selected].copy_(action)
        # Decision boundary: the next native observation compute advances the trained role's windows.
        self.readers[self.selected].mark()

    def advance(self):
        with torch.no_grad():
            if self.selected == 'planner' and self.tick % self.periods['executor'] == 0:
                value = self._infer('executor')
                self.previous['executor'].copy_(self.outputs['executor'])
                self.outputs['executor'].copy_(value)
        self.tick += 1
        return self.outputs['executor']

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in self.outputs:
            self.outputs[name][ids] = 0
            self.previous[name][ids] = 0
            if name == 'executor' and self.initial_output is not None:
                self.outputs[name][ids] = self.initial_output[ids]
                self.previous[name][ids] = self.initial_output[ids]
            self.readers[name].reset(env_ids)
        self.planner_due[ids] = -1

    def snapshot_state(self):
        # planner_due stays last; component inputs include window contents, counts and pending rows.
        return ({k: v.clone() for k, v in self.outputs.items()},
                {k: v.clone() for k, v in self.previous.items()},
                {k: v.state() for k, v in self.readers.items()}, self.planner_due.clone())

    def restore_state(self, state):
        outputs, previous, inputs, due = state
        for key in outputs:
            self.outputs[key].copy_(outputs[key])
            self.previous[key].copy_(previous[key])
            self.readers[key].restore(inputs[key])
        self.planner_due.copy_(due)

    def read(self, role, field):
        if field == 'input':
            return self.readers[role].last
        return (self.outputs if field == 'output' else self.previous)[role]


def validate_targets(targets, endpoints, num_envs, device):
    if not isinstance(targets, dict) or targets.keys() != endpoints.keys():
        raise ValueError('executor.decode must return exactly the configured target names')
    for name, endpoint in endpoints.items():
        value = targets[name]
        if (not isinstance(value, torch.Tensor) or value.shape != (num_envs, endpoint.action_dim)
                or not value.is_floating_point() or value.device != device):
            raise ValueError(f'Target {name}: expected floating tensor [{num_envs}, {endpoint.action_dim}] on {device}')
        # No host scalar extraction or CPU transfer on the GPU control path.
        torch._assert_async(torch.isfinite(value).all(), f'Non-finite control target: {name}')
