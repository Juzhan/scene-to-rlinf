"""Shared scheduling of the policy layers and tensor contracts; native managers own stepping."""
import contextlib

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


def build_layer(config, *, actions=None, model_config=None, model_factory=load_factory, layered=True):
    """A layer's model from scene.yaml: ``model`` is a class or factory called with ``parameters``; for policy
    layers ``weights`` (a file beside policy.py) is loaded here, and two arguments are passed when the model asks for
    them: ``actions`` (the number of actions decode expects by default: the controlled joints) and ``config`` (the
    recipe's ``actor.model``: train.yaml settings such as hidden_dims, and any key the project adds there)."""
    factory = model_factory(config['model'])
    parameters = dict(config.get('parameters', {}))
    weights = parameters.pop('weights', None) if layered else None
    if layered and actions is not None and accepts(factory, 'actions'):
        parameters['actions'] = actions
    if layered and accepts(factory, 'config'):
        from omegaconf import DictConfig, OmegaConf
        parameters['config'] = model_config if isinstance(model_config, DictConfig) else OmegaConf.create(model_config or {})
    model = factory(**parameters)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f'{config["model"]} must build a torch.nn.Module')
    if weights:
        load_weights(model, weights, config['model'])
    return model


def accepts(factory, name):
    import inspect
    try:
        return name in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def weights_path(weights, reference):
    """``weights`` relative to the directory of the module ``reference`` (module:symbol) names."""
    from importlib.util import find_spec
    from pathlib import Path
    path = Path(weights)
    if not path.is_absolute():
        spec = find_spec(reference.partition(':')[0])
        path = Path(spec.origin).parent / path
    if not path.is_file():
        raise FileNotFoundError(f'{path} is missing (parameters.weights of {reference})')
    return path


def load_weights(model, weights, reference):
    """A state dict of the module itself, or a training checkpoint of it (its ``model.`` keys)."""
    state = torch.load(weights_path(weights, reference), map_location='cpu', weights_only=True)
    if any(key.startswith('model.') for key in state):
        state = {key[6:]: value for key, value in state.items() if key.startswith('model.')}
    model.load_state_dict(state, strict=True)


@contextlib.contextmanager
def same_random_state():
    """Run trial steps without moving the random streams the environment's randomization draws from."""
    devices = range(torch.cuda.device_count()) if torch.cuda.is_available() and torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        yield


def parts_of(value):
    """A layer result as {part: Tensor}: a dict's entries, or {'output': tensor}."""
    if isinstance(value, torch.Tensor):
        return {'output': value}
    if isinstance(value, dict) and value and all(isinstance(key, str) and key.isidentifier() for key in value) \
            and all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        return dict(value)
    raise TypeError('A layer returns a Tensor or a dict of Tensors with Python-name keys')


class FeedbackChain:
    """The policy's layers in order (``control['components']``; the last, ``driver``, drives the robots), one of
    them trained (``selected``, its actions come from RLinf). A fixed layer runs here at its period; what it returns
    sets its output buffers on the first step. Outputs are ``{name: Tensor}`` for a Tensor result, or
    ``{name: {part: Tensor}}`` for a dict; the trained layer's output is its action Tensor."""

    def __init__(self, control, dt, env, endpoint_dimension, settings, initial_output=None, *, input_factory, model_factory=load_factory):
        self.env, self.control, self.dt = env, control, dt
        self.initial_output = initial_output
        self.selected = control['train']
        self.order = list(control['components'])
        self.driver = control.get('driver', self.order[-1])
        self.upstream = self.order[0] if len(self.order) == 2 else None
        layered = control.get('layers') is True
        self.outputs, self.previous, self.readers, self.models, self.periods = {}, {}, {}, {}, {}
        env._assembly_control_chain = self
        self.tick = 0
        self.planner_due = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        for name, config in control['components'].items():
            self.periods[name] = round(config['period'] / dt)
            trained = name == self.selected
            if not layered:  # control: declared sizes
                dimension = endpoint_dimension if name == 'executor' and config['dimension'] == 'auto' else config['dimension']
                if dimension == 'auto':
                    raise ValueError('Planner output dimension must be explicit for object inputs')
                self._allocate(name, torch.zeros(env.num_envs, dimension, device=env.device))
                self.readers[name] = input_factory(config, env, self._links(name), settings)
                if not trained:
                    self.models[name] = self._fixed(build_layer(config, model_factory=model_factory, layered=False))
                continue
            actions = endpoint_dimension if name == self.driver else None
            if trained:
                count = self._action_count(name, config, actions, input_factory, settings, model_factory)
                self._allocate(name, torch.zeros(env.num_envs, count, device=env.device))
                self.readers[name] = input_factory(config, env, self._links(name), settings)
                continue
            self.readers[name] = input_factory(config, env, self._links(name), settings)
            self.models[name] = self._fixed(build_layer(config, actions=actions, model_config=settings.get('_model_config'),
                                                        model_factory=model_factory))
            with torch.no_grad(), same_random_state():  # a trial: a sampler's draws stay the episode's
                value = parts_of(self.models[name](self.readers[name]()))
            if name == self.driver and set(value) != {'output'}:
                raise TypeError(f'{name} drives the robots: its forward returns the action Tensor, not a dict')
            self._allocate(name, value if set(value) != {'output'} else value['output'], zero=True)
        self.dimension = self.outputs[self.selected].shape[1]
        driver = control['components'][self.driver]
        self.decode = model_factory(driver['decode'])
        if 'initial' in driver:
            self.initial_output = model_factory(driver['initial'])(self.readers[self.driver]()).clone()
            if self.initial_output.shape != self.outputs[self.driver].shape:
                raise ValueError(f'{driver["initial"]} returns {tuple(self.initial_output.shape)}, '
                                 f'the actions are {tuple(self.outputs[self.driver].shape)}')
            self.outputs[self.driver].copy_(self.initial_output)
            self.previous[self.driver].copy_(self.initial_output)

    def _fixed(self, model):
        return model.to(self.env.device).requires_grad_(False).eval()

    def _allocate(self, name, value, zero=False):
        if isinstance(value, dict):
            self.outputs[name] = {part: torch.zeros_like(tensor) for part, tensor in value.items()}
            self.previous[name] = {part: torch.zeros_like(tensor) for part, tensor in value.items()}
        else:
            self.outputs[name] = torch.zeros_like(value) if zero else value
            self.previous[name] = self.outputs[name].clone()

    def _links(self, name):
        """Readers of the layer outputs ``name`` may select: its own (previous action) and the other layers'."""
        links = {}
        for other in self.outputs:
            output = self.outputs[other]
            if isinstance(output, dict):
                links.update({f'{other}.{part}': (lambda other=other, part=part: self.outputs[other][part]) for part in output})
                continue
            links[other + '.output'] = lambda other=other: self.outputs[other]
            links[other + '.previous_output'] = lambda other=other: self.previous[other]
        return links

    def _action_count(self, name, config, actions, input_factory, settings, model_factory):
        """How many actions the trained layer outputs: ``actions`` when its model takes that argument, else its
        output size on a trial step. A layer reading its own previous action is tried with as many as the robots
        have joints, then with what it returned, until the two agree."""
        factory = model_factory(config['model'])
        if actions is not None and accepts(factory, 'actions'):
            return actions
        with same_random_state():
            return self._trial_count(name, config, actions, input_factory, settings, model_factory)

    def _trial_count(self, name, config, actions, input_factory, settings, model_factory):
        config = {**config, 'parameters': {k: v for k, v in config.get('parameters', {}).items() if k != 'weights'}}
        width, problem = actions or 0, None
        for _ in range(3):
            # A fresh model each time: lazy layers take their input size from the first call.
            model = build_layer(config, actions=actions, model_config=settings.get('_model_config'),
                                model_factory=model_factory).to(self.env.device)
            self.outputs[name] = torch.zeros(self.env.num_envs, width, device=self.env.device)
            self.previous[name] = self.outputs[name].clone()
            try:
                reader = input_factory(config, self.env, self._links(name), settings)
                with torch.no_grad():
                    value = model(reader())
            except Exception as error:  # noqa: BLE001 - a wrong guess of its own action size
                problem = error
                break
            finally:
                del self.outputs[name], self.previous[name]
            value = value[0] if isinstance(value, tuple) else value  # (mean, value) of an older trained model
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise TypeError(f'{config["model"]}: the trained layer returns its actions as one Tensor [B, actions]')
            if value.shape[1] == width or name + '.output' not in config['observations']:
                return value.shape[1]
            width = value.shape[1]
        raise ValueError(f'{config["model"]}: could not tell how many actions it returns ({problem}); give its '
                         '__init__ an `actions` argument (the number of actions decode_action expects)')

    def _infer(self, name, rows=None):
        # A fixed component's decision advances its input windows for the deciding rows.
        value = self.models[name](self.readers[name](True if rows is None else rows))
        output = self.outputs[name]
        if isinstance(output, dict):
            value = parts_of(value)
            if value.keys() != output.keys() or any(value[k].shape != output[k].shape for k in output):
                raise ValueError(f'{name} returned {[(k, tuple(v.shape)) for k, v in value.items()]}, '
                                 f'expected {[(k, tuple(v.shape)) for k, v in output.items()]}')
            for part, tensor in value.items():
                torch._assert_async(torch.isfinite(tensor).all() if tensor.is_floating_point() else torch.tensor(True, device=tensor.device),
                                    f'Non-finite {name}.{part}')
            return value
        if not isinstance(value, torch.Tensor) or value.shape != output.shape:
            raise ValueError(f'{name} returned an invalid output shape or non-finite values')
        torch._assert_async(torch.isfinite(value).all(), f'Non-finite {name} action')
        return value

    @staticmethod
    def _assign(target, value, rows=None):
        if isinstance(target, dict):
            for part, tensor in target.items():
                if rows is None:
                    tensor.copy_(value[part])
                else:
                    tensor[rows] = value[part][rows]
        elif rows is None:
            target.copy_(value)
        else:
            target[rows] = value[rows]

    def observation(self, source):
        with torch.no_grad():
            upstream = self.upstream
            if upstream is not None and self.selected == self.driver and upstream in self.models:
                step = self.env.common_step_counter
                update = self.planner_due <= step
                if update.any():
                    previous_input = {k: v.clone() for k, v in self.readers[upstream].last.items()}
                    value = self._infer(upstream, update)
                    for key, cached_input in previous_input.items():
                        self.readers[upstream].last[key][~update] = cached_input[~update]
                    self._assign(self.previous[upstream], self.outputs[upstream], update)
                    self._assign(self.outputs[upstream], value, update)
                    self.planner_due[update] = step + self.periods[upstream] // self.periods[self.driver]
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
            if self.selected != self.driver and self.tick % self.periods[self.driver] == 0:
                value = self._infer(self.driver)
                self.previous[self.driver].copy_(self.outputs[self.driver])
                self.outputs[self.driver].copy_(value)
        self.tick += 1
        return self.outputs[self.driver]

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in self.outputs:
            for buffer in (self.outputs[name], self.previous[name]):
                for tensor in (buffer.values() if isinstance(buffer, dict) else (buffer,)):
                    tensor[ids] = 0
            if name == self.driver and self.initial_output is not None:
                self.outputs[name][ids] = self.initial_output[ids]
                self.previous[name][ids] = self.initial_output[ids]
            self.readers[name].reset(env_ids)
        self.planner_due[ids] = -1

    @staticmethod
    def _copy(buffer):
        return {k: v.clone() for k, v in buffer.items()} if isinstance(buffer, dict) else buffer.clone()

    def snapshot_state(self):
        # planner_due stays last; component inputs include window contents, counts and pending rows.
        return ({k: self._copy(v) for k, v in self.outputs.items()},
                {k: self._copy(v) for k, v in self.previous.items()},
                {k: v.state() for k, v in self.readers.items()}, self.planner_due.clone())

    def restore_state(self, state):
        outputs, previous, inputs, due = state
        for key in outputs:
            self._assign(self.outputs[key], outputs[key])
            self._assign(self.previous[key], previous[key])
            self.readers[key].restore(inputs[key])
        self.planner_due.copy_(due)

    def read(self, role, field):
        if field == 'input':
            return self.readers[role].last
        if isinstance(self.outputs[role], dict):
            if field not in self.outputs[role]:
                raise ValueError(f'{role} has no part {field!r}; it returns {list(self.outputs[role])}')
            return self.outputs[role][field]
        if field not in ('output', 'previous_output'):
            raise ValueError(f'{role} returns one Tensor: read {role}.output (or previous_output), not {role}.{field}')
        return (self.outputs if field == 'output' else self.previous)[role]


def validate_targets(targets, endpoints, num_envs, device):
    if not isinstance(targets, dict) or targets.keys() != endpoints.keys():
        got = sorted(targets) if isinstance(targets, dict) else type(targets).__name__
        raise ValueError(f'decode_action (policy.decode) must return one target per controlled robot {sorted(endpoints)}, '
                         f'got {got}; a robot the policy should not drive is set to none in policy.modes')
    for name, endpoint in endpoints.items():
        value = targets[name]
        if (not isinstance(value, torch.Tensor) or value.shape != (num_envs, endpoint.action_dim)
                or not value.is_floating_point() or value.device != device):
            raise ValueError(f'Target {name}: expected floating tensor [{num_envs}, {endpoint.action_dim}] on {device}')
        # No host scalar extraction or CPU transfer on the GPU control path.
        torch._assert_async(torch.isfinite(value).all(), f'Non-finite control target: {name}')
