"""Batched task expressions over Signals, independent of policy observations.

No simulator loop, Python eval, controller or trainer lives here. Backends call
evaluate once per control step and reset the selected episode slots.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math


# Required and optional fields, excluding the common ``op`` discriminator.
RULES = {
    "field": ({"input", "name"}, set()),
    "rotation_distance": ({"a", "b"}, set()),
    "inside_region": ({"input", "region"}, set()),
    "distance": ({"a", "b"}, set()),
    "subtract": ({"a", "b"}, set()),
    "add": ({"a", "b"}, set()),
    "multiply": ({"a", "b"}, set()),
    "divide": ({"a", "b"}, set()),
    "dot": ({"a", "b"}, set()),
    "angle": ({"a", "b"}, set()),
    "abs": ({"input"}, set()),
    "norm": ({"input"}, set()),
    "sum": ({"input"}, set()),
    "mean": ({"input"}, set()),
    "min": ({"input"}, set()),
    "max": ({"input"}, set()),
    "select": ({"input", "indices"}, set()),
    "rotate": ({"input", "quaternion"}, {"inverse"}),
    "transform_point": ({"input", "quaternion", "position"}, set()),
    "less": ({"a", "b"}, set()),
    "less_equal": ({"a", "b"}, set()),
    "greater": ({"a", "b"}, set()),
    "greater_equal": ({"a", "b"}, set()),
    "equal": ({"a", "b"}, set()),
    "not_equal": ({"a", "b"}, set()),
    "near": ({"a", "b", "tolerance"}, set()),
    "between": ({"input", "lower", "upper"}, set()),
    "in_set": ({"input", "choices"}, set()),
    "inside_box": ({"input", "lower", "upper"}, set()),
    "valid": ({"input"}, set()),
    "not": ({"input"}, set()),
    "all": ({"inputs"}, set()),
    "any": ({"inputs"}, set()),
    "at_least": ({"inputs", "count"}, set()),
    "all_elements": ({"input"}, set()),
    "any_elements": ({"input"}, set()),
    "count": ({"input"}, set()),
    "hold": ({"input", "seconds"}, set()),
    "ever": ({"input"}, set()),
    "sequence": ({"inputs"}, set()),
    "within": ({"input", "seconds"}, set()),
    "changed": ({"input"}, {"tolerance"}),
    "rising": ({"input"}, set()),
    "delta": ({"input"}, set()),
    "accumulate": ({"input"}, set()),
    "integral": ({"input"}, set()),
    "running_mean": ({"input"}, set()),
    "running_min": ({"input"}, set()),
    "running_max": ({"input"}, set()),
}
OPERANDS = {"a", "b", "input", "lower", "upper", "quaternion", "position", "region"}


def _fields(value, required, optional, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: expected a mapping")
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise ValueError(f"{label}: missing {sorted(missing)}, unknown {sorted(extra)}")



TASK_FIELDS = frozenset({'success', 'failure', 'reward', 'metrics'})


def evaluation_config(settings):
    """Compile the sole public variable scope and the task's output expressions."""
    task = settings.get('task', {})
    if not isinstance(task, Mapping):
        raise ValueError('task must be a mapping')
    obsolete = task.keys() & {'evaluation', 'variables', 'values', 'conditions', 'history'}
    if obsolete:
        raise ValueError(f'Obsolete task fields {sorted(obsolete)}: declare variables at scene root and task outputs directly under task')
    return {'variables': deepcopy(settings.get('variables', {})),
            **{key: deepcopy(task[key]) for key in TASK_FIELDS if key in task}}

def task_nodes(config):
    """Unified declarations; old values/conditions remain readable for experiments."""
    import keyword
    from .expressions import HELPERS, DOMAIN_FUNCTIONS
    nodes, histories, information = {}, {}, {}
    for group in ('values', 'conditions', 'variables'):
        entries = config.get(group, {})
        if not isinstance(entries, Mapping): raise ValueError(f'evaluation.{group}: expected a mapping')
        for name, value in entries.items():
            if not isinstance(name, str) or not name or name in nodes:
                raise ValueError(f'Invalid or duplicate task node: {name!r}')
            if group == 'variables':
                if not name.isidentifier() or keyword.iskeyword(name) or name.startswith('_') or name in {'torch', *HELPERS, *DOMAIN_FUNCTIONS}:
                    raise ValueError(f'Invalid variable name: {name!r}')
                value = {'expr': value} if isinstance(value, str) else deepcopy(value)
                if not isinstance(value, Mapping): raise ValueError(f'variables.{name}: expected a declaration')
                sampling = value.pop('sampling', 'decision')
                if sampling not in ('decision', 'physics'):
                    raise ValueError(f'{name}.sampling must be decision or physics')
                if sampling == 'physics' and 'history' in value:
                    raise ValueError('Physics-sampled variables are instantaneous; histories use the decision clock')
                size = value.pop('history', None)
                if size is not None:
                    if type(size) is not int or size < 1: raise ValueError(f'{name}.history must be a positive step count')
                    histories[name] = size
                if 'information' in value:
                    from ..information import Information
                    fields = dict(value.pop('information'))
                    for field in ('shape', 'axes', 'labels'):
                        if fields.get(field) is not None: fields[field] = tuple(fields[field])
                    information[name] = Information(**fields)
                if 'source' in value:
                    if len(value) != 1: raise ValueError(f'{name}: source cannot be combined with another expression')
                    value = {'signal': value['source']}
            nodes[name] = value
    return nodes, histories, information



def physics_variables(config):
    """Pure fast-feedback definitions; no second history clock or state store."""
    import ast
    from .expressions import dependencies, path
    fast = {name: value for name, value in config.get('variables', {}).items()
            if isinstance(value, Mapping) and value.get('sampling') == 'physics'}
    all_names = config.get('variables', {})
    for name, declaration in fast.items():
        if 'history' in declaration:
            raise ValueError('Physics-sampled variables cannot own history')
        if 'source' in declaration:
            continue
        if 'expr' not in declaration:
            raise ValueError('Physics-sampled variables use pure expr or source')
        refs, _, histories, _ = dependencies(declaration['expr'], all_names)
        if histories or refs - fast.keys():
            raise ValueError(f'{name}: physics variables only depend on basic signals and physics variables')
        for node in ast.walk(ast.parse(declaration['expr'], mode='eval')):
            if isinstance(node, ast.Call) and path(node.func) in {'hold', 'running_mean', 'history_mean', 'history_sum', 'history_all', 'history_any'}:
                raise ValueError(f'{name}: temporal operations use decision sampling')
    return fast

def reward_entries(config):
    """Normalize a total formula or the older optional named contributions."""
    if 'reward' in config:
        if 'rewards' in config: raise ValueError('Choose reward or rewards, not both')
        formula = config['reward']
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError('reward must be a nonempty expression string')
        return {'total': {'input': formula}}
    entries = config.get('rewards', {})
    if not isinstance(entries, Mapping): raise ValueError('rewards: expected mapping')
    return entries


def validate_evaluation(config):
    """Static shape/reference/cycle checks; does not import torch or run readers."""
    _fields(config, set(), {"variables", "values", "conditions", "success", "failure", "reward", "rewards", "metrics"}, "evaluation")
    nodes, histories, _ = task_nodes(config)
    physics_variables(config)
    signals, dependencies = set(), {}

    def expression(expr, refs, label):
        if isinstance(expr, str): expr = {'expr': expr}
        if not isinstance(expr, Mapping):
            raise ValueError(f"{label}: use {{signal: path}}, {{ref: name}}, {{value: literal}} or an op")
        if set(expr) == {'expr'}:
            from .expressions import dependencies
            references, sources, history, _ = dependencies(expr['expr'], nodes)
            if history - histories.keys():
                raise ValueError(f'{label}: history is not enabled for {sorted(history - histories.keys())}')
            refs.update(references)
            signals.update(sources)
            return
        if len(expr) == 1 and next(iter(expr)) in ("signal", "ref", "value"):
            key, value = next(iter(expr.items()))
            if key == "value":
                def literal(x):
                    return (isinstance(x, str) and bool(x) or (isinstance(x, (int, float, bool)) and math.isfinite(x))
                            or isinstance(x, list) and bool(x) and all(literal(y) for y in x))
                if not literal(value):
                    raise ValueError(f"{label}: constants must be finite numbers, booleans, category labels or nonempty arrays")
            elif not isinstance(value, str) or not value:
                raise ValueError(f"{label}: {key} must be a nonempty string")
            elif key == "signal":
                signals.add(value)
            else:
                if value not in nodes:
                    raise ValueError(f"{label}: unknown task reference {value!r}")
                refs.add(value)
            return
        op = expr.get("op")
        if op not in RULES:
            raise ValueError(f"{label}: unknown operation {op!r}")
        required, optional = RULES[op]
        _fields(expr, required | {"op"}, optional, label)
        if op == "field" and (not isinstance(expr["name"], str) or not expr["name"]):
            raise ValueError(f"{label}.name must be a nonempty field name")
        for key in OPERANDS & expr.keys():
            expression(expr[key], refs, f"{label}.{key}")
        if "inputs" in expr:
            if not isinstance(expr["inputs"], list) or not expr["inputs"]:
                raise ValueError(f"{label}.inputs must be nonempty")
            for i, entry in enumerate(expr["inputs"]):
                expression(entry, refs, f"{label}.inputs[{i}]")
        for key in ("seconds", "tolerance"):
            if key in expr and (isinstance(expr[key], bool) or not isinstance(expr[key], (int, float))
                                or not math.isfinite(expr[key]) or expr[key] < 0):
                raise ValueError(f"{label}.{key}: expected a finite nonnegative number")
        if "count" in expr and (type(expr["count"]) is not int or not 0 <= expr["count"] <= len(expr["inputs"])):
            raise ValueError(f"{label}.count: outside input count")
        if "indices" in expr:
            indices = expr["indices"]
            if not (type(indices) is int or isinstance(indices, list) and indices and all(type(i) is int for i in indices)):
                raise ValueError(f"{label}.indices: expected integer or nonempty integer list")
        if "choices" in expr:
            if not isinstance(expr["choices"], list) or not expr["choices"]:
                raise ValueError(f"{label}.choices: expected nonempty numeric list")
            expression({"value": expr["choices"]}, refs, label)
            if any(isinstance(x, list) for x in expr["choices"]):
                raise ValueError(f"{label}.choices must be flat")
        if "inverse" in expr and not isinstance(expr["inverse"], bool):
            raise ValueError(f"{label}.inverse: expected boolean")

    for name, expr in nodes.items():
        dependencies[name] = set()
        expression(expr, dependencies[name], name)
    visited, visiting = set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError(f"Cyclic task reference: {name}")
        if name not in visited:
            visiting.add(name)
            for dependency in dependencies[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
    for name in nodes:
        visit(name)
    for key in ("success", "failure"):
        if key in config:
            expression(config[key], set(), key)
    for key in ("metrics", "rewards"):
        entries = reward_entries(config) if key == 'rewards' else config.get(key, {})
        if not isinstance(entries, Mapping):
            raise ValueError(f"{key}: expected mapping")
        for name, entry in entries.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{key}: names must be nonempty strings")
            if key == "rewards":
                _fields(entry, {"input"}, {"weight"}, f"rewards.{name}")
                weight = entry.get("weight", 1.0)
                if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight):
                    raise ValueError(f"rewards.{name}.weight must be finite")
                entry = entry["input"]
            expression(entry, set(), f"{key}.{name}")
    return {"signals": sorted(signals), "nodes": list(nodes), "rules": sorted(RULES), "history": histories}


@dataclass
class Reading:
    """Optional explicit per-environment validity for a signal/evaluator result."""
    value: object
    valid: object = True


@dataclass
class TaskResult:
    success: object
    failure: object
    reward: object
    rewards: dict
    metrics: dict


class TaskEvaluator:
    def __init__(self, config, signals, *, num_envs=1, device="cpu"):
        import torch
        self.torch = torch
        self.config = deepcopy(config)
        self.report = validate_evaluation(config)
        self.signals, self.num_envs, self.device = signals, num_envs, device
        for path in self.report["signals"]:
            if path not in signals:
                raise ValueError(f"Unknown task signal: {path}")
        self.nodes, self.history_sizes, self.declarations = task_nodes(self.config)
        self.histories = {}
        self.variable_readers = {}
        self.physics_runtime = None
        from .task_types import infer_evaluation
        self.types = infer_evaluation(self.config, {path: signals.information(path) for path in self.report["signals"]})
        self.report["information"] = {name: info.describe() for name, info in self.types["nodes"].items()}
        self.report["untyped_signals"] = self.types["untyped_signals"]
        self.report["untyped_variables"] = self.types["untyped_variables"]
        self.state = {}
        self.elapsed = self._zeros()
        self._step = None
        self._result = None
        self.cache, self.read_cache = {}, {}
        self._preserved_cache = {}
        self._sample_step = self.torch.full((num_envs,), -1, dtype=self.torch.long, device=device)

    def _zeros(self, **kwargs):
        return self.torch.zeros(self.num_envs, device=self.device, **kwargs)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        if self.physics_runtime is not None:
            self.physics_runtime.reset(env_ids)
        self.elapsed[ids] = 0
        self._sample_step[ids] = -1
        for history in self.histories.values(): history.reset(ids)
        for fields in self.state.values():
            for value in fields.values():
                value[ids] = 0
        self._preserved_cache = dict(self.cache)
        self._step = None
        self._result = None

    def _reading(self, value, *, constant=False, info=None):
        t = self.torch
        reading = value if isinstance(value, Reading) else Reading(value)
        raw = reading.value
        if info is not None and info.kind == "category":
            def encode(x):
                if isinstance(x, str):
                    if x not in info.labels:
                        raise ValueError(f"Unknown category label {x!r}; expected {info.labels}")
                    return info.labels.index(x)
                if isinstance(x, (tuple, list)): return [encode(item) for item in x]
                return x
            raw = encode(raw)
        data = t.as_tensor(raw, device=self.device)
        if constant:
            data = data.unsqueeze(0).expand(self.num_envs, *data.shape)
        elif data.ndim == 0:
            data = data.expand(self.num_envs)
        if data.shape[0] != self.num_envs:
            raise ValueError(f"Signal must have leading environment dimension {self.num_envs}, got {tuple(data.shape)}")
        valid = t.as_tensor(reading.valid, dtype=t.bool, device=self.device)
        if valid.ndim == 0:
            valid = valid.expand(self.num_envs)
        if valid.shape != (self.num_envs,):
            raise ValueError("Validity must be a boolean per environment")
        finite = t.isfinite(data).reshape(self.num_envs, -1).all(-1)
        if info is not None and info.kind == "category" and info.labels:
            codes_valid = (data >= 0) & (data < len(info.labels)) & (data == data.long())
            finite = finite & codes_valid.reshape(self.num_envs, -1).all(-1)
        return Reading(data, valid & finite)

    def _boolean(self, reading, *, scalar=False):
        if reading.value.dtype != self.torch.bool or scalar and reading.value.shape != (self.num_envs,):
            raise ValueError("Condition must be boolean (one value per environment for task/time composition)")
        return reading.value

    def _begin(self, *, dt, step):
        if step == self._step:
            return
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Task dt must be finite and positive")
        if self._step is not None and step < self._step:
            raise ValueError("Task step must increase; call reset for a new episode")
        self.dt, self.current_step = dt, step
        self._preserve_rows = self._sample_step == step
        self._preserved_state = {key: {name: value.clone() for name, value in fields.items()}
                                 for key, fields in self.state.items()} if self._preserve_rows.any() else {}
        if not self._preserve_rows.any():
            self._preserved_cache = {}
        self.elapsed += dt * ~self._preserve_rows
        self._sample_step[:] = step
        self.cache, self.read_cache = {}, {}
        self._step, self._result = step, None

    def _restore_unreset_state(self):
        for key, fields in self._preserved_state.items():
            for name, value in fields.items():
                self.state[key][name][self._preserve_rows] = value[self._preserve_rows]

    def read_variable(self, name, *, dt, step):
        """Read one shared variable at the same clock as task evaluation."""
        if name in self.variable_readers:
            return self.variable_readers[name]()
        self._begin(dt=dt, step=step)
        reading = self._node(name)
        self._restore_unreset_state()
        return reading

    def evaluate(self, *, dt, step):
        """One shared sample per decision step; repeated consumers reuse it."""
        self._begin(dt=dt, step=step)
        if self._result is not None:
            return self._result
        # Nodes are evaluated even if not on the selected success branch: time
        # conditions must not miss steps because of short-circuit evaluation.
        for name in self.nodes:
            self._node(name)
        def outcome(key):
            reading = self._expr(self.config.get(key, {"value": False}), key)
            return self._boolean(reading, scalar=True) & reading.valid
        failure, success = outcome("failure"), outcome("success")
        success = success & ~failure
        rewards, total = {}, self._zeros()
        for name, entry in reward_entries(self.config).items():
            reading = self._expr(entry["input"], "rewards." + name)
            if reading.value.shape != (self.num_envs,):
                raise ValueError(f"Reward {name} must be scalar per environment")
            value = self.torch.where(reading.valid, reading.value, 0).float() * entry.get("weight", 1.0)
            rewards[name], total = value, total + value
        metrics = {name: self._expr(expr, "metrics." + name)
                   for name, expr in self.config.get("metrics", {}).items()}
        self._step = step
        self._restore_unreset_state()
        self._result = TaskResult(success, failure, total, rewards, metrics)
        return self._result

    def _node(self, name):
        if name not in self.cache:
            try:
                reading = self.variable_readers[name]() if name in self.variable_readers else self._expr(self.nodes[name], "node." + name)
                declared = self.declarations.get(name)
                if declared is not None:
                    if declared.shape is not None and (len(declared.shape) != reading.value.ndim - 1 or any(
                        wanted is not None and wanted != got for wanted, got in zip(declared.shape, reading.value.shape[1:]))):
                        raise ValueError(f'declared shape {declared.shape}, received {tuple(reading.value.shape[1:])}')
                    if declared.kind == 'boolean' and reading.value.dtype != self.torch.bool:
                        raise ValueError('declared boolean variable returned non-boolean data')
                reading = Reading(reading.value.clone(), reading.valid.clone())
                if name in self._preserved_cache:
                    old = self._preserved_cache[name]
                    reading.value[self._preserve_rows] = old.value[self._preserve_rows]
                    reading.valid[self._preserve_rows] = old.valid[self._preserve_rows]
                self.cache[name] = reading
                if name in self.history_sizes:
                    from .expressions import HistoryBuffer
                    if name not in self.histories:
                        self.histories[name] = HistoryBuffer(reading, self.history_sizes[name], self)
                    self.histories[name].append(reading, self.elapsed, self.current_step)
            except (TypeError, ValueError, RuntimeError, IndexError) as error:
                raise ValueError(f'Task variable {name}: {error}') from error
        return self.cache[name]

    def _expr(self, expr, key):
        t = self.torch
        if isinstance(expr, str): expr = {'expr': expr}
        if 'expr' in expr:
            from .expressions import execute
            return execute(expr['expr'], self, key)
        if "value" in expr:
            return self._reading(expr["value"], constant=True, info=self.types["expressions"].get(key))
        if "ref" in expr:
            return self._node(expr["ref"])
        if "signal" in expr:
            path = expr["signal"]
            if path not in self.read_cache:
                info = self.signals.information(path)
                reading = self._reading(self.signals.read(path), info=info)
                actual = tuple(reading.value.shape[1:])
                if info.shape is not None and (len(actual) != len(info.shape) or any(
                    expected is not None and expected != found for expected, found in zip(info.shape, actual)
                )):
                    raise ValueError(f"Task signal {path}: declared shape {info.shape}, received {actual}")
                if info.dtype == "bool" and reading.value.dtype != t.bool:
                    raise ValueError(f"Task signal {path}: expected boolean data")
                self.read_cache[path] = reading
            return self.read_cache[path]
        op = expr["op"]
        args = {name: self._expr(expr[name], key + "." + name) for name in OPERANDS & expr.keys()}
        inputs = [self._expr(e, key + f".inputs.{i}") for i, e in enumerate(expr.get("inputs", []))]
        valid = t.ones(self.num_envs, dtype=t.bool, device=self.device)
        for reading in [*args.values(), *inputs]:
            valid = valid & reading.valid
        def finish(value, mask=None):
            return self._reading(Reading(value, valid if mask is None else mask))
        def vector(value):
            if value.ndim < 2:
                raise ValueError(f"{key}: expected feature dimension after environment axis")
            return value.float()
        x = args["input"].value if "input" in args else None
        def quaternion(value, operand_key):
            info = self.types["expressions"].get(operand_key)
            if info is not None and info.encoding == "xyzw":
                value = value[..., [3, 0, 1, 2]]
            return value.float()
        if op == "field":
            info = self.types["expressions"][key + ".input"]
            if expr["name"] not in info.fields:
                raise ValueError(f"{key}: field extraction needs declared structured metadata")
            field = info.fields[expr["name"]]
            index = field.indices[0] if field.info.shape == () else list(field.indices)
            return finish(x[..., index])
        if op == "rotation_distance":
            a = quaternion(args["a"].value, key + ".a")
            b = quaternion(args["b"].value, key + ".b")
            if a.shape != (self.num_envs, 4) or b.shape != a.shape:
                raise ValueError("rotation_distance requires two quaternions")
            la, lb = a.norm(dim=-1), b.norm(dim=-1)
            cosine = ((a * b).sum(-1) / (la * lb).clamp_min(1e-12)).abs().clamp(0, 1)
            return finish(2 * t.acos(cosine), valid & (la > 1e-12) & (lb > 1e-12))
        if op == "inside_region":
            info = self.types["expressions"][key + ".region"]
            if info.encoding not in ("box_center_wxyz_halfsize", "rectangle_center_wxyz_halfsize"):
                raise ValueError("inside_region requires a declared box or rectangle region")
            region = args["region"].value
            if region.shape != (self.num_envs, 10) or x.shape != (self.num_envs, 3):
                raise ValueError("inside_region expects a 3D point and a 10-component region")
            q = region[:, 3:7].float()
            length = q.norm(dim=-1)
            q = q / length.clamp_min(1e-12).unsqueeze(-1)
            offset = x - region[:, :3]
            xyz = -q[:, 1:]
            cross = 2 * t.cross(xyz, offset, dim=-1)
            local = offset + q[:, :1] * cross + t.cross(xyz, cross, dim=-1)
            dimensions = 2 if info.encoding.startswith("rectangle") else 3
            halfsize = region[:, 7:7 + dimensions]
            return finish((local[:, :dimensions].abs() <= halfsize + 1e-7).all(-1),
                          valid & (length > 1e-12) & (halfsize > 0).all(-1))
        if "a" in args:
            a, b = args["a"].value, args["b"].value
            if a.shape != b.shape:
                # Broadcast a scalar per environment over feature dimensions,
                # never accidentally align environment and feature axes.
                while a.ndim < b.ndim: a = a.unsqueeze(-1)
                while b.ndim < a.ndim: b = b.unsqueeze(-1)
            binary = {"add": t.add, "subtract": t.sub, "multiply": t.mul, "divide": t.div,
                      "less": t.lt, "less_equal": t.le, "greater": t.gt, "greater_equal": t.ge,
                      "equal": t.eq, "not_equal": t.ne}
            if op in binary:
                return finish(binary[op](a, b))
            if op == "near": return finish((a - b).abs() <= expr["tolerance"])
            if op == "distance": return finish(t.linalg.vector_norm(vector(a - b), dim=-1))
            if op == "dot": return finish((vector(a) * vector(b)).sum(-1))
            if op == "angle":
                a, b = vector(a), vector(b)
                denominator = a.norm(dim=-1) * b.norm(dim=-1)
                return finish(t.acos(((a * b).sum(-1) / denominator.clamp_min(1e-12)).clamp(-1, 1)),
                              valid & (denominator > 1e-12))
        if op == "valid": return finish(args["input"].valid, t.ones_like(valid))
        if op == "abs": return finish(x.abs())
        if op == "select":
            vector(x)
            return finish(x[..., expr["indices"]])
        if op in ("norm", "sum", "mean", "min", "max"):
            x = vector(x)
            functions = {"norm": lambda y: y.norm(dim=-1), "sum": lambda y: y.sum(-1),
                         "mean": lambda y: y.mean(-1), "min": lambda y: y.amin(-1), "max": lambda y: y.amax(-1)}
            return finish(functions[op](x))
        if op in ("rotate", "transform_point"):
            q = quaternion(args["quaternion"].value, key + ".quaternion")
            if x.shape != (self.num_envs, 3) or q.shape != (self.num_envs, 4):
                raise ValueError(f"{key}: rotation expects a 3D vector and wxyz quaternion")
            length = q.norm(dim=-1)
            valid = valid & (length > 1e-12)
            q = q / length.clamp_min(1e-12).unsqueeze(-1)
            xyz = -q[:, 1:] if expr.get("inverse", False) else q[:, 1:]
            cross = 2 * t.cross(xyz, x.float(), dim=-1)
            rotated = x + q[:, :1] * cross + t.cross(xyz, cross, dim=-1)
            return finish(rotated + args["position"].value if op == "transform_point" else rotated)
        if op in ("between", "inside_box"):
            lower, upper = args["lower"].value, args["upper"].value
            while lower.ndim < x.ndim: lower = lower.unsqueeze(-1)
            while upper.ndim < x.ndim: upper = upper.unsqueeze(-1)
            result = (x >= lower) & (x <= upper)
            if op == "inside_box":
                vector(x)
                result = result.all(-1)
            return finish(result)
        if op == "in_set":
            info = self.types["expressions"][key + ".input"]
            choices = [info.labels.index(v) if isinstance(v, str) else v for v in expr["choices"]]
            return finish(t.isin(x, t.tensor(choices, device=self.device)))
        if op == "not": return finish(~self._boolean(args["input"]))
        if op in ("all_elements", "any_elements", "count"):
            data = self._boolean(args["input"])
            vector(data)
            return finish({"all_elements": lambda: data.all(-1), "any_elements": lambda: data.any(-1),
                           "count": lambda: data.sum(-1)}[op]())
        if op in ("all", "any", "at_least"):
            data = t.stack([self._boolean(r, scalar=True) for r in inputs], dim=-1)
            return finish({"all": lambda: data.all(-1), "any": lambda: data.any(-1),
                           "at_least": lambda: data.sum(-1) >= expr["count"]}[op]())
        if op in ("hold", "ever", "within", "rising"):
            self._boolean(args["input"], scalar=op != "ever")
        if op == "sequence":
            for reading in inputs: self._boolean(reading, scalar=True)
        state = self.state.setdefault(key, {})
        def memory(name, like):
            if name not in state:
                state[name] = t.zeros_like(like)
            return state[name]
        if op == "hold":
            duration = memory("duration", self.elapsed)
            duration.copy_(t.where(valid & x, duration + self.dt, 0))
            return finish(x & (duration + 1e-7 >= expr["seconds"]))
        if op == "ever":
            seen = memory("seen", x)
            seen |= valid.reshape(self.num_envs, *([1] * (x.ndim - 1))) & x
            return finish(seen.clone(), t.ones_like(valid))
        if op == "within":
            seen = memory("seen", x)
            seen |= valid & x & (self.elapsed <= expr["seconds"] + 1e-7)
            return finish(seen.clone(), t.ones_like(valid))
        if op == "sequence":
            stage = memory("stage", self.elapsed.long())
            # One stage at most per sample. Earlier completions remain latched.
            previous = stage.clone()
            for i, reading in enumerate(inputs):
                stage += ((previous == i) & reading.valid & reading.value).long()
            return finish(stage == len(inputs), t.ones_like(valid))
        if op in ("delta", "changed", "rising"):
            previous = memory("previous", x)
            initialized = memory("initialized", valid)
            usable = valid & initialized
            if op == "rising": value = x & ~previous & usable
            else:
                value = x.float() - previous.float()
                if op == "changed":
                    value = value.abs().reshape(self.num_envs, -1).amax(-1) > expr.get("tolerance", 0)
                    value &= usable
            mask = valid.reshape(self.num_envs, *([1] * (x.ndim - 1)))
            previous.copy_(t.where(mask, x, previous))
            initialized.copy_(valid)  # invalid samples break the consecutive pair
            if op == "delta":
                value = t.where(usable.reshape(self.num_envs, *([1] * (value.ndim - 1))), value, 0)
            return finish(value, usable if op == "delta" else valid)
        if op in ("accumulate", "integral", "running_mean", "running_min", "running_max"):
            total = memory("total", x.float())
            count = memory("count", self.elapsed)
            mask = valid.reshape(self.num_envs, *([1] * (x.ndim - 1)))
            first = (count == 0).reshape(mask.shape)
            if op in ("accumulate", "integral", "running_mean"):
                total.add_(t.where(mask, x.float() * (self.dt if op == "integral" else 1), 0))
            else:
                update = t.minimum(total, x) if op == "running_min" else t.maximum(total, x)
                total.copy_(t.where(mask, t.where(first, x, update), total))
            count.add_(valid.float())
            value = total / count.clamp_min(1).reshape(mask.shape) if op == "running_mean" else total.clone()
            return finish(value, count > 0)
        raise ValueError(f"Unimplemented task operation: {op}")
