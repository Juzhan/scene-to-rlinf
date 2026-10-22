"""Read-only Python expressions using selected real Torch operations.

Parsing and dependency discovery need no tensor runtime. This is an expression
interpreter, not eval(), a Python sandbox, or a second numerical library.
"""
import ast
import operator
import math
from functools import lru_cache


FUNCTIONS = {
    'torch.abs', 'torch.square', 'torch.sqrt', 'torch.exp', 'torch.log', 'torch.tanh',
    'torch.sin', 'torch.cos', 'torch.atan2', 'torch.clamp', 'torch.minimum', 'torch.maximum',
    'torch.where', 'torch.sum', 'torch.mean', 'torch.amin', 'torch.amax',
    'torch.all', 'torch.any', 'torch.diff', 'torch.stack', 'torch.cat',
    'torch.logical_and', 'torch.logical_or', 'torch.logical_not',
    'torch.isfinite', 'torch.linalg.vector_norm', 'torch.linalg.norm',
}
METHODS = {'sum', 'mean', 'amin', 'amax', 'all', 'any', 'abs', 'square', 'sqrt',
           'clamp', 'float', 'bool', 'unsqueeze', 'squeeze'}
HELPERS = {'history_mean', 'history_sum', 'history_all', 'history_any'}
DOMAIN_FUNCTIONS = {'inverse_rotate': ('input', 'quaternion'), 'inside_region': ('input', 'region'), 'hold': ('input', 'seconds'),
                    'running_mean': ('input',)}
HISTORY = {'history', 'history_valid', 'history_time'}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
          ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}
COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
           ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def path(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = path(node.value)
        return parent + '.' + node.attr if parent else None
    return None


def domain_expression(node):
    """Function syntax for existing geometric/time operations; one implementation."""
    name = path(node.func)
    fields = DOMAIN_FUNCTIONS[name]
    if len(node.args) > len(fields): raise ValueError(f'{name}: too many arguments')
    arguments = dict(zip(fields, node.args))
    for item in node.keywords:
        if item.arg not in fields or item.arg in arguments:
            raise ValueError(f'{name}: unknown or repeated argument {item.arg}')
        arguments[item.arg] = item.value
    if set(arguments) != set(fields): raise ValueError(f'{name} requires {fields}')
    output = {'op': 'rotate', 'inverse': True} if name == 'inverse_rotate' else {'op': name}
    for field, value in arguments.items():
        if field == 'seconds':
            try: number = ast.literal_eval(value)
            except (ValueError, TypeError): raise ValueError('hold seconds must be a nonnegative constant') from None
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                raise ValueError('hold seconds must be a nonnegative constant')
            output[field] = number
        else:
            output[field] = {'expr': ast.unparse(value)}
    return output


def domain_key(key, node):
    return f'{key}.call.{node.lineno}.{node.col_offset}'


@lru_cache(maxsize=512)
def parse(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('expr must be a nonempty Python expression')
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as error:
        raise ValueError(f'Invalid task expr: {error.msg}') from error
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Attribute, ast.Load,
               ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.keyword,
               ast.Subscript, ast.Slice, ast.Tuple, ast.List,
               *BINARY, *UNARY, *COMPARE)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f'Unsupported expr syntax {type(node).__name__}; use tensor operations (&, |, torch.where)')
        if isinstance(node, ast.Name) and node.id.startswith('_') or isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private names are not allowed in expr')
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool, str, type(None))) and node.value is not Ellipsis:
            raise ValueError('Unsupported expr literal')
        if isinstance(node, ast.keyword) and (node.arg is None or node.arg in {'out', 'device', 'requires_grad'}):
            raise ValueError('expr cannot mutate output buffers or change device/autograd')
        if isinstance(node, ast.Compare) and len(node.ops) != 1:
            raise ValueError('Use (a < b) & (b < c), not chained tensor comparisons')
    return tree.body


def dependencies(text, variables):
    refs, signals, histories, masks = set(), set(), set(), set()

    def reference(node):
        name = path(node)
        if name is None:
            raise ValueError('Only scene properties and declared variable history attributes are supported')
        root, *tail = name.split('.')
        if root in variables:
            refs.add(root)
            if tail:
                if len(tail) != 1 or tail[0] not in HISTORY:
                    raise ValueError(f'Unknown variable attribute: {name}')
                histories.add(root)
                if tail[0] == 'history_valid': masks.add(root)
            return
        if root in {'torch', *HELPERS, *DOMAIN_FUNCTIONS}:
            raise ValueError(f'{name} must be used as a function call')
        if not tail:
            raise ValueError(f'Unknown task variable: {name}')
        signals.add(name)

    def visit(node):
        if isinstance(node, ast.Call):
            target = path(node.func)
            if target in DOMAIN_FUNCTIONS:
                domain_expression(node)
                for item in node.args: visit(item)
                for item in node.keywords: visit(item.value)
                return
            if target in HELPERS:
                if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Name) or node.args[0].id not in variables:
                    raise ValueError(f'{target} requires one declared variable')
                name = node.args[0].id
                refs.add(name); histories.add(name)
                return
            if target not in FUNCTIONS:
                if not isinstance(node.func, ast.Attribute) or node.func.attr not in METHODS:
                    raise ValueError(f'Unsupported expression function: {target}')
                visit(node.func.value)
            for item in node.args: visit(item)
            for item in node.keywords: visit(item.value)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            reference(node)
        else:
            for child in ast.iter_child_nodes(node): visit(child)
    visit(parse(text))
    return refs, signals, histories, masks


def execute(text, engine, key='expression'):
    """Use real Torch tensors. All side effects are owned by the variable store."""
    import torch
    from .task import Reading
    refs, signals, histories, masks = dependencies(text, engine.nodes)
    valid = torch.ones(engine.num_envs, dtype=torch.bool, device=engine.device)

    def use(reading):
        nonlocal valid
        valid = valid & reading.valid
        return reading.value

    def history(name):
        engine._node(name)
        return engine.histories[name].read()

    def check_broadcast(values):
        ranks = {x.ndim for x in values if isinstance(x, torch.Tensor) and x.ndim}
        if len(ranks) > 1:
            raise ValueError('Batched operands need matching rank; use unsqueeze to make broadcasting explicit')

    def run(node):
        nonlocal valid
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = path(node)
            if name in engine.nodes: return use(engine._node(name))
            root, _, field = name.partition('.')
            if root in engine.nodes:
                values, mask, times = history(root)
                if field == 'history_valid': return mask
                valid = valid & (mask.any(dim=1) if root in masks else mask.all(dim=1))
                return times if field == 'history_time' else values
            return use(engine._expr({'signal': name}, 'expr.signal.' + name))
        if isinstance(node, ast.BinOp):
            left, right = run(node.left), run(node.right)
            check_broadcast((left, right))
            return BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp): return UNARY[type(node.op)](run(node.operand))
        if isinstance(node, ast.Compare):
            left, right = run(node.left), run(node.comparators[0])
            check_broadcast((left, right))
            return COMPARE[type(node.ops[0])](left, right)
        if isinstance(node, ast.Tuple): return tuple(run(x) for x in node.elts)
        if isinstance(node, ast.List): return [run(x) for x in node.elts]
        if isinstance(node, ast.Slice):
            return slice(*(run(x) if x is not None else None for x in (node.lower, node.upper, node.step)))
        if isinstance(node, ast.Subscript):
            value, index = run(node.value), run(node.slice)
            if isinstance(value, torch.Tensor):
                indices = index if isinstance(index, tuple) else (index,)
                expanded = []
                width = value.ndim - sum(item is not None and item is not Ellipsis for item in indices)
                for item in indices:
                    expanded.extend([slice(None)] * max(width, 0) if item is Ellipsis else [item])
                first = expanded[0] if expanded else slice(None)
                if not isinstance(first, slice) or first != slice(None):
                    raise ValueError('Keep the environment axis intact: use x[:, ...] or x[..., ...]')
            return value[index]
        if isinstance(node, ast.Call):
            target = path(node.func)
            if target in DOMAIN_FUNCTIONS:
                return use(engine._expr(domain_expression(node), domain_key(key, node)))
            if target in HELPERS:
                values, mask, _ = history(node.args[0].id)
                expanded = mask.reshape(*mask.shape, *([1] * (values.ndim - 2)))
                valid = valid & mask.any(dim=1)
                if target in {'history_all', 'history_any'}:
                    if values.dtype != torch.bool: raise ValueError(f'{target} requires boolean history')
                    return ((values | ~expanded).all(dim=1) if target == 'history_all'
                            else (values & expanded).any(dim=1))
                total = torch.where(expanded, values, 0).sum(dim=1)
                if target == 'history_sum': return total
                count = mask.sum(dim=1).clamp_min(1).reshape(engine.num_envs, *([1] * (total.ndim - 1)))
                return total / count
            if target in FUNCTIONS:
                function = torch
                for part in target.split('.')[1:]: function = getattr(function, part)
            else:
                receiver = run(node.func.value)
                if not isinstance(receiver, torch.Tensor): raise ValueError('Expression methods require a Tensor')
                function = getattr(receiver, node.func.attr)
            args, kwargs = [run(x) for x in node.args], {x.arg: run(x.value) for x in node.keywords}
            operation = target.split('.')[-1] if target in FUNCTIONS else node.func.attr
            if operation in {'where', 'minimum', 'maximum', 'logical_and', 'logical_or', 'clamp'}:
                check_broadcast([*args, *kwargs.values(), *([receiver] if target not in FUNCTIONS else [])])
            tensor = args[0] if target in FUNCTIONS and args else (receiver if target not in FUNCTIONS else None)
            dimension_operations = {'sum', 'mean', 'amin', 'amax', 'all', 'any', 'diff',
                                    'vector_norm', 'norm', 'stack', 'cat', 'squeeze', 'unsqueeze'}
            if operation in dimension_operations:
                # These calls must keep each independent environment separate.
                if operation in {'stack', 'cat'}:
                    tensor = tensor[0]
                offset = 1 if target in FUNCTIONS else 0
                position = offset + (1 if operation in {'diff', 'vector_norm', 'norm'} else 0)
                default = -1 if operation == 'diff' else 0 if operation in {'stack', 'cat'} else None
                dim = kwargs.get('dim', args[position] if len(args) > position else default)
                if dim is None: raise ValueError(f'{operation} requires explicit dim; environment reduction is forbidden')
                dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
                rank = tensor.ndim + (1 if operation in {'stack', 'unsqueeze'} else 0)
                if any(d % rank == 0 for d in dimensions):
                    raise ValueError('Cannot reduce, concatenate or insert along the environment axis')
            return function(*args, **kwargs)
        raise ValueError(f'Unsupported expression node: {type(node).__name__}')

    with torch.no_grad():
        result = run(parse(text))
    if not isinstance(result, (torch.Tensor, int, float, bool)):
        raise ValueError('expr must return a tensor or scalar')
    return engine._reading(Reading(result, valid))


class HistoryBuffer:
    """Bounded, per-environment rings. Reads are oldest to newest, left padded."""
    def __init__(self, reading, size, engine):
        import torch
        self.values = torch.zeros((engine.num_envs, size, *reading.value.shape[1:]),
                                  dtype=reading.value.dtype, device=engine.device)
        self.valid = torch.zeros((engine.num_envs, size), dtype=torch.bool, device=engine.device)
        self.times = torch.zeros_like(self.valid, dtype=torch.float64)
        self.cursor = torch.zeros(engine.num_envs, dtype=torch.long, device=engine.device)
        self.last_step = torch.full_like(self.cursor, -1)
        self.size = size

    def append(self, reading, elapsed, step):
        import torch
        if self.values.shape[2:] != reading.value.shape[1:] or self.values.dtype != reading.value.dtype:
            raise ValueError('Variable history shape/dtype changed during an episode')
        rows = torch.where(self.last_step != step)[0]
        columns = self.cursor[rows]
        self.values[rows, columns] = reading.value[rows].detach()
        self.valid[rows, columns] = reading.valid[rows]
        self.times[rows, columns] = elapsed[rows].double()
        self.cursor[rows] = (columns + 1) % self.size
        self.last_step[rows] = step

    def read(self):
        import torch
        columns = (self.cursor[:, None] + torch.arange(self.size, device=self.cursor.device)) % self.size
        rows = torch.arange(len(self.cursor), device=self.cursor.device)[:, None]
        return self.values[rows, columns], self.valid[rows, columns], self.times[rows, columns]

    def reset(self, ids):
        self.values[ids] = 0
        self.valid[ids] = False
        self.times[ids] = 0
        self.cursor[ids] = 0
        self.last_step[ids] = -1
