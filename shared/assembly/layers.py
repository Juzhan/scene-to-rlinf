"""scene.yaml ``policy:``: one or two layers, read into the internal control mapping every runtime part uses.

::

    policy:
      layers:                      # in order; the last one drives the robots
        policy: {model: .policy:Policy, period: 0.02, inputs: [robot.joint_position, destination.position]}
      decode: .policy:decode_action
      initial: .policy:initial_action    # optional
      modes: {piper: joint_velocity}     # optional: robots not driven by joint positions (or none)
      train: lower                       # with two layers: the one that learns

An input is scene information (``robot.joint_position``) or another layer's name, read as that layer's output:
``obs.upper.goal`` when it returns a dict with a ``goal`` entry, ``obs.upper.output`` when it returns a Tensor;
a layer's own name is its previous action. What a layer returns is read from policy.py without importing
it (``output_structure``); sizes the source does not state are measured on the first step.

The internal mapping is the older ``control:`` (still accepted): ``components`` in layer order with their
``observations``, ``train``, ``modes``/``output``, plus ``driver`` (the last layer), ``layers: True`` and
``structure`` (output_structure).
"""
import ast
from pathlib import Path
import re

FIELDS = {'model', 'parameters', 'period', 'inputs', 'history'}
POLICY_FIELDS = {'layers', 'train', 'decode', 'initial', 'modes', 'output'}


def control_from_policy(policy, directory=None):
    """The internal control mapping of a scene.yaml ``policy:``; ``directory`` holds policy.py."""
    if not isinstance(policy, dict) or policy.keys() - POLICY_FIELDS:
        raise ValueError(f'policy: unknown fields {sorted(set(policy) - POLICY_FIELDS) if isinstance(policy, dict) else policy}')
    layers = policy.get('layers')
    if not isinstance(layers, dict) or not 1 <= len(layers) <= 2:
        raise ValueError('policy.layers names one layer, or two (the first feeds the second)')
    names = list(layers)
    for name in names:
        if not isinstance(name, str) or not name.isidentifier() or name.startswith('_') or name in ('simulation', 'input'):
            raise ValueError(f'policy.layers: {name!r} is not a usable layer name (a Python name, not simulation or input)')
    train = policy.get('train', names[0] if len(names) == 1 else None)
    if train not in layers:
        raise ValueError(f'policy.train must name the layer that learns: one of {names}')
    if 'decode' not in policy:
        raise ValueError('policy.decode names the function turning actions into the robots\' joint targets')
    structure = output_structure(policy, directory)
    components = {}
    for name, layer in layers.items():
        if not isinstance(layer, dict) or layer.keys() - FIELDS:
            raise ValueError(f'policy.layers.{name}: unknown fields {sorted(set(layer) - FIELDS) if isinstance(layer, dict) else layer}')
        inputs = layer.get('inputs', [])
        if not isinstance(inputs, list):
            raise ValueError(f'policy.layers.{name}.inputs lists scene information and layer names')
        # An empty list is a scene being edited; compiling it says the layer has no input yet.
        sources = []
        for token in inputs:
            sources += layer_sources(token, name, layers, structure)
        cfg = {'model': layer.get('model'), 'period': layer.get('period'), 'observations': sources, 'dimension': None}
        if 'parameters' in layer:
            cfg['parameters'] = layer['parameters']
        if 'history' in layer:  # keyed by inputs entries, as inputs are
            history = layer['history']
            if not isinstance(history, dict):
                raise ValueError(f'policy.layers.{name}.history maps inputs to a number of frames')
            cfg['history'] = {source: frames for token, frames in history.items()
                              for source in layer_sources(token, name, layers, structure)}
        if name == names[-1]:
            cfg['decode'] = policy['decode']
            if 'initial' in policy:
                cfg['initial'] = policy['initial']
        components[name] = cfg
    control = {'train': train, 'components': components, 'driver': names[-1], 'layers': True, 'structure': structure}
    for key in ('modes', 'output'):
        if key in policy:
            control[key] = policy[key]
    return control


def type_name(layer):
    """Generated type prefix of a layer: planner -> Planner, low_level -> LowLevel."""
    return ''.join(part[:1].upper() + part[1:] for part in layer.split('_'))


def layer_sources(token, name, layers, structure):
    """Internal observation sources of one ``inputs`` entry of layer ``name``."""
    if not isinstance(token, str) or not token:
        raise ValueError(f'{name}.inputs: entries are scene information names or layer names')
    if token not in layers:
        return [token]
    if token == name:
        return [name + '.output']  # its previous action
    parts = (structure.get(token) or {}).get('parts')
    return [f'{token}.{part}' for part in parts] if parts else [token + '.output']


def layer_tokens(sources, layers):
    """The scene.yaml ``inputs`` of internal observation sources: a layer's output (``plan.goal``, ``arms.output``) is
    selected as the layer's name."""
    tokens = []
    for source in sources:
        head, _, tail = source.partition('.')
        token = head if head in layers and tail != 'previous_output' else source
        if token not in tokens:
            tokens.append(token)
    return tokens


def normalize(settings, directory=None):
    """``settings`` with a ``policy:`` section also as the internal ``control`` (a copy; others unchanged)."""
    if not isinstance(settings, dict) or 'policy' not in settings:
        return settings
    if 'control' in settings:
        if (settings['control'] or {}).get('layers') is True:
            return settings  # already read
        raise ValueError('scene.yaml has both policy and control; keep policy (control is the older form)')
    return {**settings, 'control': control_from_policy(settings['policy'], directory)}


# --------------------------------------------------------------------------------------------- policy.py source
def output_structure(policy, directory):
    """Layer -> {'parts': [name...], 'shapes': {part: [...]}, 'notes': {part: text}} for layers whose forward
    returns a dict literal in policy.py; {'tensor': True} otherwise; absent
    when the model cannot be found. Read from the source (no import), so the editor and observation.py work before
    the code runs; sizes a comment does not state are measured on the first step."""
    result = {}
    if directory is None:
        return result
    for name, layer in (policy.get('layers') or {}).items():
        reference = (layer or {}).get('model') if isinstance(layer, dict) else None
        found = _model_output(reference, Path(directory))
        if found:
            result[name] = found
    return result


def _source(reference, directory):
    if not isinstance(reference, str) or ':' not in reference:
        return None, None
    module, _, symbol = reference.partition(':')
    if module.startswith('.'):
        path = directory / (module[1:].replace('.', '/') + '.py')
    else:
        from importlib.util import find_spec
        try:
            spec = find_spec(module)
        except (ImportError, ValueError):
            return None, None
        path = Path(spec.origin) if spec and spec.origin else None
    if path is None or not path.is_file():
        return None, None
    return path, symbol


def _model_output(reference, directory):
    path, symbol = _source(reference, directory)
    if path is None:
        return None
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    top = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    node = top.get(symbol)
    if isinstance(node, ast.FunctionDef):  # a factory: its return annotation or the class it returns
        model = _name(node.returns) or next((_name(item.value.func) for item in ast.walk(node)
                                             if isinstance(item, ast.Return) and isinstance(item.value, ast.Call)), None)
        node = top.get(model)
    if not isinstance(node, ast.ClassDef):
        return None
    forward = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == 'forward'), None)
    if forward is None:
        return None
    # A dict returned as a literal, {'goal': ..., 'heading': ...}: its keys are the parts; a comment after an entry
    # ([B, 3] what it is) gives the part's shape and meaning.
    returned = next((item.value for item in ast.walk(forward) if isinstance(item, ast.Return)
                     and isinstance(item.value, ast.Dict)), None)
    if returned is not None and returned.keys and all(isinstance(key, ast.Constant) and isinstance(key.value, str)
                                                      for key in returned.keys):
        lines = text.splitlines()
        parts, shapes, notes = [], {}, {}
        for key in returned.keys:
            part = key.value
            parts.append(part)
            comment = lines[key.lineno - 1].partition('#')[2].strip()
            shape = re.match(r'\[([^\]]*)\]', comment)
            if shape:  # [B, 3]: the sizes after the batch axis
                sizes = [_dimension(value) for value in shape.group(1).split(',') if value.strip()]
                shapes[part] = sizes[1:] if sizes[:1] == ['B'] else sizes
                comment = comment[shape.end():].strip(' ,;')
            notes[part] = comment
        return {'parts': parts, 'shapes': shapes, 'notes': notes}
    return {'tensor': True}


def _dimension(value):
    value = value.strip().strip('\'"')
    return int(value) if value.isdigit() else ('B' if value == 'B' else None)


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
