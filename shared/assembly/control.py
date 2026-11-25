"""Compile the planner/executor boundary; physics and stepping remain native."""
from copy import deepcopy
from importlib import import_module
import math

from .assets import DEFAULT_BACKEND, backend_module
from .layers import type_name


def load_factory(path):
    module, separator, name = path.partition(':')
    if not separator or not module or not name:
        raise ValueError('Model factory must be module:function')
    factory = getattr(import_module(module), name)
    if not callable(factory):
        raise ValueError(f'Model factory is not callable: {path}')
    return factory


def component_observations(selection):
    if not isinstance(selection, list) or not selection or any(not isinstance(s, str) or not s for s in selection):
        raise ValueError('observations must be an ordered list of basic information names')
    if len(selection) != len(set(selection)):
        raise ValueError('Duplicate observations')
    return [{'source': source} for source in selection]


CONTROL_MODES = ('joint_position', 'joint_velocity', 'joint_effort')


def actuated(node):
    """Robots the executor drives: assets declaring an articulation (Go2, Piper, Franka) or entities of kind
    robot; ``node`` is a scene description node or a scene.yaml entity. Articulated objects (drawers, doors)
    only report joint_state and stay passive; a kinematic entity is moved by scene motion or the Task."""
    from collections.abc import Mapping
    from .catalog import asset_capabilities, factory_capabilities
    field = (lambda name, default=None: node.get(name, default)) if isinstance(node, Mapping) else \
        (lambda name, default=None: getattr(node, name, default))
    factory = field('factory')
    return bool(factory) and not field('kinematic') and (field('kind') == 'robot' or 'articulation' in asset_capabilities(
        factory, factory_capabilities(factory), field('parameters') or {}))


def automatic_outputs(world, modes=None):
    """One control target per robot of the scene, named after it, driving all its joints in the order of its
    joint readings (obs.<robot>.joint_position). ``modes`` (control.modes) gives a robot velocity or effort
    targets, or ``none``: not driven by the policy (it keeps its default targets)."""
    modes = dict(modes or {})
    if not isinstance(modes, dict) or any(mode not in (*CONTROL_MODES, 'none') for mode in modes.values()):
        raise ValueError(f'policy.modes (control.modes) maps robot names to one of {", ".join((*CONTROL_MODES, "none"))}')
    outputs = {}
    for path, (node, _) in world.index().items():
        if not actuated(node):
            continue
        key = path.split('/', 1)[1].replace('/', '__')
        mode = modes.pop(key, 'joint_position')
        if mode != 'none':
            outputs[key] = {'type': mode, 'asset_name': key, 'joint_names': ['.*']}
    if modes:
        raise ValueError(f'policy.modes (control.modes) names entities that are not robots: {", ".join(sorted(modes))}')
    return outputs


def effective_outputs(control, world):
    """The control targets of a scene: control.output when it lists them (older projects), else one per
    robot (automatic_outputs)."""
    if 'output' in control:
        if 'modes' in control:
            raise ValueError('policy.modes applies to automatic targets; with policy.output (control.output) set the type per target')
        return control['output']
    return automatic_outputs(world, control.get('modes'))


def validate_outputs(outputs):
    if not isinstance(outputs, dict):
        raise ValueError('control.output must map names to control bindings')
    if not outputs:
        raise ValueError('没有受策略控制的机器人（场景里没有机器人，或都在 policy.modes 中设为 none），策略无法驱动任何关节')
    for key, output in outputs.items():
        if not isinstance(key, str) or not key.isidentifier() or key.startswith('_'):
            raise ValueError('Control target names must be public Python identifiers')
        if not isinstance(output, dict) or set(output) != {'type', 'asset_name', 'joint_names'}:
            raise ValueError('Each control output binds type, asset_name and joint_names; implement scaling in policy.py')
        if output['type'] not in CONTROL_MODES:
            raise ValueError('Use position, velocity or effort; convert higher-level commands in policy.py')
        if not isinstance(output['joint_names'], list) or not output['joint_names']:
            raise ValueError('A control target must select joints')
        if any(not isinstance(joint, str) or not joint for joint in output['joint_names']):
            raise ValueError('Joint names must be nonempty strings')
    validate_target_conflicts(outputs)


def require_listed_joints(outputs):
    """Authored scenes name every controlled joint; regex selections only remain readable in old snapshots."""
    from .properties import is_joint_pattern
    for name, output in outputs.items():
        patterns = [joint for joint in output['joint_names'] if is_joint_pattern(joint)]
        if patterns:
            raise ValueError(f'控制目标 {name} 用正则 {patterns} 选择关节；请逐个列出关节名（顺序即动作维度顺序）')



def overlay_outputs(outputs, changes):
    """The same effective bindings for panels, dialogs and validation."""
    result = dict(outputs)
    for name, binding in changes.items():
        if binding is None:
            result.pop(name, None)
        else:
            result[name] = binding
    return result


def validate_target_conflicts(outputs, measured_joints=None):
    """Reject known overlaps without loading a simulator.

    Resolve patterns against measured names when available. Otherwise detect
    exact names, equal patterns and universal patterns; arbitrary regex overlap
    still receives the authoritative native check at environment initialization.
    """
    from .properties import is_joint_pattern, joint_selected
    entries = list(outputs.items())
    for index, (name, binding) in enumerate(entries):
        for other_name, other in entries[:index]:
            asset = binding['asset_name']
            if asset != other['asset_name']:
                continue
            left, right = binding['joint_names'], other['joint_names']
            candidates = set((measured_joints or {}).get(asset, ()))
            candidates.update(p for p in left + right if not is_joint_pattern(p))
            conflict = next((joint for joint in sorted(candidates)
                if any(joint_selected(p, joint) for p in left)
                and any(joint_selected(p, joint) for p in right)), None)
            if conflict is None and (set(left) & set(right) or '.*' in left or '.*' in right):
                conflict = '重叠的关节选择'
            if conflict is not None:
                raise ValueError(f'控制目标 {other_name} 与 {name} 冲突：{asset}/{conflict}。'
                                 '请编辑现有目标切换模式，或先取消原绑定；同一关节只能由一个目标控制')

def free_target_name(outputs, base):
    """A public, unused control target name derived from the controlled asset."""
    if base not in outputs:
        return base
    index = 2
    while f'{base}_{index}' in outputs:
        index += 1
    return f'{base}_{index}'


def observation_order(selected, catalog):
    """Editor list order: the selected terms keep their saved order (the model's input order); each
    unselected catalog term follows the selected term that precedes it in the catalog. Ticking a term
    again therefore puts it back where it was instead of at the end."""
    selected, chosen = list(selected), set(selected)
    order, placed = list(selected), set(selected)
    anchor = None
    for key in catalog:
        if key in chosen:
            anchor = key
        elif key not in placed:
            index = 0 if anchor is None else order.index(anchor) + 1
            while index < len(order) and order[index] not in chosen:
                index += 1  # After earlier unselected followers of the same anchor.
            order.insert(index, key)
            placed.add(key)
    return order


def default_target_joints(available, controlled, joints=()):
    """Default joints for a new target: the uncontrolled joints of the first one's kind.

    ``joints`` are kinematic joint records (``name``, ``type``, ``limits``). Arm and fingers differ in
    type (revolute vs prismatic), legs and wheels in having limits, so a new ``arm`` target leaves the
    gripper for a target of its own. Without records every uncontrolled joint is chosen.
    """
    free = [joint for joint in available if joint not in controlled]
    kinds = {joint['name']: (joint.get('type'), bool(joint.get('limits'))) for joint in joints}
    if not free or free[0] not in kinds:
        return free
    return [joint for joint in free if kinds.get(joint) == kinds[free[0]]]


def toggle_target(outputs, entity, mode, joint, enabled, joints=()):
    """Control-target changes that add or remove one actuator property.

    One binding per asset and mode keeps the scene property list and the
    controller panel reading the same named target. Explicit joint lists are
    edited in place; ``joint=None`` selects every name in ``joints`` (the
    measured joints of that asset), written out one by one.
    """
    from shared.assembly.properties import is_joint_pattern, joint_selected
    if mode not in ('joint_position', 'joint_velocity', 'joint_effort'):
        raise ValueError('Control targets select joint position, velocity or effort')
    if enabled:
        others = {name: binding for name, binding in outputs.items()
                  if binding.get('type') != mode}
        candidate = free_target_name(others, entity)
        if joint is None and not joints:
            raise ValueError(f'{entity} 的关节名尚未读出；先 inspect，再逐个选择关节')
        validate_target_conflicts({**others, candidate: {'type': mode, 'asset_name': entity,
                                                       'joint_names': [joint] if joint else list(joints)}})
    matching = {name: binding for name, binding in outputs.items()
                if binding.get('asset_name') == entity and binding.get('type') == mode}
    if not enabled:
        if joint is None:
            return {name: None for name in matching}
        changes = {}
        for name, binding in matching.items():
            patterns = [item for item in binding['joint_names'] if is_joint_pattern(item)]
            if patterns:
                raise ValueError(f'控制目标 {name} 用通配关节选择；请在那里修改，而不是逐个关节取消')
            remaining = [item for item in binding['joint_names'] if item != joint]
            if remaining == binding['joint_names']:
                continue
            changes[name] = None if not remaining else {**binding, 'joint_names': remaining}
        return changes
    if joint is None:
        missing = [item for item in joints if not any(joint_selected(pattern, item)
                   for binding in matching.values() for pattern in binding['joint_names'])]
        explicit = next((name for name, binding in matching.items()
                         if not any(is_joint_pattern(item) for item in binding['joint_names'])), None)
        if not missing:
            return {}
        if explicit is not None:
            binding = matching[explicit]
            return {explicit: {**binding, 'joint_names': [*binding['joint_names'], *missing]}}
        name = free_target_name(outputs, entity)
        return {name: {'type': mode, 'asset_name': entity, 'joint_names': missing}}
    covered = any(joint_selected(pattern, joint)
                  for binding in matching.values() for pattern in binding['joint_names'])
    if covered:
        return {}
    explicit = next((name for name, binding in matching.items()
                     if not any(is_joint_pattern(item) for item in binding['joint_names'])), None)
    if explicit is not None:
        binding = matching[explicit]
        return {explicit: {**binding, 'joint_names': [*binding['joint_names'], joint]}}
    name = free_target_name(outputs, entity)
    return {name: {'type': mode, 'asset_name': entity, 'joint_names': [joint]}}


def joint_count(output):
    """Joints a control endpoint drives; regex selections in old snapshots stay unknown."""
    joints = list(output.get('joint_names', []))
    return len(joints) if joints and all(name.replace('_', '').isalnum() for name in joints) else None


def declared_dimension(control, role):
    """Output size known from scene.yaml alone: explicit, or executor auto = total endpoint joints."""
    dimension = control['components'][role].get('dimension', 'auto')
    if dimension is None:  # a policy layer: measured from the model
        return None
    if type(dimension) is int:
        return dimension
    counts = [joint_count(output) for output in (control.get('output') or {}).values()]
    return sum(counts) if role == 'executor' and counts and None not in counts else None


def period_steps(name, seconds, dt):
    """Physics steps per decision of one component."""
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'{name}.period must be positive seconds')
    steps = seconds / dt
    if steps < 1 or not math.isclose(steps, round(steps), abs_tol=1e-8):
        raise ValueError(f'{name}.period {seconds} s 须为物理步 {dt} s 的整数倍')
    return round(steps)


def component_config(control, project_package):
    """Validated internal control: ``components`` in order (the last, ``driver``, drives the robots).

    scene.yaml ``policy:`` arrives here as converted by layers.control_from_policy (``layers: True``: one or two
    layers of any names, output sizes measured at run time); the older ``control:`` has a planner and an executor
    with declared dimensions."""
    control = deepcopy(control)
    components = control.get('components')
    layered = control.get('layers') is True
    allowed = {'train', 'components', 'output', 'modes', 'driver', 'layers', 'structure'}
    if not {'train', 'components'} <= set(control) <= allowed or not isinstance(components, dict):
        raise ValueError('control requires train and components (output or modes optional)')
    if 'output' in control and 'modes' in control:
        raise ValueError('policy.modes applies to automatic targets; with policy.output (control.output) set the type per target')
    if layered:
        if not 1 <= len(components) <= 2 or control.get('driver') != list(components)[-1]:
            raise ValueError('policy.layers names one or two layers; the last drives the robots')
    else:
        if set(components) != {'planner', 'executor'}:
            raise ValueError('Declare both planner and executor')
        components = {name: components[name] for name in ('planner', 'executor')}
        control['driver'] = 'executor'
    if 'output' in control:
        validate_outputs(control['output'])
    if control['train'] not in components:
        raise ValueError(f'control.train must select one of {list(components)}')
    control['components'] = components
    driver = control['driver']
    for name, cfg in control['components'].items():
        if cfg.keys() - {'period', 'dimension', 'model', 'parameters', 'observations', 'decode', 'initial', 'history'}:
            raise ValueError(f'{name}: unsupported component fields')
        if not isinstance(cfg.get('model'), str):
            raise ValueError(f'{name}.model must explicitly select a model class or factory')
        if cfg['model'].startswith('.'):
            if not project_package:
                raise ValueError('Relative model factories require a project package')
            cfg['model'] = project_package + cfg['model']
        load_factory(cfg['model'])
        if name == driver and 'decode' not in cfg:
            raise ValueError(f'{name}.decode must explicitly map actions to the robots\' joint targets')
        for field in ('decode', 'initial'):
            if field in cfg:
                if name != driver or not isinstance(cfg[field], str):
                    raise ValueError(f'{field} belongs to the layer driving the robots ({driver})')
                if cfg[field].startswith('.'):
                    cfg[field] = project_package + cfg[field]
                load_factory(cfg[field])

        if not isinstance(cfg.get('parameters', {}), dict):
            raise ValueError('Model parameters must be a mapping')
        if layered:
            cfg['dimension'] = None  # measured from the model on the first step
        else:
            dimension = cfg.get('dimension', 'auto')
            if dimension != 'auto' and (type(dimension) is not int or dimension <= 0):
                raise ValueError('Component dimension must be auto or a positive integer')
            cfg['dimension'] = dimension
            if name == 'planner' and dimension == 'auto':
                raise ValueError('A planner must declare its output dimension')
        if not cfg.get('observations'):
            raise ValueError(f'{name} 尚未选择输入；在编辑器“策略”里添加，至少一项')
        component_observations(cfg.get('observations'))
        if 'history' in cfg:
            from .history import validate_history
            validate_history(cfg['history'], cfg['observations'], name)
    return control


def compile_control(settings, registry, *, project_package=None, backend=DEFAULT_BACKEND, world=None):
    """``world`` (the scene description) resolves automatic control targets (effective_outputs)."""
    if 'control' not in settings:
        return settings, registry
    settings, registry = deepcopy(settings), deepcopy(registry)
    control = dict(settings['control'])
    if 'output' not in control:
        if world is None:
            raise ValueError('Automatic control targets need the scene description')
        control['output'] = effective_outputs(control, world)
        control.pop('modes', None)
    control = component_config(control, project_package)
    components, selected = control['components'], control['train']
    if settings.get('actions') or settings.get('observations'):
        raise ValueError('With control, declare observations on components and basic control in output')
    dt = float(settings.get('simulation', {}).get('dt', .005))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('simulation.dt must be positive')
    periods = {}
    from shared.assembly.entity_information import parse_signal
    from shared.assembly.targets import target_catalog
    targets = target_catalog(settings)
    for name, cfg in components.items():
        periods[name] = period_steps(name, cfg.get('period'), dt)
        cfg['_view'] = f'{project_package}.observation:{type_name(name)}Obs' if project_package else None
        cfg['_targets'] = {}
        for entry in component_observations(cfg['observations']):
            source = entry['source']
            if source in targets:
                cfg['_targets'][source] = targets[source][0]
            elif source in {name + '.output', name + '.previous_output'}:
                if cfg['dimension'] == 'auto' and name == 'planner':
                    raise ValueError('A recurrent planner must declare dimension')
                if control.get('layers') and name != selected:
                    raise ValueError(f'{name} is not trained and keeps no state: it cannot read its own output')
            elif name == 'executor' and source == 'planner.output' and 'planner' in components:
                pass
            elif control.get('layers') and source.split('.')[0] in components:
                pass  # the other layer's output (layers.layer_sources)
            elif source in settings.get('variables', {}):
                raise ValueError(f'{name}: select basic information; compute {source!r} inside the model')
            elif source != 'simulation.gravity' and not parse_signal(source):
                raise ValueError(f'{name}: unavailable input {source!r}')
    order = list(components)
    if len(order) == 2 and periods[order[0]] % periods[order[1]]:
        raise ValueError(f'{order[1]} period must divide {order[0]} period')
    world = settings.setdefault('world', {})
    if 'decimation' in world and world['decimation'] != periods[selected]:
        raise ValueError('world.decimation conflicts with selected component period')
    world['decimation'] = periods[selected]
    settings['control'] = control
    settings['observations'] = {'policy': list(components[selected]['observations'])}
    settings['actions'] = {'control_chain': {'type': 'control_chain'}}
    install = backend_module(backend, 'control').install_terms
    install(control, dt, settings, registry)
    return settings, registry
