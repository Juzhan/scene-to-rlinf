"""One editor-facing view of Signals metadata, not another signal reader.

Stable references and capabilities come from the existing catalog. Optional live
editor overlays carry unsaved authoring values; these are never simulation data.

Rows also state whether an actuator property is already a control target, so the
scene property list and the controller panel read the same ``control.output``.
"""
from dataclasses import dataclass, field
from copy import deepcopy
import re


def is_joint_pattern(value):
    return isinstance(value, str) and any(character in value for character in '*?[')


def joint_selected(pattern, joint):
    """Native control resolves joint_names as regular expressions, not globs."""
    if not isinstance(pattern, str):
        return False
    if pattern == joint:
        return True
    try:
        return re.fullmatch(pattern, joint) is not None
    except re.error:
        return False


def control_coverage(control, entity, mode, joint=None):
    """Binding names whose asset, mode and joints already cover this property."""
    if not control or not entity:
        return []
    covered = []
    for name, binding in (control.get('output') or {}).items():
        if binding.get('asset_name') != entity or binding.get('type') != mode:
            continue
        if joint is None:
            covered.append(name)
            continue
        for pattern in binding.get('joint_names', ()):
            if joint_selected(pattern, joint):
                covered.append(name)
                break
    return covered


@dataclass
class Property:
    reference: str
    owner: str
    name: str
    information: dict = field(default_factory=dict)
    value: object = None
    value_source: str = 'runtime'
    observation: str | None = None
    task: str | None = None
    edit: tuple | None = None
    children: list = field(default_factory=list)
    # Actuator properties publish what a control target would bind; rows that
    # already are one list the matching control.output names.
    controllable: dict | None = None
    control: list = field(default_factory=list)


def properties(snapshot, *, entities=None, variables=None):
    """Build the current directory from a shared snapshot and authoring overlays."""
    overlays = entities or {}
    rows = []
    control = snapshot.get('control')
    variable_definitions = snapshot.get("variables", {}) if variables is None else variables
    variable_names = set(snapshot.get("variables", {})) | set(variable_definitions)
    records = {item['path']: item for item in snapshot.get('entities', [])}
    for signal in snapshot.get('catalog', {}).get('observations', []):
        ref = signal['id']
        if ref in variable_names:
            continue
        # New entity references use dot notation. Preset/custom observation terms
        # remain observable, but are not invented as task signal readers.
        sources = signal.get('source_groups', {}).get('default', {}).get('sources', [])
        owner = signal.get('entity') or (sources[0]['path'] if len(sources) == 1 else
                                         signal.get('source_groups', {}).get('default', {}).get('group', 'custom'))
        if entities is not None and owner in records and owner not in overlays:
            continue
        record = records.get(owner, {})
        live = overlays.get(owner, {})
        prefix = record.get('native_key', '') + '.'
        name = ref[len(prefix):] if prefix != '.' and ref.startswith(prefix) else ref
        info = deepcopy(signal.get('information') or {})
        task = ref if ('.' in ref or ref.startswith('command__')) and not ref.startswith('entity__') else None
        row = Property(ref, owner, name, info, observation=ref if signal.get('available', True) else None, task=task)
        if 'value' in signal:
            row.value, row.value_source = signal['value'], 'authored target'
        initial = live.get('initial', record.get('initial_display', {}))
        if record.get('kind') == 'abstract' and name == 'size':
            row.value = live.get('size', row.value)
            row.edit = (owner, 'size')
        elif name in ('position', 'orientation'):
            row.value = live.get(name)
            row.value_source = 'authoring / local'
            row.edit = (owner, name) if name in live and live.get('editable', True) else None
        elif name in ('linear_velocity', 'angular_velocity'):
            row.value = initial.get(name)
            row.value_source = 'initial'
            row.edit = (owner, 'initial')
        elif name in ('joint_position', 'joint_velocity', 'joint_effort'):
            names = list(initial.get('joint_positions', {}))
            values = initial.get('joint_positions' if name == 'joint_position' else 'joint_velocities', {}) if name != 'joint_effort' else {}
            if names:
                row.information['shape'] = [len(names)]
            row.edit = (owner, 'initial') if name != 'joint_effort' else None
            entity = record.get('native_key')
            row.controllable = {'entity': entity, 'mode': name, 'joint': None, 'joints': names} if control and entity else None
            row.control = control_coverage(control, entity, name)
            for i, joint in enumerate(names):
                child = Property(ref + f'[..., {i}]', owner, joint,
                                 {**info, 'shape': []}, values.get(joint),
                                 'initial' if joint in values else 'runtime',
                                 task=task + f'[..., {i}]' if task else None,
                                 edit=row.edit)
                if row.controllable:
                    child.controllable = {**row.controllable, 'joint': joint}
                    child.control = control_coverage(control, entity, name, joint)
                row.children.append(child)
        rows.append(row)
    def parameter_nodes(owner, prefix, mapping):
        result = []
        for name, value in mapping.items():
            ref = prefix + '.' + name
            result.append(Property(ref, owner, name, value=None if isinstance(value, dict) else value,
                value_source='asset configuration', children=parameter_nodes(owner, ref, value) if isinstance(value, dict) else []))
        return result
    for owner, record in records.items():
        if entities is not None and owner not in overlays:
            continue
        if record.get('parameters'):
            rows.append(Property(owner + '.parameters', owner, 'Asset parameters',
                value_source='asset configuration', children=parameter_nodes(owner, owner + '.parameters', record['parameters'])))
    for name, definition in variable_definitions.items():
        rows.append(Property(name, '共享变量', name,
                             deepcopy(definition.get('information', {})) if isinstance(definition, dict) else {},
                             value_source='computed', task=name,
                             observation=None))
    return rows
