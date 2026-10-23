"""Generate project-local IDE types from the existing scene information catalog."""
import keyword
from pathlib import Path
from .objects import selected_fields

HEADER = '# Generated from scene.yaml; edit the scene and run check to refresh.\n'


def observation_fields(scene, *, catalog_directory=None, settings=None, factories=None, terms=None,
                       extra_fields=None):
    """Selectable information path -> catalog entry, as exposed by SceneInfo."""
    from .catalog import describe_scene
    from .layout import load_scene_config
    scene = Path(scene).resolve()
    settings = load_scene_config(scene) if settings is None else settings
    # A project entry knows its own package; the path alone cannot identify a
    # project added to ``projects.__path__``, so its factories are passed in.
    catalog = describe_scene(settings, directory=catalog_directory or scene.parent,
                             factories=factories, term_registry=terms)
    fields = {item['id']: item for item in catalog['observations']
              if item.get('available', True) and item.get('observation_selectable', True)
              and not item['id'].startswith('entity__')}
    for name, entry in (extra_fields or {}).items():
        if name in fields:
            raise ValueError(f'Duplicate observation metadata: {name}')
        fields[name] = entry
    # Simulation context is raw time information, not a task-derived feature.
    for key, unit in [('step_dt','s'), ('time','s')]:
        fields['simulation.' + key] = {'information': {'shape': [1], 'unit': unit}, 'description': '每个环境的仿真时间信息'}
    # component.input is a structured view, not a fictitious tensor.
    for role, selection in selected_fields(settings).items():
        fields.pop(role + '.input', None)
        if role != 'policy':
            for path in selection:
                if path in fields:
                    fields[role + '.input.' + path] = fields[path]
    return settings, fields


def _contracts(settings):
    """Return types of the project functions scene.yaml calls; unknown sizes stay None."""
    lines = ['\n\n']
    control = settings.get('control')
    if control:
        from .control import declared_dimension, joint_count
        for role in control['components']:
            shape = ['B', declared_dimension(control, role)]
            name = role.capitalize()
            if role == control['train']:
                note = (f"训练组件 forward 返回 ({name}Output 均值, Tensor ['B', 1] 价值)，模型另需参数 actor_logstd；"
                        "build 的 config 即 train.yaml 的 actor.model（action_dim、hidden_dims 等）")
            else:
                note = '固定组件 forward 返回'
            if role == 'executor':
                note += '；解码前动作，initial 返回同形状'
            lines.append(f'{name}Output = Tensor  # {shape}, {note}\n')
        outputs = control.get('output') or {}
        if outputs:
            lines.append('\n\nclass ExecutorTargets(TypedDict):  # decode(action, obs: ExecutorObs) 返回；实际单位，键与 control.output 一致\n')
            for key, output in outputs.items():
                if not key.isidentifier() or keyword.iskeyword(key):
                    raise ValueError(f'Cannot expose control output {key!r} as a Python key; rename it in the scene')
                joints = list(output.get('joint_names', []))
                lines.append(f"    {key}: Tensor  # {['B', joint_count(output)]}, {output.get('type')}, "
                             f"{output.get('asset_name')}: {', '.join(joints)}\n")
    lines.append('\n\nclass TaskResult(TypedDict, total=False):  # evaluate(obs: SceneInfo) 返回；缺省 reward=0, success/failure=False\n'
                 '    # 每决策步评价一次：reward 乘 step_dt 计入回报；success 或 failure 为真即结束该环境的回合。\n'
                 '    # 计时等跨步状态：写 Task 类，在 __init__ 中建 shared.assembly.task_memory.TaskMemory（hold 计时、mean 平均），按环境随复位清零。\n'
                 "    reward: Tensor  # ['B']\n"
                 "    success: Tensor  # ['B'], bool\n"
                 "    failure: Tensor  # ['B'], bool\n"
                 "    metrics: dict[str, Tensor]  # 名称 -> ['B']；回合最后一步的值记入训练/评估日志 env/task/名称\n")
    return ''.join(lines)


def generate(scene, *, catalog_directory=None, settings=None, factories=None, terms=None,
             extra_fields=None, write=True):
    scene = Path(scene).resolve()
    settings, fields = observation_fields(scene, catalog_directory=catalog_directory, settings=settings,
                                          factories=factories, terms=terms, extra_fields=extra_fields)
    groups = {'SceneInfo': list(fields)}
    selections = selected_fields(settings)
    for role, paths in selections.items():
        missing = set(paths) - fields.keys()
        if missing:
            raise ValueError(f'{role}: unavailable observation fields {sorted(missing)}')
        groups[role.capitalize() + 'Obs' if role != 'policy' else 'Obs'] = paths
    from .layout import world_spec_from_config
    from .robot import RobotView, robot_frames
    robots = robot_frames(world_spec_from_config(settings, directory=catalog_directory or scene.parent,
                                                 factories=factories))
    methods = {name for name in vars(RobotView) if not name.startswith('_')}
    lines = [HEADER, 'from typing import TypedDict\n\nfrom torch import Tensor\nfrom shared.assembly.objects import ObjectView\n'
             'from shared.assembly.robot import RobotView\n\n']
    for root, paths in groups.items():
        tree = {}
        for path in sorted(paths):
            node = tree
            for part in path.split('.'):
                if not part.isidentifier() or keyword.iskeyword(part) or part.startswith('_'):
                    raise ValueError(f'Cannot expose {path!r} as Python attributes; rename it in the scene')
                if node.get('_leaf'):
                    raise ValueError(f'Information path collision: {path}')
                node = node.setdefault(part, {})
            if node:
                raise ValueError(f'Information path collision: {path}')
            node['_leaf'] = path
        def emit(node, name, robot=None):
            children = {}
            for attr, child in node.items():
                if '_leaf' not in child:
                    child_name = name + '_' + attr
                    # Only a robot's own top-level node queries it; component.input.<robot> is a cached past reading.
                    emit(child, child_name, attr if name == root and attr in robots else None)
                    children[attr] = child_name
            if robot is not None:
                if node.keys() & methods:
                    raise ValueError(f'{robot}: information names {sorted(node.keys() & methods)} clash with robot methods')
                frames = ', '.join(robots[robot]) or '无'
                lines.append(f'class {name}(RobotView):\n'
                             f'    # 方法 joint_names / frame_pose / jacobian / differential_ik（环境进程内可用，被训练组件内报错）；\n'
                             f'    # frame 取本体 body 名或已命名坐标系：{frames}\n')
            else:
                lines.append(f'class {name}(ObjectView):\n')
            if not node:
                lines.append('    pass\n')
            for attr, child in node.items():
                if attr in children:
                    lines.append(f'    {attr}: {children[attr]}\n')
                else:
                    entry = fields[child['_leaf']]
                    info = entry.get('information') or {}
                    shape = ['B', *(info.get('shape') or [])]
                    comment = ', '.join(str(v) for v in [shape, info.get('dtype'), info.get('unit'), info.get('frame'), info.get('encoding'), entry.get('label')] if v is not None)
                    lines.append(f'    {attr}: Tensor  # {comment}\n')
            lines.append('\n\n')
        emit(tree, root)
    if 'control' in settings:
        lines.append(f"Obs = {settings['control']['train'].capitalize()}Obs\n")
    lines.append('SCENE_FIELDS = ' + repr(tuple(groups['SceneInfo'])) + '\n')
    lines.append(_contracts(settings))
    text = ''.join(lines)
    target = scene.parent / 'observation.py'
    if target.exists() and not target.read_text().startswith(HEADER):
        raise ValueError(f'{target} is not generated; rename it before generating observation types')
    if not write:
        return text
    if not target.exists() or target.read_text() != text:
        import os, tempfile
        fd, temporary = tempfile.mkstemp(prefix='.observation-', dir=target.parent)
        try:
            with os.fdopen(fd, 'w') as output: output.write(text)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    return target
