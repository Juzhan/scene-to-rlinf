"""Evaluate one policy under several declared conditions and tabulate the results.

A project's recipe lists its evaluation conditions; ``run.py evaluate <project> --conditions all``
runs each as its own evaluation (same checkpoint, same snapshot) and writes ``summary.json``::

    evaluation:
      conditions:
        nominal: {randomization: null}            # no randomization
        heavy:                                    # replaces env.eval.init_params.randomization
          randomization: {cube: {mass: {scale: [1.5, 2.0]}}}
        small_cube:                               # merged into the named scene entities
          scene: {cube: {parameters: {scale: [0.7, 0.7, 0.7]}}}

A condition without ``randomization`` keeps the evaluation's own (which follows training). Scene
edits write a derived scene used through ``init_params.scene_config``; the trained component and
the policy interface must stay the same (both are checked when the environment starts). The
conditions are read from the current project recipe, so a saved run can be evaluated under a
protocol written after it was trained; each condition directory records what was applied.
"""
from copy import deepcopy
import json
from pathlib import Path

CONDITION_KEYS = {'randomization', 'scene'}
SUMMARY = ('success_once', 'fail_once', 'episode_len', 'return', 'num_trajectories')


def load_conditions(recipe):
    """``evaluation.conditions`` of a recipe file, validated; empty when it declares none."""
    from omegaconf import OmegaConf
    declared = OmegaConf.select(OmegaConf.load(recipe), 'evaluation.conditions')
    if declared is None:
        return {}
    conditions = OmegaConf.to_container(declared, resolve=True)
    if not isinstance(conditions, dict):
        raise ValueError(f'{recipe}: evaluation.conditions must map names to conditions')
    for name, condition in conditions.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f'{recipe}: condition names must be identifiers, got {name!r}')
        condition = {} if condition is None else condition
        if not isinstance(condition, dict) or condition.keys() - CONDITION_KEYS:
            raise ValueError(f'{recipe}: evaluation.conditions.{name} takes only {sorted(CONDITION_KEYS)}')
        scene = condition.get('scene', {})
        if not isinstance(scene, dict) or not all(isinstance(edit, dict) and 'name' not in edit
                                                  for edit in scene.values()):
            raise ValueError(f'{recipe}: evaluation.conditions.{name}.scene maps entity names to field changes')
        conditions[name] = condition
    return conditions


def select(conditions, requested, recipe):
    if not conditions:
        raise ValueError(f'{recipe} declares no evaluation.conditions')
    names = list(conditions) if requested == 'all' else [name.strip() for name in requested.split(',') if name.strip()]
    unknown = [name for name in names if name not in conditions]
    if unknown or not names:
        raise ValueError(f'Unknown evaluation conditions {unknown}; declared: {list(conditions)}')
    return names


def _merge(base, change):
    for key, value in change.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)


def _absolute_assemblies(entity, directory):
    # Entity assemblies are the scene's only file references relative to its folder.
    if isinstance(entity.get('assembly'), str) and not Path(entity['assembly']).is_absolute():
        entity['assembly'] = str((directory / entity['assembly']).resolve())
    for child in entity.get('children', ()) or ():
        _absolute_assemblies(child, directory)


def derive_scene(source, edits, destination):
    """Copy ``source`` with entity fields changed; returns the written path."""
    import yaml
    source = Path(source).resolve()
    data = yaml.safe_load(source.read_text())
    entities = {entity.get('name'): entity for entity in data.get('entities', ())}
    for name, change in edits.items():
        if name not in entities:
            raise ValueError(f'Condition scene edit names unknown top-level entity {name!r}; '
                             f'entities: {list(entities)}')
        _merge(entities[name], change)
    for entity in data.get('entities', ()):
        _absolute_assemblies(entity, source.parent)
    path = Path(destination) / 'scene.yaml'
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return path


def apply_condition(cfg, scene_config, name, condition, destination):
    """Point ``cfg.env.eval`` at the condition and record it in ``destination``."""
    from omegaconf import OmegaConf, open_dict
    parameters = cfg.env.eval.init_params
    source = parameters.get('scene_config') or scene_config
    with open_dict(cfg):
        if 'randomization' in condition:
            parameters.randomization = condition['randomization']
        if condition.get('scene'):
            parameters.scene_config = str(derive_scene(source, condition['scene'], destination))
    applied = {'condition': name, 'declared': condition, 'source_scene': str(source),
               'scene_config': parameters.get('scene_config') or str(source),
               'randomization': OmegaConf.to_container(parameters.randomization, resolve=True)
               if OmegaConf.is_config(parameters.get('randomization')) else parameters.get('randomization')}
    (Path(destination) / 'condition.json').write_text(json.dumps(applied, indent=2, ensure_ascii=False) + '\n')


def read_scalars(directory):
    """Last value of every ``eval/*`` scalar an evaluation wrote."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    values = {}
    for events in sorted(Path(directory).rglob('events.out.tfevents.*')):
        accumulator = EventAccumulator(str(events), size_guidance={'scalars': 0})
        accumulator.Reload()
        for tag in accumulator.Tags()['scalars']:
            if tag.startswith('eval/'):
                values[tag.removeprefix('eval/')] = accumulator.Scalars(tag)[-1].value
    return values


def run_all(names, group, arguments, executable, root):
    """One evaluation subprocess per condition (a fresh simulator each), then the summary."""
    import subprocess
    results = {}
    for name in names:
        destination = Path(group) / name
        print(f'=== condition {name}: {destination}', flush=True)
        completed = subprocess.run([executable, str(Path(root) / 'run.py'), *arguments,
                                    '--condition', name, '--into', str(destination)])
        results[name] = ({'error': f'evaluation exited with {completed.returncode}'} if completed.returncode
                         else read_scalars(destination))
    summary = Path(group) / 'summary.json'
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    width = max(len(name) for name in names)
    print('condition'.ljust(width), *(key.rjust(16) for key in SUMMARY))
    for name, values in results.items():
        cells = ([values['error'].rjust(16)] if 'error' in values else
                 [(f'{values[key]:16.3f}' if key in values else '-'.rjust(16)) for key in SUMMARY])
        print(name.ljust(width), *cells)
    print(f'Summary: {summary}', flush=True)
    return 1 if any('error' in values for values in results.values()) else 0
