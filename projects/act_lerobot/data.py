"""A LeRobot dataset (v3.0, as written by lerobot-record) as ACT training samples.

Each sample is one frame: the selected camera images, ``observation.state`` and the next
``chunk_size`` actions (``action`` [chunk, dim] with ``action_is_pad`` past the episode end), the
pairing LeRobot's own ACT training uses. ``data.episodes`` (default all) selects the episodes used at all;
train and validation are split among them by episode, the last ``data.validation_episodes`` validating
unless a list is given.
"""
from pathlib import Path


def _episodes(cfg, total):
    selected = cfg.data.get('episodes')
    used = list(range(total)) if selected is None else sorted({int(e) for e in selected})
    if not used or any(e < 0 or e >= total for e in used):
        raise ValueError(f'episodes must name episodes in 0..{total - 1}, got {selected}')
    chosen = cfg.data.get('validation_episodes', 1)
    validation = sorted(int(e) for e in chosen) if not isinstance(chosen, int) else used[len(used) - chosen:]
    if not validation or any(e not in used for e in validation):
        raise ValueError(f'validation_episodes must name used episodes {used[0]}..{used[-1]}, got {chosen}')
    train = [e for e in used if e not in validation]
    if not train:
        raise ValueError('No training episode left after the validation split')
    return train, validation


def load_dataset(path, cfg):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    root = Path(path)
    if not (root / 'meta' / 'info.json').is_file():
        raise ValueError(f'{root}: not a LeRobot dataset (meta/info.json missing)')
    repo = 'local/' + root.name
    meta = LeRobotDatasetMetadata(repo, root=root)
    train, validation = _episodes(cfg, meta.total_episodes)
    # The action chunk: this frame's action and the following ones, at the dataset's own rate.
    delta = {'action': [step / meta.fps for step in range(cfg.actor.model.chunk_size)]}
    backend = cfg.data.get('video_backend', 'pyav')
    datasets = {name: LeRobotDataset(repo, root=root, episodes=episodes, delta_timestamps=delta, video_backend=backend)
                for name, episodes in (('train', train), ('validation', validation))}
    manifest = {'source': str(root), 'robot_type': meta.robot_type, 'fps': meta.fps,
                'codebase_version': meta.info.get('codebase_version'),
                'episodes': list(range(meta.total_episodes)), 'train_episodes': train,
                'validation_episodes': validation,
                'features': {key: {'dtype': value['dtype'], 'shape': list(value['shape'])}
                             for key, value in meta.features.items()}}
    return datasets, manifest


def validate(cfg, datasets, manifest):
    features = manifest['features']
    for key in ('observation.state', 'action'):
        if key not in features:
            raise ValueError(f'The dataset has no {key}; ACT reads the robot state and predicts actions')
    cameras = cfg.actor.model.get('cameras') or [key for key, value in features.items() if value['dtype'] in ('video', 'image')]
    missing = [key for key in cameras if key not in features]
    if missing or not cameras:
        raise ValueError(f'Cameras {missing or cameras} are not images of this dataset')
    model = cfg.actor.model
    if not 1 <= model.n_action_steps <= model.chunk_size:
        raise ValueError('n_action_steps must be between 1 and chunk_size')
    if len(datasets['train']) < 2:
        raise ValueError('At least two training frames are required')
