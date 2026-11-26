"""Convert verified original robot trajectories to causal 144D history inputs.

Requires recorded state/base_action/step, not the compact 39D export. Original
files remain untouched. --robot-frame supplies fixed robot-base calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def convert_episode(state, base_action, target, steps, calibration):
    from projects.franka_bowl_vla.history import RobotHistory, robot_feedback
    count = len(steps)
    if state.shape != (count, 8) or base_action.shape != (count, 7) or target.shape != (count, 7):
        raise ValueError('Expected aligned robot state, base action and residual targets')
    if not np.issubdtype(steps.dtype, np.integer) or np.any(np.diff(steps) <= 0):
        raise ValueError('Steps must be strictly increasing integers within each episode')
    if not all(np.isfinite(x).all() for x in (state, base_action, target)):
        raise ValueError('Nonfinite source trajectory')
    history = RobotHistory()
    inputs, targets = [], []
    consecutive = 0
    for index, step in enumerate(steps):
        consecutive = consecutive + 1 if index and step == steps[index-1] + 1 else 1
        feedback = robot_feedback(state[index], calibration['position'], calibration['rotation'])
        value = history.append(feedback, base_action[index], int(step))
        # Six poses plus one earlier pose for the oldest frame's velocity.
        # At real episode reset the initial velocity is explicitly zero.
        if consecutive >= 7 or (steps[0] == 0 and consecutive == index + 1):
            inputs.append(value)
            targets.append(target[index])
    if not inputs:
        raise ValueError('No complete causal windows; record a longer contiguous trajectory')
    return np.asarray(inputs, np.float32), np.asarray(targets, np.float32)


def import_report(report_path, destination, robot_frame):
    from projects.franka_bowl_vla.data import INTERFACE_METADATA, load_dataset
    from projects.franka_bowl_vla.history import INTERFACE
    report_path, destination = Path(report_path).resolve(), Path(destination).resolve()
    report = json.loads(report_path.read_text())
    calibration = json.loads(Path(robot_frame).read_text())
    rotation, position = np.asarray(calibration['rotation']), np.asarray(calibration['position'])
    if rotation.shape != (3, 3) or position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Invalid robot base calibration')
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-7)
    if report['residual_start_step'] != 10:
        raise ValueError('This task uses the ten-step residual gate')
    records = {row['seed']: row for row in report['dataset_manifest']['episodes']}
    paths = report['training_episodes'] + report['validation_episodes']
    if len(paths) != len(set(paths)) or len(records) != len(paths):
        raise ValueError('Source splits must describe unique episodes')
    destination.mkdir(parents=True, exist_ok=False)
    output = {**INTERFACE_METADATA, 'version': 2, 'interface': INTERFACE, 'residual_start_step': 10,
              'robot_frame': calibration, 'source_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
              'history_source': 'original consecutive robot state/base_action only; incomplete prefixes discarded',
              'episodes': []}
    for split, selected in (('train', report['training_episodes']), ('validation', report['validation_episodes'])):
        for source_name in selected:
            source = Path(source_name)
            seed = int(source.stem.rsplit('_', 1)[1])
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if records[seed]['sha256'] != original_hash or records[seed]['split'] != split:
                raise ValueError('Source episode hash/split mismatch')
            with np.load(source, allow_pickle=False) as row:
                if not bool(row['accepted_native']):
                    raise ValueError('Source trajectory was not accepted')
                np.testing.assert_allclose(row['base_action'] + row['residual_target'], row['applied_action'], atol=1e-7)
                obs, target = convert_episode(row['state'], row['base_action'], row['residual_target'], row['step'], calibration)
                source_samples = len(row['step'])
            path = destination / f'episode_{seed}.npz'
            np.savez_compressed(path, observation=obs, target=target)
            output['episodes'].append({'file': path.name, 'seed': seed, 'split': split,
                'samples': len(target), 'discarded_prefix_samples': source_samples-len(target),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_sha256': original_hash})
    (destination / 'manifest.json').write_text(json.dumps(output, indent=2))
    datasets, _ = load_dataset(destination / 'manifest.json')
    print({key: len(value) for key, value in datasets.items()})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--robot-frame', type=Path, required=True)
    args = parser.parse_args()
    import_report(args.report, args.destination, args.robot_frame)
