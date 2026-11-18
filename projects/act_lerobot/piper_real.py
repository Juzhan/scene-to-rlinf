"""Run an exported ACT ``pretrained_model`` on the real Piper X through robot_toolkit.

Needs an environment with LeRobot (0.4.3+), pyAgxArm, pyrealsense2 and ruckig, and ``~/ws`` on the path for
robot_toolkit, e.g.::

    cd ~/ws && PYTHONPATH=~/ws RealSense-Stereo-Trial/.venv-linux/bin/python \\
        rlinf-workspace/projects/act_lerobot/piper_real.py <run>/pretrained_model --dataset <dataset>

Without ``--execute`` the arm connection is read-only (nothing is sent on the CAN bus): the cameras open, the
policy runs on live frames and the predicted first actions are printed against the measured joints. With
``--execute`` the arm is enabled, moved slowly to the mean first-frame state of the dataset's episodes, and the
policy's chunks are streamed through robot_toolkit's continuous executor (joint limits, jerk-limited reference,
tracking-error abort, gated gripper). Ctrl+C stops and holds the measured position; the arm stays enabled.

The leader arm must be off the bus: commanding a master-slave pair pulls both out of teaching (power-cycle to
restore it, see robot_toolkit/PIPER.md) and the leader's broadcast would fight the commands.
"""
import argparse
import json
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch

TASK_CONFIG = Path.home() / 'ws/robot_toolkit/configs/tasks/piper_master_slave.json'


class ActPolicy:
    """The exported policy and its processors; ``chunk(frames, state)`` is one action chunk in dataset units."""

    def __init__(self, path):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.config = PreTrainedConfig.from_pretrained(path)
        self.policy = ACTPolicy.from_pretrained(path).to(self.config.device).eval()
        self.pre, self.post = make_pre_post_processors(self.config, pretrained_path=str(path))
        self.cameras = {key: key.removeprefix('observation.images.') for key in self.config.image_features}

    @torch.no_grad()
    def chunk(self, frames, state):
        batch = {'observation.state': torch.as_tensor(state, dtype=torch.float32)[None]}
        for key, name in self.cameras.items():
            rgb = torch.from_numpy(frames[name].rgb).permute(2, 0, 1).float() / 255.0
            batch[key] = rgb[None]
        actions = self.policy.predict_action_chunk(self.pre(batch))
        return self.post(actions)[0].cpu().numpy().astype(float)


class ActSource:
    """robot_toolkit ChunkSource: observe, predict, return the first ``steps`` actions from this index on.

    Joint targets are clipped to the arm's joint limits: the demonstrations hold joint 4 at its upper limit,
    and ACT predicts a few mrad beyond it, which the executor would otherwise reject. The clipped amount is
    logged; more than 0.05 rad beyond a limit stops the run.

    With ``ensemble`` (ACT's temporal ensembling coefficient) each returned action is the weighted mean of every
    chunk predicted so far that covers its action index, older chunks weighted ``exp(-ensemble * i)`` higher.
    """

    def __init__(self, policy, robot, cameras, steps, fps, max_width, log, ensemble=None):
        self.policy, self.robot, self.cameras = policy, robot, cameras
        self.steps, self.fps, self.max_width, self.log = steps, fps, max_width, log
        self.lower, self.upper = robot.guard.limits.lower, robot.guard.limits.upper
        self.ensemble, self.history = ensemble, []

    def blend(self, action_index, chunk):
        if self.ensemble is None:
            return chunk[:self.steps]
        self.history = [(start, old) for start, old in self.history if start + len(old) > action_index]
        self.history.append((action_index, chunk))
        blended = []
        for index in range(action_index, action_index + self.steps):
            rows = np.stack([old[index - start] for start, old in self.history if start <= index < start + len(old)])
            weights = np.exp(-self.ensemble * np.arange(len(rows)))
            blended.append(weights @ rows / weights.sum())
        return np.stack(blended)

    def observe(self):
        frames = self.cameras.snapshot(strict=True)
        sample = self.robot.state()
        if sample.q is None:
            raise RuntimeError('no joint state')
        return frames, np.r_[np.asarray(sample.q, float), float(sample.extra.get('gripper_width', 0.0))]

    def next_chunk(self, action_index, timeout=None):
        from robot_toolkit.control import ActionChunk
        frames, state = self.observe()
        began = time.perf_counter()
        chunk = self.policy.chunk(frames, state)
        predicted = self.blend(action_index, chunk)
        actions = np.c_[np.clip(predicted[:, :-1], self.lower, self.upper),
                        np.clip(predicted[:, -1], 0.0, self.max_width)]
        self.log.append({'action_index': int(action_index), 'state': state.tolist(),
                         'infer_ms': (time.perf_counter() - began) * 1000, 'chunk': chunk.tolist(),
                         'clipped': float(np.max(np.abs(actions - predicted)))})
        if self.log[-1]['clipped'] > 0.05:
            raise RuntimeError(f"policy target {self.log[-1]['clipped']:.3f} rad beyond the joint limits")
        return ActionChunk(actions, action_hz=self.fps, action_index=action_index)


def start_pose(dataset):
    """Mean first-frame state of the dataset's episodes (joints and gripper width)."""
    import pandas as pd
    files = sorted(Path(dataset).glob('data/*/*.parquet'))
    frames = pd.concat([pd.read_parquet(path, columns=['frame_index', 'observation.state']) for path in files])
    return np.stack(frames[frames.frame_index == 0]['observation.state'].to_numpy()).mean(0)


def save_frames(cameras, directory, prefix=''):
    from PIL import Image
    for name, frame in cameras.snapshot().items():
        Image.fromarray(frame.rgb).save(directory / f'{prefix}{name}.png')


def settled_state(robot, seconds=1.0):
    """Joint feedback after the per-joint CAN frames have all arrived (two reads that agree)."""
    time.sleep(seconds)
    first = robot.state()
    time.sleep(0.2)
    second = robot.state()
    if first.q is None or second.q is None or np.max(np.abs(np.subtract(first.q, second.q))) > 0.01:
        raise RuntimeError('joint feedback is missing or unsteady')
    return second


def main():
    parser = argparse.ArgumentParser(description='ACT on the real Piper X (dry run unless --execute)')
    parser.add_argument('model', type=Path, help='LeRobot pretrained_model directory')
    parser.add_argument('--dataset', type=Path, required=True, help='training dataset (fps and start pose)')
    parser.add_argument('--task-config', type=Path, default=TASK_CONFIG, help='robot_toolkit task JSON (cameras)')
    parser.add_argument('--steps', type=int, default=20, help='actions executed per inference')
    parser.add_argument('--seconds', type=float, default=30.0, help='policy run time limit')
    parser.add_argument('--max-speed', type=float, default=0.6, help='joint speed limit, rad/s')
    parser.add_argument('--max-accel', type=float, default=1.2, help='joint acceleration limit, rad/s^2')
    parser.add_argument('--ensemble', type=float, help='temporal ensembling coefficient (ACT uses 0.01); off by default')
    parser.add_argument('--start-speed', type=float, default=0.3, help='joint speed of the move to the start pose')
    parser.add_argument('--max-jump', type=float, default=0.35, help='abort when a target is this far from the reference (rad)')
    parser.add_argument('--execute', action='store_true', help='enable the arm and move it')
    parser.add_argument('--start-only', action='store_true', help='with --execute: only move to the start pose (reset between trials)')
    parser.add_argument('--output', type=Path, help='log directory (default <model>/../real/<time>)')
    args = parser.parse_args()

    from robot_toolkit.control import ContinuousConfig, min_jerk, run_continuous_stream, run_trajectory
    from robot_toolkit.gripper import AgxGripper
    from robot_toolkit.profiles import build_camera_hub, build_robot
    from robot_toolkit.robot import AgxPiperConfig

    fps = json.loads((args.dataset / 'meta/info.json').read_text())['fps']
    task = json.loads(args.task_config.read_text())
    output = args.output or args.model.parent / 'real' / time.strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    policy = ActPolicy(args.model)
    if not 1 <= args.steps <= policy.config.chunk_size:
        parser.error(f'--steps must be 1..{policy.config.chunk_size}')
    cameras = build_camera_hub([entry for entry in task['collect']['cameras']
                                if entry['name'] in policy.cameras.values()])
    missing = set(policy.cameras.values()) - set(cameras.names)
    if missing:
        parser.error(f'cameras {sorted(missing)} are not in {args.task_config}')
    options = {'model': 'piper_x', 'enable_on_connect': args.execute, 'configure_on_connect': args.execute,
               'ik_backend': 'none', 'ik_warmup': False}
    robot = build_robot({'kind': 'agx_piper', 'name': 'piper_x', 'options': options, 'safety': True})
    gripper = AgxGripper(robot, max_width=AgxPiperConfig().gripper_max_width)
    log, report = [], {'model': str(args.model), 'arguments': {key: str(value) for key, value in vars(args).items()}, 'fps': fps}
    abort = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: abort.set())
    try:
        robot.connect()
        cameras.open()
        if not cameras.warmup(min_frames=5):
            raise RuntimeError('cameras did not deliver frames')
        state = settled_state(robot)
        source = ActSource(policy, robot, cameras, args.steps, fps, gripper.max_width, log, args.ensemble)
        goal = start_pose(args.dataset)
        report.update(initial_q=np.round(state.q, 4).tolist(), start_pose=np.round(goal, 4).tolist())
        print('measured q:', report['initial_q'], ' start pose:', report['start_pose'], flush=True)
        if not args.execute:
            for index in range(5):
                source.next_chunk(index)
                entry = log[-1]
                first = np.asarray(entry['chunk'][0])
                print(f"infer {entry['infer_ms']:.0f} ms  first action {np.round(first, 3).tolist()}  "
                      f"gap to state {np.round(first - entry['state'], 3).tolist()}", flush=True)
                time.sleep(0.3)
            save_frames(cameras, output)
            print('dry run: nothing was commanded (add --execute to move)', flush=True)
            return
        if not robot.prepare_motion():
            raise RuntimeError('cannot enable the arm')
        dt = 1.0 / AgxPiperConfig().command_hz
        start = np.asarray(robot.state().q, float)
        move = run_trajectory(robot, min_jerk(start, goal[:6], dt, args.start_speed), dt,
                              start_tolerance=0.03, tolerance=0.02, abort_event=abort)
        report['start_move'] = move.as_dict()
        if not move.ok or abort.is_set():
            raise RuntimeError('move to the start pose failed or was aborted')
        # The demonstrations start with the gripper open; open it to their mean first-frame width.
        report['start_gripper'] = {'ok': gripper.move(goal[-1], wait=True, timeout=3.0), 'width': gripper.width}
        save_frames(cameras, output, 'start_')
        if args.start_only:
            print('at start pose; policy not run (--start-only)', flush=True)
            return
        print('at start pose; running the policy (Ctrl+C stops and holds)', flush=True)
        samples = []
        config = ContinuousConfig(action_hz=fps, max_joint_speed=args.max_speed, max_joint_accel=args.max_accel,
                                  max_trial_s=args.seconds, max_action_jump_rad=args.max_jump)
        result = run_continuous_stream(robot, gripper, source, config, abort_event=abort,
                                       on_sample=samples.append)
        report.update(result=result.as_dict(), samples=samples)
        print(json.dumps(result.as_dict(), default=float), flush=True)
    finally:
        report['chunks'] = log
        (output / 'log.json').write_text(json.dumps(report, default=float))
        print('log:', output / 'log.json', flush=True)
        cameras.close()
        robot.disconnect()


if __name__ == '__main__':
    main()
