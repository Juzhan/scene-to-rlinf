"""Pick the cube and hold it 4 cm clear of the ground, still, for 0.5 s; dense shaping reward.

Success: lifted 40 mm above rest, hand at the cube, cube speed (from its own position difference)
below 0.1 m/s, held 0.5 s. There is no contact sensor here, so the two-finger contact is replaced by
the fingertip centre being at the cube with the gripper closed on it.

Where to write the task: ``Task.evaluate(obs)`` runs once per decision step for all environments and
returns [B] tensors ``reward`` (multiplied by step_dt when summed), ``success`` and ``failure`` (either
ends the episode) and ``metrics``. ``obs`` (SceneInfo in observation.py) reads every entity, including
ground truth the policy does not see. Per-environment state lives in ``TaskMemory``, cleared on reset.
"""
import torch
from .observation import SceneInfo, TaskResult
from shared.assembly.task_memory import TaskMemory

CUBE_REST = .024      # cube centre height resting on the ground (m)
LIFT = .04            # lift above CUBE_REST that counts as held clear of the ground (m)
HOLD_TIME = .5        # seconds the lifted cube must stay still in the hand
STILL_SPEED = .1      # cube speed below this counts as still (m/s)
NEAR = .04            # fingertip centre to cube centre (m)
CLOSED = .08          # finger opening below this counts as closed on the cube (fully open 0.1 m)
# Rewards are multiplied by step_dt (0.02 s). Holding still at height earns about 0.27 per step, i.e. at most
# ~27 discounted (gamma 0.99) if the episode went on; the one-off success bonus (50) must exceed that,
# otherwise ending the episode by succeeding would lose return.
SUCCESS_BONUS = 50 / .02
FAILURE_PENALTY = 5 / .02


class Task:
    def __init__(self):
        self.memory = TaskMemory()

    def evaluate(self, obs: SceneInfo) -> TaskResult:
        cube = obs.cube.position
        dt = obs.simulation.step_dt[:, 0]
        tcp = obs.tcp_frame.frame_positions[:, 0]
        reach = torch.linalg.vector_norm(tcp - cube, dim=-1)
        # gripper_joint1 opens in +m, gripper_joint2 in -m; 0 is closed.
        opening = obs.piper.joint_position[:, 6] - obs.piper.joint_position[:, 7]
        height = cube[:, 2] - CUBE_REST
        # From the cube's own position change: the velocity buffer is biased while the cube is pinched.
        speed = torch.linalg.vector_norm(self.memory.rate('cube', cube, dt), dim=-1)
        near = reach < NEAR
        grasped = near & (opening < CLOSED)
        # Fraction of the way to LIFT; starts after the first 5 mm.
        lift = ((height - .005) / (LIFT - .005)).clamp(0, 1)
        still = (1 - speed / STILL_SPEED).clamp(0, 1)
        held = self.memory.hold('held', grasped & (height >= LIFT) & (speed < STILL_SPEED), dt)
        success = held >= HOLD_TIME - 1e-6
        dropped = (torch.linalg.vector_norm(cube[:, :2] - obs.cube_area.position[:, :2], dim=-1) > .6) | (cube[:, 2] < -.05)
        action_change = (obs.executor.output - obs.executor.previous_output).square().sum(-1)
        # 稠密奖励
        reward = (1 - torch.tanh(reach / .1)                                  # approach the cube
                  + .5 * (reach < .03) * (1 - opening / .1)                   # close the gripper on it
                  + 8 * grasped * lift                                        # raise it while holding it
                  + 4 * grasped * lift * still                                # and keep it still up there
                  + SUCCESS_BONUS * success - FAILURE_PENALTY * dropped
                  - .01 * action_change)
        return {'reward': reward, 'success': success, 'failure': dropped,
                'metrics': {'reach_m': reach, 'cube_lift_m': height, 'cube_speed_mps': speed.clamp(max=10),
                            'held_s': held, 'gripper_opening_m': opening}}
