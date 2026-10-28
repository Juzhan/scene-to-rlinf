"""The task: reward, success, failure and metrics, in plain Torch; thresholds live here, not in YAML.

``evaluate(obs)`` runs once per decision step for all environments. ``obs`` (SceneInfo in
observation.py) reads every entity of the scene, including simulator ground truth the policy does
not see, plus ``obs.planner``/``obs.executor`` outputs and ``obs.simulation.step_dt``. Return a dict
of [B] tensors: ``reward`` (default 0; the shared layer multiplies it by step_dt), ``success`` and
``failure`` (default False; either ends the episode), ``metrics`` (logged in training and evaluation) and optionally
``set`` ({abstract entity: [B, 3] position or [B, 7] pose} to move it). For state across steps use a
``Task`` class with ``evaluate(self, obs)`` and ``reset(env_ids)``, or
``shared.assembly.task_memory.TaskMemory``.

The example rewards approaching ``destination`` and fails on body contact; replace it with the task.
"""
import torch
from .observation import SceneInfo, TaskResult
from shared.policy_math import relative_goal, yaw


def evaluate(obs: SceneInfo) -> TaskResult:
    robot, goal = obs.robot, obs.destination
    distance = torch.linalg.vector_norm(goal.position - robot.position, dim=-1)
    _, heading = relative_goal(robot.position, robot.orientation, goal.position, yaw(goal.orientation))
    heading = heading[:,0].abs()
    fallen = torch.linalg.vector_norm(obs.contact_forces.contact_force_samples, dim=-1).amax(dim=(1,2)) > 1.
    action_change = ((obs.executor.output - obs.executor.previous_output) / .25).square().sum(-1)
    reward = .5*(1-torch.tanh(distance/2.)) + .5*(1-torch.tanh(distance/.2)) - .2*heading - 400.*fallen - .01*action_change
    return {'reward': reward, 'failure': fallen,
            'metrics': {'distance_m': distance, 'heading_error_rad': heading, 'reached': distance < .5, 'fallen': fallen}}
