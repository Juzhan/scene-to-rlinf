"""Two-cube stacking evaluation; the upper cube must rest on the lower one."""
import torch

from .observation import SceneInfo, TaskResult
from shared.assembly.task_memory import TaskMemory


class Task:
    def __init__(self):
        self.memory = TaskMemory()

    def evaluate(self, obs: SceneInfo) -> TaskResult:
        upper, lower = obs.cube_upper, obs.cube_lower
        upper_error = torch.linalg.vector_norm(upper.position - upper.target.position, dim=-1)
        lower_error = torch.linalg.vector_norm(lower.position - lower.target.position, dim=-1)
        speed = torch.linalg.vector_norm(upper.linear_velocity, dim=-1)
        stacked = (upper_error <= .02) & (lower_error <= .02) & (speed <= .05)
        held = self.memory.hold('stacked', stacked, obs.simulation.step_dt[:, 0])
        return {'reward': -upper_error - lower_error,
                'success': held >= .5,
                'metrics': {'upper_error_m': upper_error, 'lower_error_m': lower_error,
                            'upper_speed_m_s': speed, 'stacked': stacked}}
