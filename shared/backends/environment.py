"""Small common tensor boundary around a native vector environment."""
from shared.integrations.rlinf.terminal import snapshot


class Environment:
    """Small tensor boundary shared by preview and the RLinf adapter."""

    def __init__(self, native):
        self.native = native
        self.num_envs = native.num_envs
        self.device = native.device
        self.action_dim = native.action_manager.total_action_dim

    def reset(self, env_ids=None, *, seed=None):
        options = {}
        if env_ids is not None:
            options["env_ids"] = env_ids
        if seed is not None:
            options["seed"] = seed
        observations, info = self.native.reset(**options)
        return snapshot(observations), snapshot(info)

    def step(self, action):
        observations, rewards, terminated, truncated, info = self.native.step(action)
        return snapshot(observations), snapshot(rewards), snapshot(terminated), snapshot(truncated), info

    def close(self):
        self.native.close()


