"""RLinf's Gaussian PPO interface for networks defined in each project.

Subclasses provide ``actor``, ``actor_logstd`` and ``value_head`` modules or
parameters. Network architecture and checkpoint loading stay in the project.
"""

import torch
from torch import nn
from torch.distributions import Normal

from rlinf.models.embodiment.base_policy import BasePolicy


class GaussianPolicy(nn.Module, BasePolicy):
    """Single-chunk continuous actor-critic with optional adaptive-KL statistics."""

    def __init__(self, *, collect_gaussian_stats=False):
        super().__init__()
        self.collect_gaussian_stats = collect_gaussian_stats

    def prepare_inputs(self, states):
        """Project model feature extraction; replay stores the original observations."""
        return states

    def distribution(self, states):
        means = self.actor(states)
        return Normal(means, self.actor_logstd.exp().expand_as(means))

    def forward(self, **kwargs):
        return BasePolicy.forward(self, **kwargs)

    def default_forward(
        self, forward_inputs, compute_logprobs=True, compute_entropy=True,
        compute_values=True, **kwargs,
    ):
        states = self.prepare_inputs(forward_inputs["states"])
        distribution = self.distribution(states)
        result = {}
        if compute_logprobs:
            result["logprobs"] = distribution.log_prob(forward_inputs["action"])
        if compute_entropy:
            result["entropy"] = distribution.entropy()
        if compute_values:
            result["values"] = self.value_head(states)
        if self.collect_gaussian_stats:
            with torch.no_grad():
                old_mean = forward_inputs["prev_action_mean"]
                old_std = forward_inputs["prev_action_std"]
                result["gaussian_kl"] = (
                    torch.log(distribution.stddev / old_std + 1e-5)
                    + (old_std.square() + (old_mean - distribution.mean).square())
                    / (2 * distribution.stddev.square())
                    - 0.5
                ).sum(dim=-1)
        return result

    @torch.no_grad()
    def predict_action_batch(self, env_obs, mode=None, do_sample=None, **kwargs):
        if mode is None:
            mode = "eval" if do_sample is False else "train"
        states = env_obs["states"].to(next(self.parameters()))
        features = self.prepare_inputs(states)
        distribution = self.distribution(features)
        actions = distribution.mean if mode == "eval" else distribution.sample()
        forward_inputs = {"states": states, "action": actions}
        if self.collect_gaussian_stats:
            forward_inputs["prev_action_mean"] = distribution.mean
            forward_inputs["prev_action_std"] = distribution.stddev
        return actions[:, None, :], {
            "prev_logprobs": distribution.log_prob(actions)[:, None, :],
            "prev_values": self.value_head(features),
            "forward_inputs": forward_inputs,
        }
