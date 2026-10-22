"""RLinf PPO protocol around a project model whose forward accepts typed observations."""
import torch
from torch import nn
from torch.distributions import Normal
from rlinf.models.embodiment.base_policy import BasePolicy
from .observations import fields


class ObjectPolicy(nn.Module, BasePolicy):
    def __init__(self, model, observation_type, collect_stats=False):
        super().__init__()
        self.model, self.observation_type, self.collect_stats = model, observation_type, collect_stats

    def _evaluate(self, observations):
        reference = self.model.actor_logstd
        values = {key: value.to(device=reference.device, dtype=reference.dtype if value.is_floating_point() else value.dtype)
                  for key, value in observations.items() if isinstance(value, torch.Tensor)}
        means, values = self.model(self.observation_type(values))
        return Normal(means, self.model.actor_logstd.exp().expand_as(means)), values

    def forward(self, **kwargs):
        return BasePolicy.forward(self, **kwargs)

    @torch.no_grad()
    def predict_action_batch(self, env_obs, mode=None, do_sample=None, **kwargs):
        if mode is None:
            mode = 'eval' if do_sample is False else 'train'
        env_obs = fields(env_obs)
        env_obs = {key: value for key, value in env_obs.items() if isinstance(value, torch.Tensor)}
        distribution, values = self._evaluate(env_obs)
        action = distribution.mean if mode == 'eval' else distribution.sample()
        inputs = {**{'obs/' + key: value for key, value in env_obs.items()}, 'action': action}
        if self.collect_stats:
            inputs.update(prev_action_mean=distribution.mean, prev_action_std=distribution.stddev)
        return action[:,None,:], {'prev_logprobs': distribution.log_prob(action)[:,None,:],
                                  'prev_values': values, 'forward_inputs': inputs}

    def default_forward(self, forward_inputs, compute_logprobs=True, compute_entropy=True, compute_values=True, **kwargs):
        distribution, values = self._evaluate({key[4:]: value for key, value in forward_inputs.items() if key.startswith('obs/')})
        result = {}
        if compute_logprobs: result['logprobs'] = distribution.log_prob(forward_inputs['action'])
        if compute_entropy: result['entropy'] = distribution.entropy()
        if compute_values: result['values'] = values
        if self.collect_stats:
            with torch.no_grad():
                old_mean, old_std = forward_inputs['prev_action_mean'], forward_inputs['prev_action_std']
                result['gaussian_kl'] = (torch.log(distribution.stddev / old_std + 1e-5)
                    + (old_std.square() + (old_mean - distribution.mean).square()) / (2 * distribution.stddev.square()) - .5).sum(-1)
        return result
