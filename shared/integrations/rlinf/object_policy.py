"""RLinf PPO protocol around a project model whose forward accepts typed observations.

A policy layer's model returns its action means [B, actions]; this adapter adds what PPO needs: a learned
log standard deviation per action (``log_std``) and a value network: the model's ``value(obs)`` when it defines
one, else ``value_head``, an MLP over the layer's numeric inputs. RLinf gives parameters named ``value_head``
the critic's learning rate (``optim.value_lr``) and warm-up, so a model's own value network goes in an attribute
of that name. An older trained component returns
``(means, values)`` itself and owns ``actor_logstd``; it is used as it is.
"""
import torch
from torch import nn
from torch.distributions import Normal
from rlinf.models.embodiment.base_policy import BasePolicy
from .observations import fields


class DefaultCritic(nn.Module):
    """Value of the inputs a layer reads, flattened: the numbers (joints, positions, ...), not images."""

    def __init__(self, hidden=(256, 128)):
        super().__init__()
        layers = []
        for size in hidden:
            layers += [nn.LazyLinear(size), nn.ELU()]
        self.net = nn.Sequential(*layers, nn.LazyLinear(1))

    def forward(self, values):
        images = [key for key, value in values.items() if value.ndim >= 4 or not value.is_floating_point()]
        if images:
            raise ValueError(f'The layer reads images {images}: the default value network only takes numbers; '
                             'define value(self, obs) -> Tensor [B, 1] on the model')
        return self.net(torch.cat([values[key].flatten(1) for key in sorted(values)], -1))


class ObjectPolicy(nn.Module, BasePolicy):
    def __init__(self, model, observation_type, collect_stats=False, *, action_dim=None, hidden=(256, 128), log_std=-.5):
        super().__init__()
        self.model, self.observation_type, self.collect_stats = model, observation_type, collect_stats
        self.legacy = hasattr(model, 'actor_logstd')
        if not self.legacy:
            if type(action_dim) is not int or action_dim < 1:
                raise ValueError('The trained layer needs its number of actions')
            self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std)))
            self.value_head = None if callable(getattr(model, 'value', None)) else DefaultCritic(tuple(hidden))

    def _evaluate(self, observations):
        reference = self.model.actor_logstd if self.legacy else self.log_std
        values = {key: value.to(device=reference.device, dtype=reference.dtype if value.is_floating_point() else value.dtype)
                  for key, value in observations.items() if isinstance(value, torch.Tensor)}
        obs = self.observation_type(values)
        if self.legacy:
            means, value = self.model(obs)
            return Normal(means, self.model.actor_logstd.exp().expand_as(means)), value
        means = self.model(obs)
        if not isinstance(means, torch.Tensor) or means.shape[-1] != reference.shape[0]:
            raise ValueError(f'The trained layer returns {getattr(means, "shape", type(means))}, '
                             f'expected actions [B, {reference.shape[0]}]')
        value = self.model.value(obs) if self.value_head is None else self.value_head(values)
        return Normal(means, self.log_std.exp().expand_as(means)), value

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
