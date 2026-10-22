"""Wire representation for stock RLinf; project observations stay named tensors.

RLinf preserves `states` as a nested value and its batch utilities recurse into
dictionaries. Task descriptions are native transport metadata, never model input.
"""
from collections.abc import Mapping

import torch


def fields(observation):
    states = observation.get('states')
    if isinstance(states, Mapping):
        return states
    # Stock evaluation merges nested states as a list of producer dictionaries.
    # Training merges recursively. Normalize at the shared transport boundary.
    if isinstance(states, list) and states and all(isinstance(item, Mapping) for item in states):
        keys = states[0].keys()
        if any(item.keys() != keys for item in states):
            raise ValueError('Evaluation observation producers have different fields')
        return {name: torch.cat([item[name] for item in states], dim=0) for name in keys}
    return observation


def pack(observation, num_envs, task_description=''):
    if 'states' in observation:  # Existing flat-state integrations already conform.
        return observation
    return {'states': observation, 'task_descriptions': [task_description] * num_envs}


def copy_rows(destination, source, mask):
    destination, source = fields(destination), fields(source)
    for name, value in source.items():
        destination[name][mask] = value[mask]


def same_schema(left, right):
    left, right = fields(left), fields(right)
    return left.keys() == right.keys() and all(
        left[name].shape == value.shape and left[name].dtype == value.dtype
        for name, value in right.items())
