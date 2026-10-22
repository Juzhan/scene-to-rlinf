"""Authored fixed targets and abstract scene entities as readable information.

Observation and task readers share desired values, never inferred scene state.
Fixed targets are ``<entity>.target.<field>``. Abstract entities publish
``<name>.position|orientation|size|direction`` like physical entities; a value
is a constant list when static, otherwise a description of its live pose.
"""
import math
from .entity_information import signal_information
from .abstract import abstract_catalog, abstract_entities


def visualization(settings):
    """Abstract entities drawn in the simulator viewport, validated by name."""
    value = settings.get('visualization', {})
    if not isinstance(value, dict) or value.keys() - {'abstract'}:
        raise ValueError('visualization only supports an abstract entity name list')
    shown = value.get('abstract', [])
    names = abstract_entities(settings)
    if not isinstance(shown, list) or any(not isinstance(name, str) or name not in names for name in shown):
        raise ValueError('visualization.abstract must list existing abstract entities')
    return shown


def target_catalog(settings):
    visualization(settings)
    targets = settings.get('targets', {})
    if not isinstance(targets, dict):
        raise ValueError('targets must map scene entities to desired properties')
    result = {}
    for entity, properties in targets.items():
        if not isinstance(entity, str) or not entity or not isinstance(properties, dict):
            raise ValueError('targets must map scene entities to desired properties')
        for field, value in properties.items():
            if field not in ('position', 'linear_velocity', 'angular_velocity'):
                raise ValueError(f'targets.{entity}.{field}: unsupported target property')
            if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
            ):
                raise ValueError(f'targets.{entity}.{field}: expected three finite numbers')
            path = entity + '.target.' + field
            info = signal_information(entity + '.' + field)
            result[path] = (list(value), info)
    result.update(abstract_catalog(settings))
    return result
