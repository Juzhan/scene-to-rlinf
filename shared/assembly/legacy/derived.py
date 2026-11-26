"""Type discovery for legacy ``variables:`` expressions, without a simulator SDK."""
from shared.assembly.information import Information
from shared.assembly.legacy.sources import validate_sources
from shared.assembly.legacy.task import evaluation_config


def validate_variables(settings, signal_factories=None):
    config = evaluation_config(settings)
    return config, validate_sources(config, signal_factories, settings=settings)


def variable_information(settings, name, signal_factories=None):
    _, report = validate_variables(settings, signal_factories)
    values = dict(report['information'][name])
    values.pop('label', None)
    values.pop('operations', None)
    for key in ('shape', 'axes', 'labels'):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    return Information(**values)
