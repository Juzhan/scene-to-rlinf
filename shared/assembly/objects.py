"""Typed, lazy views over named tensors. No simulator objects or tensor copies."""
from collections.abc import Mapping


class ObjectView:
    def __init__(self, values=None, *, reader=None, paths=None, _prefix='', _cache=None):
        if values is None and reader is None:
            raise TypeError('Provide tensors or a signal reader')
        object.__setattr__(self, '_reader', reader if reader is not None else values.__getitem__)
        object.__setattr__(self, '_paths', frozenset(paths if paths is not None else values))
        object.__setattr__(self, '_prefix', _prefix)
        object.__setattr__(self, '_cache', {} if _cache is None else _cache)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        annotations = getattr(type(self), '__annotations__', {})
        if annotations and name not in annotations:
            raise AttributeError(f'{name!r} is not declared in {type(self).__name__}')
        path = self._prefix + name
        if path in self._paths:
            if path not in self._cache:
                self._cache[path] = self._reader(path)
            return self._cache[path]
        prefix = path + '.'
        if any(field.startswith(prefix) for field in self._paths):
            cls = type(self).__annotations__.get(name, ObjectView) if hasattr(type(self), '__annotations__') else ObjectView
            if not isinstance(cls, type) or not issubclass(cls, ObjectView):
                cls = ObjectView
            return cls(reader=self._reader, paths=self._paths, _prefix=prefix, _cache=self._cache)
        raise AttributeError(f'{path!r} is not available in this observation; check scene selection')

    def __setattr__(self, name, value):
        raise AttributeError('Observation attributes are read-only')


def selected_fields(settings):
    if 'control' in settings:
        return {role: list(cfg['observations']) for role, cfg in settings['control']['components'].items()}
    result = []
    for value in settings.get('observations', {}).values():
        result.extend(value if isinstance(value, list) else value['terms'])
    return {'policy': list(dict.fromkeys(result))}
