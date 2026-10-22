"""Lazy named queries: discover everything, fetch only explicitly selected fields.

Signals are not automatically policy observations. Custom fields (device state,
contact statistics, images, etc.) can be registered without changing Entity.
"""

from dataclasses import dataclass
from typing import Callable
from .information import Information


@dataclass(frozen=True)
class Signal:
    read: Callable
    category: str = "state"
    unit: str | None = None
    frame: str | None = None
    description: str = ""
    info: Information | None = None
    entity: str | None = None
    source: str = "unspecified"

    @property
    def information(self):
        return self.info or Information(unit=self.unit, frame=self.frame)


class Signals:
    def __init__(self, prepare=None):
        self._fields = {}
        self._prepare = prepare

    def register(self, path: str, signal: Signal):
        if path in self._fields:
            raise ValueError(f"Duplicate signal: {path}")
        self._fields[path] = signal

    def describe(self):
        return {path: {"category": s.category, "unit": s.information.unit, "frame": s.information.frame,
                       "description": s.description, "entity": s.entity, "source": s.source,
                       "information": s.information.describe()} for path, s in self._fields.items()}

    def information(self, path):
        return self._fields[path].information

    def read(self, path):
        return self.read_many((path,))[path]

    def read_many(self, paths):
        paths = tuple(dict.fromkeys(paths))
        missing = {path for path in paths if path not in self._fields}
        if missing:
            raise KeyError(f"Unknown signal fields: {sorted(missing)}")
        if self._prepare is not None:
            self._prepare(paths)
        return {path: self._fields[path].read() for path in paths}

    def __contains__(self, path):
        return path in self._fields


@dataclass
class Observation:
    """Explicit alias -> signal selection. Leaves stacking/noise/encoding to users."""
    fields: dict[str, str]

    def __post_init__(self):
        if not isinstance(self.fields, dict) or any(
            not isinstance(alias, str) or not alias or not isinstance(path, str) or not path
            for alias, path in self.fields.items()
        ):
            raise ValueError("Observation fields must map nonempty aliases to nonempty signal paths")
        self.fields = dict(self.fields)

    def read(self, signals: Signals):
        cache = signals.read_many(self.fields.values())
        return {alias: cache[path] for alias, path in self.fields.items()}
