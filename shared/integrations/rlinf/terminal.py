"""Compatibility forward; the snapshot helpers belong to the backend boundary (shared.backends.terminal)."""
from shared.backends.terminal import (  # noqa: F401 - re-exported for existing imports
    TerminalObservationMixin, snapshot, snapshot_terminal_observations, validate_terminal_observations,
)
