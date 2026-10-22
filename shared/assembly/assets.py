"""Resolve logical asset names to backend implementations, without loading SDKs."""
from importlib import import_module
from inspect import signature
from math import isfinite
from pathlib import Path
from typing import Literal, get_args

BACKENDS = ('isaac', 'mjlab')
DEFAULT_BACKEND = 'isaac'


def selected_backend(parameters):
    """Backend named by ``init_params.backend``; recipes without one use the default."""
    backend = (parameters or {}).get('backend', DEFAULT_BACKEND)
    if backend not in BACKENDS:
        raise ValueError(f'Unknown scene backend {backend!r}; choose one of {BACKENDS}')
    return backend


def backend_module(backend, name):
    """A backend's implementation of a shared hook module, e.g. ``project`` or ``control``."""
    return import_module(f'shared.backends.{selected_backend({"backend": backend})}.{name}')


def factories_for(backend=DEFAULT_BACKEND):
    return dict(backend_module(backend, 'assets').FACTORIES)


def asset_backends(name):
    return [backend for backend in BACKENDS if name in factories_for(backend)]


def workspace_file(path):
    """Imported files are relative to the workspace, not to the launch directory."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        raise FileNotFoundError(f"Asset file is missing: {path}")
    return path.resolve()


def box_parameters(size, mass, color, friction):
    """Common dimensions are full side lengths in metres; None mass is static."""
    if len(size) != 3 or any(not isfinite(v) or v <= 0 for v in size):
        raise ValueError('box.size requires three positive finite side lengths')
    if mass is not None and (not isfinite(mass) or mass <= 0):
        raise ValueError('box.mass must be positive or null for a static fixture')
    if len(color) != 3 or any(not isfinite(v) or not 0 <= v <= 1 for v in color):
        raise ValueError('box.color requires three values in [0,1]')
    if not isfinite(friction) or friction < 0:
        raise ValueError('box.friction must be nonnegative and finite')


def checked(check):
    """Attach an SDK-free value check that check/editor saves run without building the asset.

    ``check`` receives every factory parameter (authored values over defaults) by name.
    The factory still calls it itself, so direct construction is validated too.
    """
    def attach(factory):
        factory.check_parameters = check
        return factory
    return attach


def check_parameters(factory, parameters):
    """Bind authored parameters to the factory signature and run its attached value check."""
    try:
        spec = signature(factory)
    except (TypeError, ValueError):
        return
    arguments = spec.bind(**parameters)
    check = getattr(factory, 'check_parameters', None)
    if check is not None:
        arguments.apply_defaults()
        check(**arguments.arguments)


CameraData = Literal['rgb', 'distance_to_image_plane', 'distance_to_camera', 'normals']
CAMERA_DATA = get_args(CameraData)


def camera_parameters(width, height, data_types, focal_length, clipping_range, depth_clipping):
    """Pinhole camera: image size in pixels, focal length in mm, clipping planes in metres."""
    for name, value in (('width', width), ('height', height)):
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError(f'camera.{name} must be an integer pixel count in [1, 4096]')
    if (isinstance(data_types, str) or not data_types or len(set(data_types)) != len(data_types)
            or set(data_types) - set(CAMERA_DATA)):
        raise ValueError(f'camera.data_types must list distinct outputs from {list(CAMERA_DATA)}')
    if not isfinite(focal_length) or focal_length <= 0:
        raise ValueError('camera.focal_length must be a positive number of millimetres')
    if len(clipping_range) != 2 or not all(isfinite(v) for v in clipping_range) or not 0 < clipping_range[0] < clipping_range[1]:
        raise ValueError('camera.clipping_range requires 0 < near < far (m)')
    if depth_clipping not in ('max', 'zero', 'none'):
        raise ValueError("camera.depth_clipping must be 'max', 'zero' or 'none'")


def image_outputs(factory, parameters):
    """Image types an asset produces: its ``data_types`` parameter, else RGB only."""
    try:
        arguments = signature(factory).bind_partial(**parameters)
    except (TypeError, ValueError):
        return ('rgb',)
    arguments.apply_defaults()
    return tuple(arguments.arguments.get('data_types', ('rgb',)))
