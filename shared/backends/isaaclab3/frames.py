"""Quaternion order at the backend boundary: Isaac Lab 3 stores xyzw, the shared layer wxyz."""


def wxyz(quaternion):
    """Native xyzw -> shared wxyz; tensors keep their batch axes, sequences become tuples."""
    if hasattr(quaternion, 'shape'):
        return quaternion[..., [3, 0, 1, 2]]
    x, y, z, w = quaternion
    return w, x, y, z


def xyzw(quaternion):
    """Shared wxyz -> native xyzw."""
    if hasattr(quaternion, 'shape'):
        return quaternion[..., [1, 2, 3, 0]]
    w, x, y, z = quaternion
    return x, y, z, w
