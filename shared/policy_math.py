"""Pure Torch geometry helpers for project model feature extraction (wxyz quaternions)."""
import torch


def inverse_rotate(vector, quaternion):
    xyz = quaternion[..., 1:]
    cross = 2 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector - quaternion[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def yaw(quaternion):
    """Heading about +Z in radians, shape (..., ); wxyz quaternion."""
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def relative_goal(position, quaternion, goal_position, goal_heading):
    heading = yaw(quaternion)
    offset = goal_position - position
    c, s = heading.cos(), heading.sin()
    relative = torch.stack((c * offset[..., 0] + s * offset[..., 1],
                            -s * offset[..., 0] + c * offset[..., 1], offset[..., 2]), -1)
    difference = goal_heading - heading
    wrapped = (difference + torch.pi) % (2 * torch.pi)
    error = torch.where((wrapped == 0) & (difference > 0), torch.pi, wrapped - torch.pi)
    return relative, error.unsqueeze(-1)
