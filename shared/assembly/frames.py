"""Editor frame math shared by scene editors; numpy only, no simulator import.

Poses use the scene.yaml convention: position in metres and a (w, x, y, z)
quaternion. Matrices are 4x4 homogeneous transforms. These helpers only place
editor previews and compute authored offsets; physical constraints remain the
backend's job.
"""
import math

import numpy as np

IDENTITY_POSE = {'position': [0.0, 0.0, 0.0], 'orientation': [1.0, 0.0, 0.0, 0.0]}
AXES = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}


def quat_matrix(quat):
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_quat(rotation):
    m = np.asarray(rotation, dtype=float)[:3, :3]
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        quat = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        quat = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        quat = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        quat = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    quat = np.asarray(quat) / np.linalg.norm(quat)
    return quat if quat[0] >= 0 else -quat


def pose_matrix(pose):
    pose = pose or {}
    result = np.eye(4)
    result[:3, :3] = quat_matrix(pose.get('orientation', (1, 0, 0, 0)))
    result[:3, 3] = pose.get('position', (0, 0, 0))
    return result


def matrix_pose(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return {'position': [float(v) for v in matrix[:3, 3]], 'orientation': [float(v) for v in matrix_quat(matrix)]}


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def euler_matrix(degrees):
    """XYZ Euler angles in degrees, applied X then Y then Z (Blender's default)."""
    rx, ry, rz = (math.radians(float(v)) for v in degrees)
    return axis_angle(AXES['Z'], rz) @ axis_angle(AXES['Y'], ry) @ axis_angle(AXES['X'], rx)


def matrix_euler(rotation):
    m = np.asarray(rotation, dtype=float)[:3, :3]
    ry = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(m[2, 0]) < 1 - 1e-9:
        rx, rz = math.atan2(m[2, 1], m[2, 2]), math.atan2(m[1, 0], m[0, 0])
    else:
        rx, rz = math.atan2(-m[1, 2], m[1, 1]), 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def transform(position=(0, 0, 0), rotation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def same_pose(a, b, tolerance=1e-6):
    """Compare authored poses; q and -q are the same rotation."""
    a, b = a or IDENTITY_POSE, b or IDENTITY_POSE
    position = np.allclose(a.get('position', (0, 0, 0)), b.get('position', (0, 0, 0)), atol=tolerance)
    qa = np.asarray(a.get('orientation', (1, 0, 0, 0)), dtype=float)
    qb = np.asarray(b.get('orientation', (1, 0, 0, 0)), dtype=float)
    qa, qb = qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)
    return bool(position and (np.allclose(qa, qb, atol=tolerance) or np.allclose(qa, -qb, atol=tolerance)))


def align_rotation(matrix, mode):
    """One-shot axis alignment of a relative transform; never a constraint.

    ``all`` matches the parent frame; ``x``/``y``/``z`` rotates by the
    shortest arc that makes that local axis parallel to the parent's axis.
    """
    matrix = np.array(matrix, dtype=float)
    if mode in (None, 'none'):
        return matrix
    if mode == 'all':
        matrix[:3, :3] = np.eye(3)
        return matrix
    axis = np.asarray(AXES[mode.upper()])
    current = matrix[:3, :3] @ axis
    cross = np.cross(current, axis)
    sine, cosine = np.linalg.norm(cross), float(np.dot(current, axis))
    if sine < 1e-12:
        if cosine < 0:
            # Opposite: any perpendicular axis gives a shortest arc.
            perpendicular = np.cross(axis, (1, 0, 0) if abs(axis[0]) < .9 else (0, 1, 0))
            matrix[:3, :3] = axis_angle(perpendicular, math.pi) @ matrix[:3, :3]
        return matrix
    matrix[:3, :3] = axis_angle(cross, math.atan2(sine, cosine)) @ matrix[:3, :3]
    return matrix


def forward_kinematics(frames, joints, positions=None):
    """Root-local body transforms after replacing exported joint positions.

    ``frames`` are the exported root-local body poses at each joint's
    exported ``position``; only the joint offset is applied, so the result
    reproduces the native frames exactly when no values change.
    """
    positions = positions or {}
    baseline = {name: pose_matrix(pose) for name, pose in frames.items()}
    result = dict(baseline)
    pending = list(joints)
    while pending:
        children = {joint['child'] for joint in pending}
        ready = [joint for joint in pending if joint['parent'] not in children]
        if not ready:
            raise ValueError('Native joint tree contains a cycle')
        for joint in ready:
            pending.remove(joint)
            parent, child = joint['parent'], joint['child']
            if parent not in baseline or child not in baseline:
                continue
            origin = pose_matrix(joint['origin'])
            delta = float(positions.get(joint['name'], joint['position'])) - float(joint['position'])
            motion = np.eye(4)
            if joint['type'] == 'revolute':
                motion[:3, :3] = axis_angle(AXES[joint['axis']], delta)
            elif joint['type'] == 'prismatic':
                motion[:3, 3] = np.asarray(AXES[joint['axis']]) * delta
            result[child] = (result[parent] @ origin @ motion @ np.linalg.inv(origin)
                             @ np.linalg.inv(baseline[parent]) @ baseline[child])
    return result


def fixed_child_local(parent_pose, child_pose, child_body_local):
    """Child root relative to the parent body for a fixed connection."""
    return pose_matrix(parent_pose) @ np.linalg.inv(pose_matrix(child_pose)) @ np.linalg.inv(child_body_local)


def fixed_parent_pose(child_local, child_body_local, child_pose):
    """Inverse of :func:`fixed_child_local`: the authored ``parent_pose``."""
    return matrix_pose(np.asarray(child_local) @ child_body_local @ pose_matrix(child_pose))
