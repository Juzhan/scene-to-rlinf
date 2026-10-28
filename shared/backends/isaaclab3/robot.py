"""Isaac Lab 3 primitives of the shared robot interface: link poses and Newton Jacobians."""
from shared.assembly.joint_order import canonical_joints
from shared.assembly.robot import Kinematics as BaseKinematics, provider
from .frames import wxyz


def joint_layout(env, key):
    """(names, None): scene.py installs the recorded order as native joint_ordering, so columns match it."""
    from shared.asset_formats import recorded_joint_order
    native = env.scene[key]
    spawn = getattr(native.cfg, 'spawn', None)
    path = getattr(spawn, 'usd_path', None)
    names, index = canonical_joints(native.joint_names, recorded_joint_order(path) if path else None, key)
    if index is not None:
        raise RuntimeError(f'{key}: native joint_ordering does not match the recorded joint order')
    return names, None


def joint_index(env, key):
    return None


class Kinematics(BaseKinematics):
    def __init__(self, env, name, frames):
        super().__init__(name, frames)
        self.env, self.asset = env, env.scene[name]
        if not hasattr(self.asset, 'num_base_dofs'):
            raise ValueError(f'{name}: robot methods need a native articulation')
        self.names, _ = joint_layout(env, name)

    def joint_names(self):
        return self.names

    def joint_position(self):
        return self.asset.data.joint_pos.torch

    def body_names(self):
        return tuple(self.asset.body_names)

    def body_pose(self, body):
        index, data = self.asset.body_names.index(body), self.asset.data
        return (data.body_link_pos_w.torch[:, index] - self.env.scene.env_origins,
                wxyz(data.body_link_quat_w.torch[:, index]))

    def body_jacobian(self, body):
        # Referenced at the link origin; joint columns follow joint_names after the base DOFs.
        index = self.asset.body_names.index(body)
        jacobians = self.asset.data.body_link_jacobian_w.torch
        if self.asset.is_fixed_base:
            if index == 0:
                return jacobians.new_zeros(jacobians.shape[0], 6, self.asset.num_joints)
            return jacobians[:, index - 1]
        return jacobians[:, index, :, self.asset.num_base_dofs:]


def robots(env, settings):
    return provider(env, settings, Kinematics)
