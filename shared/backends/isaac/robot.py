"""Isaac primitives of the shared robot interface: native body poses and PhysX Jacobians."""
from shared.assembly.joint_order import canonical_joints
from shared.assembly.robot import Kinematics as BaseKinematics, move_point, provider


def recorded_order(asset):
    """joint_order of the asset's USD (or URDF) file; None without a record."""
    from shared.asset_formats import recorded_joint_order
    spawn = getattr(getattr(asset, 'cfg', None), 'spawn', None)
    path = getattr(spawn, 'usd_path', None) or getattr(spawn, 'asset_path', None)
    return recorded_joint_order(path) if path else None


def joint_layout(env, key):
    """(names, device index or None): joint names in the recorded order and the native column of each."""
    import torch
    native = env.scene[key]
    names, index = canonical_joints(native.joint_names, recorded_order(native), key)
    return names, None if index is None else torch.tensor(index, device=env.device)


def joint_index(env, key):
    """Device index putting native joint columns in the recorded order; None when they already are."""
    return None if recorded_order(env.scene[key]) is None else joint_layout(env, key)[1]


class Kinematics(BaseKinematics):
    def __init__(self, env, name, frames):
        super().__init__(name, frames)
        self.env, self.asset = env, env.scene[name]
        if not hasattr(self.asset, 'root_physx_view'):
            raise ValueError(f'{name}: robot methods need a native articulation')
        self.names, self.index = joint_layout(env, name)

    def joint_names(self):
        return self.names

    def joint_position(self):
        value = self.asset.data.joint_pos
        return value if self.index is None else value[:, self.index]

    def body_names(self):
        return tuple(self.asset.body_names)

    def body_pose(self, body):
        index, data = self.asset.body_names.index(body), self.asset.data
        return data.body_link_pos_w[:, index] - self.env.scene.env_origins, data.body_link_quat_w[:, index]

    def body_jacobian(self, body):
        index, data = self.asset.body_names.index(body), self.asset.data
        jacobians = self.asset.root_physx_view.get_jacobians()
        # PhysX omits the root body of a fixed base; a floating base adds 6 root DOF columns first.
        if self.asset.is_fixed_base:
            if index == 0:
                return jacobians.new_zeros(jacobians.shape[0], 6, self.asset.num_joints)
            value = jacobians[:, index - 1]
        else:
            value = jacobians[:, index, :, 6:]
        if self.index is not None:
            value = value[..., self.index]
        # PhysX linear rows refer to the link centre of mass; move them to the link origin.
        return move_point(value, data.body_link_pos_w[:, index] - data.body_com_pos_w[:, index])


def robots(env, settings):
    return provider(env, settings, Kinematics)
