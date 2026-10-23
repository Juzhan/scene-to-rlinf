"""Isaac primitives of the shared robot interface: native body poses and PhysX Jacobians."""
from shared.assembly.robot import Kinematics as BaseKinematics, move_point, provider


class Kinematics(BaseKinematics):
    def __init__(self, env, name, frames):
        super().__init__(name, frames)
        self.env, self.asset = env, env.scene[name]
        if not hasattr(self.asset, 'root_physx_view'):
            raise ValueError(f'{name}: robot methods need a native articulation')

    def joint_names(self):
        return tuple(self.asset.joint_names)

    def joint_position(self):
        return self.asset.data.joint_pos

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
        # PhysX linear rows refer to the link centre of mass; move them to the link origin.
        return move_point(value, data.body_link_pos_w[:, index] - data.body_com_pos_w[:, index])


def robots(env, settings):
    return provider(env, settings, Kinematics)
