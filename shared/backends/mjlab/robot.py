"""mjlab primitives of the shared robot interface: MuJoCo body poses and mujoco_warp Jacobians."""
from shared.assembly.joint_order import canonical_joints
from shared.assembly.robot import Kinematics as BaseKinematics, provider


def recorded_order(entity):
    """joint_order of the asset an entity was built from; entities built from a converted MJCF
    carry its path on spec_fn (assets._mjcf_entity). None without a record."""
    from shared.asset_formats import recorded_joint_order
    path = getattr(getattr(getattr(entity, 'cfg', None), 'spec_fn', None), 'asset_path', None)
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
        self.env, self.entity = env, env.scene[name]
        self.buffers = {}
        self.names, self.index = joint_layout(env, name)

    def joint_names(self):
        return self.names

    def joint_position(self):
        value = self.entity.data.joint_pos
        return value if self.index is None else value[:, self.index]

    def body_names(self):
        return tuple(self.entity.body_names)

    def _body(self, body):
        return int(self.entity.indexing.body_ids[self.entity.body_names.index(body)])

    def body_pose(self, body):
        index, data = self._body(body), self.env.sim.data
        return data.xpos[:, index] - self.env.scene.env_origins, data.xquat[:, index]

    def body_jacobian(self, body):
        import mujoco_warp as mjwarp
        import torch
        import warp as wp
        sim, index = self.env.sim, self._body(body)
        if body not in self.buffers:
            # Same native call and column selection as mjlab's DifferentialIKAction.
            count, dofs = self.env.num_envs, sim.mj_model.nv
            with wp.ScopedDevice(sim.wp_device):
                arrays = (wp.zeros((count, 3, dofs), dtype=float), wp.zeros((count, 3, dofs), dtype=float),
                          wp.zeros(count, dtype=wp.vec3), wp.zeros(count, dtype=wp.int32))
            arrays[3].fill_(index)
            self.buffers[body] = arrays, wp.to_torch(arrays[0]), wp.to_torch(arrays[1]), wp.to_torch(arrays[2]).view(count, 3)
        arrays, linear, angular, point = self.buffers[body]
        point[:] = sim.data.xpos[:, index]
        with wp.ScopedDevice(sim.wp_device):
            mjwarp.jac(sim.wp_model, sim.wp_data, *arrays)
        columns = self.entity.indexing.joint_v_adr.long()
        if self.index is not None:
            columns = columns[self.index]
        return torch.cat((linear[:, :, columns], angular[:, :, columns]), 1)


def robots(env, settings):
    return provider(env, settings, Kinematics)
