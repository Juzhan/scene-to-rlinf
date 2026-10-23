"""mjlab primitives of the shared robot interface: MuJoCo body poses and mujoco_warp Jacobians."""
from shared.assembly.robot import Kinematics as BaseKinematics, provider


class Kinematics(BaseKinematics):
    def __init__(self, env, name, frames):
        super().__init__(name, frames)
        self.env, self.entity = env, env.scene[name]
        self.buffers = {}

    def joint_names(self):
        return tuple(self.entity.joint_names)

    def joint_position(self):
        return self.entity.data.joint_pos

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
        return torch.cat((linear[:, :, columns], angular[:, :, columns]), 1)


def robots(env, settings):
    return provider(env, settings, Kinematics)
