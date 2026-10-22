"""Isaac 5.1 camera attachment using live physics poses instead of USD parents."""


def mounted_camera_class(base, body_path):
    """Keep native camera sampling; resolve the mounted body's current pose.

    Nested USD cameras can inherit stale articulation transforms with Fabric.
    A sibling camera avoids that ambiguity. Its native sensor update follows
    the rigid-body view before rendering the sample requested by the observer.
    """
    class MountedCamera(base):
        def _initialize_impl(self):
            import torch
            from isaacsim.core.simulation_manager import SimulationManager

            super()._initialize_impl()
            namespace = self.cfg.prim_path.rsplit("/", 1)[0]
            expression = body_path.replace("{ENV_REGEX_NS}", namespace).replace(".*", "*")
            self._mounted_body = SimulationManager.get_physics_sim_view().create_rigid_body_view(expression)
            targets = [body_path.replace("{ENV_REGEX_NS}", path.rsplit("/", 1)[0])
                       for path in self._view.prim_paths]
            actual = self._mounted_body.prim_paths
            if len(actual) != len(targets) or any(path not in actual for path in targets):
                raise ValueError(f"Camera mount requires one concrete rigid body per camera: {body_path}")
            self._mounted_order = torch.tensor([actual.index(path) for path in targets],
                                                device=self.device, dtype=torch.long)

        def _update_buffers_impl(self, env_ids):
            import torch
            from isaaclab.sim import SimulationContext
            from isaaclab.utils.math import (combine_frame_transforms, convert_quat,
                                            convert_camera_frame_orientation_convention)

            transforms = self._mounted_body.get_transforms()[self._mounted_order]
            positions = transforms[:, :3]
            orientations = convert_quat(transforms[:, 3:7], to="wxyz")
            offset_pos = torch.tensor(self.cfg.offset.pos, device=self.device).expand(len(transforms), 3)
            offset_rot = torch.tensor(self.cfg.offset.rot, device=self.device).expand(len(transforms), 4)
            offset_rot = convert_camera_frame_orientation_convention(
                offset_rot, origin=self.cfg.offset.convention, target="opengl")
            positions, orientations = combine_frame_transforms(positions, orientations, offset_pos, offset_rot)
            # In Isaac 5.1 the first render propagates the new USD/Fabric pose,
            # while its image/pose can still describe the previous sample.
            # Reapply and render once more before exposing a synchronized pair.
            for _ in range(2):
                self.set_world_poses(positions, orientations, convention="opengl")
                SimulationContext.instance().render()
            super()._update_buffers_impl(env_ids)

    return MountedCamera
