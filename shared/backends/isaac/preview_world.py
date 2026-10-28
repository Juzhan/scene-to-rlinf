"""Isaac Lab 2.3 / Isaac Sim 5.1 assembly backend.

Import and construct only after AppLauncher. This builds the runtime world
around an InteractiveScene, not a training environment. Actions, rewards and
training uses Isaac Lab's native environment separately.
"""

from copy import deepcopy
from importlib import import_module

from shared.assembly.signals import Signal
from .preview_runtime import RuntimeWorld
from shared.assembly.spec import Mount, Sensor


def _on_demand_camera(base):
    class OnDemandCamera(base):
        def _initialize_impl(self):
            import omni.replicator.core as rep

            super()._initialize_impl()
            self._assembly_render_products = []
            for prim, path in zip(self._sensor_prims, self.render_product_paths):
                product = rep.create.render_product(str(prim.GetPath()),
                    resolution=(self.cfg.width, self.cfg.height), force_new=False)
                if product.path != path:
                    raise RuntimeError("Camera render product lookup returned a different product")
                product.hydra_texture.set_updates_enabled(False)
                self._assembly_render_products.append(product)

        def set_rendering(self, enabled):
            for product in self._assembly_render_products:
                product.hydra_texture.set_updates_enabled(enabled)
            if enabled:
                for annotators in self._rep_registry.values():
                    for annotator, product in zip(annotators, self._assembly_render_products):
                        annotator.detach([product.path])
                        annotator.attach(product)

        def has_rendered_data(self):
            for annotators in self._rep_registry.values():
                for annotator in annotators:
                    output = annotator.get_data()
                    if isinstance(output, dict):
                        output = output.get("data")
                    if output is None or not output.shape or not all(output.shape):
                        return False
            return True

    return OnDemandCamera


class IsaacAssemblyWorld(RuntimeWorld):
    def __init__(self, world, *, num_envs=1, env_spacing=4.0, decimation=1, factories=None,
                 replicate_physics=False, viewer=None, render_mode=None):
        from shared.assembly.spec import Abstract
        if any(isinstance(node, Abstract) for node, _ in world.index().values()):
            raise NotImplementedError("Abstract entities require the native Project environment; "
                                      "the standalone assembly_demo preview does not support them")
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg, SensorBaseCfg
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        if num_envs < 1 or decimation < 1:
            raise ValueError("num_envs and decimation must be positive")
        self.sim = sim_utils.SimulationContext.instance()
        if self.sim is None:
            raise RuntimeError("Create SimulationContext before IsaacAssemblyWorld")
        self.decimation = decimation
        super().__init__(world, num_envs=num_envs, step_dt=self.sim.get_physics_dt() * decimation,
                         prepare_signals=self._prepare_signals)
        self.factories = dict(factories or {})
        self.mounts = {p: self.description.resolve_mount(p, n.mount) for p, (n, _) in self.index.items() if n.mount}
        self.connections = self.description.ordered_connections()
        self.keys = {path: f"entity_{i}" for i, (path, (node, _)) in enumerate(self.index.items()) if node.factory}
        self.prim_paths = {path: f"{{ENV_REGEX_NS}}/{key}" for path, key in self.keys.items()}
        self.configs = {}
        self._sensor_epoch = 0
        self._camera_samples = {}
        self._image_sources = {}
        cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=replicate_physics)

        # Cameras are spawned after all potential mounting assets.
        nodes = sorted(self.index.items(), key=lambda pair: isinstance(pair[1][0], Sensor))
        for path, (node, pose) in nodes:
            if not node.factory:
                continue
            native = self._asset_config(node)
            if node.reset is not None and (not isinstance(native, RigidObjectCfg)
                    or (native.spawn and native.spawn.rigid_props and native.spawn.rigid_props.kinematic_enabled)):
                raise ValueError(f"{path}: pose reset currently requires a dynamic rigid object")
            if isinstance(node, Sensor):
                if not isinstance(native, SensorBaseCfg):
                    raise TypeError(f"{path}: Sensor factory must return SensorBaseCfg")
                if node.mount and not isinstance(native, CameraCfg):
                    raise NotImplementedError(f"{path}: live mounts currently support native cameras only")
            if isinstance(native, CameraCfg):
                if node.mount:
                    # Keep the render camera outside the moving USD subtree.
                    # Its physical mount is enforced in _sync_cameras; nested
                    # Fabric transforms otherwise apply stale parent offsets.
                    pose = node.pose
                native.offset = CameraCfg.OffsetCfg(pos=pose.position, rot=pose.orientation, convention="opengl")
                native.update_latest_camera_pose = True
                native.class_type = _on_demand_camera(native.class_type)
            elif isinstance(native, AssetBaseCfg):
                native.init_state.pos, native.init_state.rot = pose.position, pose.orientation
            elif isinstance(native, SensorBaseCfg):
                pass
            else:
                raise TypeError(f"Unsupported factory result at {path}: {type(native)}")
            if (isinstance(native, AssetBaseCfg) and not isinstance(native, (ArticulationCfg, RigidObjectCfg))
                    and native.prim_path.startswith("/") and "{ENV_REGEX_NS}" not in native.prim_path):
                self.prim_paths[path] = native.prim_path
            native.prim_path = self.prim_paths[path]
            self.configs[path] = native
            setattr(cfg, self.keys[path], native)

        # Root constraints on an attached articulation must be explicitly disabled
        # in its project-owned config, rather than silently overriding the model.
        for joint in self.connections:
            native = self.configs[joint.child.entity]
            if not isinstance(native, (ArticulationCfg, RigidObjectCfg)):
                raise TypeError(f"{joint.child.entity}: fixed attachment requires a physical child")
            if isinstance(native, RigidObjectCfg) and native.spawn and native.spawn.rigid_props and native.spawn.rigid_props.kinematic_enabled:
                raise ValueError(f"{joint.child.entity}: kinematic child cannot be driven by a fixed joint")
            if isinstance(native, ArticulationCfg):
                props = native.spawn.articulation_props if native.spawn else None
                if props is None or props.fix_root_link is not False:
                    raise ValueError(f"{joint.child.entity}: attached articulation requires fix_root_link=False")

        self.scene = InteractiveScene(cfg)
        self.device = self.scene.device
        self.render_mode = render_mode
        if viewer is not None or render_mode == "rgb_array":
            from types import SimpleNamespace
            from isaaclab.envs.common import ViewerCfg
            from isaaclab.envs.ui import ViewportCameraController

            self.cfg = SimpleNamespace(viewer=ViewerCfg(origin_type="env", env_index=0, **(viewer or {})))
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        self.elapsed_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        stage = sim_utils.get_current_stage()
        for path, (node, _) in self.index.items():
            if node.mount:
                mount = self.mounts[path]
                target = self.prim_paths[mount.entity].replace("{ENV_REGEX_NS}", self.scene.env_prim_paths[0])
                if mount.body:
                    target += "/" + mount.body
                if not stage.GetPrimAtPath(target).IsValid():
                    raise ValueError(f"Missing camera mount: {target}")
                if not stage.GetPrimAtPath(target).HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(f"Mounted camera currently requires a rigid-body link: {target}; use a scene camera for static placement")

        def endpoint(mount, env_ns):
            prim_path = self.prim_paths[mount.entity].replace("{ENV_REGEX_NS}", env_ns)
            if mount.body:
                prim_path += "/" + mount.body
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Fixed joint endpoint must be a rigid body: {prim_path}")
            return prim_path

        def matrix(pose):
            result = Gf.Matrix4d(1.)
            result.SetRotate(Gf.Quatd(pose.orientation[0], Gf.Vec3d(*pose.orientation[1:])))
            result.SetTranslateOnly(Gf.Vec3d(*pose.position))
            return result

        for i, env_ns in enumerate(self.scene.env_prim_paths):
            UsdGeom.Scope.Define(stage, env_ns + "/Connections")
            for joint in self.connections:
                child_path = endpoint(joint.child, env_ns)
                parent_path = endpoint(joint.parent, env_ns) if joint.parent else None
                child_cfg = self.configs[joint.child.entity]
                if isinstance(child_cfg, ArticulationCfg):
                    # A floating articulation with a container-level root API lets
                    # PhysX choose a middle link as root. Pin the intended base.
                    asset_root = stage.GetPrimAtPath(self.prim_paths[joint.child.entity].replace("{ENV_REGEX_NS}", env_ns))
                    for prim in Usd.PrimRange(asset_root):
                        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    # fix_root_link=False explicitly releases the old world
                    # anchor; remove it so moving the asset cannot leave a stale
                    # anchor frame behind in the source USD.
                    anchors = []
                    for prim in Usd.PrimRange(asset_root):
                        if prim.IsA(UsdPhysics.FixedJoint):
                            anchor = UsdPhysics.FixedJoint(prim)
                            if bool(anchor.GetBody0Rel().GetTargets()) != bool(anchor.GetBody1Rel().GetTargets()):
                                anchors.append(prim.GetPath())
                    for anchor in anchors:
                        # Deactivate through an override: RemovePrim alone would
                        # reveal the same joint again from the referenced USD.
                        stage.GetPrimAtPath(anchor).SetActive(False)
                    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(child_path))
                    sim_utils.modify_articulation_root_properties(child_path, child_cfg.spawn.articulation_props, stage=stage)
                # Align the authored rest geometry before PhysX creates the joint.
                # Runtime reset repeats alignment after writing initial joint angles.
                cache = UsdGeom.XformCache()
                parent_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path or env_ns))
                desired_body = matrix(joint.child_pose).GetInverse() * matrix(joint.parent_pose) * parent_world
                root_path = self.prim_paths[joint.child.entity].replace("{ENV_REGEX_NS}", env_ns)
                root = stage.GetPrimAtPath(root_path)
                root_world = cache.GetLocalToWorldTransform(root)
                body_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_path))
                root_parent_world = cache.GetLocalToWorldTransform(root.GetParent())
                root_local = root_world * body_world.GetInverse() * desired_body * root_parent_world.GetInverse()
                UsdGeom.Xformable(root).MakeMatrixXform().Set(root_local)
                native = UsdPhysics.FixedJoint.Define(stage, env_ns + "/Connections/" + joint.name)
                native.CreateBody1Rel().SetTargets([child_path])
                if parent_path:
                    native.CreateBody0Rel().SetTargets([parent_path])
                    position = joint.parent_pose.position
                else:
                    origin = self.scene.env_origins[i].tolist()
                    position = tuple(a+b for a, b in zip(joint.parent_pose.position, origin))
                native.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
                native.CreateLocalRot0Attr().Set(Gf.Quatf(joint.parent_pose.orientation[0], Gf.Vec3f(*joint.parent_pose.orientation[1:])))
                native.CreateLocalPos1Attr().Set(Gf.Vec3f(*joint.child_pose.position))
                native.CreateLocalRot1Attr().Set(Gf.Quatf(joint.child_pose.orientation[0], Gf.Vec3f(*joint.child_pose.orientation[1:])))
                native.CreateExcludeFromArticulationAttr().Set(True)
                native.CreateCollisionEnabledAttr().Set(False)

        self._register_signals()

    def asset(self, path):
        return self.scene[self.keys[path]]

    def render(self, recompute=False):
        from isaaclab.envs import ManagerBasedRLEnv

        return ManagerBasedRLEnv.render(self, recompute=recompute)

    def _asset_config(self, node):
        factory = self.factories.get(node.factory)
        if callable(factory):
            return deepcopy(factory(**deepcopy(node.parameters)))
        if not isinstance(factory, str) or ":" not in factory:
            raise ValueError(f"No Isaac factory registered for {node.factory!r}")
        module, symbol = factory.split(":", 1)
        return deepcopy(getattr(import_module(module), symbol)(**deepcopy(node.parameters)))

    def set_joint_position_target(self, path, target, env_ids=None):
        import torch

        if self.keys[path] not in self.scene.articulations:
            raise TypeError(f"Joint-position control requires an articulation: {path}")
        asset = self.asset(path)
        ids = self._ids(env_ids)
        target = torch.as_tensor(target, device=self.device, dtype=asset.data.joint_pos.dtype)
        expected = (len(ids), asset.num_joints)
        if tuple(target.shape) != expected or not torch.isfinite(target).all():
            raise ValueError(f"Joint-position target must be finite with shape {expected}")
        asset.set_joint_position_target(target, env_ids=ids)

    def _register_signals(self):
        from isaaclab.assets import Articulation, RigidObject
        from isaaclab.sensors import Camera as NativeCamera
        from shared.assembly.information import Information, Field, pose_information

        for path, (node, _) in self.index.items():
            if path not in self.keys:
                continue
            asset = self.asset(path)
            if isinstance(asset, (RigidObject, Articulation)):
                self.signals.register(path + "/pose", Signal(lambda a=asset: self._local_pose(a.data.root_pose_w),
                    info=pose_information("environment", path + "/root"), entity=path, source="simulation"))
                self.signals.register(path + "/velocity", Signal(lambda a=asset: a.data.root_vel_w.clone(),
                    info=Information("record", (6,), "float", fields={
                        "linear": Field((0, 1, 2), Information("vector", (3,), "float", "m/s", "world")),
                        "angular": Field((3, 4, 5), Information("vector", (3,), "float", "rad/s", "world")),
                    }), entity=path, source="simulation"))
                self.signals.register(path + "/mass", Signal(lambda a=asset: a.root_physx_view.get_masses().clone(), "physics_parameter", "kg",
                    info=Information("number", (None,), "float", "kg", axes=("body",)), entity=path, source="simulation"))
            if isinstance(asset, Articulation):
                for field, attr, unit in (("joint_position", "joint_pos", "rad or m"),
                                          ("joint_velocity", "joint_vel", "rad/s or m/s"),
                                          ("applied_torque", "applied_torque", "Nm or N")):
                    self.signals.register(path + "/" + field, Signal(lambda a=asset, k=attr: getattr(a.data, k).clone(),
                        info=Information("number", (len(asset.joint_names),), "float", axes=("joint",), labels=tuple(asset.joint_names)),
                        entity=path, source="simulation"))
                self.signals.register(path + "/joint_names", Signal(lambda a=asset: tuple(a.joint_names), "metadata"))
                self.signals.register(path + "/joint_defaults", Signal(lambda a=asset: a.data.default_joint_pos.clone(), "physics_parameter", "rad or m"))
                self.signals.register(path + "/joint_limits", Signal(lambda a=asset: a.data.soft_joint_pos_limits.clone(), "physics_parameter", "rad or m"))
            for body_frame in node.frames:
                if not isinstance(asset, (Articulation, RigidObject)):
                    raise TypeError(f"Body frames require a rigid object or articulation: {path}")
                mount = Mount(path, body_frame.body)
                source = path + "/frames/" + body_frame.name
                self.signals.register(source + "/pose", Signal(
                    lambda m=mount, p=body_frame.pose: self._body_frame_pose(m, p),
                    info=pose_information("environment", source), entity=path, source="simulation"))
                self.signals.register(source + "/position", Signal(
                    lambda m=mount, p=body_frame.pose: self._body_frame_pose(m, p)[:, :3],
                    info=Information("position", (3,), "float", "m", "environment"), entity=path, source="simulation"))
            if isinstance(asset, NativeCamera):
                for data_type in asset.cfg.data_types:
                    source = path + "/" + data_type
                    self._image_sources[source] = path
                    self.signals.register(source, Signal(lambda a=asset, k=data_type: a.data.output[k].clone(), "sensor",
                        info=Information("image", dtype="uint8" if data_type == "rgb" else None, encoding=data_type),
                        entity=path, source="sensor", description="Captured on read; cached until physics advance or reset"))
                self.signals.register(path + "/position", Signal(lambda a=asset: self._camera_pose(a)[0] - self.scene.env_origins,
                    info=Information("position", (3,), "float", "m", "environment"), entity=path, source="sensor"))
                self.signals.register(path + "/orientation", Signal(lambda a=asset: self._camera_pose(a)[1],
                    info=Information("rotation", (4,), "float", "1", "environment", path + "/camera", "wxyz"), entity=path, source="sensor"))
            elif not isinstance(node, Sensor) and not isinstance(asset, (RigidObject, Articulation)):
                self.signals.register(path + "/pose", Signal(lambda a=asset: self._static_pose(a),
                    info=pose_information("environment", path), entity=path, source="authored"))
        self.signals.register("clock/elapsed_steps", Signal(lambda: self.elapsed_steps.clone(), "runtime", info=Information("number", (), "int", "1")))
        self.signals.register("clock/time", Signal(lambda: self.elapsed_steps * self.step_dt, "runtime", "s", info=Information("number", (), "float", "s")))

    def _prepare_signals(self, paths):
        requested = tuple(dict.fromkeys(self._image_sources[path] for path in paths if path in self._image_sources))
        pending = [path for path in requested if self._camera_samples.get(path) != self._sensor_epoch]
        if not pending:
            return
        cameras = [self.asset(path) for path in pending]
        try:
            for camera in cameras:
                camera.set_rendering(True)
            frames = 3 if any(path not in self._camera_samples for path in pending) else 1
            for _ in range(frames):
                self.sim.render()
            for _ in range(20):
                if all(camera.has_rendered_data() for camera in cameras):
                    break
                self.sim.render()
            else:
                raise RuntimeError(f"Camera annotators did not become ready: {pending}")
            for camera in cameras:
                camera.update(0.0, force_recompute=True)
            self._camera_samples.update({path: self._sensor_epoch for path in pending})
        finally:
            for camera in cameras:
                camera.set_rendering(False)

    def _camera_pose(self, camera):
        camera._update_poses(camera._ALL_INDICES)
        return camera._data.pos_w.clone(), camera._data.quat_w_opengl.clone()

    def _ids(self, env_ids):
        import torch
        return torch.tensor(self.environment_ids(env_ids), device=self.device, dtype=torch.long)

    def advance(self):
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.sim.get_physics_dt())
        self._sync_cameras()
        if self.sim.has_gui():
            self.sim.render()
        self.elapsed_steps += 1
        self._sensor_epoch += 1

    def _sync_cameras(self):
        """Use live physics poses, not a potentially stale USD/Fabric parent.

        Camera.set_world_poses updates the render camera, not just reported data.
        Sync precedes rendering so image and reported pose use the same mount.
        """
        from isaaclab.utils.math import combine_frame_transforms
        for path, mount in self.mounts.items():
            pose = self.body_pose(mount)
            offset = self.index[path][0].pose
            p, q = combine_frame_transforms(pose[:, :3], pose[:, 3:],
                pose.new_tensor(offset.position).expand(self.num_envs, 3),
                pose.new_tensor(offset.orientation).expand(self.num_envs, 4))
            self.asset(path).set_world_poses(p, q, convention="opengl")

    def _local_pose(self, pose):
        result = pose.clone()
        result[:, :3] -= self.scene.env_origins
        return result

    def _body_frame_pose(self, mount, offset):
        import torch
        from isaaclab.utils.math import combine_frame_transforms

        body = self.body_pose(mount)
        position, orientation = combine_frame_transforms(body[:, :3], body[:, 3:],
            body.new_tensor(offset.position).expand(self.num_envs, 3),
            body.new_tensor(offset.orientation).expand(self.num_envs, 4))
        return self._local_pose(torch.cat((position, orientation), dim=-1))

    def _static_pose(self, asset):
        import torch
        position, orientation = asset.get_world_poses()
        pose = torch.cat((position, orientation), dim=-1)
        if asset.prim_paths[0].startswith(self.scene.env_prim_paths[0] + "/"):
            # Lab builds static views before cloning, so they hold env_0 only; the
            # clones share its local pose because static meshes never move.
            pose = pose[:1].clone()
            pose[:, :3] -= self.scene.env_origins[:1]
            return pose.expand(self.num_envs, 7).clone()
        return self._local_pose(pose.expand(self.num_envs, 7))

    def body_pose(self, mount):
        from isaaclab.assets import Articulation, RigidObject
        asset = self.asset(mount.entity)
        if isinstance(asset, Articulation):
            body_name = mount.body.split("/")[-1]
            if body_name not in asset.body_names:
                raise ValueError(f"Articulation endpoint needs an explicit body: {mount}")
            return asset.data.body_state_w[:, asset.body_names.index(body_name), :7].clone()
        if isinstance(asset, RigidObject):
            return asset.data.root_pose_w.clone()
        raise TypeError(f"Endpoint must be a rigid object or articulation: {mount.entity}")

    def _randomize_root_poses(self, ids):
        import torch

        rules = [(path, node.reset) for path, (node, _) in self.index.items() if node.reset is not None]
        if not rules or len(ids) == 0:
            return
        from isaaclab.envs.mdp import reset_root_state_uniform
        from isaaclab.managers import SceneEntityCfg

        for path, rule in rules:
            asset = self.asset(path)
            if rule.states:
                samples = [self.rng.choice(rule.states) for _ in range(len(ids))]
                poses = torch.tensor([(*pose.position, *pose.orientation) for pose in samples],
                                     device=self.device, dtype=asset.data.default_root_state.dtype)
                poses[:, :3] += self.scene.env_origins[ids]
                asset.write_root_pose_to_sim(poses, env_ids=ids)
            else:
                device = torch.device(self.device)
                devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
                seed = self.rng.randrange(2**63)
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(seed)
                    if devices:
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(seed)
                    reset_root_state_uniform(self, ids, pose_range=rule.pose_range, velocity_range={},
                                             asset_cfg=SceneEntityCfg(self.keys[path]))

    def reset(self, env_ids=None):
        """Restore physical and semantic state for selected environments only.

        Call after SimulationContext.reset(). Fixed children are placed at their
        attachment frames before the next physics step, avoiding reset impulses.
        """
        import torch
        from isaaclab.assets import Articulation
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

        ids = self._ids(env_ids)
        if len(ids) == 0:
            return
        for asset in (*self.scene.articulations.values(), *self.scene.rigid_objects.values()):
            state = asset.data.default_root_state[ids].clone()
            state[:, :3] += self.scene.env_origins[ids]
            asset.write_root_state_to_sim(state, env_ids=ids)
            if isinstance(asset, Articulation):
                q, qd = asset.data.default_joint_pos[ids].clone(), asset.data.default_joint_vel[ids].clone()
                asset.write_joint_state_to_sim(q, qd, env_ids=ids)
                asset.set_joint_position_target(q, env_ids=ids)
                asset.set_joint_velocity_target(torch.zeros_like(q), env_ids=ids)
                asset.set_joint_effort_target(torch.zeros_like(q), env_ids=ids)
        # Apply scene reset rules once, after every asset has its default state.
        # Sampling inside the loop both consumes extra randomness and can be
        # overwritten when a later asset is restored to its defaults.
        self._randomize_root_poses(ids)
        # Update articulation link kinematics without advancing simulation time.
        from isaaclab.sim import SimulationContext
        SimulationContext.instance().forward()
        self.scene.update(0.0)

        def tensor_pose(pose):
            return (torch.tensor(pose.position, device=self.device).expand(len(ids), 3),
                    torch.tensor(pose.orientation, device=self.device).expand(len(ids), 4))

        for joint in self.connections:
            if joint.parent:
                parent = self.body_pose(joint.parent)[ids]
            else:
                parent = torch.zeros((len(ids), 7), device=self.device)
                parent[:, :3] = self.scene.env_origins[ids]
                parent[:, 3] = 1
            p, q = combine_frame_transforms(parent[:, :3], parent[:, 3:], *tensor_pose(joint.parent_pose))
            child_p, child_q = tensor_pose(joint.child_pose)
            inv_p, inv_q = subtract_frame_transforms(child_p, child_q)
            p, q = combine_frame_transforms(p, q, inv_p, inv_q)
            child = self.asset(joint.child.entity)
            if isinstance(child, Articulation) and joint.child.body.split("/")[-1] != child.body_names[0]:
                raise ValueError("Articulation attachment must target its root body")
            state = torch.zeros((len(ids), 13), device=self.device)
            state[:, :3], state[:, 3:7] = p, q
            child.write_root_state_to_sim(state, env_ids=ids)
            SimulationContext.instance().forward()
            self.scene.update(0.0)
        self.reset_semantic_state(ids)
        self.scene.reset(ids)
        self.scene.write_data_to_sim()
        self.elapsed_steps[ids] = 0
        self._sync_cameras()
        self._sensor_epoch += 1
