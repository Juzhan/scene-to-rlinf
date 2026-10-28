"""Compile the shared entity tree into Isaac Lab's native scene configuration.

This module describes and installs a scene. Isaac Lab owns stepping, reset
ordering and simulation resources through its normal managers and events.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import posixpath

from shared.assembly.spec import Mount, Pose, Sensor
from .assets import FACTORIES, FACTORY_SCOPES


@dataclass(frozen=True)
class SceneMetadata:
    keys: dict[str, str]
    prim_paths: dict[str, str]
    scopes: dict[str, str]


def scene_metadata(description):
    """Resolve names and scopes without importing or launching Isaac.

    Top-level names are native scene keys. Nested names use ``__``. Logical
    parentage composes initial transforms, while a mount follows a live body.
    A mounted sensor's path denotes the observed body. Mounted cameras spawn
    independent prims and follow live physics through their native sensor update.
    """
    index = description.index()
    keys, paths, scopes = {}, {}, {}
    root = description.root.name
    for path, (node, _) in index.items():
        if node.factory:
            key = path.removeprefix(root + "/").replace("/", "__")
            if key in keys.values():
                raise ValueError(f"Native scene key collision at {path}: {key}")
            keys[path] = key

    def scope(path):
        if path in scopes:
            return scopes[path]
        node = index[path][0]
        parent = posixpath.dirname(path)
        if node.mount:
            mount = description.resolve_mount(path, node.mount)
            value = scope(mount.entity)
            if node.scope is not None and node.scope != value:
                raise ValueError(f"Entity {path}: mount scope must match {mount.entity} ({value})")
        elif node.scope is not None:
            value = node.scope
        else:
            value = FACTORY_SCOPES.get(node.factory, "environment")
            ancestor = parent
            while ancestor in index:
                if index[ancestor][0].scope is not None:
                    value = scope(ancestor)
                    break
                ancestor = posixpath.dirname(ancestor)
        scopes[path] = value
        return value

    for path, key in keys.items():
        node = index[path][0]
        prefix = "{ENV_REGEX_NS}" if scope(path) == "environment" else "/World"
        paths[path] = f"{prefix}/{node.prim_name or key}"
    for path in keys:
        node = index[path][0]
        if node.mount:
            mount = description.resolve_mount(path, node.mount)
            paths[path] = f"{paths[mount.entity]}/{mount.body}".rstrip("/")
        for reference in node.meshes:
            target = description.resolve_mount(path, Mount(reference)).entity
            if target not in keys:
                raise ValueError(f"Entity {path}: mesh must reference a concrete asset: {reference}")
    # Two independent entities must never accidentally author the same prim.
    occupied = {}
    for path, prim in paths.items():
        if not isinstance(index[path][0], Sensor):
            if prim in occupied:
                raise ValueError(f"Native prim collision: {path} and {occupied[prim]} at {prim}")
            occupied[prim] = path
    return SceneMetadata(keys, paths, scopes)


def build_scene_config(description, *, num_envs=1, env_spacing=4.0,
                       replicate_physics=True, factories=None, **scene_options):
    """Return an ``InteractiveSceneCfg``; preserve factory poses when omitted.

    Project factories return ordinary Isaac Lab asset, terrain or sensor
    configurations. They are evaluated only after the application has started.
    Every result is copied before modifying it so reusable presets stay intact.
    """
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, SensorBaseCfg

    metadata = scene_metadata(description)
    registry = {**FACTORIES, **(factories or {})}
    if description.ordered_connections():
        # The SDK rejects prestartup USD edits when physics is replicated.
        # Connections are authored per environment before physics starts, so
        # both USD schemas and transforms must remain independently editable.
        replicate_physics = False
        scene_options["clone_in_fabric"] = False
    config = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing,
                                 replicate_physics=replicate_physics, **scene_options)
    reserved = set(vars(config))
    index = description.index()
    spawned_paths = {}
    # Native cameras spawn underneath their target; create assets before sensors.
    for path, (node, pose) in sorted(index.items(), key=lambda item: isinstance(item[1][0], Sensor)):
        if not node.factory:
            if node.initial_state:
                raise NotImplementedError(f"Entity {path}: initialize custom state through a project task event")
            continue
        if metadata.keys[path] in reserved:
            raise ValueError(f"Entity name conflicts with native scene option: {metadata.keys[path]}")
        if node.factory not in registry:
            raise ValueError(f"Entity {path}: unknown asset factory {node.factory}")
        native = deepcopy(registry[node.factory](**deepcopy(node.parameters)))
        from .initial_state import apply_initial_state
        apply_initial_state(native, node.initial_state, path)
        if not hasattr(native, "prim_path"):
            raise TypeError(f"Entity {path}: factory must return a native asset, terrain or sensor config")
        if isinstance(node, Sensor) != isinstance(native, SensorBaseCfg):
            raise TypeError(f"Entity {path}: kind sensor must match a native SensorBaseCfg")
        native.prim_path = metadata.prim_paths[path]
        if metadata.scopes[path] == "global" and hasattr(native, "collision_group"):
            # Shared physical assets must remain visible to every clone's
            # collision group (including global floors and factory equipment).
            native.collision_group = -1
        if node.reset:
            from isaaclab.assets import ArticulationCfg, RigidObjectCfg

            if metadata.scopes[path] != "environment":
                raise ValueError(f"Entity {path}: per-environment reset requires environment scope")
            rigid_props = getattr(getattr(native, "spawn", None), "rigid_props", None)
            articulation_props = getattr(getattr(native, "spawn", None), "articulation_props", None)
            free_articulation_region = (isinstance(native, ArticulationCfg)
                                       and getattr(articulation_props, "fix_root_link", None) is not True)
            if (not isinstance(native, RigidObjectCfg) and not free_articulation_region) or (
                    rigid_props is not None and rigid_props.kinematic_enabled):
                raise ValueError(f"Entity {path}: pose reset requires a dynamic rigid object; "
                                 "reset additionally supports a free-root articulation")
        if node.meshes:
            if not hasattr(native, "mesh_prim_paths"):
                raise TypeError(f"Entity {path}: native sensor does not accept mesh references")
            native.mesh_prim_paths = [metadata.prim_paths[description.resolve_mount(path, Mount(ref)).entity]
                                      for ref in node.meshes]
        if isinstance(node, Sensor):
            sensor_pose = node.pose if node.mount else pose
            if isinstance(native, CameraCfg) and node.mount:
                from .camera import mounted_camera_class
                native.class_type = mounted_camera_class(native.class_type, native.prim_path)
                prefix = "{ENV_REGEX_NS}" if metadata.scopes[path] == "environment" else "/World"
                native.prim_path = prefix + "/" + (node.prim_name or metadata.keys[path])
                native.update_latest_camera_pose = True
            offset = getattr(native, "offset", getattr(native, "source_frame_offset", None))
            if offset is not None and (node.pose_authored or sensor_pose != Pose()):
                if not node.pose_authored:
                    sensor_pose = sensor_pose.compose(Pose(tuple(offset.pos), tuple(offset.rot)))
                offset.pos, offset.rot = sensor_pose.position, sensor_pose.orientation
            elif offset is None and (node.pose_authored and node.pose != Pose()):
                raise ValueError(f"Entity {path}: this sensor has no configurable pose offset")
            # Reusable frame-transformer targets may be relative to the mounted
            # asset. Absolute/native namespace paths are left untouched.
            for frame in getattr(native, "target_frames", ()):
                if frame.prim_path.startswith("{SOURCE_ASSET}/"):
                    if not node.mount:
                        raise ValueError(f"Entity {path}: relative frame targets require a mount")
                    mount = description.resolve_mount(path, node.mount)
                    frame.prim_path = frame.prim_path.replace("{SOURCE_ASSET}", metadata.prim_paths[mount.entity], 1)
        elif hasattr(native, "init_state"):
            if not node.pose_authored:
                initial = native.init_state
                pose = pose.compose(Pose(tuple(initial.pos), tuple(initial.rot)))
            native.init_state.pos, native.init_state.rot = pose.position, pose.orientation
        elif pose != Pose():
            raise ValueError(f"Entity {path}: this native config has no initial pose; configure its factory parameters")
        if not isinstance(native, SensorBaseCfg) or isinstance(native, CameraCfg):
            if native.prim_path in spawned_paths:
                raise ValueError(f"Native prim collision: {path} and {spawned_paths[native.prim_path]} at {native.prim_path}")
            spawned_paths[native.prim_path] = path
        setattr(config, metadata.keys[path], native)
    _validate_connections(description, config, metadata)
    return config


def _validate_connections(description, config, metadata):
    if not description.ordered_connections():
        return
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg

    for joint in description.ordered_connections():
        if metadata.scopes[joint.child.entity] != "environment" or (
                joint.parent and metadata.scopes[joint.parent.entity] != "environment"):
            raise ValueError("Fixed connections require per-environment assets")
        if joint.parent and not isinstance(getattr(config, metadata.keys[joint.parent.entity]),
                                            (ArticulationCfg, RigidObjectCfg)):
            raise TypeError(f"{joint.parent.entity}: fixed attachment requires a physical parent")
        native = getattr(config, metadata.keys[joint.child.entity])
        if not isinstance(native, (ArticulationCfg, RigidObjectCfg)):
            raise TypeError(f"{joint.child.entity}: fixed attachment requires a physical child")
        if isinstance(native, RigidObjectCfg) and native.spawn and native.spawn.rigid_props and native.spawn.rigid_props.kinematic_enabled:
            raise ValueError(f"{joint.child.entity}: kinematic child cannot be driven by a fixed joint")
        if isinstance(native, ArticulationCfg):
            props = native.spawn.articulation_props if native.spawn else None
            if props is None or props.fix_root_link is not False:
                raise ValueError(f"{joint.child.entity}: attached articulation requires fix_root_link=False")


def event_configuration(description):
    """Native scene events to append after a task's normal reset events.

    No project-specific reset loop is needed. The SDK calls these through its
    EventManager; event parameters contain only serializable scene data.
    """
    from isaaclab.managers import EventTermCfg

    metadata = scene_metadata(description)
    joints = [asdict(joint) for joint in description.ordered_connections()]
    events = {}
    # Abstract entities have no native asset; their reset is a command term.
    assets = {path: node for path, (node, _) in description.index().items() if path in metadata.keys}
    rules = {metadata.keys[path]: asdict(node.reset) for path, node in assets.items()
             if node.reset is not None and node.reset.region is None}
    if rules:
        events["assembly_root_poses"] = EventTermCfg(func=reset_root_poses, mode="reset", params={"rules": rules})
    region_rules = {metadata.keys[path]: asdict(node.reset) for path, node in assets.items()
                    if node.reset is not None and node.reset.region is not None}
    if region_rules:
        from shared.assembly.abstract import area
        from .abstract_terms import reset_root_in_regions
        areas = {rule["region"]: area(description.index()[rule["region"]][0]) for rule in region_rules.values()}
        events["assembly_region_poses"] = EventTermCfg(func=reset_root_in_regions, mode="reset", params={
            "rules": region_rules, "areas": areas})
    if joints:
        params = {"joints": joints, "keys": metadata.keys, "prim_paths": metadata.prim_paths}
        events["assembly_connections"] = EventTermCfg(func=install_connections, mode="prestartup", params=params)
        events["assembly_attachment_poses"] = EventTermCfg(func=reset_connections, mode="reset",
                                                          params={"joints": joints, "keys": metadata.keys})
    return events


def authored_reset_assets(scene, description):
    """Per-environment physical assets that take the environment origin on reset.

    Static meshes (plain AssetBaseCfg) are placed once by Lab's cloning: authored
    pose plus the cloner grid origin, which is env_origins unless a terrain
    defines its own. They are never moved afterwards: Lab builds their
    XformPrimView before cloning (env_0 only) and PhysX keeps static colliders
    where it parsed them.
    """
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
    from isaaclab.terrains import TerrainImporterCfg
    metadata = scene_metadata(description)
    local = [key for path, key in metadata.keys.items()
             if metadata.scopes[path] == "environment" and isinstance(getattr(scene, key), AssetBaseCfg)]
    physical = (ArticulationCfg, RigidObjectCfg, DeformableObjectCfg)
    static = [key for key in local if not isinstance(getattr(scene, key), physical)]
    if static and any(isinstance(value, TerrainImporterCfg) for value in vars(scene).values()):
        raise NotImplementedError(
            f"Per-environment static objects {static} stay at the cloned grid origins, but the terrain defines "
            "its own environment origins; make them scope: global (one copy in world coordinates) or physical "
            "(e.g. box/usd_object with a mass)")
    return [key for key in local if key not in static]


def reset_authored_assets(env, env_ids, *, asset_names):
    """Default local layout before task randomizers, using Lab's asset API.

    Only per-environment physical assets are selected (authored_reset_assets).
    Global assets are never translated by an environment origin or partially reset.
    """
    import torch
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for name in asset_names:
        asset = env.scene[name]
        state = asset.data.default_root_state[env_ids].clone()
        state[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
        if name in env.scene.articulations:
            position = asset.data.default_joint_pos[env_ids].clone()
            velocity = asset.data.default_joint_vel[env_ids].clone()
            asset.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
            asset.set_joint_position_target(position, env_ids=env_ids)
            asset.set_joint_velocity_target(velocity, env_ids=env_ids)


def reset_root_poses(env, env_ids, *, rules):
    """Apply authored root-pose rules using the native random generator."""
    import torch
    from isaaclab.envs.mdp import reset_root_state_uniform
    from isaaclab.managers import SceneEntityCfg

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for name, rule in rules.items():
        if rule["states"]:
            asset = env.scene[name]
            options = asset.data.default_root_state.new_tensor([
                (*pose["position"], *pose["orientation"]) for pose in rule["states"]])
            sampled = options[torch.randint(len(options), (len(env_ids),), device=env.device)]
            state = asset.data.default_root_state[env_ids].clone()
            state[:, :7] = sampled
            state[:, :3] += env.scene.env_origins[env_ids]
            asset.write_root_state_to_sim(state, env_ids=env_ids)
        else:
            reset_root_state_uniform(env, env_ids, pose_range=rule["pose_range"], velocity_range={},
                                     asset_cfg=SceneEntityCfg(name))


def install_connections(env, env_ids, *, joints, keys, prim_paths):
    """Author physical attachments before the SDK initializes physics handles."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = sim_utils.get_current_stage()

    def endpoint(mount, namespace):
        path = prim_paths[mount["entity"]].replace("{ENV_REGEX_NS}", namespace)
        if mount["body"]:
            path += "/" + mount["body"]
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Fixed joint endpoint must be a rigid body: {path}")
        return path

    def matrix(pose):
        transform = Gf.Matrix4d(1.)
        transform.SetRotate(Gf.Quatd(pose["orientation"][0], Gf.Vec3d(*pose["orientation"][1:])))
        transform.SetTranslateOnly(Gf.Vec3d(*pose["position"]))
        return transform

    for index, namespace in enumerate(env.scene.env_prim_paths):
        UsdGeom.Scope.Define(stage, namespace + "/Connections")
        for joint in joints:
            child_path = endpoint(joint["child"], namespace)
            parent_path = endpoint(joint["parent"], namespace) if joint["parent"] else None
            root_path = prim_paths[joint["child"]["entity"]].replace("{ENV_REGEX_NS}", namespace)
            root = stage.GetPrimAtPath(root_path)
            child_cfg = getattr(env.cfg.scene, keys[joint["child"]["entity"]])
            if isinstance(child_cfg, ArticulationCfg):
                for prim in Usd.PrimRange(root):
                    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                anchors = []
                for prim in Usd.PrimRange(root):
                    if prim.IsA(UsdPhysics.FixedJoint):
                        anchor = UsdPhysics.FixedJoint(prim)
                        if bool(anchor.GetBody0Rel().GetTargets()) != bool(anchor.GetBody1Rel().GetTargets()):
                            anchors.append(prim.GetPath())
                for anchor in anchors:
                    stage.GetPrimAtPath(anchor).SetActive(False)
                UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(child_path))
                sim_utils.modify_articulation_root_properties(child_path, child_cfg.spawn.articulation_props, stage=stage)
            cache = UsdGeom.XformCache()
            parent_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path or namespace))
            desired_body = matrix(joint["child_pose"]).GetInverse() * matrix(joint["parent_pose"]) * parent_world
            root_world = cache.GetLocalToWorldTransform(root)
            body_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(child_path))
            root_parent_world = cache.GetLocalToWorldTransform(root.GetParent())
            UsdGeom.Xformable(root).MakeMatrixXform().Set(
                root_world * body_world.GetInverse() * desired_body * root_parent_world.GetInverse())
            native = UsdPhysics.FixedJoint.Define(stage, namespace + "/Connections/" + joint["name"])
            native.CreateBody1Rel().SetTargets([child_path])
            parent_pose, child_pose = joint["parent_pose"], joint["child_pose"]
            if parent_path:
                native.CreateBody0Rel().SetTargets([parent_path])
                position = parent_pose["position"]
            else:
                position = tuple(a + b for a, b in zip(parent_pose["position"], env.scene.env_origins[index].tolist()))
            native.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
            native.CreateLocalRot0Attr().Set(Gf.Quatf(parent_pose["orientation"][0], Gf.Vec3f(*parent_pose["orientation"][1:])))
            native.CreateLocalPos1Attr().Set(Gf.Vec3f(*child_pose["position"]))
            native.CreateLocalRot1Attr().Set(Gf.Quatf(child_pose["orientation"][0], Gf.Vec3f(*child_pose["orientation"][1:])))
            native.CreateExcludeFromArticulationAttr().Set(True)
            native.CreateCollisionEnabledAttr().Set(False)


def reset_connections(env, env_ids, *, joints, keys):
    """Place attached roots at their joint frames after the SDK resets assets."""
    import torch
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env.sim.forward()
    env.scene.update(0.0)

    def tensor_pose(pose):
        return (torch.tensor(pose["position"], device=env.device).expand(len(env_ids), 3),
                torch.tensor(pose["orientation"], device=env.device).expand(len(env_ids), 4))

    def body_pose(mount):
        asset = env.scene[keys[mount["entity"]]]
        if isinstance(asset, Articulation):
            name = mount["body"].split("/")[-1]
            if name not in asset.body_names:
                raise ValueError(f"Articulation endpoint needs an explicit body: {mount}")
            return asset.data.body_state_w[env_ids, asset.body_names.index(name), :7]
        return asset.data.root_pose_w[env_ids]

    for joint in joints:
        if joint["parent"]:
            parent = body_pose(joint["parent"])
        else:
            parent = torch.zeros((len(env_ids), 7), device=env.device)
            parent[:, :3], parent[:, 3] = env.scene.env_origins[env_ids], 1
        position, orientation = combine_frame_transforms(parent[:, :3], parent[:, 3:], *tensor_pose(joint["parent_pose"]))
        inverse = subtract_frame_transforms(*tensor_pose(joint["child_pose"]))
        position, orientation = combine_frame_transforms(position, orientation, *inverse)
        child = env.scene[keys[joint["child"]["entity"]]]
        if isinstance(child, Articulation) and joint["child"]["body"].split("/")[-1] != child.body_names[0]:
            raise ValueError("Articulation attachment must target its root body")
        state = torch.zeros((len(env_ids), 13), device=env.device)
        state[:, :3], state[:, 3:7] = position, orientation
        child.write_root_state_to_sim(state, env_ids=env_ids)
        env.sim.forward()
        env.scene.update(0.0)
