"""Compile the shared entity tree into Isaac Lab 3 native scene configuration (Newton or OvPhysX).

Keys, prim paths and scopes are resolved exactly as in the ``isaac`` backend. Isaac Lab 3 managers
own stepping and resets; this module only translates poses to its xyzw quaternions and binds
reset rules to its native events and index-based write APIs.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from math import isfinite

from shared.assembly.spec import Abstract, Mount, Pose, Sensor
from shared.backends.isaac.scene import scene_metadata
from .frames import wxyz, xyzw

SCENE_KEYS = {'schema_version', 'name', 'entities', 'connections', 'simulation', 'world', 'authoring',
              'control', 'targets', 'application', 'observations', 'visualization'}
# `newton` configures the Newton engine, `physx` the OvPhysX one; `mujoco` belongs to mjlab.
SIMULATION_KEYS = {'dt', 'gravity', 'render_interval', 'physics_material', 'physx', 'mujoco', 'newton'}
INITIAL_STATE = {'joint_positions', 'joint_velocities', 'linear_velocity', 'angular_velocity'}


def validate(description, settings):
    """Reject features this backend does not implement yet, before any SDK import."""
    extra = settings.keys() - SCENE_KEYS
    if extra:
        raise NotImplementedError(f'isaaclab3 backend does not implement: {sorted(extra)}')
    connection_scopes(description)
    simulation = settings.get('simulation', {})
    if simulation.keys() - SIMULATION_KEYS:
        raise NotImplementedError(f'isaaclab3 simulation supports only {sorted(SIMULATION_KEYS)}')
    newton = simulation.get('newton', {})
    if not isinstance(newton, dict) or not isinstance(newton.get('solver', {}), dict) or newton.keys() & {'dt', 'gravity'}:
        raise ValueError('simulation.newton holds NewtonCfg options (solver: MJWarpSolverCfg options); '
                         'dt and gravity stay in simulation')
    dt = simulation.get('dt', .005)
    gravity = simulation.get('gravity', [0., 0., -9.81])
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not isfinite(dt) or dt <= 0:
        raise ValueError('simulation.dt must be positive and finite')
    if len(gravity) != 3 or not all(isfinite(x) for x in gravity):
        raise ValueError('simulation.gravity requires three finite values')
    world = settings.get('world', {})
    if world.keys() - {'env_spacing', 'contact_materials'}:
        raise NotImplementedError('isaaclab3 scene world supports only env_spacing and contact_materials')
    # Isaac Lab 3 keeps the native material startup event on Newton and OvPhysX.
    from shared.backends.isaac.contact_materials import validate_materials
    validate_materials(world.get('contact_materials', {}), scene_metadata(description).keys.values())
    from .assets import RAY_SENSORS, SENSORS
    for path, (node, _) in description.index().items():
        if isinstance(node, Abstract):
            continue
        if isinstance(node, Sensor):
            if node.factory not in SENSORS:
                raise NotImplementedError(f'{path}: isaaclab3 implements only the sensors {list(SENSORS)}')
            if not node.mount or not node.mount.body:
                raise ValueError(f'{path}: {node.factory} needs a mount entity and body; it follows that body')
            if node.reset is not None:
                raise ValueError(f'{path}: sensors cannot be reset')
            if node.factory in RAY_SENSORS:
                ray_targets(description, path, node)
            elif node.meshes:
                raise ValueError(f'{path}: only {list(RAY_SENSORS)} cast rays at meshes')
            continue
        if node.frames and node.kind != 'robot':
            raise NotImplementedError(f'{path}: isaaclab3 body frames are read on robot entities only')
        if node.mount or node.meshes:
            raise NotImplementedError(f'{path}: isaaclab3 mounts only sensors {list(SENSORS)}')
        if node.initial_state.keys() - INITIAL_STATE:
            raise NotImplementedError(f'{path}: isaaclab3 initial_state supports only {sorted(INITIAL_STATE)}')


def connection_scopes(description):
    """Fixed connections join two bodies of the same environment, as on the isaac backend."""
    metadata = scene_metadata(description)
    for joint in description.ordered_connections():
        for mount in filter(None, (joint.parent, joint.child)):
            if metadata.scopes.get(mount.entity) != 'environment':
                raise ValueError(f'Connection {joint.name}: {mount.entity} must be a per-environment physical entity')


def ray_targets(description, path, node):
    """Check a height scanner's ``meshes`` against what Newton's ray caster can restrict to.

    Newton casts against shapes by world, not by mesh: ``global_world_only`` hits exactly the
    global (unreplicated) shapes, otherwise also every shape of the sensor's own environment,
    the robot carrying it included. Only the first is faithful to a ``meshes`` list, so the list
    must name every global entity with shapes."""
    from .assets import NO_SHAPES
    metadata = scene_metadata(description)
    index = description.index()
    targets = {description.resolve_mount(path, Mount(reference)).entity for reference in node.meshes}
    shapes = {other for other, (entity, _) in index.items() if other in metadata.keys
              and metadata.scopes[other] == 'global' and not isinstance(entity, Sensor)
              and entity.factory not in NO_SHAPES}
    if not shapes:
        raise NotImplementedError(f'{path}: isaaclab3 height_scanner needs a global entity with shapes to hit '
                                  '(a ground or terrain with scope: global); there is none')
    if not targets or targets != shapes:
        raise NotImplementedError(
            f'{path}: isaaclab3 height_scanner meshes must list exactly the global entities with shapes '
            f'{sorted(shapes)}; Newton ray casts select shapes by world, not by mesh')
    return [metadata.prim_paths[target] for target in sorted(targets)]


def _free_root(native):
    """A rigid object, or an articulation whose root is not fixed by any of its property variants."""
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    spawn = getattr(native, 'spawn', None)
    rigid = getattr(spawn, 'rigid_props', None)
    if rigid is not None and getattr(rigid, 'kinematic_enabled', None):
        return False
    if isinstance(native, RigidObjectCfg):
        return True
    props = getattr(spawn, 'articulation_props', None)
    # Isaac Lab 3 presets list one property set per physics backend.
    variants = props if isinstance(props, (list, tuple)) else [props]
    return isinstance(native, ArticulationCfg) and all(getattr(p, 'fix_root_link', None) is not True for p in variants)


def build_scene_config(description, *, num_envs=1, env_spacing=4.0, factories=None):
    """Return an ``InteractiveSceneCfg``; factory poses are kept when the scene omits them.

    Articulations with a recorded joint order (ASSET.json) get it as Isaac Lab 3's native
    ``joint_ordering``, so native joint columns, actions and Jacobians already follow it.
    """
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveSceneCfg
    from shared.asset_formats import recorded_joint_order
    from shared.backends.isaac.initial_state import apply_initial_state
    from .assets import FACTORIES
    from .spawners import NATIVE, SPAWN_FROM_USD

    metadata = scene_metadata(description)
    registry = {**FACTORIES, **(factories or {})}
    config = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    reserved = set(vars(config))
    # Sensors last, as in the isaac backend: they track prims their mount entities spawn.
    for path, (node, pose) in sorted(description.index().items(), key=lambda item: isinstance(item[1][0], Sensor)):
        if not node.factory:
            if node.initial_state:
                raise NotImplementedError(f'Entity {path}: initialize custom state through a project task event')
            continue
        key = metadata.keys[path]
        if key in reserved:
            raise ValueError(f'Entity name conflicts with native scene option: {key}')
        if node.factory not in registry:
            raise ValueError(f'Entity {path}: unknown asset factory {node.factory}')
        native = deepcopy(registry[node.factory](**deepcopy(node.parameters)))
        if not hasattr(native, 'prim_path'):
            raise TypeError(f'Entity {path}: factory must return a native Isaac Lab 3 asset config')
        native.prim_path = metadata.prim_paths[path]
        if metadata.scopes[path] == 'global' and hasattr(native, 'collision_group'):
            # Shared physical assets collide with every environment's clones.
            native.collision_group = -1
        if isinstance(node, Sensor):
            # prim_path is the mount body's prim, as in the isaac backend. A camera is spawned as a
            # child of that body, as in Isaac Lab 3's own wrist cameras.
            if getattr(native, 'spawn', None) is not None and hasattr(native, 'renderer_cfg'):
                native.prim_path += '/' + (node.prim_name or key)
            if hasattr(native, 'mesh_prim_paths'):
                # Newton reads hits from the global world only (see ray_targets).
                native.mesh_prim_paths, native.global_world_only = ray_targets(description, path, node), True
            # The sensor pose relative to its body (a ray caster's ray starts and directions, a frame
            # transformer's source frame); an unauthored pose keeps the factory's offset.
            offset = getattr(native, 'offset', getattr(native, 'source_frame_offset', None))
            if offset is not None:
                sensor_pose = node.pose if node.pose_authored else node.pose.compose(
                    Pose(tuple(offset.pos), wxyz(tuple(offset.rot))))
                offset.pos, offset.rot = sensor_pose.position, xyzw(sensor_pose.orientation)
            elif node.pose_authored and node.pose != Pose():
                raise ValueError(f'Entity {path}: this sensor has no configurable pose offset')
            # Reusable frame-transformer targets are relative to the mounted asset.
            for frame in getattr(native, 'target_frames', ()):
                if frame.prim_path.startswith('{SOURCE_ASSET}/'):
                    mount = description.resolve_mount(path, node.mount)
                    frame.prim_path = frame.prim_path.replace('{SOURCE_ASSET}', metadata.prim_paths[mount.entity], 1)
            setattr(config, key, native)
            continue
        spawn = getattr(native, 'spawn', None)
        func = getattr(spawn, 'func', None)
        if func is not None and (func if isinstance(func, str) else f'{func.__module__}:{func.__name__}') == NATIVE:
            spawn.func = SPAWN_FROM_USD
        apply_initial_state(native, node.initial_state, path)
        if node.reset:
            if metadata.scopes[path] != 'environment':
                raise ValueError(f'Entity {path}: per-environment reset requires environment scope')
            if not _free_root(native):
                raise ValueError(f'Entity {path}: pose reset requires a dynamic rigid object or a free-root articulation')
        if hasattr(native, 'init_state'):
            if not node.pose_authored:
                initial = native.init_state
                pose = pose.compose(Pose(tuple(initial.pos), wxyz(tuple(initial.rot))))
            native.init_state.pos, native.init_state.rot = pose.position, xyzw(pose.orientation)
        elif pose != Pose():
            raise ValueError(f'Entity {path}: this native config has no initial pose; configure its factory parameters')
        if isinstance(native, ArticulationCfg) and native.joint_ordering is None:
            order = recorded_joint_order(native.spawn.usd_path) if hasattr(native.spawn, 'usd_path') else None
            if order is not None:
                native.joint_ordering = list(order)
        setattr(config, key, native)
    _validate_connections(description, config, metadata)
    return config


def _validate_connections(description, config, metadata):
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    for joint in description.ordered_connections():
        for mount in filter(None, (joint.parent, joint.child)):
            if not isinstance(getattr(config, metadata.keys[mount.entity], None), (ArticulationCfg, RigidObjectCfg)):
                raise TypeError(f'Connection {joint.name}: {mount.entity} must be an articulation or rigid object')
        if not _free_root(getattr(config, metadata.keys[joint.child.entity])):
            raise ValueError(f'{joint.child.entity}: an attached entity needs a dynamic free root (fix_root_link=False)')


def event_configuration(description):
    """Native reset events for authored placement rules; Isaac Lab 3's EventManager calls them.

    Fixed connections end the list so their children follow the parents' final poses."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    metadata = scene_metadata(description)
    assets = {path: node for path, (node, _) in description.index().items() if path in metadata.keys}
    events = {}
    for path, node in assets.items():
        if node.reset is None or node.reset.region is not None:
            continue
        key, rule = metadata.keys[path], asdict(node.reset)
        if rule['states']:
            events['assembly_root_states_' + key] = EventTermCfg(
                func=reset_root_states, mode='reset', params={'name': key, 'states': rule['states']})
        else:
            # Native uniform pose reset; deltas compose with the asset's default root pose.
            events['assembly_root_poses_' + key] = EventTermCfg(
                func=mdp.reset_root_state_uniform, mode='reset',
                params={'pose_range': rule['pose_range'], 'velocity_range': {}, 'asset_cfg': SceneEntityCfg(key)})
    region_rules = {metadata.keys[path]: asdict(node.reset) for path, node in assets.items()
                    if node.reset is not None and node.reset.region is not None}
    if region_rules:
        from shared.assembly.abstract import area
        from .abstract_terms import reset_root_in_regions
        areas = {rule['region']: area(description.index()[rule['region']][0]) for rule in region_rules.values()}
        events['assembly_region_poses'] = EventTermCfg(func=reset_root_in_regions, mode='reset', params={
            'rules': region_rules, 'areas': areas})
    joints = [asdict(joint) for joint in description.ordered_connections()]
    if joints:
        events['assembly_connection_poses'] = EventTermCfg(func=reset_connections, mode='reset', params={
            'joints': joints, 'keys': metadata.keys})
    return events


def authored_reset_assets(scene, description):
    """Per-environment physical assets that take the environment origin on reset.

    Static colliders are placed once by cloning at their authored pose plus the cloner grid origin,
    which is env_origins unless a terrain defines its own (rejected, as on the isaac backend).
    """
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.terrains import TerrainImporterCfg
    metadata = scene_metadata(description)
    local = [key for path, key in metadata.keys.items()
             if metadata.scopes[path] == 'environment' and isinstance(getattr(scene, key), AssetBaseCfg)]
    static = [key for key in local if not isinstance(getattr(scene, key), (ArticulationCfg, RigidObjectCfg))]
    if static and any(isinstance(value, TerrainImporterCfg) for value in vars(scene).values()):
        raise NotImplementedError(
            f'Per-environment static objects {static} stay at the cloned grid origins, but the terrain defines '
            'its own environment origins; make them scope: global or physical (e.g. box/usd_object with a mass)')
    return [key for key in local if key not in static]


def _env_ids(env, env_ids):
    import torch
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def reset_authored_assets(env, env_ids, *, asset_names):
    """Default local layout before task randomizers, through Isaac Lab 3's index write API."""
    env_ids = _env_ids(env, env_ids)
    for name in asset_names:
        asset = env.scene[name]
        pose = asset.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=asset.data.default_root_vel.torch[env_ids].clone(),
                                               env_ids=env_ids)
        if name in env.scene.articulations:
            position = asset.data.default_joint_pos.torch[env_ids].clone()
            velocity = asset.data.default_joint_vel.torch[env_ids].clone()
            asset.write_joint_state_to_sim_index(position=position, velocity=velocity, env_ids=env_ids)
            asset.set_joint_position_target_index(target=position, env_ids=env_ids)
            asset.set_joint_velocity_target_index(target=velocity, env_ids=env_ids)


def reset_root_states(env, env_ids, *, name, states):
    """One of the authored root poses per environment, drawn with torch's seeded generator."""
    import torch
    env_ids = _env_ids(env, env_ids)
    asset = env.scene[name]
    options = env.scene.env_origins.new_tensor([(*state['position'], *xyzw(state['orientation'])) for state in states])
    pose = options[torch.randint(len(options), (len(env_ids),), device=env.device)]
    pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=asset.data.default_root_vel.torch[env_ids].clone(),
                                           env_ids=env_ids)


def _relative(parent_pose, child_pose):
    """The child root pose in the parent body frame (wxyz), as the mjlab weld uses it."""
    from shared.backends.mjlab.scene import inverse
    return parent_pose.compose(inverse(child_pose))


def _body_path(prim_paths, mount, namespace):
    path = prim_paths[mount['entity']].replace('{ENV_REGEX_NS}', namespace)
    return path + '/' + mount['body'] if mount['body'] else path


@contextmanager
def newton_connections(description, physics, dt):
    """Weld each attached child to its parent in every Newton world while the scene is replicated.

    Newton imports each asset source on its own and replicates it per world, so a USD joint between
    two assets is never imported. Like the mjlab backend, each connection becomes an explicit MuJoCo
    weld (time constant 2 dt) with contacts between the two rigid groups filtered; the child keeps its
    free root and the reset event places it. Other engines need no hook."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    joints = description.ordered_connections()
    if not joints or not isinstance(physics, NewtonCfg):
        yield
        return
    if not isinstance(physics.solver_cfg, MJWarpSolverCfg):
        raise NotImplementedError('isaaclab3 Newton connections are MuJoCo welds; use the MuJoCo-Warp solver')
    import mujoco
    import warp as wp
    from isaaclab_newton.cloner import newton_builder_world_hook
    from newton import JointType
    from newton.solvers import SolverMuJoCo
    prim_paths = scene_metadata(description).prim_paths
    welds = [(joint.name, joint.parent and _body_path(prim_paths, asdict(joint.parent), ''),
              _body_path(prim_paths, asdict(joint.child), ''), _relative(joint.parent_pose, joint.child_pose))
             for joint in joints]

    def tail(worlds, current):
        # Replication appended this world's elements last.
        index = len(worlds)
        while index and worlds[index - 1] == current:
            index -= 1
        return range(index, len(worlds))

    def hook(builder, world, position, quaternion):
        bodies = tail(builder.body_world, builder.current_world)
        joints = tail(builder.joint_world, builder.current_world)
        roots = {builder.joint_child[j] for j in joints if builder.joint_type[j] == JointType.FREE}
        fixed = [(builder.joint_parent[j], builder.joint_child[j]) for j in joints
                 if builder.joint_type[j] == JointType.FIXED and builder.joint_parent[j] != -1]

        def rigid(body):
            # Newton keeps fixed-jointed links (e.g. a dummy root above base_link) as separate bodies
            # that MJCF compilation would merge, so the filter covers the whole rigid group.
            group, grown = {body}, True
            while grown:
                grown = False
                for a, b in fixed:
                    if (a in group) != (b in group):
                        group |= {a, b}
                        grown = True
            return group

        def find(name, suffix):
            found = [body for body in bodies if builder.body_label[body].endswith(suffix)]
            if len(found) != 1:
                raise ValueError(f'Connection {name}: expected one Newton body {suffix!r} in world {world}, '
                                 f'found {len(found)}')
            return found[0]

        for name, parent_suffix, child_suffix, relative in welds:
            child = find(name, child_suffix)
            if child not in roots:
                raise ValueError(f'Connection {name}: {child_suffix} is not the free root body of its entity')
            parent = find(name, parent_suffix) if parent_suffix else -1
            if parent == -1:
                # An absent parent is the environment origin of this world.
                origin = Pose(tuple(float(x) for x in position), wxyz(tuple(float(x) for x in quaternion)))
                relative = origin.compose(relative)
            builder.add_custom_values(**{
                'mujoco:equality_constraint_type': int(SolverMuJoCo.EqType.WELD),
                'mujoco:equality_constraint_objtype': int(mujoco.mjtObj.mjOBJ_BODY),
                'mujoco:equality_constraint_body1': parent,
                'mujoco:equality_constraint_body2': child,
                'mujoco:equality_constraint_anchor': wp.vec3(),
                'mujoco:equality_constraint_torquescale': 1.,
                # relpose is body2 in body1's frame.
                'mujoco:equality_constraint_relpose': wp.transform(relative.position, xyzw(relative.orientation)),
                'mujoco:equality_constraint_joint1': -1,
                'mujoco:equality_constraint_joint2': -1,
                'mujoco:equality_constraint_polycoef': [0.] * 5,
                'mujoco:equality_constraint_label': name,
                'mujoco:equality_constraint_enabled': True,
                'mujoco:equality_constraint_world': builder.current_world,
                'mujoco:eq_solref': wp.vec2(2 * dt, 1.)})
            if parent != -1:
                for parent_body in rigid(parent):
                    for child_body in rigid(child):
                        for a in builder.body_shapes[parent_body]:
                            for b in builder.body_shapes[child_body]:
                                builder.add_shape_collision_filter_pair(a, b)

    with newton_builder_world_hook(hook):
        yield


def refresh_newton_frames(env):
    """Re-evaluate Newton frame-transform sensors from the reset kinematics.

    Lab 3's Newton frame transformers copy Newton sensors that are evaluated only at the end of a
    physics step, so after a reset they report the pre-reset frames (zeros on the first reset) until
    the next step. Isaac's PhysX transformer reads live body poses, as does Lab 3 on OvPhysX."""
    from isaaclab_newton.physics import NewtonCfg, NewtonManager
    if not isinstance(env.cfg.sim.physics, NewtonCfg) or not NewtonManager._newton_frame_transform_sensors:
        return
    env.sim.forward()
    for sensor in NewtonManager._newton_frame_transform_sensors:
        sensor.update(NewtonManager.backend.state_0)


@contextmanager
def physx_connections(env, description):
    """Author a PhysX fixed joint per environment after the scene exists, before the first
    ``sim.reset`` serializes the stage for OvPhysX.

    OvPhysX natively clones each asset on its own, which drops joints between assets. The joints
    are authored in env_0 and env_0 is cloned as a whole when every environment holds the same
    assets; otherwise each environment gets its joints and a full-stage import keeps them.
    Lab 3 rejects prestartup events while physics is replicated, so the joints are authored in
    this window of ``ManagerBasedEnv._init_sim`` instead; the wrapper lives on this environment's
    simulation context only and is removed afterwards. Other engines need no hook."""
    from isaaclab_ov.physics import OvPhysxCfg
    joints = [asdict(joint) for joint in description.ordered_connections()]
    sim = env.sim
    if not joints or not isinstance(sim.cfg.physics, OvPhysxCfg):
        yield
        return
    metadata = scene_metadata(description)
    reset = sim.reset

    def first_reset(*args, **kwargs):
        del sim.reset
        _author_physx_connections(env, joints, metadata.prim_paths)
        return reset(*args, **kwargs)

    sim.reset = first_reset
    try:
        yield
    finally:
        vars(sim).pop('reset', None)


def _author_physx_connections(env, joints, prim_paths):
    from isaaclab_ov.physics import OvPhysxManager
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = env.sim.stage
    source = f'{env.scene.env_ns}/env_0'
    for joint in joints:
        for mount in filter(None, (joint['parent'], joint['child'])):
            prim = stage.GetPrimAtPath(_body_path(prim_paths, mount, source))
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Connection {joint['name']}: endpoint must be a rigid body: {prim.GetPath()}")
    whole = _whole_environment_clone(env, joints)
    if not whole:
        OvPhysxManager.require_full_stage()
    origins = env.scene.env_origins.tolist()

    def rotation(pose):
        return Gf.Quatf(pose['orientation'][0], Gf.Vec3f(*pose['orientation'][1:]))

    for index in range(1 if whole else env.num_envs):
        namespace = f'{env.scene.env_ns}/env_{index}'
        UsdGeom.Scope.Define(stage, namespace + '/Connections')
        for joint in joints:
            _float_articulation(stage, prim_paths[joint['child']['entity']].replace('{ENV_REGEX_NS}', namespace),
                                _body_path(prim_paths, joint['child'], namespace))
            native = UsdPhysics.FixedJoint.Define(stage, f"{namespace}/Connections/{joint['name']}")
            native.CreateBody1Rel().SetTargets([_body_path(prim_paths, joint['child'], namespace)])
            parent_pose, child_pose = joint['parent_pose'], joint['child_pose']
            position = parent_pose['position']
            if joint['parent']:
                native.CreateBody0Rel().SetTargets([_body_path(prim_paths, joint['parent'], namespace)])
            else:
                position = [a + b for a, b in zip(position, origins[index])]
            native.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
            native.CreateLocalRot0Attr().Set(rotation(parent_pose))
            native.CreateLocalPos1Attr().Set(Gf.Vec3f(*child_pose['position']))
            native.CreateLocalRot1Attr().Set(rotation(child_pose))
            native.CreateExcludeFromArticulationAttr().Set(True)
            native.CreateCollisionEnabledAttr().Set(False)
    if whole:
        whole()


def _whole_environment_clone(env, joints):
    """Return a callable that replaces Lab 3's per-asset OvPhysX clones by one clone of env_0, or None.

    Per-asset native clones drop the joints between assets, and a full-stage import parses every
    environment (about 30 MB of host memory each for go2_piper_wholebody). When every environment
    holds the same assets as env_0, cloning env_0 as a whole carries its joints along, like Kit's
    replicated physics. World-anchored joints and environments with differing assets keep the
    full-stage import. Lab 3 exposes no call to withdraw queued recipes, so its list is replaced."""
    import numpy as np
    from isaaclab_ov.cloner import ovphysx_replicate
    from isaaclab_ov.physics import OvPhysxManager

    plan = env.sim.get_clone_plan()
    recipes = OvPhysxManager._clone_recipes
    if plan is None or not recipes or any(joint['parent'] is None for joint in joints):
        return None
    root = f'{env.scene.env_ns}/env_'
    for source, targets, *_ in recipes:
        if not source.startswith(root + '0/'):
            return None
        suffix = source[len(root + '0'):]
        if sorted(targets) != sorted(f'{root}{index}{suffix}' for index in range(env.num_envs)):
            return None

    def replicate():
        recipes.clear()
        ovphysx_replicate(env.sim.stage, [root + '0'], [root + '{}'], np.arange(env.num_envs),
                          np.ones((1, env.num_envs), bool), positions=plan.positions)
    return replicate


def _float_articulation(stage, root_path, body_path):
    """Free an attached articulation whose root schema sits on its world anchor joint.

    Converted fixed-base assets carry ArticulationRootAPI on the joint to the world; with
    fix_root_link=False Lab 3 only disables that joint, which leaves PhysX without an articulation.
    As on the isaac backend, the root schema and its PhysX properties move to the attached body
    and the anchor is deactivated."""
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    anchors = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)
               and prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not anchors:
        return
    body = stage.GetPrimAtPath(body_path)
    for anchor in anchors:
        for schema in anchor.GetAppliedSchemas():
            if schema in ('PhysicsArticulationRootAPI', 'PhysxArticulationAPI'):
                body.AddAppliedSchema(schema)
        for prop in anchor.GetAuthoredPropertiesInNamespace('physxArticulation'):
            prop.FlattenTo(body)
        anchor.SetActive(False)


def reset_connections(env, env_ids, *, joints, keys):
    """Place each attached root at its joint frame with zero velocity, after the parents' resets."""
    import torch
    from isaaclab.assets import Articulation
    from shared.assembly.abstract import mounted_pose
    env_ids = _env_ids(env, env_ids)
    if not len(env_ids):
        return
    env.sim.forward()
    env.scene.update(0.)
    for joint in joints:
        if joint['parent'] is None:
            parent = torch.zeros((len(env_ids), 7), device=env.device)
            parent[:, :3], parent[:, 3] = env.scene.env_origins[env_ids], 1.
        else:
            asset = env.scene[keys[joint['parent']['entity']]]
            name = joint['parent']['body'].rsplit('/', 1)[-1]
            if isinstance(asset, Articulation) and name:
                if name not in asset.body_names:
                    raise ValueError(f"Connection {joint['name']}: {joint['parent']['entity']} has no body {name!r}")
                pose = asset.data.body_link_pose_w.torch[env_ids, asset.body_names.index(name)]
            else:
                pose = asset.data.root_link_pose_w.torch[env_ids]
            parent = torch.cat((pose[:, :3], wxyz(pose[:, 3:7])), -1)
        child = env.scene[keys[joint['child']['entity']]]
        name = joint['child']['body'].rsplit('/', 1)[-1]
        if isinstance(child, Articulation) and name and name != child.body_names[0]:
            raise ValueError(f"Connection {joint['name']}: attach the root body {child.body_names[0]!r} "
                             f"of {joint['child']['entity']}")
        relative = _relative(Pose(**joint['parent_pose']), Pose(**joint['child_pose']))
        pose = mounted_pose(parent[:, :3], parent[:, 3:7], asdict(relative))
        child.write_root_pose_to_sim_index(root_pose=torch.cat((pose[:, :3], xyzw(pose[:, 3:7])), -1),
                                           env_ids=env_ids)
        child.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(env_ids), 6), device=env.device),
                                               env_ids=env_ids)
