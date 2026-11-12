"""Domain randomization bound to Isaac Lab 3's native event terms (Newton and OvPhysX).

Mass, centre of mass and friction are ``startup`` events, pushes ``interval`` events; the
sampling itself is Isaac Lab's. Friction is a PhysX material coefficient: OvPhysX combines it
with the partner's by ``simulation.physics_material.friction_combine_mode``. Newton
(MuJoCo-Warp) has no combine mode and takes the larger coefficient at equal priority, so the
randomized entity's geoms get contact priority at startup and sample the coefficient PhysX
would give the pair against a friction-1 partner (``unit_contact_friction``), as mjlab does.
"""
from shared.assembly.randomization import unit_contact_friction

# Newton MuJoCo-Warp: geoms of a randomized entity win the pair's contact parameters.
RANDOMIZED_PRIORITY = 1


def validate(randomizations, settings):
    """Every declared property has a native term on both engines; body names resolve at startup."""


def _restitution(item, settings):
    # A randomized friction replaces the entity's authored one (``world.contact_materials``,
    # applied first); its restitution is kept.
    materials = settings.get('world', {}).get('contact_materials') or {}
    for entity, material in materials.items():
        if entity.replace('.', '__') == item.key:
            return float(material['restitution'])
    return 0.


def raise_contact_priority(env, env_ids, asset_cfg, priority):
    """Give the asset's collision shapes MuJoCo contact priority (Newton only, once at startup).

    MuJoCo-Warp reads ``geom_priority`` per contact; it is per geom, shared by all worlds,
    and Newton fixes it when the model is built from USD, where replicated assets cannot
    author ``mjc:priority``. The solver is reached through the physics manager."""
    import numpy as np
    from newton.solvers import SolverMuJoCo
    solver = getattr(env.sim.physics_manager, '_solver', None)
    if not isinstance(solver, SolverMuJoCo):
        raise NotImplementedError(f'friction randomization on Newton needs the MuJoCo-Warp solver, got {type(solver).__name__}')
    model = env.sim.physics_manager.get_model()
    labels = set(env.scene[asset_cfg.name]._root_view.shape_labels)
    shapes = np.array([i for i, label in enumerate(model.shape_label) if label in labels])
    template = solver.mjc_geom_to_newton_shape.numpy()[0]
    geoms = np.flatnonzero(np.isin(template, shapes))
    if len(geoms) == 0:
        raise RuntimeError(f'{asset_cfg.name}: no MuJoCo geoms found for its collision shapes')
    values = solver.mjw_model.geom_priority.numpy()
    values[geoms] = priority
    solver.mjw_model.geom_priority.assign(values)


def events(randomizations, settings, physics):
    """Native ``EventTermCfg`` per declared property, keyed ``assembly_randomize_<entity>_<property>``."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    def entity(item):
        return SceneEntityCfg(item.key, body_names=list(item.bodies) if item.bodies else '.*', preserve_order=True)
    terms = {}
    for item in randomizations:
        name = f'assembly_randomize_{item.key}_{item.property}'
        if item.property == 'friction':
            restitution = _restitution(item, settings)
            friction = (tuple(unit_contact_friction(x, settings) for x in item.range)
                        if physics == 'newton' else item.range)
            terms[name] = EventTermCfg(func=mdp.randomize_rigid_body_material, mode='startup', params={
                'asset_cfg': entity(item), 'static_friction_range': friction, 'dynamic_friction_range': friction,
                'restitution_range': (restitution, restitution), 'num_buckets': 64, 'make_consistent': True})
            if physics == 'newton':
                terms[name + '_priority'] = EventTermCfg(func=raise_contact_priority, mode='startup', params={
                    'asset_cfg': SceneEntityCfg(item.key), 'priority': RANDOMIZED_PRIORITY})
        elif item.property == 'mass':
            terms[name] = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode='startup', params={
                'asset_cfg': entity(item), 'mass_distribution_params': item.range, 'operation': item.operation,
                'distribution': item.distribution, 'recompute_inertia': True, 'min_mass': 1e-4})
        elif item.property == 'com':
            terms[name] = EventTermCfg(func=mdp.randomize_rigid_body_com, mode='startup', params={
                'asset_cfg': entity(item), 'com_range': dict(item.axes)})
        else:  # push
            terms[name] = EventTermCfg(func=mdp.push_by_setting_velocity, mode='interval',
                                       interval_range_s=item.interval_s,
                                       params={'asset_cfg': SceneEntityCfg(item.key), 'velocity_range': dict(item.axes)})
    return terms
