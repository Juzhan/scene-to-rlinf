"""Domain randomization bound to mjlab's native event terms.

Friction and centre of mass use mjlab's ``randomize_field`` on per-world model fields; mass
scales ``body_mass`` and ``body_inertia`` together (mjlab has no mass term); pushes use
``push_by_setting_velocity`` at intervals. A randomized entity's geoms get contact priority so
its coefficient, mapped as PhysX would combine it with a friction-1 partner, governs its
contacts, as ``world.contact_materials`` does here. Derived constants MuJoCo computes at
compile time (``body_invweight0``, ``body_subtreemass``) keep their compiled values.
"""
from shared.assembly.randomization import unit_contact_friction

AXIS = {'x': 0, 'y': 1, 'z': 2}


def validate(randomizations, settings):
    """Every declared property has a native counterpart; names resolve at startup."""


def raise_contact_priority(env, env_ids, asset_cfg, priority):
    """Geom priority is shared by all worlds; set once at startup for the asset's geoms."""
    geoms = env.scene[asset_cfg.name].indexing.geom_ids
    field = env.sim.model.geom_priority
    field[..., geoms] = field[..., geoms].clamp(min=priority)


def _mass(env, env_ids, asset_cfg, operation, distribution, ranges):
    import torch
    from mjlab.envs.mdp.events import _sample_distribution
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    bodies = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids].long()
    default_mass = env.sim.get_default_field('body_mass')[bodies]
    default_inertia = env.sim.get_default_field('body_inertia')[bodies]
    shape = (len(env_ids), len(bodies))
    sample = _sample_distribution(distribution, torch.tensor([ranges[0]], device=env.device),
                                  torch.tensor([ranges[1]], device=env.device), shape, env.device)
    mass = {'scale': default_mass * sample, 'add': default_mass + sample, 'abs': sample}[operation].clamp(min=1e-4)
    rows, columns = torch.meshgrid(env_ids.to(env.device).long(), bodies, indexing='ij')
    env.sim.model.body_mass[rows, columns] = mass
    env.sim.model.body_inertia[rows, columns] = default_inertia * (mass / default_mass)[..., None]


def events(randomizations, settings):
    """Native ``EventTermCfg`` per declared property, keyed ``assembly_randomize_<entity>_<property>``."""
    from mjlab.envs.mdp.events import push_by_setting_velocity, randomize_field
    from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    # Per-world copies of the fields the mass term writes.
    mass = requires_model_fields('body_mass', 'body_inertia')(_mass)

    def bodies(item):
        return SceneEntityCfg(item.key, body_names=item.bodies or '.*', preserve_order=True)
    terms = {}
    for item in randomizations:
        name = f'assembly_randomize_{item.key}_{item.property}'
        if item.property == 'friction':
            terms[name] = EventTermCfg(func=randomize_field, mode='startup', domain_randomization=True, params={
                'field': 'geom_friction', 'operation': 'abs', 'axes': [0],
                'ranges': tuple(unit_contact_friction(x, settings) for x in item.range),
                'asset_cfg': SceneEntityCfg(item.key, geom_names='.*')})
            terms[name + '_priority'] = EventTermCfg(func=raise_contact_priority, mode='startup', params={
                'asset_cfg': SceneEntityCfg(item.key), 'priority': 1})
        elif item.property == 'mass':
            terms[name] = EventTermCfg(func=mass, mode='startup', params={
                'asset_cfg': bodies(item), 'operation': item.operation,
                'distribution': item.distribution, 'ranges': item.range})
        elif item.property == 'com':
            terms[name] = EventTermCfg(func=randomize_field, mode='startup', domain_randomization=True, params={
                'field': 'body_ipos', 'operation': 'add', 'asset_cfg': bodies(item),
                'ranges': {AXIS[axis]: bounds for axis, bounds in item.axes.items()}})
        else:  # push
            terms[name] = EventTermCfg(func=push_by_setting_velocity, mode='interval',
                                       interval_range_s=item.interval_s,
                                       params={'asset_cfg': SceneEntityCfg(item.key), 'velocity_range': dict(item.axes)})
    return terms
