"""Authored per-entity contact properties, applied by a native startup event."""
import math


def validate_materials(values, entity_keys):
    if not isinstance(values, dict):
        raise ValueError('world.contact_materials must map entities to materials')
    for entity, material in values.items():
        if entity.replace('.', '__') not in entity_keys:
            raise ValueError(f'Unknown contact material entity: {entity}')
        if not isinstance(material, dict) or set(material) != {'static_friction', 'dynamic_friction', 'restitution'}:
            raise ValueError('Contact material requires static_friction, dynamic_friction, restitution')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in material.values()):
            raise ValueError('Contact properties must be finite nonnegative numbers')
        if material['restitution'] > 1:
            raise ValueError('Restitution must not exceed one')


def events(values):
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    return {'assembly_material_' + entity.replace('.', '__'): EventTermCfg(
        func=mdp.randomize_rigid_body_material, mode='startup', params={
            'asset_cfg': SceneEntityCfg(entity.replace('.', '__'), body_names='.*'),
            'static_friction_range': (m['static_friction'],) * 2,
            'dynamic_friction_range': (m['dynamic_friction'],) * 2,
            'restitution_range': (m['restitution'],) * 2, 'num_buckets': 64})
        for entity, m in values.items()}
