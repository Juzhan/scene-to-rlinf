"""Kinematic entities: physical assets declared ``kinematic: true`` in scene.yaml.

They collide with everything but ignore forces and gravity. Reset places them like any asset
(``reset``); within an episode their scene ``motion`` (``motion.py``) moves them, or the Task, returning
``set: {key: (B, 3) position or (B, 7) xyz + wxyz}`` exactly as for abstract entities (the Task wins). The backend writes that pose after
resets and before observations, with root velocity = displacement / step_dt so contacts see the
motion; environments reset this step keep their sampled pose.
"""


def kinematic_entities(settings):
    """Observation keys (``parent__child``) of the entities declared kinematic."""
    found = []

    def visit(entries, prefix):
        for entry in entries or ():
            if not isinstance(entry, dict) or entry.get('kind') == 'abstract':
                continue
            key = prefix + entry['name']
            if entry.get('kinematic') is True:
                found.append(key)
            visit(entry.get('children'), key + '__')
    visit(settings.get('entities'), '')
    return found


def kinematic_poses(settings):
    """Authored ``{'position', 'orientation'}`` of each kinematic entity, by observation key."""
    found = {}

    def visit(entries, prefix):
        for entry in entries or ():
            if not isinstance(entry, dict) or entry.get('kind') == 'abstract':
                continue
            key = prefix + entry['name']
            if entry.get('kinematic') is True:
                pose = entry.get('pose') or {}
                found[key] = {'position': list(pose.get('position', (0., 0., 0.))),
                              'orientation': list(pose.get('orientation', (1., 0., 0., 0.)))}
            visit(entry.get('children'), key + '__')
    visit(settings.get('entities'), '')
    return found
