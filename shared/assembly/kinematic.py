"""Kinematic entities: physical assets declared ``kinematic: true`` in scene.yaml.

They collide with everything but ignore forces and gravity. Reset places them like any asset
(``reset``); within an episode only the Task moves them, returning ``set: {key: (B, 3) position
or (B, 7) xyz + wxyz}`` exactly as for abstract entities. The backend writes that pose after
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
