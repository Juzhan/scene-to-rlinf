"""Scene motion (shared/assembly/motion.py): validation, batched wander/path state, scene and editing wiring."""
import math

import pytest
import torch

from shared.assembly.abstract import AbstractState, commanded, region_sources
from shared.assembly.editing import SceneDocument
from shared.assembly.layout import world_spec_from_config
from shared.assembly.motion import Motion, check, entity_motions

AREA = {'world/area': {'shape': 'rectangle', 'size': [0.2, 0.12], 'pose': {'position': [0.5, -0.36, 0.0],
                                                                             'orientation': [1., 0., 0., 0.]}}}
BASE = {'position': [0., 0., 0.], 'orientation': [1., 0., 0., 0.]}


def scene(*entities):
    return {'schema_version': 1, 'name': 'world', 'entities': list(entities)}


@pytest.mark.parametrize('motion, message', [
    ({'type': 'fly'}, 'type wander or path'),
    ({'type': 'wander', 'speed': 1}, 'needs a region'),
    ({'type': 'wander', 'speed': [2, 1], 'region': 'world/area'}, 'speed'),
    ({'type': 'wander', 'speed': 1, 'region': 'world/area', 'points': []}, 'unknown: points'),
    ({'type': 'path', 'speed': 1, 'points': [[0, 0, 0]]}, 'at least two points'),
    ({'type': 'path', 'speed': 1, 'points': [[0, 0, 0], [0, 0, 0]], 'loop': False}, 'zero length'),
    ({'type': 'path', 'speed': 1, 'points': [[0, 0, 0], [1, 0, 0]], 'start': 'middle'}, 'start'),
])
def test_motion_is_validated_with_the_entity_named(motion, message):
    with pytest.raises(ValueError, match=f'Entity cart: motion .*{message}'):
        check('cart', motion)


def test_path_replaces_reset_and_wander_defaults_to_the_reset_region():
    with pytest.raises(ValueError, match='remove its reset'):
        check('cart', {'type': 'path', 'speed': 1, 'points': [[0, 0, 0], [1, 0, 0]]}, has_reset=True)
    wander = check('crate', {'type': 'wander', 'speed': [.05, .15]}, reset_region='world/area', has_reset=True)
    assert wander['region'] == 'world/area' and wander['turn'] == pytest.approx(.15 ** 2 / .04)


def test_wander_stays_in_its_region_at_its_drawn_speed():
    motion = Motion(check('crate', {'type': 'wander', 'region': 'world/area', 'speed': [.05, .15], 'turn': .5}),
                    BASE, AREA, 8, 'cpu', generator=torch.Generator().manual_seed(0))
    motion.reset(torch.arange(8))
    speed = motion.speed.clone()
    assert ((speed >= .05) & (speed <= .15)).all()
    pose = torch.tensor([[.5, -.36, 0., 1., 0., 0., 0.]]).repeat(8, 1)
    track = []
    for _ in range(1500):
        pose = motion.step(pose, .02, torch.ones(8, dtype=torch.bool))
        track.append(pose[:, :2])
    track = torch.stack(track)
    # The top speed overshoots an edge by speed^2 / (2 turn) = 2.25 cm.
    assert (track[..., 0] - .5).abs().max() < .1 + .023 and (track[..., 1] + .36).abs().max() < .06 + .023
    # Back inside, each axis runs at the speed it left with.
    inside = ((track[-1] - torch.tensor([.5, -.36])).abs() < torch.tensor([.1, .06])).all(-1)
    assert inside.any() and motion.velocity.norm(dim=-1)[inside] == pytest.approx(speed[inside], abs=1e-5)


def test_path_goes_round_or_back_and_faces_its_way():
    points = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]
    motion = Motion(check('cart', {'type': 'path', 'points': points, 'speed': .5, 'loop': False, 'face': True}),
                    BASE, {}, 2, 'cpu')
    motion.reset(torch.arange(2))
    pose = motion.start(torch.arange(2))
    assert pose[:, :3].tolist() == [[0., 0., 0.]] * 2
    active = torch.tensor([True, False])
    for _ in range(150):  # 1.5 m: along x, then 0.5 m up y
        pose = motion.step(pose, .02, active)
    assert pose[0, :3].tolist() == pytest.approx([1., .5, 0.], abs=1e-4)
    assert pose[0, 3:].tolist() == pytest.approx([math.cos(math.pi / 4), 0., 0., math.sin(math.pi / 4)], abs=1e-4)
    assert pose[1, :3].tolist() == [0., 0., 0.]  # an inactive row keeps its place
    for _ in range(100):  # 1 m more: bounced at the end, 0.5 m back down
        pose = motion.step(pose, .02, active)
    assert pose[0, :3].tolist() == pytest.approx([1., .5, 0.], abs=1e-4)
    assert pose[0, 3:].tolist() == pytest.approx([math.cos(math.pi / 4), 0., 0., -math.sin(math.pi / 4)], abs=1e-4)
    loop = Motion(check('cart', {'type': 'path', 'points': points, 'speed': 1.}), BASE, {}, 1, 'cpu')
    loop.reset([0])
    pose = loop.start([0])
    for _ in range(50):  # 2 m + 0.5 m into the closing diagonal of length sqrt(2)
        pose = loop.step(pose, .05, torch.ones(1, dtype=torch.bool))
    assert pose[0, :3].tolist() == pytest.approx([1 - .5 / math.sqrt(2), 1 - .5 / math.sqrt(2), 0.], abs=1e-4)


def test_abstract_state_follows_its_motion_unless_the_task_sets_it():
    entity = {'name': 'beacon', 'shape': 'point', 'size': [], 'pose': BASE, 'reset': None, 'mount': None,
              'motion': check('beacon', {'type': 'path', 'points': [[0, 0, 0], [1, 0, 0]], 'speed': 1.})}
    state = AbstractState(entity, {}, 2, 'cpu')
    state.resample(torch.arange(2))
    state.advance(.1)  # fresh rows keep their reset pose
    state.advance(.1)
    assert state.pose[:, 0].tolist() == pytest.approx([.1, .1])
    assert state.velocity[:, 0].tolist() == pytest.approx([1., 1.])
    state.advance(.1, torch.tensor([[.5, .5, 0.], [.5, .5, 0.]]))
    assert state.pose[:, :2].tolist() == [[.5, .5]] * 2


def test_scene_wiring_requires_kinematic_and_keeps_regions_static(tmp_path):
    area = {'name': 'area', 'kind': 'abstract', 'shape': 'rectangle', 'size': [.2, .12]}
    crate = {'name': 'crate', 'factory': 'dex_cube', 'reset': {'region': 'world/area'},
             'motion': {'type': 'wander', 'speed': [.05, .15]}}
    with pytest.raises(ValueError, match='declare kinematic: true'):
        world_spec_from_config(scene(area, crate), directory=tmp_path)
    data = scene(area, {**crate, 'kinematic': True},
                 {'name': 'beacon', 'kind': 'abstract', 'shape': 'point',
                  'motion': {'type': 'wander', 'region': 'world/area', 'speed': .1}})
    world_spec_from_config(data, directory=tmp_path)
    assert 'area' in region_sources(data) and 'area' not in commanded(data) and 'beacon' in commanded(data)
    assert set(entity_motions(data)) == {'crate', 'beacon'}
    moving_area = scene({**area, 'motion': {'type': 'path', 'points': [[0, 0, 0], [1, 0, 0]], 'speed': 1}},
                        {**crate, 'kinematic': True})
    with pytest.raises(ValueError, match='must be static'):
        world_spec_from_config(moving_area, directory=tmp_path)
    with pytest.raises(ValueError, match='visible is true or false'):
        world_spec_from_config(scene({'name': 'cube', 'factory': 'dex_cube', 'visible': 'no'}), directory=tmp_path)


def test_editing_writes_motion_and_visible_only_when_set(tmp_path):
    path = tmp_path / 'scene.yaml'
    path.write_text('schema_version: 1\nname: world\nentities:\n'
                    '  - {name: cube, factory: dex_cube, kinematic: true}\n')
    document = SceneDocument(path)
    motion = {'type': 'path', 'points': [[0, 0, 0], [1, 0, 0]], 'speed': .3}
    document.set_entity('world/cube', {'motion': motion, 'visible': False})
    raw = document.entities['world/cube']
    assert raw['motion'] == motion and raw['visible'] is False
    with pytest.raises(ValueError, match='at least two points'):
        document.set_entity('world/cube', {'motion': {**motion, 'points': [[0, 0, 0]]}})
    document.set_entity('world/cube', {'motion': None, 'visible': True})
    assert 'motion' not in document.entities['world/cube'] and 'visible' not in document.entities['world/cube']


def test_gravity_compensation_is_written_only_when_on_and_read_into_the_scene(tmp_path):
    from shared.assembly.layout import load_world_spec
    path = tmp_path / 'scene.yaml'
    path.write_text('schema_version: 1\nname: world\nentities:\n  - {name: arm, factory: piper_x, kind: robot}\n')
    document = SceneDocument(path)
    document.set_entity('world/arm', {'gravity_compensation': True})
    assert document.entities['world/arm']['gravity_compensation'] is True
    with pytest.raises(ValueError, match='true or false'):
        document.set_entity('world/arm', {'gravity_compensation': 'yes'})
    document.set_entity('world/arm', {'gravity_compensation': False})
    assert 'gravity_compensation' not in document.entities['world/arm']
    path.write_text('schema_version: 1\nname: world\nentities:\n  - {name: arm, factory: piper_x, gravity_compensation: true}\n')
    assert load_world_spec(path, backend='isaaclab3').root.children[0].gravity_compensation is True


def test_custom_motion_calls_the_project_function_with_the_scene(monkeypatch):
    import sys
    import types
    module = types.ModuleType('npc_project.behaviors')

    def keep_away(scene, motion):
        away = motion.pose[:, :2] - scene.robot[:, :2]
        step = away / away.norm(dim=-1, keepdim=True) * motion.parameters['speed'] * motion.dt
        motion.memory.hold('moving', torch.ones(len(step), dtype=torch.bool), torch.full((len(step),), motion.dt))
        return torch.cat((motion.pose[:, :2] + step, motion.pose[:, 2:3]), -1)

    class Shy:
        def __init__(self, parameters):
            self.parameters, self.resets = parameters, []

        def step(self, scene, motion):
            return motion.pose.clone()

        def reset(self, env_ids):
            self.resets.append(list(env_ids))

    module.keep_away, module.Shy = keep_away, Shy
    monkeypatch.setitem(sys.modules, 'npc_project', types.ModuleType('npc_project'))
    monkeypatch.setitem(sys.modules, 'npc_project.behaviors', module)
    spec = check('npc', {'type': 'custom', 'function': '.behaviors:keep_away', 'speed': 1.})
    assert spec == {'type': 'custom', 'function': '.behaviors:keep_away', 'parameters': {'speed': 1.}}
    scene = types.SimpleNamespace(robot=torch.zeros(2, 3))
    motion = Motion(spec, BASE, {}, 2, 'cpu', package='npc_project', scene=lambda: scene, name='npc')
    motion.reset(torch.arange(2))
    assert motion.start(torch.arange(2)) is None  # reset places a custom-moving entity
    pose = torch.tensor([[1., 0., 0., 1., 0., 0., 0.], [0., 2., 0., 1., 0., 0., 0.]])
    after = motion.step(pose, .5, torch.tensor([True, False]))
    assert after[0, :3].tolist() == pytest.approx([1.5, 0., 0.]) and after[1].tolist() == pose[1].tolist()
    assert motion.last[0].tolist() == pytest.approx([1., 0., 0.]) and motion.last[1].tolist() == [0., 0., 0.]
    shy = Motion(check('npc', {'type': 'custom', 'function': 'npc_project.behaviors:Shy', 'shyness': 2}),
                 BASE, {}, 2, 'cpu', scene=lambda: scene, name='npc')
    shy.reset([1])
    assert shy.custom_reset.__self__.resets == [[1]] and shy.custom_reset.__self__.parameters == {'shyness': 2}
    module.keep_away = lambda scene, motion: torch.zeros(2, 4)
    bad = Motion(spec, BASE, {}, 2, 'cpu', package='npc_project', scene=lambda: scene, name='npc')
    with pytest.raises(ValueError, match=r'npc: motion function \.behaviors:keep_away must return'):
        bad.step(pose, .5, torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match='module:attribute'):
        check('npc', {'type': 'custom', 'function': 'keep_away'})
