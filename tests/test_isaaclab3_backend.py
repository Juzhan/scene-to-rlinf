"""Isaac Lab 3 (Newton / OvPhysX) backend: static rules that hold without the SDK."""
from pathlib import Path
import subprocess
import sys
import pytest
from shared.assembly.backend_scene import prepare
from shared.backends.isaaclab3.frames import wxyz, xyzw
from shared.backends.isaaclab3.scene import validate

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'projects/dual_backend_scene/scene.yaml'


def test_shared_layout_without_sdk_imports():
    reports = [prepare(SCENE, backend)[3] for backend in ('isaac', 'isaaclab3')]
    assert reports[0]['entities'] == reports[1]['entities']
    subprocess.run([sys.executable, '-c',
        "from shared.assembly.backend_scene import prepare; import sys; "
        f"prepare({str(SCENE)!r}, 'isaaclab3'); "
        "assert not {'isaaclab','isaaclab_newton','warp','torch'} & sys.modules.keys()"],
        cwd=ROOT, check=True)


def test_quaternion_boundary():
    assert xyzw((1., 2., 3., 4.)) == (2., 3., 4., 1.)
    assert wxyz(xyzw((1., 2., 3., 4.))) == (1., 2., 3., 4.)


def test_unimplemented_features_fail_before_simulator():
    settings, world, _, _ = prepare(SCENE, 'isaaclab3')
    validate(world, settings)
    with pytest.raises(NotImplementedError, match='unknown_option'):
        validate(world, {**settings, 'unknown_option': {}})
    with pytest.raises(ValueError, match='contact material entity: pad'):
        validate(world, {**settings, 'world': {'contact_materials': {'pad': {}}}})
    validate(world, {**settings, 'world': {'contact_materials': {'robot': {
        'static_friction': .8, 'dynamic_friction': .6, 'restitution': 0.}}}})
    for newton in ({'dt': .01}, {'solver': []}, []):
        with pytest.raises(ValueError, match='simulation.newton'):
            validate(world, {**settings, 'simulation': {'newton': newton}})
    validate(world, {**settings, 'simulation': {'newton': {'num_substeps': 1, 'solver': {'njmax': 600}}}})


def test_newton_overrides_use_the_native_preset():
    pytest.importorskip('isaaclab_newton')
    from shared.backends.isaaclab3.environment import physics_config
    physics = physics_config({'newton': {'num_substeps': 1, 'solver': {'njmax': 600}}})
    assert physics.num_substeps == 1 and physics.solver_cfg.njmax == 600
    with pytest.raises(ValueError, match='unknown'):
        physics_config({'newton': {'solver': {'not_a_field': 1}}})


def test_ovphysx_uses_the_native_preset_and_physx_block():
    pytest.importorskip('isaaclab_ov')
    from shared.backends.isaaclab3.environment import physics_config
    physics = physics_config({'physx': {'gpu_max_rigid_patch_count': 327680}, 'newton': {'num_substeps': 1}}, 'ovphysx')
    assert type(physics).__name__ == 'OvPhysxCfg' and physics.gpu_max_rigid_patch_count == 327680
    with pytest.raises(NotImplementedError, match='bounce_threshold_velocity'):
        physics_config({'physx': {'bounce_threshold_velocity': .01}}, 'ovphysx')
    with pytest.raises(ValueError, match='physics'):
        physics_config({}, 'physx')



def test_renderer_is_newton_or_ovrtx_without_sdk():
    from shared.backends.isaaclab3.environment import renderer_name
    assert renderer_name({}) == 'newton' and renderer_name({'renderer': 'ovrtx'}) == 'ovrtx'
    with pytest.raises(ValueError, match='renderer'):
        renderer_name({'renderer': 'kit'})

def test_marker_axes_become_rotations_of_z_prototypes():
    # Lab 3's Newton markers ignore CylinderCfg/ConeCfg.axis, so each axis is a rotation of local Z.
    import numpy as np
    from shared.backends.isaaclab3.abstract_terms import _AXIS_ROTATIONS

    def rotate(quaternion, vector):
        x, y, z, w = quaternion
        q = np.array([x, y, z])
        return vector + 2 * np.cross(q, np.cross(q, vector) + w * vector)
    for axis, expected in (('X', (1, 0, 0)), ('Y', (0, 1, 0)), ('Z', (0, 0, 1))):
        assert np.allclose(rotate(_AXIS_ROTATIONS[axis], np.array([0., 0., 1.])), expected)


def _with_scanner(tmp_path, **scanner):
    import yaml
    settings = yaml.safe_load(SCENE.read_text())
    settings['entities'].append({'name': 'height_scanner', 'kind': 'sensor', 'factory': 'height_scanner',
                                 'parameters': {'pattern': {'resolution': .1, 'size': [1.6, 1.0]}}, **scanner})
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def test_height_scanner_targets_the_global_world_only(tmp_path):
    # Newton casts by world, not by mesh: only "every global shape" is a faithful meshes list.
    from shared.backends.isaaclab3.scene import ray_targets
    mount = {'entity': 'world/robot', 'body': 'base'}
    settings, world, _, _ = prepare(_with_scanner(tmp_path, mount=mount, meshes=['world/ground']), 'isaaclab3')
    validate(world, settings)
    node = world.index()['world/height_scanner'][0]
    assert ray_targets(world, 'world/height_scanner', node) == ['/World/ground']
    for scanner, error, match in (({'mount': mount, 'meshes': ['world/steps/first']}, NotImplementedError, 'global'),
                                  ({'mount': mount, 'meshes': []}, NotImplementedError, 'global'),
                                  ({'meshes': ['world/ground']}, ValueError, 'mount')):
        with pytest.raises(error, match=match):
            settings, world, _, _ = prepare(_with_scanner(tmp_path, **scanner), 'isaaclab3')
            validate(world, settings)


def test_connections_join_per_environment_entities(tmp_path):
    import yaml
    from shared.assembly.spec import Pose
    from shared.backends.isaaclab3.scene import _relative
    settings = yaml.safe_load(SCENE.read_text())
    joint = {'name': 'mount', 'parent': {'entity': 'world/robot', 'body': 'base'},
             'child': {'entity': 'world/steps/first', 'body': ''}}
    path = tmp_path / 'scene.yaml'
    path.write_text(yaml.safe_dump({**settings, 'connections': [joint]}))
    validate(*reversed(prepare(path, 'isaaclab3')[:2]))
    path.write_text(yaml.safe_dump({**settings, 'connections': [{**joint, 'parent': {'entity': 'world/ground'}}]}))
    with pytest.raises(ValueError, match='per-environment'):
        validate(*reversed(prepare(path, 'isaaclab3')[:2]))
    # Weld pose: the child root in the parent body frame.
    relative = _relative(Pose((1., 0., 0.), (0., 0., 0., 1.)), Pose((0., 0., .5)))
    assert relative.position == pytest.approx((1., 0., -.5)) and relative.orientation == (0., 0., 0., 1.)


def test_camera_shares_the_portable_signature_without_sdk():
    from inspect import signature
    from shared.assembly.assets import check_parameters
    from shared.backends.isaac import assets as isaac
    from shared.backends.isaaclab3 import assets
    assert signature(assets.camera) == signature(isaac.camera) and 'camera' in assets.SENSORS
    with pytest.raises(ValueError, match='data_types'):
        check_parameters(assets.camera, {'data_types': ['thermal']})


def test_xform_colliders_move_onto_their_geometry():
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdGeom, UsdPhysics
    from shared.backends.isaaclab3.spawners import gprim_colliders
    stage = Usd.Stage.CreateInMemory()
    # Isaac Sim's URDF importer layout: an instanceable `collisions` holding a collider Xform.
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, '/robot/link').GetPrim())
    collisions = UsdGeom.Xform.Define(stage, '/robot/link/collisions').GetPrim()
    holder = UsdGeom.Xform.Define(stage, '/robot/link/collisions/node').GetPrim()
    UsdPhysics.CollisionAPI.Apply(holder)
    UsdPhysics.MeshCollisionAPI.Apply(holder).CreateApproximationAttr('convexHull')
    UsdGeom.Mesh.Define(stage, '/robot/link/collisions/node/mesh')
    collisions.SetInstanceable(True)
    gprim_colliders(stage, '/robot')
    mesh = stage.GetPrimAtPath('/robot/link/collisions/node/mesh')
    assert not collisions.IsInstanceable() and not holder.HasAPI(UsdPhysics.CollisionAPI)
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == 'convexHull'
