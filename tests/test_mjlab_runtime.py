"""Native GPU acceptance is opt-in, ordinary tests never allocate a simulator."""
import os
import pytest
import torch


def test_inspected_interfaces_are_cached_per_backend(tmp_path):
    from shared.assembly.interfaces import save_interface, load_interface
    scene = tmp_path / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities: []\n')
    save_interface(scene, {'backend':'isaac', 'action_dim':9})
    save_interface(scene, {'backend':'mjlab', 'action_dim':12}, backend='mjlab')
    assert load_interface(scene)['action_dim'] == 9
    assert load_interface(scene, backend='mjlab')['action_dim'] == 12


@pytest.mark.skipif(os.environ.get('RUN_MJLAB_GPU') != '1', reason='requires mjlab CUDA environment')
def test_native_batch_partial_reset_and_terminal_observation():
    from projects.dual_backend_scene import project
    from tools.inspect_scene import native_interface
    env, owner = project.make_environment({'backend':'mjlab', 'device':'cuda:0'},2,42,3)
    try:
        observations, _ = env.reset(seed=42)
        assert observations['robot.joint_position'].shape == (2,12)
        report = native_interface(env.native)
        assert report['backend'] == 'mjlab' and report['action_dim'] == 12
        # Go2's MJCF is depth-first per leg; the report keeps the asset's (Isaac) order.
        assert report['articulations']['robot']['joint_names'][:4] == [
            'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint']
        env.native.episode_length_buf[:] = torch.tensor([2,0], device=env.device)
        observations, reward, terminated, truncated, info = env.step(torch.ones(2,12,device=env.device))
        assert truncated.tolist() == [True,False]
        assert info['autoreset'] is True and not terminated.any()
        assert env.native.episode_length_buf.tolist() == [0,1]
        final = info['terminal_observation']
        assert final.keys() == observations.keys()
        assert not torch.equal(final['robot.joint_velocity'][0], observations['robot.joint_velocity'][0])
        torch.testing.assert_close(observations['robot.joint_velocity'][0],torch.zeros(12,device=env.device))
        saved_reward = reward.clone()
        env.step(torch.zeros(2,12,device=env.device))
        torch.testing.assert_close(reward,saved_reward)
        assert torch.isfinite(reward).all()
    finally:
        env.close()
        owner.close()


@pytest.mark.skipif(os.environ.get('RUN_MJLAB_GPU') != '1', reason='requires mjlab CUDA environment')
def test_native_camera_depth_matches_geometry():
    from mjlab.scene import Scene
    from mjlab.sim import Simulation, SimulationCfg
    from pathlib import Path
    from shared.assembly.assets import factories_for
    from shared.assembly.layout import world_spec_from_config
    from shared.backends.mjlab.readers import depth, image_plane_cosine
    from shared.backends.mjlab.scene import build_scene_config
    # A camera 1 m above a table top looks straight down (MuJoCo camera frame: -Z forward).
    settings = {'name': 'world', 'entities': [
        {'name': 'table', 'factory': 'box', 'parameters': {'size': [4, 4, .1]}, 'pose': {'position': [0, 0, -.05]}},
        {'name': 'eye', 'factory': 'camera', 'kind': 'sensor', 'pose': {'position': [0, 0, 1]},
         'parameters': {'width': 32, 'height': 24, 'data_types': ['distance_to_camera', 'distance_to_image_plane']}}]}
    world = world_spec_from_config(settings, directory=Path('.'), cameras=True, backend='mjlab')
    cfg, _ = build_scene_config(world, settings, factories_for('mjlab'))
    scene = Scene(cfg, device='cuda:0')
    sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=scene.compile(), device='cuda:0')
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    sim.set_sensor_context(scene.sensor_context)
    sim.forward()
    sim.sense()
    camera = scene['eye']
    cosine = image_plane_cosine(camera.cfg, 'cuda:0')
    ray = depth(camera.data.depth, camera.cfg, 'distance_to_camera')
    plane = depth(camera.data.depth, camera.cfg, 'distance_to_image_plane', cosine)
    torch.testing.assert_close(plane, torch.ones_like(plane), atol=1e-3, rtol=0)
    torch.testing.assert_close(ray, 1 / cosine.expand_as(ray), atol=1e-3, rtol=0)


@pytest.mark.skipif(os.environ.get('RUN_MJLAB_GPU') != '1', reason='requires mjlab CUDA environment')
def test_anymal_lstm_actuator_follows_isaac_network_and_resets_per_env():
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import Simulation, SimulationCfg
    from shared.backends.mjlab.assets import FACTORIES
    scene = Scene(SceneCfg(num_envs=2, entities={'ground': FACTORIES['ground_plane'](),
                                                 'robot': FACTORIES['anymal_c']()}), device='cuda:0')
    sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=scene.compile(), device='cuda:0')
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    robot = scene['robot']
    actuator, = robot.actuators
    target = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    scene.reset()
    sim.forward()
    # Isaac ActuatorNetLSTM: one sequence step per env x joint over (target - position, velocity).
    hidden = torch.zeros(2, 24, 8, device='cuda:0')
    cell = torch.zeros_like(hidden)
    for _ in range(20):
        robot.set_joint_position_target(target)
        error, velocity = target - robot.data.joint_pos, robot.data.joint_vel
        with torch.no_grad():
            expected, (hidden, cell) = actuator.network(
                torch.stack((error, velocity), -1).reshape(24, 1, 2), (hidden, cell))
        scene.write_data_to_sim()
        torch.testing.assert_close(sim.data.ctrl[:], expected.reshape(2, 12).clamp(-80, 80))
        sim.step()
        scene.update(sim.cfg.mujoco.timestep)
    scene.reset(torch.tensor([1], device='cuda:0'))
    assert actuator._hidden[:, 1].abs().sum() == 0 and actuator._cell[:, 1].abs().sum() == 0
    assert actuator._hidden[:, 0].abs().sum() > 0
