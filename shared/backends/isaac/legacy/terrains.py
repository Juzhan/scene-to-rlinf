"""Robot-independent locomotion scene recipes (dimensions in meters)."""

import isaaclab.terrains as terrains


def configure_stairs_evaluation(config, height, speed, *, flat_mesh=False, reset_base="reset_base", reset_joints="reset_robot_joints", perturbations=("base_external_force_torque", "push_robot")):
    if not 0.0 <= height <= 0.3 or not 0.0 < speed <= 1.0:
        raise ValueError("Expected stair height in [0, 0.3] and speed in (0, 1]")
    generator = stairs_terrain()
    ascending = generator.sub_terrains["narrow_down"]
    ascending.step_height_range = (height, height)
    ascending.proportion = 1.0
    generator.sub_terrains = {"ascending": ascending}
    generator.num_rows = 1
    generator.num_cols = config.scene.num_envs
    generator.curriculum = False
    generator.difficulty_range = (0.0, 0.0)
    config.scene.terrain.terrain_generator = generator
    config.scene.terrain.max_init_terrain_level = 0
    if flat_mesh and height != 0.0:
        raise ValueError("Flat mesh evaluation requires height zero")
    if flat_mesh:
        generator.sub_terrains = {"flat": terrains.MeshPlaneTerrainCfg(proportion=1.0)}
    elif height == 0.0:
        config.scene.terrain.terrain_type = "plane"
        config.scene.terrain.terrain_generator = None
    config.curriculum.terrain_levels = None
    if hasattr(config.curriculum, "gait_level"):
        config.curriculum.gait_level = None
    for name in perturbations:
        setattr(config.events, name, None)
    reset = getattr(config.events, reset_base)
    reset.params["pose_range"] = {
        "x": (-0.05, 0.05), "y": (-0.08, 0.08), "yaw": (-0.05, 0.05),
    }
    reset.params["velocity_range"] = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    getattr(config.events, reset_joints).params["position_range"] = (0.95, 1.05)
    command = config.commands.base_velocity
    command.resampling_time_range = (1000.0, 1000.0)
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.lin_vel_x = (speed, speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.debug_vis = False
    config.viewer.origin_type = "env"
    config.viewer.eye = (4.5, 4.0, 2.8 + 3.5 * height)
    config.viewer.lookat = (1.7, 0.0, 0.3 + 3.5 * height)
    config.viewer.resolution = (960, 720)


def stairs_terrain() -> terrains.TerrainGeneratorCfg:
    variants = {}
    for width_name, width in (("narrow", 0.28), ("wide", 0.40)):
        for direction, terrain_class in (
            ("up", terrains.MeshPyramidStairsTerrainCfg),
            ("down", terrains.MeshInvertedPyramidStairsTerrainCfg),
        ):
            variants[f"{width_name}_{direction}"] = terrain_class(
                proportion=0.25, step_height_range=(0.04, 0.20), step_width=width,
                platform_width=3.0, border_width=1.0, holes=False,
            )
    return terrains.TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=10.0, num_rows=10, num_cols=16,
        curriculum=True, difficulty_range=(0.0, 1.0), use_cache=False,
        sub_terrains=variants,
    )


def platforms_terrain() -> terrains.TerrainGeneratorCfg:
    return terrains.TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=10.0, num_rows=10, num_cols=16,
        curriculum=True, difficulty_range=(0.0, 1.0), use_cache=False,
        sub_terrains={
            "single_up": terrains.MeshPitTerrainCfg(
                proportion=0.25, pit_depth_range=(0.05, 0.35), platform_width=3.0,
            ),
            "single_down": terrains.MeshBoxTerrainCfg(
                proportion=0.25, box_height_range=(0.05, 0.35), platform_width=3.0,
            ),
            "double_up": terrains.MeshPitTerrainCfg(
                proportion=0.25, pit_depth_range=(0.05, 0.22), platform_width=3.0, double_pit=True,
            ),
            "double_down": terrains.MeshBoxTerrainCfg(
                proportion=0.25, box_height_range=(0.05, 0.22), platform_width=3.0, double_box=True,
            ),
        },
    )