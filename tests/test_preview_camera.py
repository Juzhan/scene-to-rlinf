"""Default preview camera framing from the scene description, without a simulator."""
import math

from shared.assembly.framing import complete_viewer, preview_camera
from shared.assembly.layout import world_spec_from_config


def world(*entities, tmp_path):
    return world_spec_from_config({"schema_version": 1, "name": "world", "entities": list(entities)},
                                  directory=tmp_path)


def at(name, factory, position, **fields):
    return {"name": name, "factory": factory, "pose": {"position": list(position)}, **fields}


def test_framing_covers_robot_objects_and_areas_but_not_ground(tmp_path):
    description = world(
        at("robot", "go2", (0., 0., .4), kind="robot"),
        at("ground", "ground_plane", (0., 0., 0.)),
        at("cube", "dex_cube", (4., 0., .05)),
        {"name": "goal", "kind": "abstract", "shape": "rectangle", "size": [2., 2.],
         "pose": {"position": [0., 6., 0.]}},
        tmp_path=tmp_path)
    camera = preview_camera(description)
    assert camera["follow"] == "robot"
    # Relative to the followed robot root: the centre of x -1..4, y 0..7, z 0..0.4.
    assert camera["lookat"] == (1.5, 3.5, -.2)
    eye, lookat = camera["eye"], camera["lookat"]
    distance = math.dist(eye, lookat)
    reach = max(math.dist(lookat, (x, y, z - .4)) for x, y, z in
                [(0, 0, .4), (4, 0, .05), (-1, 5, 0), (1, 7, 0)])
    assert distance >= 2 * reach and eye[0] > lookat[0] and eye[1] > lookat[1] and eye[2] > lookat[2]


def test_small_scene_keeps_a_minimum_distance_and_scene_without_robot_uses_env_origin(tmp_path):
    camera = preview_camera(world(at("cube", "dex_cube", (1., 0., 0.)), tmp_path=tmp_path))
    assert camera["follow"] is None and camera["lookat"] == (1., 0., 0.)
    assert math.isclose(math.dist(camera["eye"], camera["lookat"]), 2.5, abs_tol=1e-3)
    assert preview_camera(world(at("ground", "ground_plane", (0., 0., 0.)), tmp_path=tmp_path)) == {}


def test_explicit_camera_options_are_kept(tmp_path):
    description = world(at("robot", "go2", (0., 0., .4), kind="robot"), tmp_path=tmp_path)
    assert complete_viewer(None, description) is None
    explicit = {"resolution": (640, 480), "eye": (1., 2., 3.)}
    assert complete_viewer(explicit, description) == explicit
    assert complete_viewer({"resolution": (640, 480)}, description)["follow"] == "robot"
