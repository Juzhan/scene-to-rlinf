"""Abstract-entity markers: one shared shape/size spec, drawn by Isaac and mjlab previews alike."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from shared.assembly import abstract
from shared.assembly.abstract import ARROW_LENGTH, EDGE_WIDTH, POINT_RADIUS, marker_edges
from shared.backends.isaac import abstract_terms as isaac_terms
from shared.backends.mjlab import abstract_terms as mjlab_terms


def scene(*entities, **fields):
    return {"schema_version": 1, "name": "world", "entities": list(entities), **fields}


def test_edges_close_a_box_and_a_rectangle_only():
    offsets, axes = marker_edges({"shape": "box", "size": [2., 1., .5]})
    assert len(offsets) == 12 and sorted(axes) == [0] * 4 + [1] * 4 + [2] * 4
    assert {tuple(o) for o, a in zip(offsets, axes) if a == 0} == {
        (0., y, z) for y in (-.5, .5) for z in (-.25, .25)}
    offsets, axes = marker_edges({"shape": "rectangle", "size": [2., 1.]})
    assert sorted(axes) == [0, 0, 1, 1]
    assert {tuple(o) for o in offsets} == {(0., -.5, 0.), (0., .5, 0.), (-1., 0., 0.), (1., 0., 0.)}
    assert marker_edges({"shape": "point", "size": []}) is None


def test_isaac_parts_place_shaft_and_head_along_x():
    offsets, indices = isaac_terms.marker_parts({"shape": "direction", "size": []})
    shaft = (1 - abstract.ARROW_HEAD_LENGTH) * ARROW_LENGTH
    assert indices == [0, 1]
    assert offsets[0] == [shaft / 2, 0., 0.]
    # The head cone ends at the arrow tip.
    assert math.isclose(offsets[1][0] + abstract.ARROW_HEAD_LENGTH * ARROW_LENGTH / 2, ARROW_LENGTH)
    assert isaac_terms.marker_parts({"shape": "point", "size": []}) == ([[0., 0., 0.]], [0])
    assert isaac_terms.marker_parts({"shape": "box", "size": [1., 1., 1.]}) == marker_edges(
        {"shape": "box", "size": [1., 1., 1.]})


def test_isaac_markers_exist_only_with_a_viewer(monkeypatch):
    monkeypatch.setattr(isaac_terms, "_config_type", lambda: SimpleNamespace)
    settings = scene({"name": "area", "kind": "abstract", "shape": "rectangle", "size": [1., 1.]},
                     {"name": "goal", "kind": "abstract", "shape": "point",
                      "reset": {"region": "world/area", "interval": 4}},
                     visualization={"abstract": ["area", "goal"]})
    output = {"commands": SimpleNamespace()}
    isaac_terms.attach_abstract(output, settings, visualize=False)
    # A randomized entity keeps its term for its readings; nothing is drawn.
    assert {name: term.debug_vis for name, term in vars(output["commands"]).items()} == {
        "assembly_abstract_goal": False}
    output = {"commands": SimpleNamespace()}
    isaac_terms.attach_abstract(output, settings, visualize=True)
    assert {name: term.debug_vis for name, term in vars(output["commands"]).items()} == {
        "assembly_abstract_area": True, "assembly_abstract_goal": True}


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


def test_mjlab_draws_shared_shapes_in_world_pose():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])  # +90 deg yaw
    position = np.array([1., 2., 3.])
    drawn = Recorder()
    mjlab_terms.draw_marker(drawn, {"shape": "box", "size": [2., 1., .5]}, position, rotation)
    assert len(drawn.calls) == 12 and {call[0] for call in drawn.calls} == {"add_cylinder"}
    start, end, radius, color = drawn.calls[0][1]
    # First edge: along local x (world +y after the yaw), at local (0, -0.5, -0.25).
    centre = position + rotation @ np.array([0., -.5, -.25])
    np.testing.assert_allclose((start + end) / 2, centre)
    np.testing.assert_allclose(end - start, [0., 2. + EDGE_WIDTH, 0.], atol=1e-12)
    assert radius == EDGE_WIDTH / 2 and color == (*abstract.MARKER_COLOR, 1.)

    drawn = Recorder()
    mjlab_terms.draw_marker(drawn, {"shape": "direction", "size": []}, position, rotation)
    (name, (start, end, color), kwargs), = drawn.calls
    # Twice the length: MuJoCo draws an arrow connector over half of from-to.
    assert name == "add_arrow" and kwargs == {"width": abstract.ARROW_WIDTH}
    np.testing.assert_allclose(end - start, [0., 2 * ARROW_LENGTH, 0.], atol=1e-12)

    drawn = Recorder()
    mjlab_terms.draw_marker(drawn, {"shape": "point", "size": []}, position, rotation)
    assert drawn.calls == [("add_sphere", (position, POINT_RADIUS, (*abstract.MARKER_COLOR, 1.)), {})]


def test_mjlab_terms_cover_commanded_entities_and_shown_ones_when_visualizing(monkeypatch):
    monkeypatch.setattr(mjlab_terms, "_config_type", lambda: SimpleNamespace)
    settings = scene({"name": "robot", "factory": "go2", "kind": "robot"},
                     {"name": "area", "kind": "abstract", "shape": "rectangle", "size": [1., 1.]},
                     {"name": "heading", "kind": "abstract", "shape": "direction",
                      "mount": {"entity": "world/robot"}},
                     {"name": "goal", "kind": "abstract", "shape": "point",
                      "reset": {"region": "world/area", "interval": 2}},
                     {"name": "still", "kind": "abstract", "shape": "point"},
                     visualization={"abstract": ["area", "heading"]})
    # Training: unmounted, non-region entities own state (the Task may move them); shown-only ones are not built.
    commands = {}
    mjlab_terms.attach_abstract(commands, settings)
    assert set(commands) == {"assembly_abstract_goal", "assembly_abstract_still"}
    assert commands["assembly_abstract_still"].resampling_time_range == (1e9, 1e9)
    goal = commands["assembly_abstract_goal"]
    assert goal.resampling_time_range == (2., 2.) and not goal.debug_vis
    assert "world/area" in goal.areas
    commands = {}
    mjlab_terms.attach_abstract(commands, settings, visualize=True)
    assert set(commands) == {"assembly_abstract_area", "assembly_abstract_heading", "assembly_abstract_goal",
                             "assembly_abstract_still"}
    assert commands["assembly_abstract_heading"].entity["mount"] == "robot"
    assert commands["assembly_abstract_area"].debug_vis and not commands["assembly_abstract_goal"].debug_vis
    with pytest.raises(ValueError, match="Reserved"):
        mjlab_terms.attach_abstract(commands, settings, visualize=True)


def test_mjlab_visualizer_refuses_to_overflow_the_scene():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mjlab")
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    scene_ = mujoco.MjvScene(model, maxgeom=2)
    visualizer = mjlab_terms.guarded_visualizer(scene_, model)
    for _ in range(2):
        visualizer.add_sphere(np.zeros(3), .1, (1., 1., 1., 1.))
    with pytest.raises(RuntimeError, match="Preview scene is full"):
        visualizer.add_cylinder(np.zeros(3), np.ones(3), .1, (1., 1., 1., 1.))
    assert scene_.ngeom == 2


def test_mujoco_arrow_matches_a_cylinder_of_the_marker_length():
    """The arrow compensation in draw_marker against MuJoCo's own renderer (cylinders draw full length)."""
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("mjlab")
    try:
        from mujoco.egl import GLContext
        context = GLContext(640, 480)
        context.make_current()
    except Exception as error:  # No EGL device in this environment.
        pytest.skip(f"offscreen OpenGL unavailable: {error}")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><global offwidth="640" offheight="480" orthographic="true" fovy="1"/></visual></mujoco>')
    data = mujoco.MjData(model)
    renderer = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    try:
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderer)
        scene_ = mujoco.MjvScene(model, maxgeom=100)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (ARROW_LENGTH / 2, 0., 0.)
        camera.distance, camera.azimuth, camera.elevation = 2., 90., -90.
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene_)
        visualizer = mjlab_terms.guarded_visualizer(scene_, model)
        mjlab_terms.draw_marker(visualizer, {"shape": "direction", "size": []},
                                np.array([0., .1, 0.]), np.eye(3))
        visualizer.add_cylinder(np.array([0., -.1, 0.]), np.array([ARROW_LENGTH, -.1, 0.]), .015, (1., 0., 0., 1.))
        viewport = mujoco.MjrRect(0, 0, 640, 480)
        mujoco.mjr_render(viewport, scene_, renderer)
        pixels = np.empty((480, 640, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, viewport, renderer)
    finally:
        renderer.free()
        context.free()
    pixels = pixels.astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]

    def extent(mask):
        columns = np.flatnonzero(mask.any(0))
        rows = np.flatnonzero(mask.any(1))
        assert len(columns) and len(rows)
        return columns[-1] - columns[0] + 1, (rows[0] + rows[-1]) / 2

    arrow, arrow_row = extent((green > 60) & (green > 2 * red) & (green > 2 * blue))
    shaft, shaft_row = extent((red > 60) & (red > 2 * green) & (red > 2 * blue))
    assert arrow_row != shaft_row
    assert abs(arrow - shaft) <= 0.03 * shaft, (arrow, shaft)
