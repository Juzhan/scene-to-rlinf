"""Translate the common scene.yaml into a Newton model.

Only primitives that Newton simulates differentiably are translated: the static
fixtures and the free rigid boxes.  The Franka arm, cameras and sensors stay in
scene.yaml for preview and for the Isaac backends; Newton's differentiable path
in this workspace covers rigid contacts of free bodies, not articulated drives.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import warp as wp

import newton

from shared.assembly.layout import load_world_spec

SCENE = Path(__file__).resolve().parents[1] / "scene.yaml"

# Contact parameters tuned for 0.2 kg cubes at dt = 1/240 with the semi-implicit
# solver: stiff enough to avoid sinking, soft enough to stay stable.
CONTACT_KE = 2.0e3
CONTACT_KD = 6.0
FRICTION = 0.8
# Viscous friction gain; Newton's default 1000 makes an off-centre cube-on-cube contact explode.
CONTACT_KF = 10.0

# Scene entities that carry no Newton geometry: lighting, the ground plane
# (added explicitly), sensors and the robot (its sources are MJCF/USD assets).
NON_PHYSICAL = {"ground_plane", "dome_light", "clear_sky", "frame_transformer", "camera",
                "contact_sensor", "height_scanner"}


@dataclass
class SceneBodies:
    """Body indices of the translated scene, relative to one world."""
    world_positions: dict[str, tuple[float, float, float]]
    sizes: dict[str, tuple[float, float, float]]
    masses: dict[str, float | None]
    body_of: dict[str, int]
    body_count: int

    def index(self, name: str, world: int = 0) -> int:
        return self.body_of[name] + world * self.body_count


def read_scene(path: Path = SCENE) -> SceneBodies:
    """Collect every box entity with its world pose, size and mass."""
    description = load_world_spec(path)
    body_of, positions, sizes, masses = {}, {}, {}, {}
    next_body = 0
    for path_name, (entity, pose) in description.index().items():
        if entity.factory is None or entity.kind in ("robot", "sensor"):
            continue
        if entity.factory in NON_PHYSICAL:
            continue
        if entity.factory != "box":
            raise ValueError(f"Newton translation supports box fixtures only; {path_name} uses {entity.factory}")
        name = path_name.rsplit("/", 1)[-1]
        size = tuple(float(v) for v in entity.parameters["size"])
        mass = entity.parameters.get("mass")
        positions[name] = tuple(float(v) for v in pose.position)
        sizes[name] = size
        masses[name] = None if mass is None else float(mass)
        body_of[name] = -1 if mass is None else next_body
        if mass is not None:
            next_body += 1
    return SceneBodies(positions, sizes, masses, body_of, next_body)


def builder(scene: SceneBodies, *, requires_grad: bool = True) -> newton.ModelBuilder:
    """Build ground, static fixtures and free boxes; static shapes share body -1."""
    build = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    build.default_shape_cfg.ke = CONTACT_KE
    build.default_shape_cfg.kd = CONTACT_KD
    build.default_shape_cfg.mu = FRICTION
    build.default_shape_cfg.kf = CONTACT_KF
    build.add_ground_plane()
    for name, size in scene.sizes.items():
        half = tuple(0.5 * v for v in size)
        position = scene.world_positions[name]
        mass = scene.masses[name]
        body = -1
        density = 1000.0
        if mass is not None:
            body = build.add_body(xform=wp.transform(wp.vec3(*position), wp.quat_identity()))
            density = mass / (size[0] * size[1] * size[2])
        cfg = newton.ModelBuilder.ShapeConfig(ke=CONTACT_KE, kd=CONTACT_KD, kf=CONTACT_KF, mu=FRICTION,
                                              density=density)
        # A static shape carries its own world pose; a shape attached to a body is body-local.
        xform = wp.transform(wp.vec3(*position), wp.quat_identity()) if body < 0 else wp.transform_identity()
        build.add_shape_box(body=body, xform=xform, hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=name)
    return build


def model(build: newton.ModelBuilder, *, worlds: int = 1, spacing: float = 3.0):
    """Finalize one differentiable model holding `worlds` identical copies of the scene."""
    if worlds > 1:
        replicated = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
        replicated.replicate(build, worlds, spacing=(spacing, 0.0, 0.0))
        return replicated.finalize(requires_grad=True)
    return build.finalize(requires_grad=True)
