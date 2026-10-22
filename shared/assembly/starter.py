"""Default content of a new environment, shared by the browser and Blender editors.

Uses shared asset factories only: a ground plane, a dome light, a fixed-base
Franka arm at the origin and a cube 0.5 m in front of it. It is a scene, not a
training project; delete or replace entities in the editor as needed.
"""
import json
from pathlib import Path


def starter_scene():
    return {'schema_version': 1, 'name': 'world', 'entities': [
        {'name': 'ground', 'factory': 'ground_plane'},
        {'name': 'light', 'factory': 'dome_light'},
        {'name': 'robot', 'kind': 'robot', 'factory': 'franka_panda'},
        # DexCube is 6 cm scaled by 0.8; start just above the ground.
        {'name': 'cube', 'factory': 'dex_cube', 'pose': {'position': [0.5, 0.0, 0.03]}},
    ]}


def write_starter_scene(path):
    path = Path(path)
    path.write_text(json.dumps(starter_scene(), indent=2))
    return path
