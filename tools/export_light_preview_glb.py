"""Convert an assembly lightweight mesh cache into a visual-inspection GLB.

Run through Blender, for example::

    blender --background --factory-startup --python tools/export_light_preview_glb.py -- \
        --input preview.light-v1.mesh.json.gz --output preview.glb

Assembly cache vertices already contain the project's authored initial joint
pose in entity-root coordinates.  This converter deliberately flattens that
snapshot: it does not reconstruct an editable articulation or a rest-pose
robot asset.  The source cache remains disposable and the simulation asset is
never changed.
"""

import argparse
import gzip
import json
from pathlib import Path
import re
import sys

import bpy


def _safe_name(value: str, index: int) -> str:
    name = value.rsplit("/", 1)[-1] or f"part_{index}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def export_glb(source: Path, destination: Path) -> None:
    with gzip.open(source, "rt") as stream:
        geometry = json.load(stream)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    material = bpy.data.materials.new("editor_preview")
    material.diffuse_color = (0.13, 0.36, 0.62, 1.0)
    material.metallic = 0.15
    material.roughness = 0.48

    for index, part in enumerate(geometry["parts"]):
        name = _safe_name(part.get("name", ""), index)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(part["vertices"], [], part["faces"])
        mesh.update(calc_edges=True)
        obj = bpy.data.objects.new(name, mesh)
        obj.data.materials.append(material)
        bpy.context.collection.objects.link(obj)

    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(destination),
        export_format="GLB",
        use_selection=False,
        export_materials="EXPORT",
        export_yup=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    export_glb(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
