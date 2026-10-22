"""Build disposable, material-free Blender previews; never alter source assets.

Run in background Blender. Each part stays separate, so link identity, small
attachments and exact native mounting frames survive simplification. Several
jobs may share one manifest: job ``i`` of ``n`` simplifies only its parts
(``mesh_preview.assign``) and writes them to ``--output``; ``merge_partials``
then writes the lightweight meshes and the manifest.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import bmesh
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.assembly.mesh_preview import assign, part_targets, pending, read_mesh_source  # noqa: E402


def simplify_part(vertices, faces, target):
    count = len(faces)
    mesh = bpy.data.meshes.new('preview')
    # Bulk-copy the arrays; from_pydata over Python lists is slow for CAD meshes.
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set('co', vertices.astype('f4').ravel())
    mesh.loops.add(3 * count)
    mesh.loops.foreach_set('vertex_index', faces.astype('i4').ravel())
    mesh.polygons.add(count)
    mesh.polygons.foreach_set('loop_start', np.arange(0, 3 * count, 3, dtype='i4'))
    if not bpy.types.MeshPolygon.bl_rna.properties['loop_total'].is_readonly:  # Blender < 4 needs explicit sizes
        mesh.polygons.foreach_set('loop_total', np.full(count, 3, dtype='i4'))
    mesh.update()
    # Many CAD exports contain triangle soup: merge duplicate vertices
    # before collapse, otherwise disconnected triangles cannot simplify.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-6)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new('preview', mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    if count > target:
        modifier = obj.modifiers.new('Editor preview only', 'DECIMATE')
        modifier.ratio = target / count
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    reduced = obj.data
    reduced.calc_loop_triangles()
    vertices = np.empty(3 * len(reduced.vertices), dtype='f4')
    reduced.vertices.foreach_get('co', vertices)
    triangles = np.empty(3 * len(reduced.loop_triangles), dtype='i4')
    reduced.loop_triangles.foreach_get('vertices', triangles)
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(reduced)
    if mesh != reduced and mesh.name in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    return vertices.reshape(-1, 3), triangles.reshape(-1, 3)


def main(manifest, jobs, job, output):
    work = pending(json.loads(manifest.read_text()))
    geometries = [read_mesh_source(source) for _, source in work]
    owners = assign(geometries, jobs)
    results = {}
    for g, geometry in enumerate(geometries):
        for i, (part, target) in enumerate(zip(geometry['parts'], part_targets(geometry))):
            if target and owners[g, i] == job:
                results[f'v_{g}_{i}'], results[f'f_{g}_{i}'] = simplify_part(part['vertices'], part['faces'], target)
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **results)
    temporary.replace(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--job', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(args.manifest, args.jobs, args.job, args.output)
