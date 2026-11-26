"""Generate one Isaac Lab sub-terrain tile as raw triangle meshes.

Isaac Lab 3's terrain generators are plain Python (numpy), so this runs in the IL3
environment without starting an application or simulator. shared.terrain_assets
launches it and turns the meshes into backend-neutral assets:

    python -m shared.backends.isaaclab3.terrain <variant> <output.npz> <size> <difficulty> <seed> <parameters json>
"""
import dataclasses
import json
import sys
import traceback
from pathlib import Path


def main(argv):
    variant, output, size, difficulty, seed, parameters = argv
    try:
        import numpy as np
        from shared.backends.isaaclab3.assets import terrain_variant

        np.random.seed(int(seed))
        config = terrain_variant(variant, json.loads(parameters))
        config.size = (float(size), float(size))
        missing = [field.name for field in dataclasses.fields(config)
                   if type(getattr(config, field.name)).__name__ == '_MISSING_TYPE']
        if missing:
            raise ValueError(f"terrain '{variant}' needs parameters: {', '.join(missing)} (pass them with --param JSON)")
        meshes, origin = config.function(float(difficulty), config)
        np.savez(Path(output), origin=np.asarray(origin, dtype=float),
                 **{f'{kind}{index}': np.asarray(getattr(mesh, kind))
                    for index, mesh in enumerate(meshes) for kind in ('vertices', 'faces')})
    except BaseException:
        # The caller detects failure by the missing output file and names the variant.
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
