"""Generate one Isaac Lab sub-terrain tile as raw triangle meshes.

Runs in the Isaac interpreter (it starts a headless Kit app); shared.terrain_assets
launches it and turns the meshes into backend-neutral assets:

    python -m shared.backends.isaac.terrain <variant> <output.npz> <size> <difficulty> <seed> <parameters json>
"""
import dataclasses
import json
import sys
import traceback
from pathlib import Path


def main(argv):
    variant, output, size, difficulty, seed, parameters = argv
    from isaaclab.app import AppLauncher
    application = AppLauncher(headless=True).app
    try:
        import numpy as np
        from shared.backends.isaac.assets import terrain_variant

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
        # Kit exits with status 0 when closed; the caller detects failure by the missing output file.
        traceback.print_exc()
    finally:
        application.close()


if __name__ == '__main__':
    main(sys.argv[1:])
