"""Backend-selectable project check and scene-only native inspection/view."""
import argparse
import json
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('check', 'inspect', 'view'))
    parser.add_argument('project')
    parser.add_argument('--backend', required=True, choices=('isaac', 'mjlab'))
    args = parser.parse_args(argv)
    if not args.project.isidentifier():
        parser.error('Project must be a Python identifier')
    path = Path(__file__).resolve().parents[1] / 'projects' / args.project / 'scene.yaml'
    if args.operation == 'check' and (path.parent / 'train.yaml').is_file():
        from run import compose_project
        cfg, module = compose_project(args.project, overrides=[
            f'env.train.init_params.backend={args.backend}'], register=False)
        print(json.dumps(module.check(cfg), indent=2, ensure_ascii=False))
        return
    if args.operation != 'check':
        import os, sys
        from shared.assembly.project import backend_interpreter
        interpreter = backend_interpreter(args.backend)
        if interpreter != sys.executable:
            os.execv(interpreter, [interpreter, str(path.parents[2] / 'run.py'), *argv])
    from shared.assembly.backend_scene import prepare
    settings, description, factories, report = prepare(path, args.backend)
    if args.operation != 'check':
        if args.backend != 'mjlab':
            parser.error('Explicit --backend currently supports Isaac static check only; use the existing Isaac project view for runtime')
        from shared.backends.mjlab.scene import compile_scene
        scene, model, data, keys = compile_scene(description, settings, factories)
        report.update(simulator_started=True, runtime='mjlab scene compiled with CPU MuJoCo',
                      model={'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
                             'bodies': model.nbody, 'geometries': model.ngeom})
        for entry in report['entities']:
            if keys[entry['path']] in scene.sensors:
                entry.update(sensor=type(scene.sensors[keys[entry['path']]]).__name__)
                continue
            entity = scene.entities[keys[entry['path']]]
            body = model.body(entity.root_body.name)
            entry.update(actual_position=data.xpos[body.id].tolist(),
                         actual_orientation=data.xquat[body.id].tolist(),
                         joints=list(entity.joint_names))
        if args.operation == 'view':
            import mujoco.viewer
            # Native viewer owns interaction. No policy or training is running.
            mujoco.viewer.launch(model=model, data=data)
    print(json.dumps(report, indent=2, ensure_ascii=False))
