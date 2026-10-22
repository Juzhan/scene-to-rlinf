"""Backend-selectable project check and scene-only native inspection/view."""
import argparse
import json
from pathlib import Path

from shared.assembly.assets import BACKENDS, backend_module


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('check', 'inspect', 'view'))
    parser.add_argument('project')
    parser.add_argument('--backend', required=True, choices=BACKENDS)
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
        inspect = getattr(backend_module(args.backend, 'project'), 'inspect_scene', None)
        if inspect is None:
            parser.error(f'Explicit --backend {args.backend} supports static check only; '
                         'use run.py inspect/view <project> for the runtime')
        inspect(description, settings, factories, report, view=args.operation == 'view')
    print(json.dumps(report, indent=2, ensure_ascii=False))
