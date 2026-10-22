"""Small single-environment evaluation driver for data-trained project policies.

Does not implement training, and does not depend on Isaac or RLinf workers.
Projects supply env.make_environment and policy.build_policy.
"""
from importlib import import_module
from pathlib import Path
import json
import hashlib
import numpy as np
from omegaconf import OmegaConf


def prepare(project, cfg, *, checkpoint=None):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    output = Path(cfg.evaluation.output).expanduser()
    cfg.evaluation.output = str((output if output.is_absolute() else project.directory / output).resolve())
    import_module(project.package + '.env').validate(cfg)
    if cfg.evaluation.episodes < 1:
        raise ValueError('Evaluation episodes must be positive')
    # Component names and model formats belong to the project's policy.
    import_module(project.package + '.policy').prepare_policy(cfg, project.directory, checkpoint=checkpoint)
    return cfg



def create_output(directory, operation):
    """Allow repeated previews/evaluations without overwriting earlier results."""
    from datetime import datetime
    from tempfile import mkdtemp

    output = Path(directory)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not output.is_dir():
            raise
        prefix = f"{output.name}-{operation}-{datetime.now():%Y%m%d-%H%M%S}-"
        output = Path(mkdtemp(prefix=prefix, dir=output.parent))
    return output


def run(project, cfg, operation, options=None):
    cfg = prepare(project, cfg, checkpoint=options.checkpoint if options is not None else None)
    if operation == "smoke":
        cfg.evaluation.episodes = 1
    if operation == "view":
        import os
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("view requires a desktop DISPLAY; use record for headless video")
    if options is not None:
        if options.num_envs != 1:
            raise ValueError('MuJoCo project preview currently supports one environment')
        if options.camera_eye or options.camera_target:
            raise ValueError('This MuJoCo preview uses the scene cameras')
    output = create_output(cfg.evaluation.output, operation)
    cfg.evaluation.output = str(output)
    print(f'Results: {output}', flush=True)
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
    provenance = {}
    for label, filename in import_module(project.package + '.policy').policy_artifacts(cfg).items():
        if filename:
            with Path(filename).open('rb') as stream:
                provenance[label] = {'path': str(filename), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    (output / 'models.json').write_text(json.dumps(provenance, indent=2))
    environment = None
    results = []
    mode = options.actions if options is not None and getattr(options, 'actions_explicit', False) else 'policy'
    policy = import_module(project.package + '.policy').build_policy(cfg) if mode == 'policy' else None
    try:
        environment = import_module(project.package + '.env').make_environment(cfg)
        for episode in range(cfg.evaluation.episodes):
            seed = cfg.evaluation.seed + episode
            observation = environment.reset(seed)
            if hasattr(environment, 'observation_catalog'):
                (output / 'observations.json').write_text(json.dumps(environment.observation_catalog(), ensure_ascii=False, indent=2))
            if policy is not None:
                policy.reset(seed)
            random = np.random.default_rng(seed)
            writer = None
            cancelled = False
            try:
                if operation == 'record' or cfg.evaluation.video:
                    import imageio.v2 as imageio
                    writer = imageio.get_writer(output / f'episode_{seed}.mp4', fps=environment.control_hz, codec='libx264')
                while True:
                    if writer is not None or operation == 'view':
                        frame = environment.render()
                        if writer is not None:
                            writer.append_data(frame)
                        if operation == 'view':
                            import cv2
                            cv2.imshow(project.package.rsplit('.', 1)[-1], frame[..., ::-1])
                            if cv2.waitKey(1) & 0xff in (27, ord('q')):
                                cancelled = True
                                break
                    action = (policy.act(observation) if policy is not None else np.zeros(environment.action_shape, np.float32)
                              if mode == 'zero' else random.uniform(-options.random_scale, options.random_scale, environment.action_shape).astype(np.float32))
                    observation, terminated, truncated, info = environment.step(action)
                    preview_limit = (operation == 'smoke' and info['steps'] >= 2) or (
                        options is not None and options.steps > 0 and info['steps'] >= options.steps)
                    if terminated or truncated or preview_limit:
                        if writer is not None:
                            writer.append_data(environment.render())
                        results.append(dict(info, seed=seed, terminated=terminated, truncated=bool(truncated or preview_limit)))
                        break
            finally:
                if writer is not None:
                    writer.close()
            report = {'episodes': results, 'completed_episodes': len(results), 'cancelled': cancelled,
                      'success_rate': sum(row['is_success'] for row in results) / len(results) if results else None}
            (output / 'results.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if cancelled:
                break
        return report
    finally:
        if environment is not None:
            environment.close()
        if operation == 'view':
            import cv2
            cv2.destroyAllWindows()
