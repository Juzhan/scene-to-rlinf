"""Thin project entry for supervised training and optional online evaluation.

Projects supply data.load_dataset/validate and policy.build_model. No simulator
or training worker is imported during configuration discovery.
"""
import json
from importlib import import_module
from pathlib import Path
from omegaconf import OmegaConf, open_dict


class SftProject:
    def __init__(self, source, package, *, model_type, export_prefix='model', online=False):
        self.online = online
        self.directory = Path(source).resolve().parent
        self.package, self.model_type, self.export_prefix = package, model_type, export_prefix

    def register(self):
        from rlinf.config import SupportedModel
        SupportedModel.register(self.model_type, force=True)

    def prepare(self, cfg):
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        def local_path(value):
            path = Path(value).expanduser()
            return str((path if path.is_absolute() else self.directory / path).resolve())
        with open_dict(cfg):
            for name in ('train_data_paths', 'val_data_paths'):
                cfg.data[name] = local_path(cfg.data[name])
            if cfg.data.train_data_paths != cfg.data.val_data_paths:
                raise ValueError('Use one manifest with disjoint train/validation episodes')
            cfg.experiment.output = local_path(cfg.experiment.output)
            if cfg.experiment.get('initial_weights'):
                cfg.experiment.initial_weights = local_path(cfg.experiment.initial_weights)
            cfg.sft_project = {'package': self.package, 'export_prefix': self.export_prefix}
            cfg.actor.model.model_type = self.model_type
        data = import_module(self.package + '.data')
        datasets, manifest = data.load_dataset(cfg.data.train_data_paths)
        data.validate(cfg, datasets, manifest)
        if cfg.actor.global_batch_size < 1 or cfg.actor.micro_batch_size < 1 or cfg.actor.global_batch_size % cfg.actor.micro_batch_size:
            raise ValueError('Global batch must be a positive multiple of micro batch')
        if cfg.runner.max_epochs < 1 and cfg.runner.max_steps < 1:
            raise ValueError('Set a positive epoch or step budget')
        if cfg.runner.get('resume_dir'):
            raise ValueError('Full training resume is not supported by this SFT adapter; initial_weights is weights-only')
        if cfg.experiment.get('initial_weights'):
            import_module(self.package + '.policy').build_model(cfg, datasets['train'])
        return cfg, datasets, manifest

    def run_recipe(self, cfg, operation, *, options=None):
        if self.online and operation in ("evaluate", "view", "record", "smoke"):
            from .online_evaluation import run
            return run(self, cfg, operation, options)
        if operation not in ('check', 'train'):
            raise ValueError('Offline supervised projects support check/train')
        cfg, datasets, manifest = self.prepare(cfg)
        summary = {'project': self.package.rsplit('.', 1)[-1],
                   'training_samples': len(datasets['train']), 'validation_samples': len(datasets['validation']),
                   'episodes': len(manifest['episodes']), 'simulator_started': False,
                   'output': cfg.experiment.output}
        if operation == 'check':
            if self.online:
                from .online_evaluation import prepare
                online = prepare(self, cfg)
                summary['online'] = {'scene': online.scene.suite, 'task_id': online.scene.task_id,
                                     'models': import_module(self.package + '.policy').policy_artifacts(online)}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return summary
        import ray
        from rlinf.runners.sft_runner import SFTRunner
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement
        from .sft_worker import SupervisedWorker
        self.register()
        output = Path(cfg.experiment.output)
        output.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
        (output / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2))
        try:
            cluster = Cluster(cluster_cfg=cfg.cluster)
            placement = HybridComponentPlacement(cfg, cluster)
            actor = SupervisedWorker.create_group(cfg).launch(
                cluster, name=cfg.actor.group_name, placement_strategy=placement.get_strategy('actor'))
            runner = SFTRunner(cfg=cfg, actor=actor)
            runner.init_workers()
            runner.run()
        finally:
            ray.shutdown()
