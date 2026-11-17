"""ACT (LeRobot's implementation) as the model RLinf's SFT worker trains.

``build_model(cfg, training)`` returns a module whose ``forward(data=batch)`` normalizes a batch with the
same processor LeRobot uses and returns ACT's loss: in training L1 on the action chunk plus the VAE's KL
term, in validation the L1 alone (normalized actions, zero latent as when deployed). State and action are normalized with statistics of the training episodes only; images with
ImageNet's, the statistics the pretrained ResNet backbone expects. ``export_pretrained`` writes a LeRobot
``pretrained_model`` directory (config, weights and processors) that ``lerobot-record --policy.path`` runs.
"""
from pathlib import Path
import torch
from torch import nn

IMAGENET = {'mean': torch.tensor([.485, .456, .406]).view(3, 1, 1), 'std': torch.tensor([.229, .224, .225]).view(3, 1, 1)}
SETTINGS = ('chunk_size', 'n_action_steps', 'vision_backbone', 'dim_model', 'n_heads', 'dim_feedforward',
            'n_encoder_layers', 'n_decoder_layers', 'use_vae', 'latent_dim', 'n_vae_encoder_layers', 'dropout', 'kl_weight')


def act_config(cfg, training):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    features = training.meta.features
    model = cfg.actor.model
    cameras = list(model.get('cameras') or training.meta.camera_keys)
    image = lambda key: PolicyFeature(type=FeatureType.VISUAL, shape=(features[key]['shape'][2], *features[key]['shape'][:2]))
    inputs = {key: image(key) for key in cameras}
    inputs['observation.state'] = PolicyFeature(type=FeatureType.STATE, shape=tuple(features['observation.state']['shape']))
    outputs = {'action': PolicyFeature(type=FeatureType.ACTION, shape=tuple(features['action']['shape']))}
    return ACTConfig(input_features=inputs, output_features=outputs, device='cuda' if torch.cuda.is_available() else 'cpu',
                     **{name: model[name] for name in SETTINGS if name in model})


def training_statistics(training, config):
    """Mean/std/min/max of state and action over the training episodes' frames; ImageNet for images."""
    columns = training.hf_dataset.with_format('torch')
    stats = {}
    for key in ('observation.state', 'action'):
        values = torch.stack(list(columns[key])).float()
        stats[key] = {'mean': values.mean(0), 'std': values.std(0).clamp_min(1e-4),
                      'min': values.amin(0), 'max': values.amax(0)}
    for key in config.image_features:
        stats[key] = {name: value.clone() for name, value in IMAGENET.items()}
    return stats


class ACTTraining(nn.Module):
    def __init__(self, config, stats):
        super().__init__()
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
        self.policy = ACTPolicy(config)
        self.config, self.stats = config, stats
        self.preprocessor, self.postprocessor = make_act_pre_post_processors(config, dataset_stats=stats)

    def forward(self, data):
        batch = self.preprocessor(data)
        if self.training:
            loss, _ = self.policy.forward(batch)
            return loss
        # Validation: L1 of the action chunk as deployed (no VAE encoder, zero latent). LeRobot's forward
        # adds the KL term, which needs the encoder's output and fails in eval mode.
        batch['observation.images'] = [batch[key] for key in self.config.image_features]
        actions = self.policy.model(batch)[0]
        valid = ~batch['action_is_pad'].unsqueeze(-1)
        return (torch.nn.functional.l1_loss(actions, batch['action'], reduction='none') * valid).mean()


def build_model(cfg, training):
    config = act_config(cfg, training)
    model = ACTTraining(config, training_statistics(training, config))
    weights = cfg.experiment.get('initial_weights')
    if weights:
        model.policy.load_state_dict(_policy_weights(weights), strict=True)
    output = Path(cfg.experiment.output)
    if output.is_dir():  # the run directory: config and processors for export_pretrained
        _save_processors(model, output / 'lerobot')
    return model


def _policy_weights(path):
    """ACT weights from a LeRobot pretrained_model directory or an act_*.pt this project saved (weights only)."""
    path = Path(path)
    if path.is_dir():
        from safetensors.torch import load_file
        return load_file(str(path / 'model.safetensors'))
    state = torch.load(path, map_location='cpu', weights_only=True)
    return {key.removeprefix('policy.'): value for key, value in state.items() if key.startswith('policy.')}


def _save_processors(model, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model.config._save_pretrained(directory)
    model.preprocessor.save_pretrained(directory)
    model.postprocessor.save_pretrained(directory)


def export_pretrained(run, checkpoint='act_best.pt', output=None):
    """A LeRobot pretrained_model directory from a training run (RUN/lerobot plus RUN/<checkpoint>)."""
    import shutil
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    run = Path(run)
    output = Path(output) if output else run / 'pretrained_model'
    config = PreTrainedConfig.from_pretrained(run / 'lerobot')
    policy = ACTPolicy(config)
    policy.load_state_dict(_policy_weights(run / checkpoint), strict=True)
    policy.save_pretrained(output)
    for item in (run / 'lerobot').iterdir():
        if item.name != 'config.json':
            shutil.copy2(item, output / item.name)
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export a trained run as a LeRobot pretrained_model directory')
    parser.add_argument('run', type=Path)
    parser.add_argument('--checkpoint', default='act_best.pt')
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    print(export_pretrained(arguments.run, arguments.checkpoint, arguments.output))
