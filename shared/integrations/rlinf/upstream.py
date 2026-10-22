"""Reject legacy training extensions instead of silently ignoring their settings."""
def validate_runtime(cfg):
    from pathlib import Path
    from rlinf.algorithms import utils
    legacy_root = Path(__file__).resolve().parents[3] / 'third_party/RLinf'
    historical_runtime = Path(utils.__file__).resolve().is_relative_to(legacy_root)
    algorithm, runner, actor = cfg.get('algorithm', {}), cfg.get('runner', {}), cfg.get('actor', {})
    for environment in cfg.get('env', {}).values():
        if not hasattr(environment, 'get'):
            continue
        if (environment.get('env_type') == 'frankasim' and not historical_runtime
                and any(environment.get(name) is not None for name in ('render_size', 'resize_to'))):
            raise ValueError('This FrankaSim recipe uses legacy render_size/resize_to extensions. '
                             'Select the historical RLINF_ROOT until its image adapter is migrated.')
    if algorithm.get('loss_type') == 'embodied_sac' and runner.get('resume_dir') and not historical_runtime:
        raise ValueError('Historical SAC resume depends on its critic update-counter patch; '
                         'select the historical RLINF_ROOT. Stock SAC resume is not verified here.')
    if algorithm.get('adaptive_kl_target') is not None and not hasattr(utils, 'adaptive_kl_learning_rate'):
        raise ValueError('This recipe requires the legacy adaptive-KL worker patch. '
                         'Use native RLinf learning-rate settings (remove adaptive_kl_target), '
                         'or explicitly select the historical RLINF_ROOT for replay.')
    if runner.get('max_train_samples', -1) > 0:
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
        if not hasattr(AsyncEmbodiedRunner, '_sample_budget_reached'):
            raise ValueError('max_train_samples requires the historical patched RLinf; '
                             'stock RLinf uses max_epochs/max_steps.')
    if actor.get('min_new_samples_per_training_step', 0) > 0:
        from rlinf.workers.actor.async_fsdp_sac_policy_worker import AsyncEmbodiedSACFSDPPolicy
        if not hasattr(AsyncEmbodiedSACFSDPPolicy, '_wait_for_new_samples'):
            raise ValueError('min_new_samples_per_training_step requires the historical patched RLinf.')
