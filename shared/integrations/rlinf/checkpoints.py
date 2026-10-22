"""Workspace checkpoint retention for RLinf embodied training."""

import shutil
from pathlib import Path

from rlinf.runners.embodied_runner import EmbodiedRunner


class RetainingEmbodiedRunner(EmbodiedRunner):
    """Use RLinf's saver, then retain only the configured recent checkpoints."""

    def _save_checkpoint(self):
        super()._save_checkpoint()
        self._prune_checkpoints()

    def _prune_checkpoints(self):
        keep = int(self.cfg.runner.get("keep_last_checkpoints", -1))
        if keep < 0:
            return
        if keep == 0:
            raise ValueError("runner.keep_last_checkpoints must be at least 1")

        root = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "checkpoints"
        )
        checkpoints = []
        for path in root.glob("global_step_*"):
            try:
                step = int(path.name.removeprefix("global_step_"))
            except ValueError:
                continue
            if path.is_dir():
                checkpoints.append((step, path))

        for _, path in sorted(checkpoints)[:-keep]:
            shutil.rmtree(path)
            self.logger.info(f"Removed superseded checkpoint {path}.")
