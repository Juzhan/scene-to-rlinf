"""Only dataset/model/loss/evaluation hooks; RLinf owns gradient updates."""
import json
from pathlib import Path
import torch
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from importlib import import_module


class SupervisedWorker(FSDPSftWorker):
    def build_dataloader(self, data_paths, eval_dataset=False):
        datasets, _ = import_module(self.cfg.sft_project.package + '.data').load_dataset(data_paths)
        loader = torch.utils.data.DataLoader(
            datasets['validation' if eval_dataset else 'train'],
            batch_size=self.eval_batch_size if eval_dataset else self.micro_batch_size,
            shuffle=not eval_dataset, generator=torch.Generator().manual_seed(self.cfg.actor.seed))
        return loader, None

    def model_provider_func(self):
        return import_module(self.cfg.sft_project.package + '.policy').build_model(
            self.cfg, self.data_loader.dataset)

    def init_worker(self):
        if self._world_size != 1:
            raise ValueError('This project currently supports a single SFT actor rank')
        torch.manual_seed(self.cfg.actor.seed)
        super().init_worker()
        self.output = Path(self.cfg.experiment.output)
        self.best_loss = float('inf')
        self.history = []
        self.run_eval()

    def get_train_model_output(self, batch):
        with self.amp_context:
            loss = self.model(data=batch)
        return loss, {'loss': loss.detach().item()}

    def get_eval_model_output(self, batch):
        with torch.no_grad(), self.amp_context:
            return self.model(data=batch)

    def run_eval(self):
        with self.worker_timer():
            return self._evaluate()

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total, count = 0., 0
        for batch in self.eval_data_loader:
            samples = len(next(iter(batch.values())))
            total += self.get_eval_model_output(batch).item() * samples
            count += samples
        loss = total / count
        if not torch.isfinite(torch.tensor(loss)):
            raise ValueError('Non-finite validation loss')
        state = self.get_model_state_dict(cpu_offload=True, full_state_dict=True)
        torch.save(state, self.output / (self.cfg.sft_project.export_prefix + '_last.pt'))
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(state, self.output / (self.cfg.sft_project.export_prefix + '_best.pt'))
        self.history.append({'step': self.optimizer_steps, 'validation_mse': loss,
                             'learning_rate': self.optimizer.param_groups[0]['lr']})
        (self.output / 'validation.json').write_text(json.dumps(self.history, indent=2))
        return {'val_loss': loss, 'best_val_loss': self.best_loss}
