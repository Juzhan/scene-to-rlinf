import json
import urllib.request

import pytest
from omegaconf import OmegaConf

from tools.board import board_directory, start_board, stop_board


def config(path, backends=('tensorboard',)):
    return OmegaConf.create({'runner': {'logger': {'log_path': str(path), 'logger_backends': list(backends)}}})


def test_board_reads_all_runs_of_a_frozen_experiment(tmp_path):
    assert board_directory(config(tmp_path / 'runs' / 'b')) == tmp_path / 'runs'
    assert board_directory(config(tmp_path / 'plain')) == tmp_path / 'plain'
    with pytest.raises(ValueError, match='tensorboard'):
        board_directory(config(tmp_path, ['wandb']))


def test_board_serves_runs_and_stops(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    for run in ('a', 'b'):
        writer = SummaryWriter(tmp_path / 'runs' / run / 'tensorboard')
        writer.add_scalar('rollout/rewards', 1., 0)
        writer.close()
    cfg = config(tmp_path / 'runs' / 'b')
    process = start_board(cfg, 6106)
    assert process is not None
    try:
        port = process.args[process.args.index('--port') + 1]
        runs = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/data/runs', timeout=10).read())
        assert runs == ['a/tensorboard', 'b/tensorboard']
    finally:
        stop_board(process, cfg)
    assert process.returncode is not None
