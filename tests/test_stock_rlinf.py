"""Run in a fresh process against the unmodified dependency, including wire routing."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_stock_rlinf_named_observations_and_ppo_update():
    code = r'''
import run
import torch
from pathlib import Path
import rlinf.data.schema.embodied_types as transport
assert Path(transport.__file__).is_relative_to(run.RLINF)
assert run.RLINF.name == 'RLinf-upstream'
import sys
cfg, project = run.compose_project('assembly_lift')
project.check(cfg)
assert 'shared.assembly.legacy.task' not in sys.modules
from shared.integrations.rlinf.upstream import validate_runtime
validate_runtime(cfg)
from omegaconf import open_dict
with open_dict(cfg): cfg.algorithm.adaptive_kl_target = .01
try:
    validate_runtime(cfg)
except ValueError as error:
    assert 'legacy adaptive-KL' in str(error)
else:
    raise AssertionError('Legacy training extension was silently ignored')
from shared.integrations.rlinf.observations import pack, fields, copy_rows, same_schema
from shared.integrations.rlinf.object_policy import ObjectPolicy
from shared.assembly.objects import ObjectView
from rlinf.data.schema.embodied_types import EnvOutput, PolicyInput, Trajectory
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(5, 2)
        self.value = torch.nn.Linear(5, 1)
        self.actor_logstd = torch.nn.Parameter(torch.zeros(2))
    def forward(self, obs):
        assert obs.camera.rgb.dtype == torch.uint8
        x = torch.cat([obs.robot.position, obs.camera.rgb.float().mean((1,2))[:,:2]], -1)
        return self.actor(x), self.value(x)
raw = {'robot.position': torch.randn(3,3), 'camera.rgb': torch.randint(255,(3,8,10,3),dtype=torch.uint8)}
packed = pack(raw, 3, 'test task')
assert packed['states']['camera.rgb'] is raw['camera.rgb']
wire = EnvOutput(obs=packed, final_obs=packed).to_dict()
assert wire['obs']['states']['camera.rgb'].shape == (3,8,10,3)
assert transport._observation_batch_size(wire['obs']) == 3
# Stock routing merges optional flags and splits nested sensor tensors.
a = PolicyInput(obs=wire['obs'], intervene_flags=torch.ones(3,1,dtype=torch.bool))
b = PolicyInput(obs=wire['obs'])
merged = PolicyInput.merge([a,b])
assert merged.obs['states']['camera.rgb'].shape == (6,8,10,3)
assert not merged.intervene_flags[3:].any()
model = ObjectPolicy(Model(), ObjectView)
first, rollout = model.predict_action_batch(wire['obs'], do_sample=False)
second, _ = model.predict_action_batch(wire['obs'], do_sample=False)
torch.testing.assert_close(first, second)
# Exercise the actual unmodified rollout dispatch, without launching Ray/GPU.
from types import SimpleNamespace
from inspect import unwrap
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
worker = SimpleNamespace(cfg=cfg, model_cfg=cfg.actor.model, algorithm_cfg=cfg.algorithm,
                         expert_model=None, enable_dagger=False, hf_model=model)
MultiStepRolloutWorker.setup_sample_params(worker)
predict = unwrap(MultiStepRolloutWorker.predict)
evaluated, _ = predict(worker, wire['obs'], mode='eval')
torch.testing.assert_close(first, evaluated)
# Exercise the actual evaluation merger with multiple producer batches.
worker._merge_optional_flag_tensors = MultiStepRolloutWorker._merge_optional_flag_tensors.__get__(worker)
eval_wire = MultiStepRolloutWorker._merge_obs_batches(worker, [wire, wire])
eval_action, _ = predict(worker, eval_wire['obs'], mode='eval')
torch.testing.assert_close(eval_action, torch.cat([first, first]))
assert fields(eval_wire['obs'])['camera.rgb'].shape == (6,8,10,3)
random_action, rollout = predict(worker, wire['obs'], mode='train')
assert not torch.equal(first, random_action)
trajectory = Trajectory(forward_inputs={k:v[None] for k,v in rollout['forward_inputs'].items()})
batch = Trajectory.to_batch([trajectory,trajectory])
assert batch['forward_inputs']['obs/camera.rgb'].shape == (1,6,8,10,3)
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
before = model.model.actor.weight.detach().clone()
result = model(forward_inputs=rollout['forward_inputs'])
loss = -result['logprobs'].mean() + result['values'].square().mean()
loss.backward(); optimizer.step()
assert not torch.equal(before, model.model.actor.weight)
assert all(torch.isfinite(p).all() for p in model.parameters())
assert same_schema(wire['obs'], wire['final_obs'])
other = pack({k:torch.zeros_like(v) for k,v in raw.items()}, 3)
copy_rows(other, packed, torch.tensor([True,False,True]))
assert not other['states']['robot.position'][1].any()
torch.testing.assert_close(other['states']['robot.position'][0],raw['robot.position'][0])
print('stock transport, routing, deterministic evaluation and gradient update passed')
'''
    env = dict(os.environ, RLINF_ROOT=str(ROOT/'third_party/RLinf-upstream'))
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
