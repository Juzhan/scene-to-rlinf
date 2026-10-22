"""Project models receive object observations; tensor layout is local to each model."""
import sys
from pathlib import Path
import importlib
import pytest
import torch
import yaml
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/RLinf'))
from shared.assembly.objects import selected_fields
from shared.integrations.rlinf.object_policy import ObjectPolicy


def samples(project, role=None):
    settings=yaml.safe_load((Path('projects')/project/'scene.yaml').read_text())
    fields=selected_fields(settings)
    role=role or settings.get('control',{}).get('train','policy')
    data={}
    for path in fields[role]:
        field=path.split('.')[-1]
        if field=='orientation': shape=(4,)
        elif field=='heading':shape=(1,)
        elif 'joint_' in field:shape=(8 if path.startswith('piperx.') else 9 if project=='assembly_lift' else 12,)
        elif field=='ray_hit_positions':shape=(187,3)
        elif field=='frame_positions':shape=(1,3)
        elif path=='planner.output':shape=(4 if project=='planner_executor' else 3,)
        elif path=='executor.output':shape=(12,)
        else:shape=(3,)
        data[path]=torch.randn(4,*shape)
        if field=='orientation': data[path]=torch.nn.functional.normalize(data[path],dim=-1)
    return data


@pytest.mark.parametrize('project,actions',[('planner_executor',12),('assembly_go2_navigation',3),('assembly_lift',8),('assembly_go2',12)])
def test_object_models_keep_rollout_and_update_consistent(project,actions):
    module=importlib.import_module('projects.'+project+'.policy')
    schema=importlib.import_module('projects.'+project+'.observation')
    config=OmegaConf.create({'hidden_dims':[16], 'action_dim':actions})
    model=(module.Executor if project=='planner_executor' else module.Policy)(config)
    data=samples(project)
    model(schema.Obs(data))
    adapter=ObjectPolicy(model,schema.Obs,True)
    before={key:value.clone() for key,value in data.items()}
    _,rollout=adapter.predict_action_batch(data)
    update=adapter(forward_inputs=rollout['forward_inputs'])
    torch.testing.assert_close(update['logprobs'],rollout['prev_logprobs'][:,0])
    torch.testing.assert_close(update['values'],rollout['prev_values'])
    for key in data:torch.testing.assert_close(data[key],before[key])
    (-update['logprobs'].mean()+update['values'].square().mean()).backward()
    assert model.actor[0].weight.grad.abs().sum()>0


def test_frozen_executor_accepts_object_and_validates_checkpoint_features():
    from projects.assembly_go2_navigation.policy import build_executor
    from projects.assembly_go2_navigation.observation import ExecutorObs
    checkpoint=Path('outputs/go2_locomotion_rsl_4096/rlinf_weights.pt')
    if not checkpoint.exists():pytest.skip('local reference checkpoint unavailable')
    model=build_executor(checkpoint)
    data=samples('assembly_go2_navigation','executor')
    a=model(ExecutorObs(data));b=model(ExecutorObs(data))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert a.shape==(4,12) and all(not p.requires_grad for p in model.parameters())
    data['height_scanner.ray_hit_positions']=data['height_scanner.ray_hit_positions'][:,:10]
    with pytest.raises(ValueError,match='187'):model(ExecutorObs(data))


@pytest.mark.parametrize('project', ['assembly_go2', 'planner_executor'])
def test_executor_distribution_uses_actual_joint_units(project):
    module = importlib.import_module('projects.' + project + '.policy')
    schema = importlib.import_module('projects.' + project + '.observation')
    model = module.build_executor(OmegaConf.create({'hidden_dims': [16], 'action_dim': 12}), torch.float32)
    data = samples(project)
    model(schema.Obs(data))
    with torch.no_grad():
        for parameter in model.actor.parameters(): parameter.zero_()
    adapter = ObjectPolicy(model, schema.Obs)
    distribution, _ = adapter._evaluate(data)
    torch.testing.assert_close(distribution.mean, data['robot.default_joint_position'])
    torch.testing.assert_close(distribution.stddev, torch.full((4, 12), .25))
    action, _ = adapter.predict_action_batch(data, mode='eval')
    torch.testing.assert_close(action[:, 0], data['robot.default_joint_position'])


def test_frozen_executor_outputs_physical_angles_with_original_actor_history():
    from projects.assembly_go2_navigation.policy import LocomotionNetwork
    from projects.assembly_go2_navigation.observation import ExecutorObs
    model = LocomotionNetwork()
    data = samples('assembly_go2_navigation', 'executor')
    data['executor.output'] = data['robot.default_joint_position'].clone()
    captured = []
    handle = model[0].register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    output = model(ExecutorObs(data))
    handle.remove()
    torch.testing.assert_close(captured[0][:, 36:48], torch.zeros(4, 12))
    raw = torch.nn.Sequential.forward(model, captured[0])
    torch.testing.assert_close(output, data['robot.default_joint_position'] + .25 * raw)
