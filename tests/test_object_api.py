from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import torch
import pytest
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/RLinf'))
from shared.assembly.objects import ObjectView
from shared.assembly.object_schema import generate
from shared.assembly.task_memory import TaskMemory


def test_typed_access_is_lazy_zero_copy_and_respects_selection():
    class Robot(ObjectView):
        position: torch.Tensor
    class Obs(ObjectView):
        robot: Robot
    value = torch.randn(3,3)
    calls=[]
    obs=Obs(reader=lambda name: calls.append(name) or value,paths=['robot.position','robot.secret'])
    assert not calls
    assert isinstance(obs.robot, Robot)
    assert obs.robot.position is value
    assert obs.robot.position is value
    assert calls == ['robot.position']
    with pytest.raises(AttributeError): _ = obs.robot.secret
    with pytest.raises(AttributeError): obs.robot.position = value


def test_generated_template_has_types_task_and_refreshes_without_touching_code(tmp_path):
    from tools.create import create_project
    target, files = create_project('typed_example', destination=tmp_path)
    assert {'task.py','observation.py'} <= set(files)
    initial=(target/'policy.py').read_bytes()
    task=(target/'task.py').read_bytes()
    scene=yaml.safe_load((target/'scene.yaml').read_text())
    assert 'variables' not in scene and 'task' not in scene
    scene['control']['components']['executor']['observations'].remove('robot.angular_velocity')
    (target/'scene.yaml').write_text(yaml.safe_dump(scene,sort_keys=False))
    generate(target/'scene.yaml')
    source=(target/'observation.py').read_text()
    tree={}
    exec(compile(source,'observation.py','exec'),tree)
    assert 'angular_velocity' not in tree['ExecutorObs_robot'].__annotations__
    assert 'angular_velocity' in tree['SceneInfo_robot'].__annotations__
    assert (target/'policy.py').read_bytes()==initial
    assert (target/'task.py').read_bytes()==task


def test_object_ppo_keeps_images_and_nested_rollout_inputs():
    from torch import nn
    from shared.integrations.rlinf.object_policy import ObjectPolicy
    class Camera(ObjectView):
        rgb: torch.Tensor
    class Robot(ObjectView):
        joint_position: torch.Tensor
    class Obs(ObjectView):
        camera: Camera
        robot: Robot
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor=nn.Linear(5,2)
            self.critic=nn.Linear(5,1)
            self.actor_logstd=nn.Parameter(torch.zeros(2))
        def forward(self, obs: Obs):
            assert obs.camera.rgb.dtype==torch.uint8
            assert obs.camera.rgb.shape[1:]==(8,10,3)
            x=torch.cat((obs.camera.rgb.float().mean((1,2))/255,obs.robot.joint_position),-1)
            return self.actor(x),self.critic(x)
    policy=ObjectPolicy(Model(),Obs,True)
    raw={'camera.rgb':torch.randint(256,(4,8,10,3),dtype=torch.uint8),'robot.joint_position':torch.randn(4,2)}
    action,result=policy.predict_action_batch(raw)
    assert result['forward_inputs']['obs/camera.rgb'] is raw['camera.rgb']
    updated=policy(forward_inputs=result['forward_inputs'])
    torch.testing.assert_close(updated['logprobs'],result['prev_logprobs'][:,0])
    torch.testing.assert_close(updated['values'],result['prev_values'])
    (-updated['logprobs'].mean()+updated['values'].square().mean()).backward()
    assert policy.model.actor.weight.grad.abs().sum()>0
    from rlinf.data.schema.embodied_types import Trajectory
    trajectory = Trajectory(forward_inputs={key: value[None] for key,value in result['forward_inputs'].items()})
    batch = Trajectory.to_batch([trajectory, trajectory])
    assert batch['forward_inputs']['obs/camera.rgb'].shape == (1,8,8,10,3)
    # RLinf's actual time/batch utility preserves pixel axes.
    from rlinf.utils.nested_dict_process import flatten_nested_tensor_time_batch
    time_inputs={key: value[None].repeat(2,*([1]*value.ndim)) for key,value in raw.items()}
    packed=flatten_nested_tensor_time_batch({'observations':time_inputs})
    assert packed['observations']['camera.rgb'].shape==(8,8,10,3)
    assert packed['observations']['camera.rgb'].dtype==torch.uint8


def test_memory_partial_reset_does_not_mix_episodes():
    memory=TaskMemory()
    torch.testing.assert_close(memory.mean('x',torch.tensor([2.,4.]),2),torch.tensor([2.,4.]))
    memory.mean('x',torch.tensor([4.,8.]),2)
    memory.reset(torch.tensor([0]))
    torch.testing.assert_close(memory.mean('x',torch.tensor([10.,12.]),2),torch.tensor([10.,10.]))
    held=memory.hold('h',torch.tensor([True,False]),torch.tensor([.2,.2]))
    torch.testing.assert_close(held,torch.tensor([.2,0.]))


def test_task_runtime_is_once_per_step_lazy_and_resettable(monkeypatch):
    from shared.backends.isaac.python_task import Runtime
    class Task:
        def __init__(self):self.memory=TaskMemory();self.calls=0
        def evaluate(self,obs):
            self.calls+=1
            return {'reward':obs.robot.position[:,0], 'metrics':{'mean':self.memory.mean('x',obs.robot.position[:,0])}}
    task=Task()
    modules={'example.task':SimpleNamespace(Task=lambda:task),
             'example.observation':SimpleNamespace(SceneInfo=ObjectView,SCENE_FIELDS=('robot.position','camera.rgb'))}
    monkeypatch.setattr('shared.backends.isaac.python_task.import_module',modules.__getitem__)
    env=SimpleNamespace(num_envs=2,device='cpu',common_step_counter=1,extras={})
    runtime=Runtime(env,{'module':'example'},{})
    value=torch.tensor([[1.,0.,0.],[2.,0.,0.]])
    from shared.assembly.signals import Signal
    runtime.signals.register('robot.position',Signal(lambda:value))
    runtime.compute();runtime.compute()
    assert task.calls==1 and 'camera.rgb' not in runtime.signals
    runtime.reset(torch.tensor([0]));env.common_step_counter=2
    value[:,0]=torch.tensor([10.,4.])
    result=runtime.compute()
    torch.testing.assert_close(result['metrics']['mean'],torch.tensor([10.,3.]))


def test_real_python_tasks_read_scene_objects():
    from projects.assembly_go2_navigation.task import evaluate
    from projects.assembly_go2_navigation.observation import SceneInfo
    data={'robot.position':torch.zeros(2,3),'robot.orientation':torch.tensor([[1.,0,0,0]]).repeat(2,1),
          'destination.position':torch.tensor([[1.,0,0],[0.,0,0]]), 'destination.orientation':torch.tensor([[1.,0,0,0]]).repeat(2,1),
          'contact_forces.contact_force_samples':torch.zeros(2,3,1,3),'planner.output':torch.zeros(2,3),
          'planner.previous_output':torch.zeros(2,3)}
    result=evaluate(SceneInfo(data))
    assert result['metrics']['reached'].tolist()==[False,True]
    assert not result['failure'].any()


def test_rlinf_transport_keeps_custom_fields_including_terminal_images():
    from rlinf.data.schema.embodied_types import EnvOutput, PolicyInput
    from rlinf.utils.obs_compression import infer_obs_batch_size
    raw={'camera.rgb':torch.randint(256,(3,4,5,3),dtype=torch.uint8),'robot.position':torch.randn(3,3)}
    from shared.integrations.rlinf.observations import pack
    wire=pack(raw,3)
    packed=EnvOutput(obs=wire, final_obs=wire).to_dict()
    assert infer_obs_batch_size(packed['obs'])==3
    for key,value in raw.items():
        torch.testing.assert_close(packed['obs']['states'][key],value)
        torch.testing.assert_close(packed['final_obs']['states'][key],value)
    parts=PolicyInput(obs=packed['obs']).split([1,2])
    assert parts[1].obs['states']['camera.rgb'].shape==(2,4,5,3)
