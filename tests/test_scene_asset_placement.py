"""Extra robots/objects follow terrain origins, including partial reset."""
from types import SimpleNamespace
import torch
from shared.backends.isaac.scene import reset_authored_assets

class Asset:
    def __init__(self, position, joints=False):
        state=torch.zeros(2,13); state[:,:3]=torch.tensor(position); state[:,3]=1
        self.data=SimpleNamespace(default_root_state=state,default_joint_pos=torch.ones(2,2),default_joint_vel=torch.zeros(2,2))
        self.pose=torch.full((2,7),99.); self.velocity=torch.full((2,6),99.)
        self.joints=torch.full((2,2),99.)
    def write_root_pose_to_sim(self,value,env_ids):self.pose[env_ids]=value
    def write_root_velocity_to_sim(self,value,env_ids):self.velocity[env_ids]=value
    def write_joint_state_to_sim(self,pos,vel,env_ids):self.joints[env_ids]=pos
    def set_joint_position_target(self,value,env_ids):pass
    def set_joint_velocity_target(self,value,env_ids):pass

class Scene(dict):
    pass

def test_partial_reset_of_added_assets_uses_current_terrain_origin():
    arm,box,global_object=Asset([-.5,1,0],True),Asset([0,1,.03]),Asset([5,5,0])
    scene=Scene(arm=arm,box=box,global_object=global_object)
    scene.articulations={'arm':arm}
    scene.env_origins=torch.tensor([[-28.,-60.,.286],[-20.,4.,-.369]])
    env=SimpleNamespace(scene=scene,num_envs=2,device='cpu')
    reset_authored_assets(env,torch.tensor([1]),asset_names=['arm','box'])
    torch.testing.assert_close(arm.pose[1,:3],torch.tensor([-20.5,5.,-.369]))
    torch.testing.assert_close(box.pose[1,:3],torch.tensor([-20.,5.,-.339]))
    assert (arm.pose[0]==99).all() and (box.pose[0]==99).all()
    assert (global_object.pose==99).all()
    assert (arm.joints[1]==1).all() and (arm.velocity[1]==0).all()
    # Moving curriculum origins must relocate equipment on the next reset.
    scene.env_origins[1]+=torch.tensor([8.,0.,.1])
    reset_authored_assets(env,torch.tensor([1]),asset_names=['arm','box'])
    torch.testing.assert_close(arm.pose[1,:3],torch.tensor([-12.5,5.,-.269]))
    torch.testing.assert_close(arm.data.default_root_state[1,:3],torch.tensor([-.5,1.,0.]))


def test_static_pose_reads_env0_view_for_every_environment():
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld
    origins=torch.tensor([[0.,0.,0.],[0.,4.,0.],[4.,0.,0.]])
    class View:  # Lab builds static views before cloning: env_0 or the single global prim.
        def __init__(self,path,position):self.prim_paths=[path];self.position=torch.tensor([position])
        def get_world_poses(self):return self.position,torch.tensor([[1.,0,0,0]])
    world=SimpleNamespace(num_envs=3,scene=SimpleNamespace(env_origins=origins,env_prim_paths=[f'/World/envs/env_{i}' for i in range(3)]))
    world._local_pose=lambda pose:IsaacAssemblyWorld._local_pose(world,pose)
    local=IsaacAssemblyWorld._static_pose(world,View('/World/envs/env_0/steps',[1.,2.,.1]))
    torch.testing.assert_close(local[:,:3],torch.tensor([[1.,2.,.1]]*3))
    shared=IsaacAssemblyWorld._static_pose(world,View('/World/floor',[1.,2.,.1]))
    torch.testing.assert_close(shared[:,:3],torch.tensor([1.,2.,.1])-origins)
