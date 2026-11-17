"""Guard ordering and safe-boundary persistence checks; no simulator."""
from pathlib import Path
from types import SimpleNamespace
import ast,json,tempfile,os
import torch
from fall_recovery import preserve_boundary,check_after_physics
from episode_semantics import EnvironmentEpisodeFault

def main():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);s={'buffer':{'dones':[True]},'steps':10}
        torch.save(s,p/'session_state.pt');assert preserve_boundary(p,s)
        before=(p/'recovery_boundary.pt').read_bytes()
        torch.save({'buffer':{'dones':[False]},'steps':11},p/'next.tmp')
        (p/'next.tmp').replace(p/'session_state.pt')
        assert not preserve_boundary(p,{'buffer':{'dones':[False]}})
        assert (p/'recovery_boundary.pt').read_bytes()==before
        state=SimpleNamespace(root_pos=torch.ones(4,3),dof_pos=torch.zeros(4,69))
        world=SimpleNamespace(dt=1/30,age=torch.ones(4));policy=SimpleNamespace(steps=10,initial_steps=10,updates=1)
        check_after_physics(world,state,torch.zeros(4,69),torch.zeros(4,dtype=torch.bool),1,None,p,policy,{})
        assert not (p/'FALL_PHYSICS_SNAPSHOT.pt').exists()
        try:check_after_physics(world,state,torch.zeros(4,69),torch.tensor([True,False,False,False]),2,None,p,policy,{})
        except EnvironmentEpisodeFault as error:
            assert error.reason=='pedestrian_fall'
            assert error.diagnostics['physics_frame_consumed_by_policy'] is False
        else:raise AssertionError('Fall ignored')
        saved=torch.load(p/'FALL_PHYSICS_SNAPSHOT.pt',weights_only=False,map_location='cpu')
        assert saved['injected_test_fault'] is False and saved['steps']==10
    src=(Path(__file__).parent/'show_newmetro_robot_preview.py').read_text()
    start=src.index('state, fallen, truncated = world.step(action)')
    assert start<src.index('check_after_physics(world,state',start)<src.index('training_bridge.post_physics(',start)
    report=dict(status='PASS_CPU_FALL_GUARD_AND_BOUNDARY',bad_frame_before_policy=True,
        safe_boundary_preserved_across_nonterminal_checkpoints=True,no_fabricated_reward_or_done=True,
        real_reset='NOT_EXECUTED')
    (Path(__file__).resolve().parents[1]/'FALL_RECOVERY_CPU_CHECK.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':main()
