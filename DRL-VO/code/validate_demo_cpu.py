"""Validate frozen-model identity and reuse of the trained observation/action path."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
import torch
from demo_policy import DemoPolicy,fingerprint
from task_sampling import make_navigation

B=Path(__file__).resolve().parents[1]

def method(path,cls,name):
    tree=ast.parse(path.read_text());c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    return ast.dump(next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)

def main():
    provenance=json.loads((B/'PROVENANCE.json').read_text());source=Path(provenance['training_experiment'])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in provenance['protected_sha256'].items())
    out=B/'runs/cpu_validation';out.mkdir(exist_ok=False)
    config=dict(model_checkpoint=str(B/'frozen_policy.pt'),model_sha256=provenance['frozen_policy_sha256'],local_goal_lookahead_m=.5)
    (out/'ROBOT_CONTROL.json').write_text(json.dumps(config))
    policy=DemoPolicy(out)
    assert policy.trained_lookahead==1.5 and policy.cfg['local_goal_lookahead_m']==.5
    from local_goal import local_goal_base
    assert np.allclose(local_goal_base([[0,0],[3,0]],[0,0,0],lookahead=policy.cfg['local_goal_lookahead_m']),[.5,0])
    state=torch.load(source/'training/session_state.pt',map_location='cpu',weights_only=False)
    assert state['steps']==270848 and state['updates']==529
    assert set(state['policy'])==set(policy.model.state_dict())
    assert all(torch.equal(v.cpu(),state['policy'][n]) for n,v in policy.model.state_dict().items())
    assert policy.loaded_fingerprint=='d5abb3e46251825fa1ef1e0f66fbfe2f9d273f9068dad9588b947219fdf2b81a'
    inputs=sorted((source/'runs/main/policy_inputs').glob('*.npy'))[:4]
    assert len(inputs)==4
    context=dict(actual_v=.2,actual_w=.1,goal_base_xy=[1.5,0.],pedestrians=np.zeros((0,4)),full_scan_minimum=2.)
    actions=[]
    for p in inputs:
        obs=np.load(p,allow_pickle=False);policy.reset_episode()
        a=policy.command(obs,distance=3.,minimum=2.,control=1,sim_time=1.,reward_context=context)
        policy.reset_episode()
        b=policy.command(obs,distance=3.,minimum=2.,control=1,sim_time=1.,reward_context=context)
        assert a['raw_action']==b['raw_action'] and a['clipped_action']==b['clipped_action']
        assert 0<=a['desired_v_m_s']<=.5 and abs(a['desired_w_rad_s'])<=2
        actions.append(a)
    assert policy.verify_frozen()==policy.loaded_fingerprint
    assert not any(x.requires_grad for x in policy.model.parameters())
    assert policy.updates==0
    assert method(B/'code/demo_driver.py','DemoDriver','command')==method(source/'code/training_driver.py','TrainingPolicyDriver','command')
    for n in ['drlvo_observation_adapter.py','local_goal.py','reward_logic.py','lidar_preview_adapter.py']:
        assert (B/'code'/n).read_bytes()==(source/'code'/n).read_bytes(),n
    p=make_navigation(B);p.current_yaw=np.pi/2
    for goal in [[-9.5,-1.65],[-10.5,-1.65]]:
        route=p.astar_path([-6.85,-1.65],goal,time_budget_s=6.)
        assert route is not None
        assert all(p.sweep_clear(a,b) for a,b in zip(p.last_poses[:-1],p.last_poses[1:]))
    report=dict(status='FROZEN_MODEL_AND_INPUT_PATH_PASS',training_steps=270848,training_updates=529,
        model_fingerprint=policy.loaded_fingerprint,strict_state_match=True,actual_training_inputs=4,
        source_trained_lookahead_m=policy.trained_lookahead,demo_lookahead_m=policy.cfg['local_goal_lookahead_m'],
        deterministic_inference=True,weights_unchanged=True,optimizer_updates=0,
        observation_action_command_AST_identical_to_training=True,protected_sources_unchanged=True,
        actual_PhysX_demo='NOT_EXECUTED',contact_safety='UNVERIFIED',actions=actions)
    (B/'OFFLINE_DEMO_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='actions'}))

if __name__=='__main__':main()
