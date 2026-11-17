"""Preserve resumable boundaries and stop before consuming fallen physics."""
from pathlib import Path
import json, os
import torch
from episode_semantics import EnvironmentEpisodeFault

def preserve_boundary(session, state):
    """Immutable inode link, not a second serialization or an in-place write."""
    dones=state['buffer']['dones']
    if state.get('interrupted_nonterminal_buffer') or (dones and not dones[-1]):
        return False
    temporary=Path(session)/'recovery_boundary.tmp'
    if temporary.exists():temporary.unlink()
    os.link(Path(session)/'session_state.pt',temporary)
    temporary.replace(Path(session)/'recovery_boundary.pt')
    return True

def check_after_physics(world,state,action,fallen,control,runtime,out,policy,report):
    limit=int(os.environ.get('NEWMETRO_TEST_FALL_AFTER_NEW_STEPS','0'))
    injected=bool(limit and policy.steps-policy.initial_steps>=limit)
    if not bool(fallen.any()) and not injected:return
    def cpu(value):
        return value.detach().cpu().clone() if isinstance(value,torch.Tensor) else value
    snapshot=dict(schema='newmetro.fall_diagnostic_state.v1',control=control,
        sim_time=control*world.dt,steps=policy.steps,updates=policy.updates,
        injected_test_fault=injected,applied_action=cpu(action),
        state={k:cpu(getattr(state,k)) for k in ('root_pos','root_rot','root_vel','root_ang_vel',
            'dof_pos','dof_vel','dof_forces') if hasattr(state,k)},
        world_history={k:cpu(getattr(world,k)) for k in ('amp_history','planar_history','age','episode_ids') if hasattr(world,k)},
        sfm_targets=cpu(getattr(runtime,'last_targets',None)),
        exact_PhysX_solver_restore=False,contact_force_recorded=False)
    torch.save(snapshot,Path(out)/'FALL_PHYSICS_SNAPSHOT.pt')
    actors=torch.where(fallen)[0].tolist()
    report['falls']=report.get('falls',0)+len(actors)
    report.setdefault('fall_events',[]).append(dict(control=control,actors=actors,injected_test_fault=injected))
    # Never feed this frame into history, reward, rollout, or a checkpoint.
    raise EnvironmentEpisodeFault('pedestrian_fall',dict(control=control,actors=actors,
        injected_test_fault=injected,physics_frame_consumed_by_policy=False,
        diagnostic_snapshot='FALL_PHYSICS_SNAPSHOT.pt'))
