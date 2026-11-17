"""Fault-injection checks of real boundary methods; no simulator or PPO imports."""
from pathlib import Path
from types import SimpleNamespace
import json, tempfile, time
import numpy as np
from validate_random_tasks_cpu import source_class, check_boundaries, BATCH
from task_sampling import make_navigation, TaskSampler, TaskSamplingError, TaskAdmissionError
from episode_semantics import EnvironmentEpisodeFault

Bridge = source_class('sfm_training_bridge.py','SfmTrainingBridge')

def fixture(out, navigation):
    calls=dict(checkpoints=0,moves=0,resets=0,samples=0,stops=0,saves=0)
    def increment(name):calls[name]+=1
    class Tensor:
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return np.full((4,4),100.)
    pose=np.array([-6.85,-1.65,np.pi/2])
    robot=SimpleNamespace(stop=lambda:increment('stops'),
        observe=lambda:dict(root_position_m=[*pose[:2],.02],yaw_rad=pose[2]),
        initialize=lambda **kw:increment('moves'),reset_count=0)
    policy=SimpleNamespace(steps=100,updates=1,episodes=1,reason='goal',
        cfg={'total_timesteps':1000},checkpoint=lambda:increment('checkpoints'))
    b=Bridge.__new__(Bridge)
    b.out=out;b.phase='long';b.start_wall=time.monotonic();b.config={'wall_limit_s':300.}
    b.reset_count=b.task_switch_count=b.task_admission_recoveries=0
    b.driver=SimpleNamespace(policy=policy,reset_episode=lambda **kw:increment('resets'))
    b.m=SimpleNamespace(robot=robot,world=SimpleNamespace(planar=lambda:Tensor()),report={},save=lambda:increment('saves'))
    b.navigation=navigation;b.sampler=TaskSampler(BATCH,navigation,45)
    b.query_task_geometry=lambda *a:None
    return b,calls

def main():
    navigation=make_navigation(BATCH)
    recovered_boundaries=check_boundaries(navigation,admission_failures=1)
    maximum_boundaries=check_boundaries(navigation,admission_failures=3)
    tests=[]
    for mode in ('exhausted','expired_budget','configuration_fault','environment_fault','geometry_fault','occupied_spawn'):
        with tempfile.TemporaryDirectory(prefix='newmetro_recovery_fault_') as temp:
            out=Path(temp);b,calls=fixture(out,navigation)
            original=b.sampler.sample
            def sample(*args,**kw):
                calls['samples']+=1
                if mode=='configuration_fault':raise TaskSamplingError('bad configuration')
                if mode=='environment_fault':raise EnvironmentEpisodeFault('sensor fault')
                if mode=='exhausted':raise TaskAdmissionError('bounded no route',diagnostics={'rejected_candidates':[{'goal_xy':[1.,2.],'reason':'injected'}]})
                return original(*args,**kw)
            b.sampler.sample=sample
            if mode=='expired_budget':b.start_wall=time.monotonic()-400.
            if mode=='occupied_spawn':
                b.driver.policy.reason='collision';b.spawn_is_occupied=lambda pose:True
            if mode=='geometry_fault':
                def reject_geometry(*args):raise EnvironmentEpisodeFault('full_height_query_rejected')
                b.query_task_geometry=reject_geometry
            try:b.handle_terminal(100,3.3)
            except (TaskSamplingError,EnvironmentEpisodeFault) as error:
                if mode in ('exhausted','expired_budget'):assert type(error) is TaskAdmissionError
                elif mode=='configuration_fault':assert type(error) is TaskSamplingError
                else:assert type(error) is EnvironmentEpisodeFault
            else:raise AssertionError('Fault swallowed: '+mode)
            expected={'exhausted':4,'expired_budget':0,'configuration_fault':1,'environment_fault':1,'geometry_fault':1,'occupied_spawn':0}
            assert calls['samples']==expected[mode],calls
            assert calls['moves']==calls['resets']==0,calls
            assert (b.driver.policy.steps,b.driver.policy.updates,b.driver.policy.episodes)==(100,1,1)
            assert calls['checkpoints']==(2 if mode in ('exhausted','expired_budget') else 1)
            if mode=='exhausted':
                events=[json.loads(x) for x in (out/'TASK_ADMISSION_EVENTS.jsonl').read_text().splitlines()]
                assert len(events)==5 and events[-1]['status']=='EXHAUSTED'
                assert all('sampler_before' in e and 'sampler_after' in e and 'diagnostics' in e for e in events[:-1])
                assert not (out/'RESET_EVENTS.jsonl').exists()
            tests.append(dict(case=mode,status='PASS',calls=calls))
    # Real sampler must retain all 16 rejections and restore its old reference.
    sampler=TaskSampler(BATCH,navigation,99);old_astar=navigation.astar_path
    reference=np.array([[0.,0.,0.],[1.,0.,0.]])
    navigation.reference_poses=reference;navigation.current_yaw=.42
    navigation.astar_path=lambda *a,**kw:None
    try:
        sampler.sample([*sampler.starts[0]['xy_m'],sampler.starts[0]['yaw_rad']])
    except TaskAdmissionError as error:
        assert len(error.diagnostics['rejected_candidates'])==16
        assert all('goal_xy' in x for x in error.diagnostics['rejected_candidates'])
    else:raise AssertionError('No route admitted')
    finally:navigation.astar_path=old_astar
    assert navigation.reference_poses is reference and navigation.current_yaw==.42
    report=dict(status='PASS_OFFLINE_MOCKED_RECOVERY',recovered_boundaries=recovered_boundaries,
        maximum_retry_boundaries=maximum_boundaries,fault_cases=tests,
        actual_no_route_diagnostics='PASS_16_GOALS_RETAINED',
        real_physics_reset='NOT_EXECUTED',contact_safety='UNVERIFIED',training_started=False)
    (BATCH/'TASK_RECOVERY_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
