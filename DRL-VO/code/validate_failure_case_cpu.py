"""Regression of frozen failure evidence against the changed actual planner."""
from pathlib import Path
import importlib.util, json, math, time
import numpy as np
from task_sampling import make_navigation
from drlvo_heading_planner import HeadingPlanner

BATCH=Path(__file__).resolve().parents[1]
FIX=BATCH.parent
ROOT=FIX.parents[1]
OLD=ROOT/'runs/long_training_20261016T1939MYT'

def main():
    spec=importlib.util.spec_from_file_location('old_geometry_checker',OLD/'audit/reachability_20261017/verify_routes.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    spec=importlib.util.spec_from_file_location('frozen_original_planner',OLD/'experiment/code/drlvo_heading_planner.py')
    oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
    data=json.loads((OLD/'FAILURE_CANDIDATES_RECONSTRUCTED.json').read_text())
    start=np.asarray(data['source_pose'])
    old=make_navigation(BATCH);old.__class__=oldmodule.HeadingPlanner
    new=make_navigation(BATCH)
    assert new.__class__ is HeadingPlanner
    assert Path(__import__('inspect').getfile(HeadingPlanner)).resolve()==BATCH/'code/drlvo_heading_planner.py'
    results=[]
    for c in data['candidates']:
        old.current_yaw=new.current_yaw=start[2]
        a=old.astar_path(start[:2],c['goal_xy'],time_budget_s=6.)
        t=time.monotonic();b=new.astar_path(start[:2],c['goal_xy'],time_budget_s=6.)
        elapsed=time.monotonic()-t
        assert a is None and old.expanded==1
        assert b is not None
        poses=new.last_poses.copy()
        np.testing.assert_allclose(poses[0],start)
        np.testing.assert_allclose(poses[-1,:2],c['goal_xy'])
        assert all(old.sweep_clear(a,b) for a,b in zip(poses[:-1],poses[1:]))
        independent=checker.dense_check(new,poses)
        assert independent['continuous_original_geometric_gap_025m_verified']
        results.append(dict(order=c['order'],goal_xy=c['goal_xy'],old_found=False,new_found=True,
            new_search_seconds=elapsed,independent_check=independent,poses=poses.tolist()))
    local=[]
    for dx in (-.02,0.,.02):
        for dy in (-.02,0.,.02):
            for angle in (-10.,0.,10.):
                pose=start+np.array([dx,dy,math.radians(angle)])
                assert new.pose_clear(pose[:2],pose[2],.030)
                old.current_yaw=new.current_yaw=pose[2]
                a=old.astar_path(pose[:2],data['candidates'][0]['goal_xy'],time_budget_s=6.)
                b=new.astar_path(pose[:2],data['candidates'][0]['goal_xy'],time_budget_s=6.)
                assert b is not None
                assert all(old.sweep_clear(x,y) for x,y in zip(new.last_poses[:-1],new.last_poses[1:]))
                local.append(dict(source_pose=pose.tolist(),old_found=a is not None,new_found=True))
    report=dict(status='PASS_STATIC_FAILURE_REGRESSION',original_found=0,new_found=16,
        local_cases=27,local_original_found=sum(x['old_found'] for x in local),local_new_found=27,
        dense_independent_checks_passed=16,original_geometry_checks_unchanged=True,
        physics_and_training='NOT_EXECUTED',contact_safety='UNVERIFIED')
    (BATCH/'FAILURE_REGRESSION.json').write_text(json.dumps(dict(summary=report,candidates=results,local_cases=local),indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
