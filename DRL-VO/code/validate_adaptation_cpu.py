"""Executable source comparisons for reward, done, normalization and timing."""
import ast
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace, MethodType
import warnings
import numpy as np
import numpy.matlib
from reward_logic import OriginalReward
from local_goal import ActionClock, local_goal_base
from drlvo_observation_adapter import build_observation

BATCH = Path(__file__).resolve().parents[1]


def original_methods(seed):
    source = BATCH/'reference/drl_nav_env.py'
    names = ('_goal_reached_reward','_obstacle_collision_punish','_angular_velocity_punish','_theta_reward','_is_done','_get_observation')
    nodes = [n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(np=np,random=random.Random(seed),rospy=SimpleNamespace(logwarn=lambda *a:None,logdebug=lambda *a:None),Twist=lambda:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),namespace)
    obj = SimpleNamespace(ROBOT_RADIUS=.3,GOAL_RADIUS=.3,DIST_NUM=10,bump_num=0,num_iterations=0,
                          dist_to_goal_reg=np.zeros(10),_cmd_vel_pub=SimpleNamespace(publish=lambda *a:None))
    for name in names: setattr(obj,name,MethodType(namespace[name],obj))
    return obj


def main():
    rng = np.random.default_rng(16)
    kernel = OriginalReward(42)
    original = original_methods(42)
    max_error = 0.
    for case in range(600):
        distance = float(rng.uniform(.1,7))
        minimum = float(rng.choice([.01,.15,.3,.30001,.6,1.,8.]))
        v,w = rng.uniform(-.1,.6),rng.uniform(-2,2)
        goal = rng.uniform(-2,2,size=2)
        angle = rng.uniform(-math.pi,math.pi,size=3)
        radius = rng.uniform(1.,8.,size=3)
        peds = np.c_[radius*np.cos(angle),radius*np.sin(angle),rng.uniform(-1,1,size=(3,2))]
        original.curr_pose=SimpleNamespace(position=SimpleNamespace(x=distance,y=0,z=0))
        original.goal_position=SimpleNamespace(x=0,y=0,z=0)
        original.cnn_data=SimpleNamespace(scan=np.full(720,minimum))
        tracks=[SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=p[0],y=p[1]))),
                 twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=p[2],y=p[3])))) for p in peds]
        expected=(original._goal_reached_reward(20.,3.2)+original._obstacle_collision_punish(original.cnn_data.scan,-.2,-20.)+
                  original._angular_velocity_punish(w,-.1,1.3)+original._theta_reward(goal,SimpleNamespace(tracks=tracks),v,.6,math.pi/6))
        expected_done=original._is_done(expected)
        reward,done,reason,terms=kernel.step(distance=distance,minimum_scan=minimum,actual_v=v,actual_w=w,
                                           goal_base_xy=goal,pedestrians_base_xy_vxy=peds)
        max_error=max(max_error,abs(expected-reward))
        assert abs(expected-reward)<1e-10 and expected_done==done
        assert kernel.bump_num==original.bump_num and kernel.num_iterations==original.num_iterations
        if done:
            kernel.reset_task()
            original.num_iterations=0
            original.dist_to_goal_reg=np.zeros(10)
    def step(k,minimum=2.,w=0.,distance=5.,peds=()):
        return k.step(distance=distance,minimum_scan=minimum,actual_v=.2,actual_w=w,
                      goal_base_xy=[1.,0.],pedestrians_base_xy_vxy=peds)
    k=OriginalReward()
    for i in range(512):
        reward,done,reason,terms=step(k)
        assert not done and terms['goal']==0.
    reward,done,reason,terms=step(k)
    assert done and reason=='timeout' and terms['goal']==-20. and k.num_iterations==513
    k=OriginalReward()
    for minimum,expected in [(.2,False),(1.,False),(.2,False),(1.,False),(.2,True)]:
        _,done,reason,_=step(k,minimum=minimum)
        assert done==expected
    assert reason=='collision' and k.bump_num==0
    k=OriginalReward()
    assert step(k,w=1.3)[3]['angular']==0.
    assert abs(step(k,w=1.3001)[3]['angular']+.13001)<1e-12
    step(k,minimum=.2)
    k.reset_task()
    assert k.bump_num==1 and k.num_iterations==0 and np.all(k.distances==0)
    near=step(k,peds=[[.2,0.,0.,0.]])
    assert math.isfinite(near[0]) and near[3]['vo_close_radicand_guards']>0
    obs_error=0.
    original=original_methods(7)
    for _ in range(24):
        peds=rng.uniform(-3.,3.,size=(2,80,80)).astype(np.float32)
        history=rng.uniform(0.,30.,size=(10,720)).astype(np.float32)
        goal=rng.uniform(-2,2,size=2).astype(np.float32)
        original.cnn_data=SimpleNamespace(ped_pos_map=peds,scan=history.reshape(-1),goal_cart=goal,vel=[0.,0.])
        expected=original._get_observation()
        actual=build_observation(peds,history,goal,clip_pedestrian=False)
        error=float(np.max(np.abs(expected-actual)));obs_error=max(obs_error,error)
        assert error<3e-7
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        np.testing.assert_allclose(local_goal_base(np.array([[0.,0.],[10.,0.]]),np.array([0.,0.,0.])),[1.5,0.])
        corner=local_goal_base(np.array([[0.,0.],[1.,0.],[1.,4.]]),np.array([0.,0.,0.]))
        np.testing.assert_allclose(corner,[1.,math.sqrt(1.5**2-1.)])
    clock=ActionClock(.05)
    times=[]
    for control in range(1801):
        t=control/30
        if clock.due(t):times.append(t);clock.commit(t)
    assert len(times)==1201 and abs((len(times)-1)/(times[-1]-times[0])-20.)<1e-9
    assert set(np.round(np.diff(times),6))=={.033333,.066667}
    report=dict(status='SOURCE_REWARD_DONE_OBSERVATION_TIMING_PASS',reward_comparison_cases=600,
        reward_max_abs_error=max_error,angular_threshold_reference_override=1.3,
        exact_source_bump_nonconsecutive_and_carry='PASS',timeout_transition_513='PASS',
        VO_near_radicand_repair_finite='PASS_INTENTIONAL_SOURCE_DIFFERENCE',
        observation_cases=24,observation_max_abs_error=obs_error,pure_pursuit_straight_and_corner='PASS',
        local_goal_lookahead_m=1.5,action_clock_average_hz=20.,physical_quantized_intervals_s=[1/30,2/30],
        original_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BATCH/'reference').glob('*.py')},
        tested_core_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BATCH/'code').glob('*.py')},
        runtime_physics='NOT_EXECUTED')
    (BATCH/'audit/ADAPTATION_CPU_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('original_sources','tested_core_hashes')},ensure_ascii=False))


if __name__=='__main__':main()
