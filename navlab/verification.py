"""Checks use independent geometry and recompute metrics from actual traces."""
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box, LineString

from .worlds import ROOT, Scene, clearance, astar, generate, save_scene, DT
from .planners import wrap


def verify():
    import irsim
    checks=[]
    rectangles=[[7.,8.,2.,4.],[12.,13.,1.5,0.8]]
    rng=np.random.default_rng(2701)
    for point in rng.uniform(0.3,19.7,(200,2)):
        shapes=[box(x-w/2,y-h/2,x+w/2,y+h/2) for x,y,w,h in rectangles]
        expected=min([Point(point).distance(shape)-0.2 for shape in shapes]+
                     [min(*point,*(20-point))-0.2])
        assert abs(float(clearance(point,rectangles))-expected)<1e-10
    checks.append("200 independent Shapely point/rectangle distance checks")
    for kind in ("random","u_shape","narrow"):
        for seed in range(1000,1010):
            scene=generate(kind,seed)
            path=astar(scene)
            line=LineString(path)
            for x,y,w,h in scene.rectangles:
                assert line.distance(box(x-w/2,y-h/2,x+w/2,y+h/2))>0.2
    checks.append("30 test maps: independent continuous path clearance > robot radius")
    scratch=ROOT/"tmp"/"verification"
    scene=Scene("integration_check",1,[2.,2.,0.],[18.,18.,0.],rectangles)
    env=irsim.make(str(save_scene(scene,scratch)),headless=True,log_level="ERROR")
    # Check that IR-SIM's actual rectangle orientation and dimensions match planner geometry.
    for obstacle,rect in zip(env.obstacle_list,rectangles):
        x,y,w,h=rect
        assert np.allclose(obstacle.geometry.bounds,[x-w/2,y-h/2,x+w/2,y+h/2])
    for action in ([0.1,0.2],[0.2,0.4],[0.3,0.6],[0.15,0.3]):
        before=env.robot.state[:,0].copy()
        expected=before+DT*np.array([action[0]*np.cos(before[2]),action[0]*np.sin(before[2]),action[1]])
        expected[2]=wrap(expected[2])
        env.step(np.asarray(action).reshape(2,1))
        assert np.allclose(env.robot.state[:,0],expected,atol=1e-10)
    checks.append("IR-SIM geometry alignment and differential-drive integration")
    collision_scene=Scene("collision_check",1,[2.,2.,0.],[18.,2.,0.],[[3.,2.,0.4,2.]])
    env2=irsim.make(str(save_scene(collision_scene,scratch)),headless=True,log_level="ERROR")
    for _ in range(30):
        env2.step(np.array([[0.5],[0.]]))
        if env2.robot.collision_flag:
            break
    assert env2.robot.collision_flag,"Platform collision detector was not triggered"
    checks.append("Deliberate wall impact triggers IR-SIM collision flag")
    csv_file=ROOT/"results"/"episodes.csv"
    if csv_file.exists():
        with csv_file.open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==100,len(rows)
        assert len({r["trace_id"] for r in rows})==100
        main=[r for r in rows if float(r["obstacle_weight"])==0.4]
        assert len(main)==60
        for kind in ("random","u_shape","narrow"):
            for method in ("dwa","astar_dwa"):
                assert {int(r["seed"]) for r in main if r["scene"]==kind and r["method"]==method}==set(range(1000,1010))
        for row in rows:
            trace=np.load(ROOT/"results"/"traces"/(row["trace_id"]+".npz"))
            state,commands,applied=trace["states"],trace["actions"],trace["applied"]
            assert len(state)-1==int(row["steps"])==len(commands)==len(applied)
            assert len(state)-1<=1200
            assert np.isfinite(state).all() and np.isfinite(commands).all()
            assert ((commands[:,0]>=-1e-8)&(commands[:,0]<=1+1e-8)).all()
            assert (np.abs(commands[:,1])<=1.5+1e-8).all()
            deltas=np.diff(np.vstack([np.zeros(2),commands]),axis=0)
            assert (np.abs(deltas)<=np.array([0.15,0.3])+1e-7).all()
            assert np.allclose(commands,applied,atol=1e-7)
            expected_delta=DT*np.column_stack([applied[:,0]*np.cos(state[:-1,2]),
                                               applied[:,0]*np.sin(state[:-1,2])])
            assert np.allclose(np.diff(state[:,:2],axis=0),expected_delta,atol=1e-9)
            length=np.linalg.norm(np.diff(state[:,:2],axis=0),axis=1).sum()
            assert abs(length-float(row["path_length_m"]))<1e-8
            assert abs(DT*(len(state)-1)-float(row["sim_time_s"]))<1e-8
            assert abs(trace["decision_ms"].mean()-float(row["decision_mean_ms"]))<1e-8
            scene=generate(row["scene"],int(row["seed"]))
            distance=np.linalg.norm(state[-1,:2]-scene.goal[:2])
            assert (int(row["success"])==1)==(distance<0.3 and int(row["collision"])==0)
            if not int(row["collision"]):
                for x,y,w,h in scene.rectangles:
                    assert LineString(state[:,:2]).distance(box(x-w/2,y-h/2,x+w/2,y+h/2))>0.2-1e-8
        checks.append("100 trace/CSV consistency checks: paired seeds, actions, accelerations, simulator motion, outcomes and metrics")
    report={"passed":True,"checks":checks}
    target=ROOT/"results"/"verification.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    for check in checks:
        print("PASS",check)
