"""IR-SIM integration, paired evaluation, raw per-episode and per-step records."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

os.environ.setdefault("MPLBACKEND","Agg")
import irsim
import numpy as np

from .worlds import ROOT, DT, Scene, astar, generate, save_scene, clearance
from .planners import DWA, Parameters

METHODS = ("dwa", "astar_dwa")
KINDS = ("random", "u_shape", "narrow")
SEEDS = list(range(1000,1010))
WEIGHTS = (0.1,0.4,1.0)


def run(scene, method, parameters, output, display=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    scene_file = save_scene(scene,ROOT/"scenes")
    init_start = perf_counter()
    path = astar(scene) if method == "astar_dwa" else None
    planning_ms = (perf_counter()-init_start)*1000
    planner = DWA(scene,parameters,path)
    env = irsim.make(str(scene_file),display=display,headless=not display,log_level="ERROR",seed=scene.seed)
    robot = env.robot
    states = [robot.state[:3,0].copy()]
    commands, times, applied = [], [], []
    reason = "timeout"
    start_wall = perf_counter()
    min_clearance = float(clearance(states[0][:2],scene.rectangles))
    try:
        if planning_ms > 5000:
            reason = "planning_budget"
        else:
            for _ in range(1200):
                tick = perf_counter()
                action = planner.decide(states[-1],np.asarray(robot.velocity[:2,0]))
                elapsed = (perf_counter()-tick)*1000
                times.append(elapsed)
                if elapsed > 1000:  # same generous hard wall-time budget for both methods
                    reason = "decision_budget"
                    break
                commands.append(action.copy())
                env.step(action.reshape(2,1))
                current = robot.state[:3,0].copy()
                applied.append(np.asarray(robot.velocity[:2,0]).copy())
                segment = np.linspace(states[-1][:2],current[:2],11)
                segment_clearance = float(np.min(clearance(segment,scene.rectangles)))
                min_clearance = min(min_clearance,segment_clearance)
                states.append(current)
                if display:
                    env.render(0.001)
                if robot.collision_flag or segment_clearance <= 0:
                    reason = "collision"
                    break
                if np.linalg.norm(current[:2]-scene.goal[:2]) < 0.3:
                    reason = "success"
                    break
            if reason == "timeout" and len(states) > 101:
                if np.max(np.linalg.norm(np.asarray(states[-100:])[:,:2]-states[-1][:2],axis=1)) < 0.3:
                    reason = "timeout_stalled"
    finally:
        if display:
            env.end(0)
    states = np.asarray(states)
    trace_id = f"{scene.name}__{method}__w{parameters.obstacle_weight:g}"
    np.savez_compressed(output/f"{trace_id}.npz",states=states,
                        actions=np.asarray(commands).reshape(-1,2),
                        applied=np.asarray(applied).reshape(-1,2),
                        decision_ms=np.asarray(times),
                        global_path=path if path is not None else np.empty((0,2)))
    length = float(np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1).sum())
    row = {
        "trace_id":trace_id,"method":method,"scene":scene.kind,"seed":scene.seed,
        "obstacle_weight":parameters.obstacle_weight,"success":int(reason=="success"),
        "failure_reason":"" if reason=="success" else reason,"collision":int(reason=="collision"),
        "steps":len(states)-1,"sim_time_s":round((len(states)-1)*DT,6),
        "path_length_m":length,"goal_distance_m":float(np.linalg.norm(states[-1,:2]-scene.goal[:2])),
        "min_clearance_m":min_clearance,"decision_mean_ms":float(np.mean(times)) if times else 0.,
        "decision_std_ms":float(np.std(times,ddof=1)) if len(times)>1 else 0.,
        "decision_max_ms":max(times,default=0.),"global_planning_ms":planning_ms,
        "episode_wall_s":perf_counter()-start_wall,"no_safe_steps":planner.no_safe_steps,
        "parameters_json":json.dumps(parameters.dict(),sort_keys=True),
    }
    return row


def write_csv(rows,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def provenance():
    packages = {p:importlib.metadata.version(p) for p in
                ("ir-sim","numpy","matplotlib","scipy","shapely","pyyaml","pillow","pymupdf","pypdf")}
    data = {"timestamp_local":datetime.now().astimezone().isoformat(),
            "user_date":"2026-11-25","timezone":"Asia/Shanghai",
            "python":sys.version,"executable":sys.executable,"os":platform.platform(),
            "machine":platform.machine(),"cpu":platform.processor(),"logical_cpus":os.cpu_count(),
            "packages":packages,"test_seeds":SEEDS,"pilot_seeds":[11,12,13],
            "dt":DT,"max_steps":1200,"decision_budget_ms":1000,"planning_budget_ms":5000,
            "code_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT/"navlab").glob("*.py"))}}
    folder = ROOT/"results"
    folder.mkdir(exist_ok=True)
    (folder/"environment.json").write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    frozen = subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (ROOT/"requirements-lock.txt").write_text(frozen,encoding="utf-8")


def pilot():
    rows=[]
    for seed,kind in zip((11,12,13),KINDS):
        scene = generate(kind,seed)
        for method in METHODS:
            row=run(scene,method,Parameters(),ROOT/"results"/"pilot"/"traces")
            rows.append(row)
            write_csv(rows,ROOT/"results"/"pilot"/"episodes.csv")
            print(f"PILOT {scene.name} {method}: {row['success']} {row['failure_reason']} "
                  f"t={row['sim_time_s']:.1f}s d={row['goal_distance_m']:.2f}m "
                  f"decision={row['decision_mean_ms']:.2f}ms",flush=True)
    provenance()


def evaluate():
    destination = ROOT/"results"/"episodes.csv"
    if destination.exists():
        raise FileExistsError("results/episodes.csv already exists; use a fresh output archive before rerunning")
    # Freeze before seeing final test outcomes. The manifest includes every tested map.
    manifest=[]
    scenes=[]
    for kind in KINDS:
        for seed in SEEDS:
            scene=generate(kind,seed)
            validation_path=astar(scene)
            save_scene(scene,ROOT/"scenes")
            manifest.append({**asdict(scene),"feasible_grid_path_length_m":
                             float(np.linalg.norm(np.diff(validation_path,axis=0),axis=1).sum())})
            scenes.append(scene)
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results"/"test_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (ROOT/"results"/"frozen_parameters.json").write_text(json.dumps(Parameters().dict(),indent=2),encoding="utf-8")
    provenance()
    rows=[]
    tasks = [(scene,weight,method) for i,scene in enumerate(scenes)
             for weight in (WEIGHTS if scene.kind=="narrow" else (0.4,))
             for method in (METHODS if i%2==0 else METHODS[::-1])]
    # Alternate order to avoid systematically assigning all cold/early runs to one method.
    for index,(scene,weight,method) in enumerate(tasks):
        row=run(scene,method,Parameters(obstacle_weight=weight),ROOT/"results"/"traces")
        rows.append(row)
        write_csv(rows,destination)
        print(f"[{index+1}/{len(tasks)}] {row['trace_id']}: {row['success']} "
              f"{row['failure_reason']} t={row['sim_time_s']:.1f}s "
              f"decision={row['decision_mean_ms']:.2f}ms",flush=True)
    print("All episodes saved, including failures.",flush=True)


def demo(kind="u_shape",seed=1000,method="astar_dwa",display=False):
    row=run(generate(kind,seed),method,Parameters(),ROOT/"results"/"demo",display)
    print(json.dumps(row,ensure_ascii=False,indent=2))
