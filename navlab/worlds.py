from dataclasses import dataclass, asdict
from pathlib import Path
import heapq
import json
import math

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RADIUS = 0.2
DT = 0.1


@dataclass
class Scene:
    kind: str
    seed: int
    start: list
    goal: list
    rectangles: list  # [center_x, center_y, full_width, full_height]

    @property
    def name(self):
        return f"{self.kind}_{self.seed}"


def clearance(points, rectangles):
    """Exact point-to-axis-aligned-rectangle distance minus robot radius.

    Inside rectangles returns -RADIUS; sufficient for collision rejection.
    Also treats the boundary of [0,20]^2 as an obstacle.
    """
    p = np.asarray(points, dtype=float)
    boundary = np.minimum(p[..., 0], p[..., 1])
    boundary = np.minimum(boundary, np.minimum(20-p[..., 0], 20-p[..., 1]))
    result = boundary - RADIUS
    # Loop over obstacles, vectorize over points: bounded memory for DWA batches.
    for x, y, w, h in rectangles:
        dx = np.maximum(np.abs(p[..., 0]-x)-w/2, 0)
        dy = np.maximum(np.abs(p[..., 1]-y)-h/2, 0)
        result = np.minimum(result, np.hypot(dx, dy)-RADIUS)
    return result


def line_clear(a, b, rects, margin=0.12):
    n = max(2, int(np.linalg.norm(np.asarray(b)-a)/0.05)+2)
    return bool(np.all(clearance(np.linspace(a, b, n), rects) > margin))


def astar(scene, resolution=0.2):
    """8-neighbor A*, inflated occupancy, no diagonal corner cutting."""
    axis = np.arange(0, 20+resolution/2, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    # Half-cell diagonal protects edges between occupied grid samples.
    free = clearance(np.stack([xx, yy], -1), scene.rectangles) > 0.10+resolution/2**0.5
    start = tuple(np.rint(np.asarray(scene.start[:2])/resolution).astype(int))
    goal = tuple(np.rint(np.asarray(scene.goal[:2])/resolution).astype(int))
    if not free[start] or not free[goal]:
        raise ValueError("Start/goal not free on inflated grid")
    frontier = [(math.dist(start, goal), 0., start)]
    costs, parent = {start: 0.}, {}
    moves = [(i,j) for i in (-1,0,1) for j in (-1,0,1) if i or j]
    while frontier:
        _, g, node = heapq.heappop(frontier)
        if g > costs[node]+1e-9:
            continue
        if node == goal:
            route = [node]
            while route[-1] != start:
                route.append(parent[route[-1]])
            path = np.asarray(route[::-1], dtype=float)*resolution
            path[0], path[-1] = scene.start[:2], scene.goal[:2]
            if not all(line_clear(a,b,scene.rectangles,0.10) for a,b in zip(path[:-1],path[1:])):
                raise ValueError("Grid path failed continuous clearance check")
            return path
        for dx,dy in moves:
            nxt = node[0]+dx, node[1]+dy
            if not (0 <= nxt[0] < len(axis) and 0 <= nxt[1] < len(axis)) or not free[nxt]:
                continue
            if dx and dy and (not free[node[0]+dx,node[1]] or not free[node[0],node[1]+dy]):
                continue
            ng = g+math.hypot(dx,dy)
            if ng < costs.get(nxt, float("inf")):
                costs[nxt], parent[nxt] = ng, node
                heapq.heappush(frontier, (ng+math.dist(nxt,goal),ng,nxt))
    raise ValueError("No feasible inflated path")


def generate(kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "random":
        for _ in range(100):
            rects = []
            for _ in range(22):
                x,y = rng.uniform(3,17,2)
                w,h = rng.uniform(0.65,1.8,2)
                rects.append([float(x),float(y),float(w),float(h)])
            scene = Scene(kind,seed,[2.,2.,math.pi/4],[18.,18.,0.],rects)
            try:
                astar(scene)
                return scene
            except ValueError:
                pass
        raise RuntimeError("Could not generate a connected random map")
    if kind == "u_shape":
        cx = float(rng.uniform(9,11))
        top = float(rng.uniform(11.5,13.0))
        bottom = float(rng.uniform(5.2,6.8))
        width = float(rng.uniform(5.0,6.5))
        rects = [[cx,top,width+0.4,0.4],
                 [cx-width/2,(top+bottom)/2,0.4,top-bottom],
                 [cx+width/2,(top+bottom)/2,0.4,top-bottom]]
        return Scene(kind,seed,[cx+float(rng.uniform(-0.4,0.4)),9.,math.pi/2],
                     [cx,17.5,0.],rects)
    if kind == "narrow":
        center = float(rng.uniform(8.8,11.2))
        gap = float(rng.uniform(1.2,1.7))
        # Horizontal barrier, one genuine narrow doorway, edges touch world boundary.
        left_end, right_begin = center-gap/2, center+gap/2
        rects = [[left_end/2,10,left_end,0.8],
                 [(20+right_begin)/2,10,20-right_begin,0.8]]
        return Scene(kind,seed,[center+float(rng.uniform(-0.3,0.3)),2,math.pi/2],
                     [center+float(rng.uniform(-0.3,0.3)),18,0.],rects)
    raise ValueError(kind)


def save_scene(scene, folder):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/f"{scene.name}.json").write_text(json.dumps(asdict(scene),indent=2),encoding="utf-8")
    # IR-SIM's rectangle shape is centered at state; theta=0, length along x.
    world = {
        "world": {"width":20,"height":20,"step_time":DT,"sample_time":DT,
                  "collision_mode":"stop","control_mode":"auto"},
        "robot": {"kinematics":{"name":"diff"},"shape":{"name":"circle","radius":RADIUS},
                  "state":scene.start,"goal":scene.goal,"goal_threshold":0.3,
                  "vel_min":[0,-1.5],"vel_max":[1,1.5],"acce":[1.5,3.0],
                  "plot":{"show_trajectory":True,"show_goal":True}},
        "obstacle": [{"static":True,"shape":{"name":"rectangle","length":w,"width":h},
                      "state":[x,y,0]} for x,y,w,h in scene.rectangles]
    }
    path = folder/f"{scene.name}.yaml"
    path.write_text(yaml.safe_dump(world,sort_keys=False),encoding="utf-8")
    return path
