from dataclasses import dataclass, asdict
import numpy as np
from .worlds import DT, clearance, line_clear


@dataclass(frozen=True)
class Parameters:
    horizon: float = 2.0
    linear_acceleration: float = 1.5
    angular_acceleration: float = 3.0
    v_samples: int = 7
    w_samples: int = 13
    safety_margin: float = 0.07
    obstacle_weight: float = 0.4
    goal_weight: float = 2.5
    heading_weight: float = 0.6
    speed_weight: float = 0.35
    turn_weight: float = 0.05
    lookahead: float = 2.5

    def dict(self):
        return asdict(self)


def wrap(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi


class DWA:
    """Self-written DWA, optionally guided by A* intermediate goals.

    Both variants use identical local sampling/scoring and the same full static map.
    No PythonRobotics source is copied; it is an algorithm reference only.
    """
    def __init__(self, scene, parameters, path=None):
        self.scene, self.p, self.path = scene, parameters, path
        self.progress = 0
        self.no_safe_steps = 0

    def target(self, state):
        if self.path is None:
            return np.asarray(self.scene.goal[:2])
        distances = np.linalg.norm(self.path[self.progress:]-state[:2],axis=1)
        nearest = self.progress+int(np.argmin(distances))
        self.progress = max(self.progress,nearest)
        target = self.path[self.progress]
        # Look forward along the path, never jump through an obstacle.
        for point in self.path[self.progress+1:]:
            if np.linalg.norm(point-state[:2]) > self.p.lookahead:
                break
            if line_clear(state[:2],point,self.scene.rectangles,0.12):
                target = point
        return target

    def decide(self, state, previous):
        p = self.p
        target = self.target(state)
        vlo,vhi = max(0,previous[0]-p.linear_acceleration*DT),min(1,previous[0]+p.linear_acceleration*DT)
        wlo,whi = max(-1.5,previous[1]-p.angular_acceleration*DT),min(1.5,previous[1]+p.angular_acceleration*DT)
        vs = np.unique(np.r_[np.linspace(vlo,vhi,p.v_samples),np.clip(previous[0],vlo,vhi)])
        ws = np.unique(np.r_[np.linspace(wlo,whi,p.w_samples),np.clip(0,wlo,whi)])
        vv,ww = np.meshgrid(vs,ws,indexing="ij")
        v,w = vv.ravel(),ww.ravel()
        # Euler integration, matches IR-SIM differential-drive integrator.
        k = np.arange(int(round(p.horizon/DT)))
        theta = state[2]+w[:,None]*DT*k
        x = state[0]+np.cumsum(v[:,None]*np.cos(theta)*DT,axis=1)
        y = state[1]+np.cumsum(v[:,None]*np.sin(theta)*DT,axis=1)
        trajectories = np.stack([x,y],axis=-1)
        free_distance = np.min(clearance(trajectories,self.scene.rectangles),axis=1)
        # Conservative clearance also covers braking distance and inter-step travel.
        admissible = free_distance > np.maximum(p.safety_margin,v*v/(2*p.linear_acceleration)+0.025)
        endpoints = trajectories[:,-1]
        distance = np.linalg.norm(endpoints-target,axis=1)
        # Compare final orientation with the current bearing, avoiding a singular
        # terminal heading cost when a predicted endpoint almost hits the target.
        bearing = np.arctan2(target[1]-state[1],target[0]-state[0])
        heading = np.abs(wrap(state[2]+w*p.horizon-bearing))
        cost = (p.goal_weight*distance+p.heading_weight*heading
                +p.obstacle_weight/(free_distance+0.15)
                +p.speed_weight*(1-v)+p.turn_weight*np.abs(w))
        cost[~admissible] = np.inf
        if np.any(admissible):
            return np.array([v[np.argmin(cost)],w[np.argmin(cost)]])
        self.no_safe_steps += 1
        # Strongest reachable braking command; never violate acceleration bounds.
        return np.array([vlo,np.clip(0,wlo,whi)])
