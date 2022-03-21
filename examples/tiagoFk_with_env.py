import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.tiagoFk import TiagoFk

import gym
import urdfenvs.tiago_reacher
from MotionPlanningGoal.staticSubGoal import StaticSubGoal


def getGoal(pos):
    goalDict = {
        "m": 3,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1, 2],
        "parent_link": 0,
        "child_link": 3,
        "desired_position": pos,
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="goal", contentDict=goalDict)
    return goal
        

def main():
    env = gym.make("tiago-reacher-vel-v0", dt=0.01, render=True)
    fkTiago = TiagoFk()
    pos0 = np.zeros(20)
    pos0[3] = 0.1
    pos0[2] = 0.8
    pos0[6] = 0.5
    pos0[8] = -0.7
    ob = env.reset(pos=pos0)
    n_ee = 11
    fk = fkTiago.fk(ob['x'], n_ee, positionOnly=True)
    env.add_goal(getGoal(fk))
    action = np.zeros(env.n())
    for _ in range(1000):
        ob, _, _, _ = env.step(action)
        q_np = ob['x']
        fkNumpy = fkTiago.fk(q_np, n_ee, positionOnly=True)
        print(f"fkNumpy : {fkNumpy}")
    

if __name__ == "__main__":
    main()
