import pybullet as p
import time
import pybullet_data
import gym
import math as m
import numpy as np

class ModularEnv(gym.Env):
    def __init__(self, render=False, robot_id='spiderV1_2'):
        if render:
            physicsClient = p.connect(p.GUI)
            self.render = True
        else:
            self.render = False
            physicsClient = p.connect(p.DIRECT)
        robot_type = robot_id.split('V')[0]
        robot_vers = robot_id.split('_')[0]
        self.urdf = robot_type+"_series/"+robot_vers+"/"+robot_id+"/urdf/"+robot_id+".urdf"
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = 5.236 # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.sleep_time = 1./240
        self.n_sim_steps = 40
        obs = self.reset()
        self.action_space = gym.spaces.Box(low=-np.ones(self.dof), high=np.ones(self.dof))
        self.observation_space = gym.spaces.Box(low=-np.ones_like(obs)*np.inf, high=np.ones_like(obs)*np.inf)

    def _render(self, mode='human', close=False):
        pass

    def get_obs(self):
        self.last_p = self.p
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.p, self.q = np.array(self.p), np.array(self.q)
        self.v = self.p - self.last_p
        jointInfo = [p.getJointState(self.robotid, i) for i in range(self.dof)]
        jointVals = np.array([[joint[0], joint[1]] for joint in jointInfo]).flatten()
        obs = np.concatenate([self.v, self.q, jointVals])
        return obs

    def act(self, action):
        for i in range(self.n_sim_steps):
            p.stepSimulation()
            if self.render:
                time.sleep(self.sleep_time)
        for i in range(len(action)):
            p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=action[i], force = 1.4, maxVelocity =self.maxVelocity)


    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        # planeId = p.loadURDF("assets/plane/plane.urdf")
        planeId = p.loadURDF("plane.urdf")
        robotStartPos = [0,0,0.2]
        robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
        # self.robotid = p.loadURDF("assets/spiderV1/urdf/spiderV1.urdf", robotStartPos, robotStartOrientation)
        self.robotid = p.loadURDF(self.urdf, robotStartPos, robotStartOrientation)
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.p, self.q = np.array(self.p), np.array(self.q)
        self.i = 0
        self.dof = p.getNumJoints(self.robotid)
        return self.get_obs()

    def step(self, action):
        self.act(action)
        obs = self.get_obs()
        done = False
        r = 100*obs[0] - 50*np.abs(obs[1])
        return obs, r, done, {}

if __name__ == '__main__':
    import os
    print(os.getcwd())
    env = ModularEnv(render=True)
    ep_r = 0
    start = time.time()

    for j in range(1):
        all_obs = []
        obs = env.reset()
        all_obs.append(obs)
        time.sleep(100)

        # print(time.time()-start)

        for i in range(1000):
            action = env.action_space.sample()
            # action = np.zeros(8)
            action[0] = i
            obs, r, done, info = env.step(action)
            time.sleep(0.1)
            # print(obs)
            all_obs.append(obs)
            ep_r += r
        # print(ep_r)
        # print(np.max(all_obs))
        # print(np.min(all_obs))
        # print(time.time()-start)
        # print('')
        ep_r = 0