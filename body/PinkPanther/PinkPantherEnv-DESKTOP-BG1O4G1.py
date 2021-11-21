import pybullet as p
import time
import pybullet_data
import gym
import time
import numpy as np
import xml.etree.ElementTree as ET
import random
import pickle
import os
import math

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw])

class PinkPantherEnv(gym.Env):
    def __init__(self, render=False):
        if render:
            self.physicsClient = p.connect(p.GUI)
            self.render = True
        else:
            self.render = False
        self.physicsClient = p.connect(p.DIRECT)
        self.mode = p.POSITION_CONTROL

        self.params = {
            'APS': 10,
            'maxForce': 4,
            'gravity': -10,
            'act_noise': 0.,
            'delay': 0.5
        }
        self.ep_len = 100

        self.urdf = "body/PinkPanther/PinkPanther_CML/urdf/PinkPanther_CML.urdf"
        obs = self.reset()

        obs_high = np.ones_like(obs)
        self.observation_space = gym.spaces.Box(high=obs_high, low=-obs_high)
        act_high = np.ones(12)
        self.action_space = gym.spaces.Box(high=act_high, low=-act_high)

    def close(self):
        p.disconnect(self.physicsClient)
        # p.resetSimulation()

    def save_config(self, path):
        pass

    def load_config(self, path):
        pass

    def _render(self, mode='human', close=False):
        pass

    def get_obs(self):
        self.last_p = self.p
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.p, self.q = np.array(self.p), np.array(self.q)
        self.rpy = quaternion_to_euler(*self.q)
        rp = self.rpy[:-1]
        self.v = self.p - self.last_p
        jointInfo = [p.getJointState(self.robotid, i) for i in range(12)]
        jointVals = np.array([joint[0] for joint in jointInfo]).flatten() # Pos only

        # 0-11 pos 12 roll 13 pitch, 14 delta y 15 delta x
        self.obs = np.concatenate([jointVals, rp, np.array([self.v[1], self.v[0]])])
        # print(self.v)
        return self.obs

    def act(self, action):
        # action = self.tg(params)
        action = action + np.random.normal(0, max(self.params['act_noise'], 0))
        n_sim_steps = int(240/self.params['APS'])
        delay_step = int((n_sim_steps-1)*self.params['delay'])
        for i in range(n_sim_steps):
            p.stepSimulation()
            if self.render:
                time.sleep(1./240.)
            if i == delay_step:
                for i in range(len(action)):
                    pos, vel, forces, torque = p.getJointState(self.robotid, i)
                    p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=pos + 0.1*action[i], force=self.params['maxForce'])
                    # p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=action[i], force=self.params['maxForce'])
        # if self.render:
        # #     time.sleep(1./self.params['APS'])
        #     print(time.time())

    def reset(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,self.params['gravity'])
        planeId = p.loadURDF("body/PinkPanther/assets/PinkPanther/plane/plane.urdf")
        robotStartPos = [0,0,0.2]
        robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.robotid = p.loadURDF(self.urdf, robotStartPos, robotStartOrientation)
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.p, self.q = np.array(self.p), np.array(self.q)
        self.i = 0
        p.setTimeStep(1./240.)
        # print(self.p)
        return self.get_obs()

    def tg(self, params):
        current_p = self.obs[:12]
        desired_p = np.zeros(12)
        w = params[0]
        params = params[1:]
        a, b, c = params[0::3], params[1::3], params[2::3]
        v = a + b * np.sin(self.i * w + c)
        pos = [1, 10, 2, 11]
        neg = [4, 7, 5, 8]
        desired_p[pos+neg] = v
        # desired_p[pos] = v
        # desired_p[neg] = -v

        delta_p = desired_p - current_p
        delta_p = np.clip(delta_p, -1, 1)
        return delta_p

    def step(self, action):
        self.act(action)
        obs = self.get_obs()
        self.i += 1
        done = self.i >= self.ep_len
        # r = 100*obs[0] - 50*np.abs(obs[1])
        r = 100*obs[-1] # positive x direction
        # r = 100*obs[-2] # positive y direction (aka turn left)
        # r = -100*obs[-2] # positive y direction (aka turn right)
        # r = 100*obs[-1] - 100*np.abs(obs[-2]) # positive x direction
        # print(self.p)
        return obs, r, done, {}

class RandPinkPantherEnv(gym.Env):
    def __init__(self, render=False, noise_percent=50, urdf_path=None):
        self.env = PinkPantherEnv(render)
        self.tree = ET.parse(self.env.urdf)
        root = self.tree.getroot()
        attrs = {
            'mass': ['value'],
            'inertia': ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'],
            'limit': ['lower', 'upper', 'effort', 'velocity'],
        }
        noise = {
            'APS': 5,
            'maxForce': 2,
            'gravity': 3,
            'act_noise': 0.1,
            'delay': 0.3
        }
        mins = {
            'APS': 0.5,
            'maxForce': 0.5,
            'gravity': -np.inf,
            'act_noise': 0,
            'delay': 0
        }
        maxes = {
            'APS': 50,
            'maxForce': np.inf,
            'gravity': -1,
            'act_noise': np.inf,
            'delay': 1
        }
        noise_percent /= 100
        for key in attrs.keys():
            for attr in root.iter(key):
                for sub_key in attrs[key]:
                    value = float(attr.attrib[sub_key])
                    value += random.gauss(0, noise_percent*value)
                    attr.set(sub_key, str(value))
        for key in self.env.params.keys():
            delta = random.gauss(0, noise[key])
            self.env.params[key] += delta
            self.env.params[key] = min(maxes[key], max(mins[key], self.env.params[key]))
        if urdf_path is None:
            rand_str = ''.join([str(random.randint(0,9)) for _ in range(10)])
            urdf_path = 'body/PinkPanther/PinkPanther_CML/urdf/PinkPantherRand_'+rand_str+'.urdf'
        self.tree.write(urdf_path)
        self.env.urdf = urdf_path
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_config(self):
        return self.env.params, self.tree

    def set_config(self, config):
        urdf_path, self.env.params = config
        self.tree = ET.parse(self.env.urdf)
        rand_str = ''.join([str(random.randint(0,9)) for _ in range(10)])
        self.env.urdf = 'body/PinkPanther/PinkPanther_CML/urdf/PinkPantherRand_'+rand_str+'.urdf'
        self.tree.write(self.env.urdf)

    def save_config(self, path):
        pickle.dump(self.env.params, open(path+'.pkl', 'wb+'))
        self.tree.write(path+'.urdf')

    def load_config(self, path):
        self.set_config((path+'.urdf', pickle.load(open(path+'.pkl', 'rb'))))

    def close(self):
        self.env.close()
        os.remove(self.env.urdf)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

# Could make this more efficient my turning this into a wrapper for any PP env
class ConstPinkPantherEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        act_high = np.ones(8)
        self.action_space = gym.spaces.Box(high=act_high, low=-act_high)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        a = np.zeros(12)
        a[1::3] = action[0::2]
        a[2::3] = action[1::2]
        return self.env.step(a)

class RandConstPinkPantherEnv(RandPinkPantherEnv):
    def __init__(self, render=False, noise_percent=50, urdf_path=None):
        super(RandConstPinkPantherEnv, self).__init__(render, noise_percent, urdf_path)
        act_high = np.ones(8)
        self.action_space = gym.spaces.Box(high=act_high, low=-act_high)

    def step(self, action):
        a = np.zeros(12)
        a[1::3] = action[0::2]
        a[2::3] = action[1::2]
        return self.env.step(a)

class HardPinkPantherEnv(gym.Env):
    def __init__(self, render=False, frozen_joints=None, values=None):
        if frozen_joints is None:
            frozen_joints = np.random.choice(12, 4)
        self.frozen_joints = frozen_joints
        if values is None:
            self.frozen_values = np.random.uniform(-1, 1, len(frozen_joints))
        else:
            self.frozen_values = values
        self.env = PinkPantherEnv(render)
        act_high = np.ones(8)
        self.action_space = gym.spaces.Box(high=act_high, low=-act_high)
        self.observation_space = self.env.observation_space

    def save_config(self, path):
        np.save(path + '.npy', self.get_config())

    def load_config(self, path):
        self.set_config(np.load(path+'.npy'))

    def close(self):
        self.env.close()

    def get_config(self):
        return np.concatenate([self.frozen_joints, self.frozen_values]).reshape(2, -1)

    def set_config(self, config):
        self.frozen_joints, self.frozen_values = config[0].astype(np.int), config[1]

    def step(self, action):
        new_act = np.zeros(12)
        i, j = 0, 0
        while i < 12 and j < 8:
            while i in self.frozen_joints:
                i += 1
                if not i < 12: break
            new_act[i] = action[j]
            i += 1
            j += 1
            # if j >= 8 and i < 12:
            #     print()
        for i in range(len(self.frozen_joints)):
            j = self.frozen_joints[i]
            new_act[j] = self.frozen_values[i]
        return self.env.step(new_act)

    def reset(self):
        return self.env.reset()

def EasyPinkPantherEnv(render=False, values=None):
        shoulders = np.array([0, 3, 6, 9])
        return HardPinkPantherEnv(render=render, frozen_joints=shoulders, values=values)

if __name__ == '__main__':
    env = RandPinkPantherEnv(noise_percent=10)
    env = RandConstPinkPantherEnv(noise_percent=10)
    path = './'
    env_string = 'randConst'
    idx = 0
    for i in range(10000):
        done = False
        while not done:
            obs, r, done, info = env.step(env.action_space.sample())
        print(i)
    env.save_config(path+env_string+'_test_config_'+str(idx))
    env.close()
