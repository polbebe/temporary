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

class PinkPantherEnv(gym.Env):
	def __init__(self, render=True):
		if render:
			self.physicsClient = p.connect(p.GUI)
			self.render = True
		else:
			self.render = False
		self.physicsClient = p.connect(p.DIRECT)
		self.mode = p.POSITION_CONTROL

		self.params = {
			'APS': 7, # Actions Per Second
			'maxForce': 1.667,
			'maxVel': 6.545,
			'gravity': -9.8,
			'act_noise': 0.01,
			'step': 2,
		}
		# TODO
		# setTimeStep should be uses here to modify the APS
		# Should keep a param of the joint states to be more in line with real life
		# need to figure out what the limits of the position control are

		self.stepper = 0

		self.ep_len = 0 # Ignore

		self.friction_values = [0.2, 0.4, 0.4, 0.2]

		#self.urdf = "C:/Users/polbe/OneDrive/Desktop/RESEARCH/PinkPanther/git-fork-multi_robot/body/PinkPanther/PinkPanther_CML/urdf/pp_urdf_final.urdf" #PC
		self.urdf = "body/PinkPanther/PinkPanther_CML/urdf/pp_urdf_final.urdf" #MAC

		obs = self.reset(self.friction_values)

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

	def _render(self, mode='human', close=True):
		pass

	def get_obs(self):
		self.last_p = self.p
		self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
		self.p, self.q = np.array(self.p), np.array(self.q)
		self.v = self.p - self.last_p

		jointInfo = [p.getJointState(self.robotid, i) for i in range(12)]
		# jointVals = np.array([[joint[0], joint[1]] for joint in jointInfo]).flatten()
		# Pos only
		jointVals = np.array([joint[0] for joint in jointInfo]).flatten()
		obs = np.concatenate([jointVals, self.q, np.array([self.v[1], self.v[0]])])
		# print(self.v)
		return obs

	def act(self, action):
		action = action + np.random.normal(0, max(self.params['act_noise'], 0)) #??? Introduces noise ?
		n_sim_steps = int(240/self.params['APS'])
		for i in range(n_sim_steps):
			p.stepSimulation()
		for i in range(len(action)):
			#pos, vel, forces, torque = p.getJointState(self.robotid, i)
			if i>5:
				p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=action[i], force=self.params['maxForce']/1.6, maxVelocity=self.params['maxVel']/1.)
			else:
				p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=action[i], force=self.params['maxForce']/1., maxVelocity=self.params['maxVel']/1.)

			#p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=action[i], force=self.params['maxForce'], maxVelocity=self.params['maxVel'])
		if self.render:
			time.sleep(1./self.params['APS'])

	def reset(self, friction_values):
		p.resetSimulation()
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0,0,self.params['gravity'])
		#planeId = p.loadURDF("C:/Users/polbe/OneDrive/Desktop/RESEARCH/PinkPanther/git-fork-multi_robot/body/PinkPanther/plane/plane.urdf") #PC
		planeId = p.loadURDF("body/PinkPanther/plane/plane.urdf") #MAC
		#standId = p.loadURDF("C:/Users/polbe/OneDrive/Desktop/RESEARCH/PinkPanther/git-fork-multi_robot/body/PinkPanther/PP_Stand_Thin/urdf/PP_Stand_Thin.urdf")
		robotStartPos = [0,0,0.2]
		robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
		self.robotid = p.loadURDF(self.urdf, robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION)
		# Setting Friction
		# Plane
		p.changeDynamics(planeId, -1, lateralFriction=friction_values[0], spinningFriction=friction_values[1], rollingFriction=0.0001)
		# Forearm links of robot
		p.changeDynamics(self.robotid, 2, lateralFriction=friction_values[2], spinningFriction=friction_values[3], rollingFriction=0.0001)
		p.changeDynamics(self.robotid, 5, lateralFriction=friction_values[2], spinningFriction=friction_values[3], rollingFriction=0.0001)
		p.changeDynamics(self.robotid, 8, lateralFriction=friction_values[2], spinningFriction=friction_values[3], rollingFriction=0.0001)
		p.changeDynamics(self.robotid, 11, lateralFriction=friction_values[2], spinningFriction=friction_values[3], rollingFriction=0.0001)
		
		self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
		self.p, self.q = np.array(self.p), np.array(self.q)
		self.i = 0
		self.set()
		# print(self.p)
		return self.get_obs()

	def set(self):
		n_sim_steps = int(240/self.params['APS'])
		# Reset down
		for j in range(20):
			for i in range(n_sim_steps):
				p.stepSimulation()
			pos = [0., 0.40854271, -0.9, 0., 0.4248062, -0.9, 0., 0.40854271, -0.9, 0., 0.4248062, -0.9]
			for i in range(12):
				p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=pos[i], force=self.params['maxForce'], maxVelocity=self.params['maxVel']/7)
			if self.render:
				time.sleep(1./self.params['APS'])
		# Reset up
		for j in range(20):
			for i in range(n_sim_steps):
				p.stepSimulation()
			vel = [1500, 1500, 1500, 1500, 1500, 1500, 1000, 1000, 1000, 1000, 1000, 1000]
			for x in range(len(vel)):
				vel[x] = vel[x]/150
			#pos = [0., 0.15, 0.09572864, 0., 0.15, 0.10310078, 0., 0.15, 0.09572864, 0., 0.15, 0.10310078] #Normal Stand up
			pos = [0., 0., 0., 0., 0., 0., 0., 0.15, 0.15, 0., 0.15, 0.15] # Params at t=0
			for i in range(12):
				p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=pos[i], force=self.params['maxForce'], maxVelocity=self.params['maxVel']/vel[i])
			if self.render:
				time.sleep(1./self.params['APS'])

	def finish(self):
		n_sim_steps = int(240/self.params['APS'])
		# Reset down
		for j in range(20):
			for i in range(n_sim_steps):
				p.stepSimulation()
			pos = [0., 0.40854271, -0.9, 0., 0.4248062, -0.9, 0., 0.40854271, -0.9, 0., 0.4248062, -0.9]
			for i in range(12):
				p.setJointMotorControl2(self.robotid, i, controlMode=self.mode, targetPosition=pos[i], force=self.params['maxForce'], maxVelocity=self.params['maxVel']/7)
			if self.render:
				time.sleep(1./self.params['APS'])


	def step(self, action):
		if ((self.stepper != 0) and (self.stepper%self.params['step'] == 0)):
			self.act(action)
			obs = self.get_obs()
			self.i += 1
			self.stepper = 0
			done = self.i >= self.ep_len
			# r = 100*obs[0] - 50*np.abs(obs[1])
			r = 100*obs[-1] # positive x direction
			# r = 100*obs[-2] # positive y direction (aka turn left)
			# r = -100*obs[-2] # positive y direction (aka turn right)
			# r = 100*obs[-1] - 100*np.abs(obs[-2]) # positive x direction
			# print(self.p)
			return obs, r, done, {}
		else:
			self.stepper += 1
			obs = self.get_obs()
			done = self.i >= self.ep_len
			r = 100*obs[-1]
			return obs, r, done, {}

	def get_dist_and_time(self):
		return p.getBasePositionAndOrientation(self.robotid), 400/(self.params['APS']*self.params['step'])

def get_action(steps):
	params = np.array([0.15, 0.0, 0.2, 0.15, 0.2]) # Smooth Criminal, Jul 31, 7pm
	return act(steps, *params)

def act(t, a, b, c, d, e):
	# Calculate desired position
	desired_p = np.zeros(12)

	pos_front_v = a * np.sin(t * e) + b
	neg_front_v = -a * np.sin(t * e) + b
	pos_back_v = c * np.sin(t * e) + d
	neg_back_v = -c * np.sin(t * e) + d

	front_pos = [1, 2]
	front_neg = [4, 5]
	back_pos = [10, 11]
	back_neg = [7, 8]

	zero = [0, 3, 6, 9]

	# Assign	
	desired_p[front_pos] = pos_front_v
	desired_p[front_neg] = neg_front_v
	desired_p[back_pos] = pos_back_v
	desired_p[back_neg] = neg_back_v
	desired_p[zero] = 0

	# Return desired new position
	return desired_p

def find_fric_params(env):
	rank = []
	h=1
	for i0 in np.arange(0.2, 1.1, 0.1):
		for i1 in np.arange(0.2, 1.1, 0.1):
			for i2 in np.arange(0.2, 1.1, 0.1):
				for i3 in np.arange(0.2, 1.1, 0.1):
					friction_values = [i0, i1, i2, i3]
					speed = test_fric_params(friction_values, env)
					friction_values.append(speed)
					rank.append(friction_values)
					print(h)
					h+=1
	rank = np.array(rank)
	r = np.argsort(rank[:,4])[::-1]
	return rank[r[0:25]]

def test_fric_params(friction_values, env):
	obs = env.reset(friction_values)
	for i in range(300):
		action = get_action(i)
		done = False
		while not done:
			obs, r, done, info = env.step(action)
	speed = (env.get_dist_and_time()[0][0][0])/(env.get_dist_and_time()[1])
	return speed


if __name__ == '__main__':

	env = PinkPantherEnv()
	#print(find_fric_params(env))
	
	obs = env.get_obs()
	start = time.time()
	for i in range(300):
		action = get_action(i)
		done = False
		while not done:
			obs, r, done, info = env.step(action)

	print("{} m".format(env.get_dist_and_time()[0][0][0]))
	print("{} s".format(env.get_dist_and_time()[1]))
	print('-----------------------')
	print('-----------------------')
	print("{} m/s".format((env.get_dist_and_time()[0][0][0])/(env.get_dist_and_time()[1])))
	print('-----------------------')
	print('-----------------------')
	
	env.finish()
	
	env.close()

'''
def find_params(env, episodes):
	params = np.random.uniform(-1, 1, (episodes, 4))
	params[:,0] = 0
	for i in range(len(params)):
		print("{} %".format(round((i/len(params)*100))))
		params[i,0] = test_params(params[i,1:], env)
	p = np.argsort(params[:,0])[::-1]
	return params[p[0:3],1:]

def test_params(params, env, steps=400):
	ep_r = 0
	obs = env.reset()
	for t in range(steps):
		action = act(obs, t, *params)
		obs, r, done, info = env.step(action)
		ep_r += r
	return ep_r
'''