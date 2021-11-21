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

	def norm_act_space(self, action):
		return action

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

		obs = np.concatenate([jointVals, self.q, np.array([self.p[1], self.p[0]])])
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

	def reset(self, friction_values=[0.2, 0.4, 0.4, 0.2]):
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
	params = np.array([0.6162537877133565, -0.17955599878997153, -1.1469615868995122, -0.788777796338048, -0.9211594446016895, 3.788643986541802, 0.7831364145614903, -0.07644673916626318, 0.7963725522597993, -0.6087149179143061, -0.3691551342676434, -0.214704334811314, 0.08208945210299254, 0.5596633398663683, 0.057377333275989165, -0.24670969407966084])
	# sin_gait w=0.21 3
	#params = np.array([-0.1452217682136046, 0.5370238181034812, 5.453212634461867, 4.161790918008703, -1.6280157636978125, -2.764998743492415, -0.5724522688587933, 0.7226947508679249, -0.6998402793502201, 0.5072764835093281, 0.03661892351135113, 0.4627483024891589, 0.21236724167077375, 0.1380141387384276, -0.27684548026527517, -0.3643201944698517])
	# sin_gait w=0.21 1
	#params = np.array([0.0879295418444952, -0.5217358566212154, -0.25308960328228675, -0.28096463695472734, 0.3295681259564287, -2.886586337197183, 0.13908990158080425, -0.4865371925717228, 0.8783518529321535, -0.6994161938402221, -0.09986364047166477, -0.29092427861412606, -0.6957321060869033, 0.09587544814407088, -0.01097088041686454, -0.09192734983501984])
	#
	#params = np.array([0.9920927057932182, 0.5164030883375238, -3.322256571095119, 4.2768464224682585, -1.0927790196300982, -2.277696702294537, -0.1858473850320759, 0.5075329536360368, 0.6162669503446215, 0.6713007949350737, 0.007501607616797997, -0.34863016799330565, -0.6187162351887475, -0.030707793972982803, 0.0006323727582827828, -0.3255434312328852]) # Trained Gait 3, Oct 11 10:22
	#params = np.array([-0.7122210428763858, -0.22491948355929436, -4.030003511967614, 2.607989002553376, 0.880079337451659, 2.471760506420538, 0.0163464519508258, -0.10750315599363436, 
	#	0.48876663062682146, -0.9656945442456197, -0.1490729571499612, -0.01692894291012759, -0.7445498714426267, -0.49400815730450554, -0.25758313754708495, 0.02863263311193996]) # Trained Gait 1, Oct 1 3:32 (w=0.2 fixed)
	#params = np.array([0.31470387082742857, -0.8154062769257271, -2.0772560424688575, 2.2393159128234656, 5.595364311113712, -4.455943474961648, 0.8324012319915999, 0.8384610746398726,
	#	-0.962714136526561, 0.5549220113983584, -0.17926266842689542, -0.10680555082706357, -0.01921413464859436, -0.10046876925877438, 0.026450428076952335, 0.3312599623115464]) # Trained Gait 2, Oct 1 1:24 FAIL - forcing 0.2 without removing the offset is pointless, will try that next
	#params = np.array([0.35332285718099654, 0.5736420744741431, -5.979138131230912, -5.948577956716892, -0.36527256953492687, -3.7619673655447716, -0.8234816592801515, -0.2733447408644949, 
	#-0.8013625337201749, -0.7227010354901618, -0.10443282694344454, -0.4021021656079176, -0.17608276569715917, -0.5704636488269477, 0.08423802606152105, 0.006208623699152057]) # Trained Gait 1, Oct 1 00:56 FAIL - thesis that frequency of joints is too much, pivoting to a test with w=0.2
	return act(steps, *params)

def act(t, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15):
	# Calculate desired position
	desired_p = np.zeros(12)
	#LF
	desired_p[0] = 0 #p0 * np.sin(t*0.2) + p10
	desired_p[1] = p6 * np.sin(t*0.2) - p12
	desired_p[2] = p8 * np.sin(t*0.2) + p14
	#RF
	desired_p[3] = 0 #p1 * np.sin(t*0.2) + p11
	desired_p[4] = p7 * np.sin(t*0.2) - p13
	desired_p[5] = p9 * np.sin(t*0.2) + p15
	#LB
	desired_p[6] = 0 #p1 * np.sin(t*0.2) - p11
	desired_p[7] = p7 * np.sin(t*0.2) - p13
	desired_p[8] = p9 * np.sin(t*0.2) + p15
	#RB
	desired_p[9] = 0 #p0 * np.sin(t*0.2) - p10
	desired_p[10] = p6 * np.sin(t*0.2) - p12
	desired_p[11] = p8 * np.sin(t*0.2) + p14
	

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