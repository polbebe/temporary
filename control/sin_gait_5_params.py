import numpy as np
import torch
import random
from collections import deque
device = 'cuda'

def sin_move(ti, para):

	s_action = np.zeros(12)

	pos_front_v = para[0] * np.sin(ti * para[4]) + para[1]
	neg_front_v = -para[0] * np.sin(ti * para[4]) + para[1]
	pos_back_v = para[2] * np.sin(ti * para[4]) + para[3]
	neg_back_v = -para[2] * np.sin(ti * para[4]) + para[3]

	front_pos = [1, 2]
	front_neg = [4, 5]
	back_pos = [10, 11]
	back_neg = [7, 8]
	
	zero = [0, 3, 6, 9]

	# Assign    
	s_action[front_pos] = pos_front_v
	s_action[front_neg] = neg_front_v
	s_action[back_pos] = pos_back_v
	s_action[back_neg] = neg_back_v
	s_action[zero] = 0

	return s_action
# Change the range that each parameter can achieve
def random_para():
	para = np.zeros(16)
	for i in range(16):
		para[i] = random.uniform(-1, 1)
	# Amplitude for front shoulder/elbow
	for i in [0]:
		para[i] = abs(para[i]*0.15)
	# Amplitude for back shoulder/elbow
	for i in [2]:
		para[i] = abs(para[i]*0.3)
	# Offset
	for i in [1,3]:
		para[i] *= 0.1
	# Forcing frequency
	for i in [4]:
		para[i] *= 0.21
	return para


def batch_random_para(para_batch):
	for i in range(16):
		para_batch[i][i] = random.uniform(-1, 1)
		if i in [0]:
			para_batch[i][i] = abs(para_batch[i][i]*0.15)
		elif i in [2]:
			para_batch[i][i] = abs(para_batch[i][i]*0.3)
		elif i in [1,3]:
			para_batch[i][i] *= 0.1
		elif i in [4]:
			para_batch[i][i] *= 0.21
		elif i in [6,7,8,9]:
			para_batch[i][i] *= 1-abs(para_batch[i][i+6])
		elif i in [10,11]:
			para_batch[i][i] *= 1-abs(para_batch[i][i-10])
		elif i in range(12,16):
			para_batch[i][i] *= 1-abs(para_batch[i][i-6])
		# if para_batch[i][8] + para_batch[i][14]>1:
		#     print('debug',i,para_batch[i][8],para_batch[i][14],para_batch[i][i-6])
	return para_batch


def find_params(env, epochs=100, rew_fn=lambda x: 100*x[0]-50*x[1], verbose=False):
	para_batch = [random_para() for _ in range(20)]
	best_para = []
	last_result = best_result = - np.inf
	epsilon = 1
	early_stop = 20
	improvements = deque(maxlen=20)
	for epoch in range(epochs):
		if verbose:
			print(epoch)
		if env.render == 0:
			# Random 16 individuals with different parameter region.
			para_batch = batch_random_para(para_batch)

			# Random all parameters of these 4 robots.
			for i in range(16, 20):
				para_batch[i] = random_para()

		update_para_batch = 0

		for individual in range(20):
			fail = 0
			para = para_batch[individual]
			result = 0
			env.reset()
			for i in range(100):
				action = sin_move(i, para)
				action = env.norm_act_space(action)
				obs, r, done, info = env.step(action)
				result = rew_fn(obs)

			if fail == 0:
				if result > best_result:
					if verbose:
						print(epoch, result, best_result)
					update_para_batch = 1
					best_result = result
					best_para = para
					# np.savetxt("log%s/%d.csv"%(name,epoch),para)
					# np.savetxt("log%s/0.csv" % name, para)

		if update_para_batch == 1:
			para_batch = np.asarray([best_para] * 20)
		improvements.append(best_result - last_result)
		last_result = best_result
		if len(improvements) == early_stop and max(improvements) < epsilon:
			break
	return best_para, best_result

def test_model(model, env, parameters=None, rew_fn = lambda x: 100 * x[0] - 50 * x[1], seq_num=4, num_traj=50):
	if parameters is None:
		print('Discovering good initial parameters...')
		parameters = find_params(env, rew_fn=rew_fn)
		print('Parameters found')
	real_reward = 0
	chosen_actions = []

	env.reset()
	for step in range(100):
		reward_traj = []
		all_action = []
		action_idx = 0
		for traj in range(num_traj):

			first_action = []
			s = env.get_obs()

			reward = 0
			for seq in range(seq_num):
				a = sin_move(step + seq, parameters)
				a = np.random.normal(loc=a, scale=0.1, size=None)
				a = env.norm_act_space(a)
				a = np.array([a]).astype(np.float32)
				if seq == 0:
					first_action = a
				a = torch.from_numpy(a).to(device)
				s = np.array([s]).astype(np.float32)
				s = torch.from_numpy(s).to(device)
				pred_ns = model(s, a)

				# transfer the next state from tensor format to numpy format
				# for simple network
				pred_ns_numpy = pred_ns.cpu().detach().numpy().flatten()
				s = pred_ns_numpy

				# Compute Rewards
				# Sum up the rewards on each future step.
				# reward += (100 * pred_ns_numpy[1] - 50 * np.abs(pred_ns_numpy[0]))  # *((0.65)**seq)
				reward += rew_fn(pred_ns_numpy)

			reward_traj.append(reward)
			# Only choose the first action
			all_action.append(first_action[0])
			action_idx = int(np.argmax(reward_traj))

		choose_next_action = all_action[action_idx]
		chosen_actions.append(choose_next_action)
		obs, real_reward, done, info = env.step(choose_next_action)

	return real_reward, chosen_actions

if __name__ == '__main__':
	from body.PinkPanther.PinkPantherEnv_Read import PinkPantherEnv
	rew_fn = lambda x: 100*x[-1] - abs(50*x[-2])
	
	env = PinkPantherEnv(render=False)
	state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
	params, best_result = find_params(env, rew_fn=rew_fn, verbose=True)
	print('Params found with best score: '+str(best_result))
	print(list(params))

	# Moving Forward
	# params = [0.03489315684881553, 0.322809426373365, -0.6470588782990053, -3.222442427918316,
	# -5.7752608974377, 2.6483466496320314, 0.9421123317438469, -0.8062854925739611,
	# 0.4749567185462493, -0.8466514900531514, -0.008048548586370155, 0.024656979394326463,
	# -0.041132682893396624, -0.09650443132141479, 0.07038930201197911, 0.036132520948350365]
	import time
	env = PinkPantherEnv(render=False)
	all_high, highs = 0, []
	for j in range(10):
		x, y = [], []
		obs = env.reset()
		x.append(obs[-1])
		y.append(obs[-2])
		for i in range(100):
			action = sin_move(i, params)
			action = env.norm_act_space(action)
			obs, r, done, info = env.step(action)
			x.append(obs[-1])
			y.append(obs[-2])
			result = rew_fn(obs)
			# time.sleep(0.01)
		import matplotlib.pyplot as plt
		print(j, result)
		plt.plot(x, y)
		highs.append(max(np.max(np.abs(x)), np.max(np.abs(y)))*1.1)
		all_high = max(highs)
	plt.xlim(-all_high, all_high)
	plt.ylim(-all_high, all_high)
	plt.show()