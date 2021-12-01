import numpy as np
import torch
import random
import time
import os


# Create a random set of parameters
def random_para():
	para = np.zeros(6)
	for i in range(6):
		p = random.uniform(-1, 1)
		para[i] = weight_para(p, i)
	return para


# Given index of parameter, multiply by appropriate weight
def weight_para(para, indx):
	# amplitude for motors
	if indx in [0, 2]:
		para *= 0.4
	# offset for motors
	if indx in [1, 3]:
		para *= 0.1
	# frequency of motion
	if indx in [4]:
		para *= 0.4
	# offset frequencies
	if indx in [5]:
		para *= np.pi
	return para


# Run simulation with given set of params and return reward
def run_sim(env, para):
	obs = env.reset()
	reward = 0
	for i in range(100):
		action = act(i, para)
		obs, r, done, info, rew = env.step(action)
		reward += rew
	#reward += 10 * (env.get_dist_and_time()[0][0][0]/0.1)
	return reward


# Calculate action
def act(t, para):

	# front positive
	f_pos = para[0] * np.sin(t * para[4]) + para[1]
	# front negative
	f_neg = -para[0] * np.sin(t * para[4]) + para[1]
	# back positive
	b_pos = para[2] * np.sin(t * para[4] + para[5]) + para[3]
	# back negative
	b_neg = -para[2] * np.sin(t * para[4] + para[5]) + para[3]

	desired_p = np.array([0, f_pos, f_pos, 0, f_neg, f_neg, 0, b_pos, b_pos, 0, b_neg, b_neg])

	# Return desired new position
	return desired_p

# Create neighbour sets of parameters
def neighbour_para(current_best):
	para_n_batch = []
	for i in range(len(current_best)-1):
		new_neighbour = []
		for j in range(len(current_best)):
			new_neighbour.append(current_best[j])
		# Sequentially change a param by a known amount
		sign = random.randrange(2)
		if sign==0:
			new_neighbour[i] += 0.01
		else:
			new_neighbour[i] -= 0.01
		# Randomly change other params - hillclimber only adjusts one element in the vector at a time (Wikipedia), so won't use it for now
		x = 0
		while (x < 0):
			r = random.randrange(8)
			if r == i:
				continue
			else:
				new_neighbour[r] += random.uniform(-0.1, 0.1)
				x += 1
		para_n_batch.append(np.array(new_neighbour))
	return para_n_batch


# Get best neighbour
def best_neighbour(env, para_batch):
	best_rew = - np.inf
	best_para = []
	r = 0
	for i in para_batch:
		r = run_sim(env, i)
		if r > best_rew:
			best_rew = r
			best_para = i
	return [best_rew, best_para]


# Discrete space hill climbing algorithm (https://en.wikipedia.org/wiki/Hill_climbing)
def discrete_hill_climber(env):
	para = random_para()
	rew = run_sim(env, para)
	current_best = [rew, para]
	init_batch = neighbour_para(current_best[1])
	next_best = best_neighbour(env, init_batch)
	while next_best[0] > current_best[0]:
		current_best = next_best
		next_batch = neighbour_para(current_best[1])
		next_best = best_neighbour(env, next_batch)
	return current_best


# Continuous space hill climbing algorithm (https://en.wikipedia.org/wiki/Hill_climbing)
def continuous_hill_climber(env):
	e = 0
	currentPoint = random_para()
	bestScore = run_sim(env, currentPoint)
	stepSize = np.zeros(6)
	for i in range(len(stepSize)):
		stepSize[i] = 0.01
	accel = 1.2
	candidate = np.array([-accel, -1/accel, 1/accel, accel])
	epsilon = 1
	beforeScore = bestScore-2*epsilon
	while ((bestScore - beforeScore) > epsilon):
		print('{} - {}'.format(e, bestScore))
		beforeScore = bestScore
		for i in range(len(stepSize)):
			beforePoint = currentPoint[i]
			bestStep = 0
			for j in candidate:
				step = stepSize[i]*j
				currentPoint[i] = beforePoint+step
				score = run_sim(env, currentPoint)
				if score > bestScore:
					bestScore = score
					bestStep = step
			if bestStep == 0:
				currentPoint[i] = beforePoint
			else:
				currentPoint[i] = beforePoint + bestStep
				stepSize[i] = bestStep
		e+=1
	return [bestScore, currentPoint]


if __name__ == '__main__':
	from body.PinkPanther.PinkPantherEnv_6 import PinkPantherEnv
	env = PinkPantherEnv(render=False)
	epochs = 20
	epoch_master = [-np.inf, []]
	epoch_runnerup = [-np.inf, []]
	print()
	print()
	for i in range(epochs):
		print()
		print('EPOCH {}'.format(i+1))
		# perform hill climber
		best = continuous_hill_climber(env)
		path = os.path.join('body/PinkPanther/params/train1', 'best_epoch{}'.format(i+1))
		np.save(path, best[1])
		print('Score of best gait: {}'.format(best[0]))
		# update best so far
		if best[0] > epoch_master[0]:
			epoch_runnerup = epoch_master
			epoch_master = best
		elif best[0] > epoch_runnerup[0]:
			epoch_runnerup = best
	print()
	print()
	print('BEST GAIT OVERALL: {}'.format(epoch_master[0]))
	path = os.path.join('body/PinkPanther/params/train0', 'best_overall')
	np.save(path, epoch_master[1])
	print()
	print('RUNNERUP: {}'.format(epoch_runnerup[0]))
	path = os.path.join('body/PinkPanther/params/train0', 'runnerup_overall')
	np.save(path, epoch_runnerup[1])