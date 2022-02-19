import numpy as np
from fns import *
import sys


# TEST FOR CLIPPING LIKE IN SIMULATION FOR REAL ROBOT

# MAX delta_pos allowed in any given movement
delta_p = 0.21


# Call corresponding function to convert sim2real/real2sim
def convFns(pos, convType):
	conv =	[left_armpit, left_elbow, left_shoulder, right_armpit, right_elbow, right_shoulder, 
			left_armpit, left_elbow, left_shoulder, right_armpit, right_elbow, right_shoulder]
	targ = np.zeros(12)
	for i in range(len(pos)):
		if i==0:
			targ[i] = conv[i](pos[i], convType, "front")
		elif i==6:
			targ[i] = conv[i](pos[i], convType, "back")
		else:
			targ[i] = conv[i](pos[i], convType)
	return targ

# Front and back legs diff
# Return target position
def act(t, a, b, c, d, e, f, obs):
	# Calculate desired position
	f_pos = a * np.sin(t * e) + b
	f_neg = -a * np.sin(t * e) + b
	b_pos = c * np.sin(t * e + f) + d
	b_neg = -c * np.sin(t * e + f) + d

	# Convert obs to sim
	obs = convFns(obs, "real2sim")
	# Desired action
	desired_p = [0, f_pos, f_pos, 0, f_neg, f_neg, 0, b_pos, b_pos, 0, b_neg, b_neg]
	# Delta action from current position
	delta = [desired_p[i]-obs[i] for i in range(len(desired_p))]
	# Clip delta to desired max delta pos
	delta = np.clip(delta, -delta_p, delta_p)
	# Update desired action to only change by delta
	desired_p = [obs[i]+delta[i] for i in range(len(delta))]

	# Return desired new position
	return convFns(desired_p, "sim2real")

# Return position to take
def get_action(steps, obs):
	#params = np.array(np.load('params/HillClimber/23_01_2022/best_overall.npy'))
	params = np.array([ 0.31607133, -0.04617572, -0.25435251,  0.09736614, -8.81590009,  6.12591908]) # 12_02_2022 params trained on envs auto-tuned to be close to the env I manually tuned
	#params = np.array([-0.16476964, 0.02548534, 0.16893791, 0.09441782, 9.44620473, -6.1950588]) # 27_01_2022 params trained on envs auto-tuned to be close to the env I manually tuned
	#params = np.array([ 0.22853782, 0.06146434, 0.25060128, 0.09051928, 10.81942692, 2.98455422]) # 31_01_2022 params trained on envs auto-tuned to be close to the env I manually tuned
	#params[4]-=4
	#params = np.array([0.24495851730947005, 0.18187873796178136, 0.2020333429029758, -0.3852743697870839, -0.2094960812992037]) # Trained sin_gait 7, Oct 11 19:01
	#params = np.array([0.2980418533307479, 0.01878523690431866, 0.022546654023646796, -0.2685025304630598, -0.2080157428428239]) # Trained sin_gait 5, Oct 12 13:21
	#params = np.array([0.15, 0.0, 0.2, 0.15, 0.2]) # Smooth Criminal
	#params = np.array([0.15, 0.0, 0.19, 0.2, 0.23, 2.05])
	return act(steps, *params, obs)




# RESET position and stand down & up before walking
pos = [500, 750, 608, 500, 250, 390, 500, 750, 608, 500, 250, 390]

real_poses = []

# WALK
j = 1
real_pos = pos
while j < 100:
	# Keep track of previous real robot pos
	prev_pos = real_pos
	# Get target position (with Clipping)
	action = get_action(j, prev_pos)
	# Move robot to target position
	pos = action
	# Update real pos (read)
	real_pos = pos

	real_poses.append(real_pos)

	j += 1


delta_max_poses = []

for j in range(len(real_poses[0])):
	delta_max_finder = []
	for i in range(len(real_poses)-1):
		delta_max_finder.append(abs(real_poses[i+1][j] - real_poses[i][j]))
	delta_max_poses.append(max(delta_max_finder))

print(delta_max_poses)

print()



'''
# TEST FOR MAX DELTA POSITION IN SIMULATION

actions = np.load("body/PinkPanther/params/Tests/12_02_2022_actual_actions.npy")

real_poses = []


for i in range(len(actions)):
	sim_pos = actions[i]
	real_pos = left_shoulder(sim_pos, "sim2real")
	real_poses.append(real_pos)
	print('Sim: {}	Real: {}'.format(sim_pos, real_pos))

print(max(real_poses))
print(min(real_poses))


# Call corresponding function to convert sim2real/real2sim
def convFns(pos, convType):
	conv =	[left_armpit, left_elbow, left_shoulder, right_armpit, right_elbow, right_shoulder, 
			left_armpit, left_elbow, left_shoulder, right_armpit, right_elbow, right_shoulder]
	targ = np.zeros(12)
	for i in range(len(pos)):
		if i==0:
			targ[i] = conv[i](pos[i], convType, "front")
		elif i==6:
			targ[i] = conv[i](pos[i], convType, "back")
		else:
			targ[i] = conv[i](pos[i], convType)
	return targ


for i in range(len(actions)):
	sim_pos = actions[i]
	real_pos = convFns(sim_pos, "sim2real")
	real_poses.append(real_pos)
	#print(real_pos)



delta_max_poses = []

for j in range(len(actions[0])):
	delta_max_finder = []
	for i in range(len(actions)-1):
		delta_max_finder.append(abs(actions[i+1][j] - actions[i][j]))
	delta_max_poses.append(max(delta_max_finder))

print(delta_max_poses)

print()

max_poses = []
min_poses = []

for j in range(len(real_poses[0])):
	max_min_finder = []
	for i in range(len(real_poses)):
		max_min_finder.append(real_poses[i][j])
	max_poses.append(max(max_min_finder))
	min_poses.append(min(max_min_finder))

print(max_poses)
print(min_poses)

print()



# TEST FOR DELTA POSITION FROM SIMULATION

folder = '23_01_2022'
gait = 'best_overall'
actions = np.load("body/PinkPanther/params/HillClimber/{}/{}_actions.npy".format(folder, gait))
#actions = np.load('body/PinkPanther/params/ROB/new-0.npy')
print(actions)

delta_actions = []

for i in range(len(actions)-1):
	delta_action = []
	for j in range(len(actions[0])):
		delta_action.append(abs(actions[i+1][j] - actions[i][j]))
	delta_actions.append(delta_action)

max_delta_actions = []
for j in range(len(delta_actions[0])):
	m = 0
	for i in range(len(delta_actions)):
		n = delta_actions[i][j]
		if n>m:
			m=n
	max_delta_actions.append(m)

print(max_delta_actions)
'''