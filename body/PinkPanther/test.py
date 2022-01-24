import numpy as np
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
