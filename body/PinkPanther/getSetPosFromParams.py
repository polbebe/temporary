import numpy as np

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

print(get_action(0))