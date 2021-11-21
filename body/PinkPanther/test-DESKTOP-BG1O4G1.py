import numpy as np
'''
rank = [[0.1, 0.2],
		[0.1, 0.1],
		[0.1, 0.0],
		[0.1, 0.3],
		[0.1, 0.25],
		[0.1, 0.4]]

rank.append([0.1, 0.5])
rank = np.array(rank)

for i in range(len(rank)):
	rank[i,1] = int(rank[i,1]*100)

r = np.argsort(rank[:,1])[::-1]

print(rank[r[:2]])
'''

print(repr(np.load('body/PinkPanther/params/test.npy')))