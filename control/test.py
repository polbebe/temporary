import numpy as np
import os

a = np.array([0,0,0])
path = os.path.join('body/PinkPanther/params/', 'test')
np.save(path, a)