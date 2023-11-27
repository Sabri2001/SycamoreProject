import pickle
import torch as th
import numpy as np


a = np.array([1., 2.])
print(a)

b = np.array([[1., 1.], [1., 1.]])
print(b)

b[1, :] += a
print(b)
