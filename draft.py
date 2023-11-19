from torch import nn
import torch as th
import gymnasium as gym
import numpy as np
import torch


coeff = nn.Parameter(torch.randn(6), requires_grad=True)
print(np.array(coeff.data))
