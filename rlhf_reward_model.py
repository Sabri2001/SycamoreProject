"""
@author: elamrani
"""

from torch import nn
import torch.nn.functional as F
import torch as th
import abc
import numpy as np
import torch


from internal_models import StateEncoderOE


class RewardModel():
    """Minimal abstract reward model.

    Only requires the implementation of a forward pass (calculating rewards given
    a batch of states, actions, next states and dones).
    """

    def __init__(
        self
    ):
        """Initialize the Reward Model"""


    @abc.abstractmethod
    def forward(
        self
    ):
        """Compute rewards for a trajectory."""


class RewardLinearNoTorch(RewardModel):
    """Reward is a linear combination of handcrafted features (cf. Gabriel's thesis)"""

    def __init__(
        self,
        gamma,
        coeff = None
    ):
        """
        Initialize the Linear Reward Model.
        """
        self.gamma = gamma
        if coeff== None:
            self.coeff = np.array([0.,0.,0.,0.,0.,0.])
        else:
            self.coeff = coeff

    def get_coeff(self):
        return self.coeff

    def reward_trajectory(
        self,
        trajectory
    ):
        """Compute reward for a trajectory."""
        reward = 0
        for i, transition in enumerate(trajectory):
            reward += self.gamma**i * self.reward_transition(transition)
        return reward
    
    def reward_transition(
        self,
        transition
    ):
        return np.dot(self.get_coeff(), transition.reward_features)

    def reward_array_features(
        self,
        reward_array
    ):
        return np.dot(self.get_coeff(), reward_array)


class RewardLinear(nn.Module):
    def __init__(self, gamma, logger, device, coeff=None, seed=None):
        super(RewardLinear, self).__init__()
        self.gamma = gamma
        self.logger = logger
        self.device = device
        if coeff is None:
            if seed:
                torch.manual_seed(seed)
            self.coeff = nn.Parameter(torch.randn(6, device=device), requires_grad=True)
            self.logger.info(f"\n \n ---> Initial reward: {self.get_reward_coeff()} \n")
        else:
            self.coeff = nn.Parameter(torch.tensor(coeff, device=device, dtype=torch.float32), requires_grad=True)

    def reward_trajectory(self, trajectory):
        """Compute reward for a trajectory."""
        reward = 0.
        for i, transition in enumerate(trajectory):
            reward += self.gamma ** i * self.reward_transition(transition)
        return reward

    def reward_transition(self, transition):
        return torch.dot(self.coeff, transition.reward_features)

    def reward_array_features(self, reward_array):
        return torch.dot(self.coeff, reward_array)
    
    def get_reward_coeff(self):
        return np.array(th.Tensor.cpu(self.coeff.data))

    def normalize_reward(self):
        # self.coeff.data -= th.mean(self.coeff.data)
        norm_factor = th.linalg.norm(self.coeff.data)
        if norm_factor > 5.40: # same l2-norm as Gab's modular reward
            self.coeff.data /= norm_factor
            self.coeff.data *= 5.40


class RewardLinearEnsemble(nn.Module):
    def __init__(self, gamma, nb_rewards, logger, device):
        self.nb_rewards = nb_rewards
        self.reward_list = [RewardLinear(gamma, logger, device, seed=30+i) for i in range(nb_rewards)]
        self.device = device
        self.logger = logger
        self.gamma = gamma

    def reward_trajectory(self, trajectory):
        """Compute reward for a trajectory."""
        reward = 0
        for i, transition in enumerate(trajectory):
            reward += self.gamma ** i * self.reward_transition(transition)
        return reward

    def reward_transition(self, transition):
        reward = 0
        for reward_model in self.reward_list:
            reward += torch.dot(reward_model.coeff, transition.reward_features)
        return reward/self.nb_rewards

    def reward_array_features(self, reward_array):
        reward = 0
        for reward_model in self.reward_list:
            reward += torch.dot(reward_model.coeff, reward_array)
        return reward/self.nb_rewards
    
    def get_reward_coeff(self):
        coeff_list = []
        for reward_model in self.reward_list:
            coeff_list.append(np.array(th.Tensor.cpu(reward_model.coeff.data)))
        return coeff_list

    def normalize_reward(self):
        for reward_model in self.reward_list:
            reward_model.normalize_reward()

    def reward_disagreement(self, trajectory_pair):
        # Compute reward for each traj according to each reward model
        reward_ar = th.zeros((2,self.nb_rewards), device=self.device)
        for traj_idx in range(2):
            for i, transition in enumerate(trajectory_pair[traj_idx].get_transitions()):
                reward_ar[traj_idx, :] += self.gamma ** i * self.reward_transition_per_reward(transition)

        # Deduce preferences according to each reward model
        pref_ar = np.zeros(self.nb_rewards)
        for idx in range(self.nb_rewards):
            if reward_ar[0, idx] > reward_ar[1,idx]:
                pref_ar[idx] = 0
            elif reward_ar[0, idx] < reward_ar[1,idx]:
                pref_ar[idx] = 1
            else:
                pref_ar[idx] = 0.5

        # Return std of preferences (= disagreement)
        return np.std(pref_ar)
    
    def reward_transition_per_reward(self, transition):
        reward_ar = th.zeros(self.nb_rewards, device=self.device)
        for idx, reward_model in enumerate(self.reward_list):
            reward_ar[idx] += torch.dot(reward_model.coeff, transition.reward_features)
        return reward_ar


class RewardCNN(nn.Module):
    def __init__(self,
                 gamma,
                 logger,
                 device,
                 maxs_grid,
                 n_robots,
                 n_regions,
                 config):
        super().__init__()
        #unpack the config file
        n_fc_layer = config['SAC_n_fc_layer']
        n_neurons = config['SAC_n_neurons']
        batch_norm = config['SAC_batch_norm']
        device = config['torch_device']
        encoder_args = {'n_channels':config['SEnc_n_channels'],
                        'n_internal_layer':config['SEnc_n_internal_layer'],
                        'stride':config['SEnc_stride']}
        
        self.state_encoder = None

        if config['SEnc_order_insensitive']:
            self.state_encoder = StateEncoderOE(maxs_grid,
                                          n_robots,
                                          n_regions,
                                          config['agent_last_only'],
                                          device=device,
                                          **encoder_args)
            
        self.input_norm = nn.BatchNorm2d(self.state_encoder.out_dims[0],device=device)
        self.FC = nn.ModuleList([nn.Linear(np.prod(self.state_encoder.out_dims),n_neurons,device=device)])
        self.FC+=nn.ModuleList([nn.Linear(n_neurons,n_neurons,device=device) for i in range(n_fc_layer-1)])
        self.out_reward = nn.Linear(n_neurons,1,device=device) # output of size 1 for reward
        self.tanh = nn.Tanh()
        self.device=device
        self.batch_norm = batch_norm
        self.gamma = gamma
        self.logger = logger

    def forward(self,grids,inference = False):
        with torch.inference_mode(inference):
            if inference:
                self.eval()
            else:
                self.train()
            if self.batch_norm:
                normed_rep = self.input_norm(self.state_encoder(grids))
                rep = torch.flatten(normed_rep,1)
            else:
                rep = torch.flatten(self.state_encoder(grids),1)
            for layer in self.FC:
                rep = F.relu(layer(rep))
            reward = self.tanh(self.out_reward(rep))*5.40 # reward of same magnitude as Gab's
            return reward.view([])

    def reward_trajectory(self, trajectory):
        """Compute reward for a trajectory."""
        reward = 0
        for i, transition in enumerate(trajectory):
            reward += self.gamma ** i * self.reward_transition(transition)
        return reward

    def reward_transition(self, transition):
        grid = transition.new_state['grid']
        return self.forward([grid])

    def reward_array_features(self, grid):
        return self.forward([grid], inference=True)


class RewardCNNEnsemble(nn.Module):
    def __init__(self,
                 gamma,
                 nb_rewards,
                 logger,
                 device,
                 maxs_grid,
                 n_robots,
                 n_regions,
                 config):
        self.nb_rewards = nb_rewards
        self.reward_list = [RewardCNN(gamma, logger, device, maxs_grid, n_robots, n_regions, config) for _ in range(nb_rewards)]
        self.device = device
        self.logger = logger
        self.gamma = gamma

    def reward_trajectory(self, trajectory):
        """Compute reward for a trajectory."""
        reward = 0
        for i, transition in enumerate(trajectory):
            reward += self.gamma ** i * self.reward_transition(transition)
        return reward

    def reward_transition(self, transition):
        reward = 0
        for reward_model in self.reward_list:
            reward += reward_model.reward_transition(transition)
        return reward/self.nb_rewards

    def reward_array_features(self, grid):
        reward = 0
        for reward_model in self.reward_list:
            reward += reward_model.reward_array_features(grid)
        return reward/self.nb_rewards

    def reward_disagreement(self, trajectory_pair):
        # Compute reward for each traj according to each reward model
        reward_ar = np.zeros((2,self.nb_rewards))
        for traj_idx in range(2):
            for i, transition in enumerate(trajectory_pair[traj_idx].get_transitions()):
                reward_ar[traj_idx, :] += self.gamma ** i * self.reward_transition_per_reward(transition)

        # Deduce preferences according to each reward model
        pref_ar = np.zeros(self.nb_rewards)
        for idx in range(self.nb_rewards):
            if reward_ar[0, idx] > reward_ar[1,idx]:
                pref_ar[idx] = 0
            elif reward_ar[0, idx] < reward_ar[1,idx]:
                pref_ar[idx] = 1
            else:
                pref_ar[idx] = 0.5

        # Return std of preferences (= disagreement)
        return np.std(pref_ar)
    
    def reward_transition_per_reward(self, transition):
        reward_ar = np.zeros(self.nb_rewards)
        for idx, reward_model in enumerate(self.reward_list):
            reward_ar[idx] += reward_model.reward_transition(transition)
        return reward_ar
