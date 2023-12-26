from torch import nn
import torch.nn.functional as F
import torch as th
import gymnasium as gym
import abc
import numpy as np
import torch
from typing import (
    Tuple,
    Iterable
)

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
            self.coeff.data /= norm_factor * 5.40


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


class RewardNet(nn.Module, abc.ABC, RewardModel):
    """Minimal abstract reward network.

    Only requires the implementation of a forward pass (calculating rewards given
    a batch of states, actions, next states and dones).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
    ):
        """Initialize the RewardNet.

        Args:
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            normalize_images: whether to automatically normalize
                image observations to [0, 1] (from 0 to 255). Defaults to True.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images

    @abc.abstractmethod
    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.

        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.

        Returns:
            Preprocessed transitions: a Tuple of tensors containing
            observations, actions, next observations and dones.
        """
        # TODO: FIND A WAY AROUND AROUND ENTIRE CODE
        # state_th = util.safe_to_tensor(state).to(self.device)
        # action_th = util.safe_to_tensor(action).to(self.device)
        # next_state_th = util.safe_to_tensor(next_state).to(self.device)
        # done_th = util.safe_to_tensor(done).to(self.device)

        # del state, action, next_state, done  # unused

        # # preprocess
        # # we only support array spaces, so we cast
        # # the observation to torch tensors.
        # state_th = cast(
        #     th.Tensor,
        #     preprocessing.preprocess_obs(
        #         state_th,
        #         self.observation_space,
        #         self.normalize_images,
        #     ),
        # )
        # action_th = cast(
        #     th.Tensor,
        #     preprocessing.preprocess_obs(
        #         action_th,
        #         self.action_space,
        #         self.normalize_images,
        #     ),
        # )
        # next_state_th = cast(
        #     th.Tensor,
        #     preprocessing.preprocess_obs(
        #         next_state_th,
        #         self.observation_space,
        #         self.normalize_images,
        #     ),
        # )
        # done_th = done_th.to(th.float32)

        # n_gen = len(state_th)
        # assert state_th.shape == next_state_th.shape
        # assert len(action_th) == n_gen

        # return state_th, action_th, next_state_th, done_th

    def predict_th(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> th.Tensor:
        """Compute th.Tensor rewards for a batch of transitions without gradients.

        Preprocesses the inputs, output th.Tensor reward arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed th.Tensor rewards of shape `(batch_size,`).
        """
        #TODO: FIND A WAY AROUND THIS
        # with networks.evaluating(self):
        #     # switch to eval mode (affecting normalization, dropout, etc)

        #     state_th, action_th, next_state_th, done_th = self.preprocess(
        #         state,
        #         action,
        #         next_state,
        #         done,
        #     )
        #     with th.no_grad():
        #         rew_th = self(state_th, action_th, next_state_th, done_th)

        #     assert rew_th.shape == state.shape[:1]
        #     return rew_th

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.

        Converting th.Tensor rewards from `predict_th` to NumPy arrays.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Computed rewards of shape `(batch_size,)`.
        """
        rew_th = self.predict_th(state, action, next_state, done)
        return rew_th.detach().cpu().numpy().flatten()

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute the processed rewards for a batch of transitions without gradients.

        Defaults to calling `predict`. Subclasses can override this to normalize or
        otherwise modify the rewards in ways that may help RL training or other
        applications of the reward function.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            kwargs: additional kwargs may be passed to change the functionality of
                subclasses.

        Returns:
            Computed processed rewards of shape `(batch_size,`).
        """
        del kwargs
        return self.predict(state, action, next_state, done)

    @property
    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            # if the model has no parameters, we use the CPU
            return th.device("cpu")

    @property
    def dtype(self) -> th.dtype:
        """Heuristic to determine dtype of module."""
        try:
            first_param = next(self.parameters())
            return first_param.dtype
        except StopIteration:
            # if the model has no parameters, default to float32
            return th.get_default_dtype()
        

class RewardNetWithVariance(RewardNet):
    """A reward net that keeps track of its epistemic uncertainty through variance."""

    @abc.abstractmethod
    def predict_reward_moments(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the mean and variance of the reward distribution.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            **kwargs: may modify the behavior of subclasses

        Returns:
            * Estimated reward mean of shape `(batch_size,)`.
            * Estimated reward variance of shape `(batch_size,)`.
        """
    

class RewardEnsemble(RewardNetWithVariance):
    """A mean ensemble of reward networks.

    A reward ensemble is made up of individual reward networks. To maintain consistency
    the "output" of a reward network will be defined as the results of its
    `predict_processed`. Thus for example the mean of the ensemble is the mean of the
    results of its members predict processed classes.
    """

    members: nn.ModuleList

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        members: Iterable[RewardNet],
    ):
        """Initialize the RewardEnsemble.

        Args:
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            members: the member networks that will make up the ensemble.

        Raises:
            ValueError: if num_members is less than 1
        """
        super().__init__(observation_space, action_space)

        members = list(members)
        if len(members) < 2:
            raise ValueError("Must be at least 2 member in the ensemble.")

        self.members = nn.ModuleList(
            members,
        )

    @property
    def num_members(self):
        """The number of members in the ensemble."""
        return len(self.members)

    def predict_processed_all(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Get the results of predict processed on all of the members.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            kwargs: passed along to ensemble members.

        Returns:
            The result of predict processed for each member in the ensemble of
                shape `(batch_size, num_members)`.
        """
        batch_size = state.shape[0]
        rewards_list = [
            member.predict_processed(state, action, next_state, done, **kwargs)
            for member in self.members
        ]
        rewards: np.ndarray = np.stack(rewards_list, axis=-1)
        assert rewards.shape == (batch_size, self.num_members)
        return rewards

    @th.no_grad()
    def predict_reward_moments(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the standard deviation of the reward distribution for a batch.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
            **kwargs: passed along to predict processed.

        Returns:
            * Reward mean of shape `(batch_size,)`.
            * Reward variance of shape `(batch_size,)`.
        """
        batch_size = state.shape[0]
        all_rewards = self.predict_processed_all(
            state,
            action,
            next_state,
            done,
            **kwargs,
        )
        mean_reward = all_rewards.mean(-1)
        var_reward = all_rewards.var(-1, ddof=1)
        assert mean_reward.shape == var_reward.shape == (batch_size,)
        return mean_reward, var_reward

    def forward(self, *args) -> th.Tensor:
        """The forward method of the ensemble should in general not be used directly."""
        raise NotImplementedError

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Return the mean of the ensemble members."""
        return self.predict(state, action, next_state, done, **kwargs)

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ):
        """Return the mean of the ensemble members."""
        mean, _ = self.predict_reward_moments(state, action, next_state, done, **kwargs)
        return mean