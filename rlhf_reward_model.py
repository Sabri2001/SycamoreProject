from torch import nn
import torch as th
import gymnasium as gym
import abc
import numpy as np
from typing import (
    Tuple,
    Iterable
)


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


class RewardLinear(RewardModel):
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

    def forward(
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
            * Estimated reward variance of shape `(batch_size,)`. # noqa: DAR202
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