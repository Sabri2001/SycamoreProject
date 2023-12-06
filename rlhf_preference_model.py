import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rlhf_reward_model import RewardModel


class PreferenceModelArchived():
    """Class to convert two trajectories' rewards into preference probability."""

    def __init__(
        self,
        reward_model: RewardModel,
        noise_prob: float = 0.0,
        threshold: float = 50,
    ):
        """Create Preference Prediction Model.

        Args:
            model: base model to compute reward.
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss).
        """
        self.reward_model = reward_model
        self.noise_prob = noise_prob # TODO: implement this! (10% chance uniform response)
        self.threshold = threshold

    def forward(
        self,
        trajectory_pair
    ):
        """
        Computes the preference probability of the first trajectory for all pairs, 
        using softmax.

        Args:
            trajectory_pairs: pair of trajectories.

        Returns:
            Preference probability for the first element of
            each trajectory pair in trajectory_pairs.
        """
        traj1 = trajectory_pair[0].get_transitions()
        traj2 = trajectory_pair[1].get_transitions()
        reward1 = self.reward_model.forward(traj1)
        reward2 = self.reward_model.forward(traj2)
        proba = self.probability(reward1, reward2)
        return proba

    def probability(self, rews1, rews2):
        """Computes the Boltzmann rational probability the first trajectory is best.

        Args:
            rews1: scalar reward for the first trajectory.
            rews2: scalar reward for the second trajectory.

        Returns:
            The softmax of the difference between the return of the
            first and second trajectory.
        """
        probability = np.exp(rews1)/(np.exp(rews1) + np.exp(rews2))
        return probability


class PreferenceModel(nn.Module):
    """Class to convert two trajectories' rewards into preference probability."""

    def __init__(self, reward_model, noise_prob=0.0, threshold=0.1):
        """Create Preference Prediction Model.

        Args:
            reward_model: base model to compute reward.
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss).
            threshold: threshold for softmax stability.
        """
        super(PreferenceModel, self).__init__()
        self.reward_model = reward_model
        self.noise_prob = noise_prob # TODO: uncertainty due to human fb, perhaps for later
        self.threshold = threshold # TODO: check whether threshold value ok

    def get_reward_coeff(self):
        return self.reward_model.get_reward_coeff()

    def normalize_reward(self):
        self.reward_model.normalize_reward()

    def forward(self, trajectory_pair, reward_nb = None):
        """
        Computes the preference probability of the first trajectory for all pairs, 
        using softmax.

        Args:
            trajectory_pairs: pair of trajectories.

        Returns:
            Preference probability for the first element of
            each trajectory pair in trajectory_pairs.
        """
        traj1 = trajectory_pair[0].get_transitions()
        traj2 = trajectory_pair[1].get_transitions()
        if reward_nb: # reward ensemble
            reward1 = self.reward_model.reward_list[reward_nb].reward_trajectory(traj1)
            reward2 = self.reward_model.reward_list[reward_nb].reward_trajectory(traj2)
        else:
            reward1 = self.reward_model.reward_trajectory(traj1)
            reward2 = self.reward_model.reward_trajectory(traj2)
        proba = self.probability(reward1, reward2)
        return proba

    def compute_reward_pair(self, trajectory_pair):
        traj1 = trajectory_pair[0].get_transitions()
        traj2 = trajectory_pair[1].get_transitions()
        reward1 = self.reward_model.forward(traj1)
        reward2 = self.reward_model.forward(traj2)
        return torch.tensor([reward1, reward2], dtype=torch.float32)

    def probability(self, rews1, rews2):
        """Computes the Boltzmann rational probability the first trajectory is best.

        Args:
            rews1: scalar reward for the first trajectory.
            rews2: scalar reward for the second trajectory.

        Returns:
            The softmax of the difference between the return of the
            first and second trajectory.
        """
        # Stack the tensors along a new dimension (e.g., dimension 0)
        rewards = torch.stack([rews1, rews2])

        # Apply softmax to the tensor along dimension 0
        softmax_values = F.softmax(rewards, dim=0)
        return softmax_values[0]
