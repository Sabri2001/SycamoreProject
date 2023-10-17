import numpy as np

from rlhf_reward_model import RewardModel, RewardLinear


class PreferenceModel():
    """Class to convert two trajectories' rewards into preference probability."""

    def __init__(
        self,
        model: RewardModel,
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
        super().__init__()
        self.model = model
        self.noise_prob = noise_prob # TODO: implement this! (10% chance uniform response)
        self.threshold = threshold

    def forward(
        self,
        trajectory_pair
    ):
        """
        Computes the preference probability of the first fragment for all pairs, 
        using softmax.

        Args:
            trajectory_pairs: pair of trajectories.

        Returns:
            Preference probability for the first element of
            each trajectory pair in trajectory_pairs.
        """
        traj1, traj2 = trajectory_pair
        rews1 = RewardLinear(traj1)
        rews2 = RewardLinear(traj2)
        proba = self.probability(rews1, rews2)

        return proba

    def probability(self, rews1, rews2):
        """Computes the Boltzmann rational probability the first trajectory is best.

        Args:
            rews1: scalar reward for the first trajectory fragment.
            rews2: scalar reward for the second trajectory fragment.

        Returns:
            The softmax of the difference between the return of the
            first and second trajectory.
        """
        probability = np.exp(rews1)/(np.exp(rews1) + np.exp(rews2))
        return probability
