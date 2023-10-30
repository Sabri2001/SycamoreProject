import numpy as np

from rlhf_preference_dataset import PreferenceDataset
from rlhf_preference_model import PreferenceModel


class RewardTrainer():
    """Abstract base class for training reward models using preference comparisons.

    This class contains only the actual reward model training code,
    it is not responsible for gathering trajectories and preferences
    or for agent training (see :class: `PreferenceComparisons` for that).
    """

    def __init__(
        self,
        preference_model: PreferenceModel
    ):
        """Initialize the reward trainer.

        Args:
            preference_model: the preference model to train the reward network.
        """
        self.preference_model = preference_model

    def train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Train the reward model on a batch of trajectory pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """
        pass


class LinearRewardTrainer(RewardTrainer):
    """Class for training linear reward model using preference comparisons."""

    def __init__(
        self,
        preference_model: PreferenceModel,
        gamma,
        logger = None
    ):
        """Initialize the reward trainer.

        Args:
            preference_model: the preference model to train the reward network.
            gamma: discount factor
            logger: log
        """
        self.preference_model = preference_model
        self.gamma = gamma
        self.logger = logger

    def train(self, dataset, epoch_multiplier = 1., learning_rate = 0.08):
        """Train the reward model on a batch of trajectory pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """
        # NOTE: quite sensitive to learning rate + not reliably going down...
        num_epochs = int(epoch_multiplier*10)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for sample in dataset:
                trajectory_pair, pref_traj1 = sample # NOTE: here batch of size 1 -> to change?

                # Calculate reward values using the preference model
                proba_traj1 = self.preference_model.forward(trajectory_pair)

                # Compute the cross-entropy loss
                total_loss -= pref_traj1*np.log(proba_traj1) + (1-pref_traj1)*np.log(1-proba_traj1)

                # Compute the gradient and update the coefficients
                features_traj1 = trajectory_pair[0].rollout_reward_features(self.gamma)
                features_traj2 = trajectory_pair[1].rollout_reward_features(self.gamma)
                gradient = -pref_traj1*( \
                    proba_traj1*(1-proba_traj1)*features_traj1 \
                    - proba_traj1*(1-proba_traj1)*features_traj2) \
                    - (1-pref_traj1)*(\
                        proba_traj1*(1-proba_traj1)*features_traj2 \
                    - proba_traj1*(1-proba_traj1)*features_traj1
                        )

                self.preference_model.reward_model.coeff -= learning_rate * gradient

            # Print the average loss for this epoch
            if epoch%10 == 0:
                average_loss = total_loss/ len(dataset)
                self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] \
                    Loss: {average_loss:.4f}")
                
        self.logger.info(f"---> Current reward coefficients: {self.preference_model.reward_model.coeff}")
