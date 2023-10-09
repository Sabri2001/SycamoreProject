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
    ) -> None:
        """Initialize the reward trainer.

        Args:
            preference_model: the preference model to train the reward network.
        """
        self._preference_model = preference_model

    def train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Train the reward model on a batch of fragment pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """
        #TODO: Train
