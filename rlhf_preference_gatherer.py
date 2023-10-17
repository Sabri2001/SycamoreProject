import abc
import numpy as np
from relative_single_agent import modular_reward


class PreferenceGatherer(abc.ABC):
    """Base class for gathering preference comparisons between trajectory fragments."""

    def __init__(
        self,
    ) -> None:
        """Initializes the preference gatherer."""

    @abc.abstractmethod
    def __call__(self, fragment_pairs) -> np.ndarray:
        """Gathers the preference for each pair of trajectory in 'trajectory_pairs'.

        Args:
            trajectory_pairs: sequence of pairs of trajectories

        Returns:
            A numpy array with shape (b, ), where b is the length of the input
            (i.e. batch size). Each item in the array is the preference score
            for the corresponding pair of trajectories.

            Preference score: 
                1 better? => 1
                2 better? => 0
                Equality? => 0.5
                Don't know? => not taken into account (only relevant with humans)
        """ 


class SyntheticPreferenceGatherer(PreferenceGatherer):
    def __init__(self, config):
        self.config = config

    def __call__(self, trajectory_pairs):
        """
        Gathers the synthetic preference for each pair of trajectory in 'trajectory_pairs'.

        Args:
            trajectory_pairs: A list of tuples, where each tuple contains two trajectories.
                Each trajectory is a list of transitions.

        Returns:
            A numpy array with shape (b,), where b is the length of the input (i.e., batch size).
            Each item in the array is the preference score for the corresponding pair of trajectories.
            Preference score:
                1 better? => 1
                2 better? => 0
                Equality? => 0.5
        """
        preferences = []

        for trajectory_pair in trajectory_pairs:
            trajectory1, trajectory2 = trajectory_pair
            reward1, reward2 = 0, 0

            # Calculate reward1 for the first trajectory
            for transition in trajectory1:
                reward_features = transition.reward_features
                action = reward_features[0]
                closer = reward_features[1]
                success = reward_features[2]
                failure = reward_features[3]
                n_sides = reward_features[4]

                reward1 += modular_reward(action,True,closer,success,failure,self.config,n_sides)

            # Calculate reward2 for the second trajectory
            for transition in trajectory2:
                reward_features = transition.reward_features
                action = reward_features[0]
                closer = reward_features[1]
                success = reward_features[2]
                failure = reward_features[3]
                n_sides = reward_features[4]

                reward2 += modular_reward(action,True,closer,success,failure,self.config,n_sides)

            # Compute preference scores
            if reward1 > reward2:
                preferences.append(1)
            elif reward1 < reward2:
                preferences.append(0)
            else:
                preferences.append(0.5)

        return np.array(preferences)

