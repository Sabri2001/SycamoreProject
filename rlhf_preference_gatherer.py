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
    def __init__(self, coeff):
        self.coeff = coeff

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
            trajectory1 = trajectory_pair[0].get_transitions()
            trajectory2 = trajectory_pair[1].get_transitions()

            # Calculate reward for both trajectories
            reward1 = self.reward_trajectory(trajectory1)
            reward2 = self.reward_trajectory(trajectory2)

            # Compute preference scores
            if reward1 > reward2:
                preferences.append(1)
            elif reward1 < reward2:
                preferences.append(0)
            else:
                preferences.append(0.5)

        return np.array(preferences)

    def reward_trajectory(self, trajectory):
        """Compute reward for a trajectory."""
        reward = 0
        for transition in trajectory:
            reward += self.reward_transition(transition)
        return reward
    
    def reward_transition(self, transition):
        return np.dot(self.coeff, transition.reward_features)
    