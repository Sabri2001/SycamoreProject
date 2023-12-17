import abc
import numpy as np


class PairGenerator(abc.ABC):
    """Class for creating pairs of trajectory from a set of trajectories."""

    def __init__(self):
        """
        Initialize the pair generator.
        """


    @abc.abstractmethod
    def __call__(
        self,
        trajectories,
        num_pairs,
        pair_oversampling = None
    ):
        """Create pairs out of a sequence of trajectories.

        Args:
            trajectories: collection of trajectories that will be paired
            num_pairs: the number of pairs to sample

        Returns:
            a sequence of pairs of trajectories
        """


class RandomPairGenerator(PairGenerator):
    def __init__(self):
        """Init random pair generator"""

    def __call__(self, trajectories, num_pairs, pair_oversampling=None):
        """
        Create pairs of complete trajectories from a collection of trajectories.

        Args:
            trajectories: A collection of trajectories (each trajectory is a list of Transition objects).
            num_pairs: The number of trajectory pairs to sample.

        Returns:
            A list of trajectory pairs, where each pair is represented as a tuple of two complete trajectories.
        """
        trajectory_pairs = []

        for _ in range(num_pairs):
            # Randomly select two trajectories from the list of trajectories
            selected_trajectories = np.random.choice(trajectories, size=2, replace=False)
            trajectory_pairs.append(tuple(selected_trajectories))

        return trajectory_pairs


class DisagreementPairGenerator(PairGenerator):
    def __init__(self,
                 reward_model
                 ):
        """Init random pair generator"""
        self.reward_model = reward_model

    def __call__(self, trajectories, num_pairs, pair_oversampling):
        """
        Create pairs of complete trajectories from a collection of trajectories.

        Args:
            trajectories: A collection of trajectories (each trajectory is a list of Transition objects).
            num_pairs: The number of trajectory pairs to sample.

        Returns:
            A list of trajectory pairs, where each pair is represented as a tuple of two complete trajectories.
        """
        trajectory_pairs = []
        disagreement = []

        # Oversample pairs + compute disagreement (= std of preferences) for each
        for _ in range(num_pairs*pair_oversampling):
            selected_trajectories = np.random.choice(trajectories, size=2, replace=False)
            trajectory_pairs.append(tuple(selected_trajectories))
            disagreement.append(self.reward_model.reward_disagreement(selected_trajectories))

        # Keep the top num_pairs pairs (most disagreement)
        disagreement = np.array(disagreement)
        sorted_indices = np.argsort(disagreement)
        top_pairs_indices = sorted_indices[-num_pairs:]
        top_trajectory_pairs = [trajectory_pairs[i] for i in top_pairs_indices]

        return top_trajectory_pairs
