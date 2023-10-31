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

    def __call__(self, trajectories, num_pairs):
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
            selected_trajectories = np.random.choice(trajectories, size=2, replace=True)
            trajectory_pairs.append(tuple(selected_trajectories))

        return trajectory_pairs

