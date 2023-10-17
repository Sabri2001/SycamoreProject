import abc
import numpy as np
from typing import(
    Sequence
)


class Fragmenter(abc.ABC):
    """Class for creating pairs of trajectory fragments from a set of trajectories."""

    def __init__(self):
        """
        Initialize the fragmenter.
        """


    @abc.abstractmethod
    def __call__(
        self,
        trajectories: Sequence,
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence:
        """Create fragment pairs out of a sequence of trajectories.

        Args:
            trajectories: collection of trajectories that will be split up into
                fragments
            fragment_length: the length of each sampled fragment
            num_pairs: the number of fragment pairs to sample

        Returns:
            a sequence of fragment pairs
        """  # noqa: DAR202


class RandomFragmenter(Fragmenter):
    def __init__(self):
        pass

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

