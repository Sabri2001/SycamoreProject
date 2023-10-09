import abc
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
        