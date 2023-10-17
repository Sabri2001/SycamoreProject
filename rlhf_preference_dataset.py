import numpy as np


class PreferenceDataset():
    """A Dataset for preference comparisons.

    Each item is a preference (for traj1) -> 1, 0 or 0.5.

    This dataset is meant to be generated piece by piece during the
    training process, which is why data can be added via the .push()
    method.
    """

    def __init__(self, max_size=10000) -> None:
        """Builds an empty PreferenceDataset.

        Args:
            max_size: Maximum number of preference comparisons to store in the dataset.
                If None (default), the dataset can grow indefinitely. Otherwise, the
                dataset acts as a FIFO queue, and the oldest comparisons are evicted
                when `push()` is called and the dataset is at max capacity.
        """
        self.fragments1 = []
        self.fragments2 = []
        self.max_size = max_size
        self.preferences: np.ndarray = np.array([])

    def push(
        self,
        fragments: [],
        preferences: np.ndarray,
    ):
        """Add more samples to the dataset.

        Args:
            fragments: list of pairs of trajectories to add
            preferences: corresponding preference 
        """
        fragments1, fragments2 = zip(*fragments)

        self.fragments1.extend(fragments1)
        self.fragments2.extend(fragments2)
        self.preferences = np.concatenate((self.preferences, preferences))

        # Evict old samples if the dataset is at max capacity
        if self.max_size is not None:
            extra = len(self.preferences) - self.max_size
            if extra > 0:
                self.fragments1 = self.fragments1[extra:]
                self.fragments2 = self.fragments2[extra:]
                self.preferences = self.preferences[extra:]

    def __getitem__(self, key):
        return (self.fragments1[key], self.fragments2[key]), self.preferences[key]

    def __len__(self) -> int:
        assert len(self.fragments1) == len(self.fragments2) == len(self.preferences)
        return len(self.fragments1)
