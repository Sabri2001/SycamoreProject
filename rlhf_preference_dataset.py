import numpy as np
import pickle
from imitation.data.types import AnyPath


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
        self.traj_list1 = []
        self.traj_list2 = []
        self.max_size = max_size
        self.preferences: np.ndarray = np.array([])

    def push(
        self,
        pairs: [],
        preferences: np.ndarray,
    ):
        """Add more samples to the dataset.

        Args:
            pairs: list of pairs of trajectories to add
            preferences: corresponding preference 
        """
        traj_list1, traj_list2 = zip(*pairs)

        self.traj_list1.extend(traj_list1)
        self.traj_list2.extend(traj_list2)
        self.preferences = np.concatenate((self.preferences, preferences))

        # Evict old samples if the dataset is at max capacity
        if self.max_size is not None:
            extra = len(self.preferences) - self.max_size
            if extra > 0:
                self.traj_list1 = self.traj_list1[extra:]
                self.traj_list2 = self.traj_list2[extra:]
                self.preferences = self.preferences[extra:]

    def __getitem__(self, key):
        return (self.traj_list1[key], self.traj_list2[key]), self.preferences[key]

    def __len__(self) -> int:
        assert len(self.traj_list1) == len(self.traj_list2) == len(self.preferences)
        return len(self.traj_list1)

    def save(self, path: AnyPath) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load(path: AnyPath) -> "PreferenceDataset":
        with open(path, "rb") as file:
            return pickle.load(file)
