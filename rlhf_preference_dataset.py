"""
@author: elamrani
"""

import numpy as np
import pickle
from imitation.data.types import AnyPath
import torch as th
import random


class PreferenceDataset():
    """A Dataset for preference comparisons.

    Each item is a preference (for traj1) -> 1, 0 or 0.5.

    This dataset is meant to be generated piece by piece during the
    training process, which is why data can be added via the .push()
    method.
    """

    def __init__(self, max_size=50) -> None:
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


class PreferenceDatasetNoDiscard():
    """A Dataset for preference comparisons.

    Each item is a preference (for traj1) -> 1, 0 or 0.5.

    This dataset is meant to be generated piece by piece during the
    training process, which is why data can be added via the .push()
    method.
    """

    def __init__(self, max_size=50, device = None):
        """Builds an empty PreferenceDataset.

        Args:
            max_size: Maximum number of preference comparisons to store in the dataset.
                If None (default), the dataset can grow indefinitely. Otherwise, the
                dataset acts as a FIFO queue, and the oldest comparisons are kept for
                saving, but are not used anymore.
        """
        # Complete set for saving (no discarding)
        self.traj_list1 = []
        self.traj_list2 = []
        self.preferences = th.tensor([], device=device)

        self.max_size = max_size
        # Preferences to consider in reward training
        self.current_traj_list1 = []
        self.current_traj_list2 = []
        self.current_preferences = th.tensor([], device=device)

    def push(
        self,
        pairs,
        preferences
    ):
        """Add more samples to the dataset.

        Args:
            pairs: list of pairs of trajectories to add
            preferences: corresponding preference 
        """
        traj_list1, traj_list2 = zip(*pairs)

        self.traj_list1.extend(traj_list1)
        self.traj_list2.extend(traj_list2)
        self.preferences = th.cat([self.preferences, preferences], dim=0)

        # Only consider latest max_size samples for reward training
        if self.max_size is not None:
            extra = len(self.preferences) - self.max_size
            if extra > 0:
                self.current_traj_list1 = self.traj_list1[extra:]
                self.current_traj_list2 = self.traj_list2[extra:]
                self.current_preferences = self.preferences[extra:]
            else:
                self.current_traj_list1 = self.traj_list1
                self.current_traj_list2 = self.traj_list2
                self.current_preferences = self.preferences

    def __getitem__(self, key):
        return (self.current_traj_list1[key], self.current_traj_list2[key]), self.current_preferences[key]

    def __len__(self) -> int:
        assert len(self.current_traj_list1) == len(self.current_traj_list2) == len(self.current_preferences)
        return len(self.current_traj_list1)

    def sample(self, batch_size=32):
        """Randomly sample items from the dataset.

        Args:
            batch_size: Number of samples to randomly select.

        Returns:
            Iterator: An iterator over tuples containing the sampled pairs of trajectories and preferences.
        """
        if len(self.current_traj_list1) <= batch_size:
            # If the dataset size is smaller than the batch_size, sample the entire dataset
            sampled_indices = list(range(len(self.current_traj_list1)))
        else:
            # Randomly sample batch_size indices without replacement
            sampled_indices = random.sample(range(len(self.current_traj_list1)), batch_size)

        # Use the sampled indices to create an iterator over corresponding samples
        sampled_data_iter = iter(
            ((self.current_traj_list1[i], self.current_traj_list2[i]), self.current_preferences[i])
            for i in sampled_indices
        )

        return sampled_data_iter

    # TODO: check that dumping although some data for other device not issue... (torch.save instead?)
    def save(self, path: AnyPath):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load(self, path: AnyPath):
        with open(path, "rb") as file:
            return pickle.load(file)
