import pickle
import torch as th
import numpy as np

from rlhf_preference_dataset import PreferenceDatasetNoDiscard, PreferenceDataset

dataset = PreferenceDatasetNoDiscard(50)
print(f"Dataset: {dataset}")

path = "02_12_rlhf_pref_dataset_local_1.pickle"
a = dataset.load(path)
print(f"Loaded: {len(a)}")
