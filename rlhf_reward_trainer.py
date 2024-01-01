"""
@author: elamrani
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rlhf_preference_dataset import PreferenceDataset
from rlhf_preference_model import PreferenceModel


class RewardTrainer():
    """Abstract base class for training reward models using preference comparisons.

    This class contains only the actual reward model training code,
    it is not responsible for gathering trajectories and preferences
    or for agent training (see :class: `PreferenceComparisons` for that).
    """

    def __init__(
        self,
        preference_model: PreferenceModel
    ):
        """Initialize the reward trainer.

        Args:
            preference_model: the preference model to train the reward network.
        """
        self.preference_model = preference_model

    def train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Train the reward model on a batch of trajectory pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """
        pass


class LinearRewardTrainerNoTorch(RewardTrainer):
    """Class for training linear reward model using preference comparisons."""

    def __init__(
        self,
        preference_model: PreferenceModel,
        gamma,
        logger = None
    ):
        """Initialize the reward trainer.

        Args:
            preference_model: the preference model to train the reward network.
            gamma: discount factor
            logger: log
        """
        self.preference_model = preference_model
        self.gamma = gamma
        self.logger = logger

    def train(self, dataset, epoch_multiplier = 1., learning_rate = 0.08):
        """Train the reward model on a batch of trajectory pairs and preferences.

        Args:
            dataset: the dataset of preference comparisons to train on.
            epoch_multiplier: how much longer to train for than usual
                (measured relatively).
        """
        # NOTE: quite sensitive to learning rate + not reliably going down...
        num_epochs = int(epoch_multiplier*1000)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for sample in dataset:
                trajectory_pair, pref_traj1 = sample # NOTE: here batch of size 1 -> to change?

                # Calculate reward values using the preference model
                proba_traj1 = self.preference_model.forward(trajectory_pair)

                # Compute the cross-entropy loss
                total_loss -= pref_traj1*np.log(proba_traj1) + (1-pref_traj1)*np.log(1-proba_traj1)

                # Compute the gradient and update the coefficients
                features_traj1 = trajectory_pair[0].rollout_reward_features(self.gamma)
                features_traj2 = trajectory_pair[1].rollout_reward_features(self.gamma)
                gradient = -pref_traj1*( \
                    proba_traj1*(1-proba_traj1)*features_traj1 \
                    - proba_traj1*(1-proba_traj1)*features_traj2) \
                    - (1-pref_traj1)*(\
                        proba_traj1*(1-proba_traj1)*features_traj2 \
                    - proba_traj1*(1-proba_traj1)*features_traj1
                        )

                self.preference_model.reward_model.coeff -= learning_rate * gradient

            # Print the average loss for this epoch
            if epoch%10 == 0:
                average_loss = total_loss/ len(dataset)
                self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] \
                    Loss: {average_loss:.4f}")
                
        self.logger.info(f"---> Current reward coefficients: {self.preference_model.reward_model.coeff}")


class LinearRewardTrainer(RewardTrainer):
    def __init__(self, preference_model, gamma, learning_rate=0.001, logger=None):
        self.preference_model = preference_model
        self.gamma = gamma
        self.logger = logger

        # Initialize the loss function
        self.loss_fn = nn.BCELoss()

        # Initialize the optimizer
        self.optimizer = optim.Adam([{'params': preference_model.reward_model.coeff, 'lr': learning_rate}])

    def train_step(self, trajectory_pair, pref_traj1):
        self.optimizer.zero_grad()

        # Calculate reward values using the preference model
        proba_traj1 = self.preference_model.forward(trajectory_pair)

        # Compute the cross-entropy loss
        loss = self.loss_fn(proba_traj1, pref_traj1)

        loss.backward()
        self.optimizer.step()
        self.preference_model.normalize_reward()

        return loss.detach().cpu().numpy()

    def train(self, dataset, epoch_multiplier=1.):
        num_epochs = int(epoch_multiplier * 1000)

        total_loss = 0.
        counter = 0
        for epoch in range(num_epochs):    
            for sample in dataset.sample(batch_size=32):
                trajectory_pair, pref_traj1 = sample

                loss = self.train_step(trajectory_pair, pref_traj1)
                total_loss += loss
                counter += 1

            # Print the average loss for this epoch
            if epoch % 50 == 0:
                average_loss = total_loss / counter
                self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {average_loss:.4f}")
                self.logger.info(f"   --->  Reward coeff: {self.preference_model.get_reward_coeff()}")


class LinearRewardEnsembleTrainer(RewardTrainer):
    def __init__(self, preference_model, gamma, learning_rate=0.001, logger=None):
        self.preference_model = preference_model
        self.gamma = gamma
        self.logger = logger

        # Initialize the loss function
        self.loss_fn = nn.BCELoss()

        # Initialize the optimizers
        self.optimizer_list = []
        for reward in preference_model.reward_model.reward_list:
            self.optimizer_list.append(optim.Adam([{'params': reward.coeff, 'lr': learning_rate}]))

    def train_step(self, trajectory_pair, pref_traj1, reward_nb):
        
        self.optimizer_list[reward_nb].zero_grad()

        # Calculate reward values using the preference model
        proba_traj1 = self.preference_model.forward(trajectory_pair, reward_nb)

        # Compute the cross-entropy loss
        loss = self.loss_fn(proba_traj1, pref_traj1)

        loss.backward()
        self.optimizer_list[reward_nb].step()

        self.preference_model.normalize_reward()

        return loss.detach().cpu().numpy()
    
    def train(self, dataset, epoch_multiplier=1.):
        num_epochs = int(epoch_multiplier * 1000)

        total_loss = 0.0
        counter = 0
        for epoch in range(num_epochs):
            for reward_nb in range(len(self.optimizer_list)):
                for sample in dataset.sample(batch_size=32):
                    trajectory_pair, pref_traj1 = sample

                    loss = self.train_step(trajectory_pair, pref_traj1, reward_nb)
                    total_loss += loss
                    counter += 1

            # Print the average loss for this epoch
            if epoch % 50 == 0:
                average_loss = total_loss / counter
                self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {average_loss:.4f}")
                for i in range(len(self.optimizer_list)):
                    self.logger.info(f"   --->  Reward coeff {i}: {self.preference_model.get_reward_coeff()[i]}")


class RewardTrainerCNN(RewardTrainer):
    def __init__(self, preference_model, gamma, learning_rate=0.0001, logger=None):
        self.preference_model = preference_model
        self.gamma = gamma
        self.logger = logger

        # Initialize the loss function
        self.loss_fn = nn.BCELoss()

        # Initialize the optimizer
        self.optimizer = torch.optim.NAdam(self.preference_model.reward_model.parameters(),lr=learning_rate, weight_decay=0.0001)

    def train_step(self, trajectory_pair, pref_traj1):
        self.optimizer.zero_grad()

        # Calculate reward values using the preference model
        proba_traj1 = self.preference_model.forward(trajectory_pair)

        # Compute the cross-entropy loss
        loss = self.loss_fn(proba_traj1, pref_traj1)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, dataset, epoch_multiplier=1.):
        num_epochs = int(epoch_multiplier * 1000)

        total_loss = 0.0
        counter = 0
        for epoch in range(num_epochs):
            for sample in dataset.sample(batch_size=32):
                trajectory_pair, pref_traj1 = sample

                loss = self.train_step(trajectory_pair, pref_traj1)
                total_loss += loss
                counter += 1

            # Print the average loss for this epoch
            if epoch % 50 == 0:
                average_loss = total_loss / counter
                self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {average_loss:.4f}")


class RewardTrainerCNNEnsemble(RewardTrainer):
    def __init__(self, preference_model, gamma, learning_rate=0.0001, logger=None):
        self.preference_model = preference_model
        self.gamma = gamma
        self.logger = logger

        # Initialize the loss function
        self.loss_fn = nn.BCELoss()
        
        # Initialize the optimizers
        self.optimizer_list = []
        for reward in preference_model.reward_model.reward_list:
            self.optimizer_list.append(torch.optim.NAdam(reward.parameters(),lr=learning_rate, weight_decay=0.0001))

    def train_step(self, trajectory_pair, pref_traj1, reward_nb):

        self.optimizer_list[reward_nb].zero_grad()

        # Calculate reward values using the preference model
        proba_traj1 = self.preference_model.forward(trajectory_pair)

        # Compute the cross-entropy loss
        loss = self.loss_fn(proba_traj1, pref_traj1)

        loss.backward()
        self.optimizer_list[reward_nb].step()

        return loss.item()

    def train(self, dataset, epoch_multiplier=1.):
        num_epochs = int(epoch_multiplier * 1000)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for reward_nb in range(len(self.optimizer_list)):
                for sample in dataset.sample(batch_size=32):
                    trajectory_pair, pref_traj1 = sample

                    loss = self.train_step(trajectory_pair, pref_traj1, reward_nb)
                    total_loss += loss

            # Print the average loss for this epoch
            if epoch % 50 == 0:
                average_loss = total_loss / len(dataset) / len(self.optimizer_list)
                self.logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {average_loss:.4f}")
