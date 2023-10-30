from typing import (
    Optional,
    Union,
    Dict,
    Callable
)
import numpy as np
import math
from stable_baselines3.common import type_aliases
from imitation.util import util

from rlhf_reward_model import RewardNet
from rlhf_preference_dataset import PreferenceDataset
from rlhf_reward_trainer import RewardTrainer
from rlhf_preference_gatherer import PreferenceGatherer
from rlhf_pair_generator import PairGenerator


QUERY_SCHEDULES: Dict[str, type_aliases.Schedule] = {
    "constant": lambda t: 1.0,
    "hyperbolic": lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
}


class PreferenceComparisons():
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        gym,
        reward_model: RewardNet,
        num_iterations: int,
        pair_generator: Optional[PairGenerator] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
        comparison_queue_size: Optional[int] = None,
        transition_oversampling = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        query_schedule: Union[str, type_aliases.Schedule] = "hyperbolic",
        draw_freq = 100
    ):
        
        # Init all attributes
        self.model = reward_model
        self.reward_trainer = reward_trainer
        self.preference_gatherer = preference_gatherer
        self.pair_generator = pair_generator
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.num_iterations = num_iterations
        self.transition_oversampling = transition_oversampling
        self.draw_freq = draw_freq # draw_freq = 1 when asking for human feedback
        
        # Init schedule
        if callable(query_schedule):
            self.query_schedule = query_schedule
        elif query_schedule in QUERY_SCHEDULES:
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            raise ValueError(f"Unknown query schedule: {query_schedule}")

        # Init preference dataset
        self.dataset = PreferenceDataset(max_size=comparison_queue_size)

        # Init gym
        self.gym = gym

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int
    ):
        """Train the reward model and the policy if applicable.

        Args:
            total_timesteps: number of environment interaction steps
            total_comparisons: number of preferences to gather in total

        Returns:
            A dictionary with final metrics such as loss and accuracy
            of the reward model.
        """
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons -= initial_comparisons

        # Compute the number of comparisons to request at each iteration in advance (with schedule).
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_comparisons)
        schedule = [initial_comparisons] + shares.tolist()
        print(f"Query schedule: {schedule}")
        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps,
            self.num_iterations,
        )

        # Some inits
        reward_loss = None
        reward_accuracy = None

        # MAIN LOOP
        for i, num_pairs in enumerate(schedule):
            #############################################
            # Generate trajectories with trained policy #
            #############################################
            # Generate trajectories
            nb_traj = self.transition_oversampling* 2 * num_pairs
            print(f"Collecting {nb_traj} trajectories")
            trajectories = self.gym.generate_trajectories(nb_traj, draw_freq=self.draw_freq)

            # Create pairs of trajectories (to be compared)
            print("Creating trajectory pairs")
            pairs = self.pair_generator(trajectories, num_pairs)
            print("Pair formation done")
            
            ##########################
            # Gather new preferences #
            ##########################    
            # Gather synthetic or human preferences
            print("Gathering preferences")
            preferences = self.preference_gatherer(pairs)
            print("Gathering over")
            print("Preferences gathered: ", preferences)

            # Store preferences in Preference Dataset
            self.dataset.push(pairs, preferences)
            print(f"Dataset now contains {len(self.dataset)} comparisons")

            ##########################
            # Train the reward model #
            ##########################
            # On the first iteration, we train the reward model for longer,
            # as specified by initial_epoch_multiplier.
            epoch_multip = 1.0
            if i == 0:
                epoch_multip = self.initial_epoch_multiplier # default: 200

            print("Training reward model")
            self.reward_trainer.train(self.dataset, epoch_multiplier=epoch_multip)
            print("Reward training finished")

            ###################
            # Train the agent #
            ###################
            num_steps = timesteps_per_iteration
            # if the number of timesteps per iterations doesn't exactly divide
            # the desired total number of timesteps, we train the agent a bit longer
            # at the end of training (where the reward model is presumably best)
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps
            
            print("Training agent")
            self.gym.training()
            print("Training finished")

        return {"reward_loss": reward_loss, "reward_accuracy": reward_accuracy}
