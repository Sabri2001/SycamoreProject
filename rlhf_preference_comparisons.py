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

from rlhf_reward_net import RewardNet
from rlhf_preference_dataset import PreferenceDataset
from rlhf_reward_trainer import RewardTrainer
from rlhf_preference_gatherer import PreferenceGatherer
from rlhf_trajectory_generator import TrajectoryGenerator
from rlhf_fragmenter import Fragmenter


QUERY_SCHEDULES: Dict[str, type_aliases.Schedule] = {
    "constant": lambda t: 1.0,
    "hyperbolic": lambda t: 1.0 / (1.0 + t),
    "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
}


class PreferenceComparisons():
    """Main interface for reward learning using preference comparisons."""

    def __init__(
        self,
        trajectory_generator: TrajectoryGenerator,
        reward_model: RewardNet,
        num_iterations: int,
        fragmenter: Optional[Fragmenter] = None,
        preference_gatherer: Optional[PreferenceGatherer] = None,
        reward_trainer: Optional[RewardTrainer] = None,
        comparison_queue_size: Optional[int] = None,
        fragment_length: int = 100,
        transition_oversampling = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        rng: Optional[np.random.Generator] = None,
        query_schedule: Union[str, type_aliases.Schedule] = "hyperbolic",
    ) -> None:
        """Initialize the preference comparison trainer.

        Args:
            trajectory_generator: generates trajectories while optionally training
                an RL agent on the learned reward function (can also be a sampler
                from a static dataset of trajectories though).
            reward_model: a RewardNet instance to be used for learning the reward
            num_iterations: number of times to train the agent against the reward model
                and then train the reward model against newly gathered preferences.
            fragmenter: takes in a set of trajectories and returns pairs of fragments
                for which preferences will be gathered. These fragments could be random,
                or they could be selected more deliberately (active learning).
                Default is a random fragmenter.
            preference_gatherer: how to get preferences between trajectory fragments.
                Default (and currently the only option) is to use synthetic preferences
                based on ground-truth rewards. Human preferences could be implemented
                here in the future.
            reward_trainer: trains the reward model based on pairs of fragments and
                associated preferences. Default is to use the preference model
                and loss function from DRLHP.
            comparison_queue_size: the maximum number of comparisons to keep in the
                queue for training the reward model. If None, the queue will grow
                without bound as new comparisons are added.
            fragment_length: number of timesteps per fragment that is used to elicit
                preferences
            transition_oversampling: factor by which to oversample transitions before
                creating fragments. Since fragments are sampled with replacement,
                this is usually chosen > 1 to avoid having the same transition
                in too many fragments.
            initial_comparison_frac: fraction of the total_comparisons argument
                to train() that will be sampled before the rest of training begins
                (using a randomly initialized agent). This can be used to pretrain the
                reward model before the agent is trained on the learned reward, to
                help avoid irreversibly learning a bad policy from an untrained reward.
                Note that there will often be some additional pretraining comparisons
                since `comparisons_per_iteration` won't exactly divide the total number
                of comparisons. How many such comparisons there are depends
                discontinuously on `total_comparisons` and `comparisons_per_iteration`.
            initial_epoch_multiplier: before agent training begins, train the reward
                model for this many more epochs than usual (on fragments sampled from a
                random agent).
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
            rng: random number generator to use for initializing subcomponents such as
                fragmenter.
                Only used when default components are used; if you instantiate your
                own fragmenter, preference gatherer, etc., you are responsible for
                seeding them!
            query_schedule: one of ("constant", "hyperbolic", "inverse_quadratic"), or
                a function that takes in a float between 0 and 1 inclusive,
                representing a fraction of the total number of timesteps elapsed up to
                some time T, and returns a potentially unnormalized probability
                indicating the fraction of `total_comparisons` that should be queried
                at that iteration. This function will be called `num_iterations` times
                in `__init__()` with values from `np.linspace(0, 1, num_iterations)`
                as input. The outputs will be normalized to sum to 1 and then used to
                apportion the comparisons among the `num_iterations` iterations.

        Raises:
            ValueError: if `query_schedule` is not a valid string or callable.
        """

        # for keeping track of the global iteration, in case train() is called
        # multiple times
        self._iteration = 0
        self.rng = rng

        self.model = reward_model
        self.reward_trainer = reward_trainer
        self.trajectory_generator = trajectory_generator
        self.preference_gatherer = preference_gatherer
        self.fragmenter = fragmenter

        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.num_iterations = num_iterations
        self.transition_oversampling = transition_oversampling
        if callable(query_schedule):
            self.query_schedule = query_schedule
        elif query_schedule in QUERY_SCHEDULES:
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            raise ValueError(f"Unknown query schedule: {query_schedule}")

        self.dataset = PreferenceDataset(max_size=comparison_queue_size)

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> dict:
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

        # Compute the number of comparisons to request at each iteration in advance.
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
        reward_loss = None
        reward_accuracy = None

        for i, num_pairs in enumerate(schedule):
            ##########################
            # Gather new preferences #
            ##########################
            num_steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * self.fragment_length,
            )

            print(f"Collecting {2 * num_pairs} fragments ({num_steps} transitions)")
            trajectories = self.trajectory_generator.sample(num_steps)
            # This assumes there are no fragments missing initial timesteps
            # (but allows for fragments missing terminal timesteps).
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            self._check_fixed_horizon(horizons)

            print("Creating fragment pairs")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)
            
            print("Gathering preferences")
            preferences = self.preference_gatherer(fragments)
            self.dataset.push(fragments, preferences)
            print(f"Dataset now contains {len(self.dataset)} comparisons")

            ##########################
            # Train the reward model #
            ##########################

            # On the first iteration, we train the reward model for longer,
            # as specified by initial_epoch_multiplier.
            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier # default: 200

            self.reward_trainer.train(self.dataset, epoch_multiplier=epoch_multiplier)

            # Print loss/accuracy -> comparison with baseline reward
            # -------------------
            # TODO
            # -------------------
            # base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
            # assert f"{base_key}/loss" in self.logger.name_to_value
            # assert f"{base_key}/accuracy" in self.logger.name_to_value
            # reward_loss = self.logger.name_to_value[f"{base_key}/loss"]
            # reward_accuracy = self.logger.name_to_value[f"{base_key}/accuracy"]

            ###################
            # Train the agent #
            ###################
            num_steps = timesteps_per_iteration
            # if the number of timesteps per iterations doesn't exactly divide
            # the desired total number of timesteps, we train the agent a bit longer
            # at the end of training (where the reward model is presumably best)
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps
            
            print(f"Training agent for {num_steps} timesteps")
            self.trajectory_generator.train(steps=num_steps)

            ########################
            # Additional Callbacks #
            ########################
            if callback:
                callback(self._iteration)
            self._iteration += 1

        return {"reward_loss": reward_loss, "reward_accuracy": reward_accuracy}
