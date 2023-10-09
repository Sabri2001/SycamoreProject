import abc
from typing import (
    Sequence,
    Any,
    Union,
    Optional
)
from imitation.data.types import (
    TrajectoryWithRew,
)
from imitation.data import rollout
import numpy as np

from rlhf_reward_net import RewardNet
from rlhf_reward_fn import RewardFn

from relative_single_agent import SupervisorRelative
from single_agent_gym import ReplayDiscreteGymSupervisor


class TrajectoryGenerator(abc.ABC):
    """Generator of trajectories with optional training logic."""

    def __init__(self):
        """Builds TrajectoryGenerator."""

    @abc.abstractmethod
    def sample(self, steps: int) -> Sequence[TrajectoryWithRew]:
        """Sample a batch of trajectories.

        Args:
            steps: All trajectories taken together should
                have at least this many steps.

        Returns:
            A list of sampled trajectories with rewards (which should
            be the environment rewards, not ones from a reward model).
        """  # noqa: DAR202

    def train(self, steps: int, **kwargs: Any) -> None:
        """Train an agent if the trajectory generator uses one.

        By default, this method does nothing and doesn't need
        to be overridden in subclasses that don't require training.

        Args:
            steps: number of environment steps to train for.
            **kwargs: additional keyword arguments to pass on to
                the training procedure.
        """


class MyAgentTrainer(TrajectoryGenerator):
    """Wrapper for training an SB3 algorithm on an arbitrary reward function."""

    def __init__(
        self,
        agent: SupervisorRelative,
        reward_fn: Union[RewardFn, RewardNet],
        gym: ReplayDiscreteGymSupervisor,
        rng: np.random.Generator,
        exploration_frac: float = 0.0,
    ) -> None:
        """Initialize the agent trainer.

        Args:
            algorithm: the stable-baselines algorithm to use for training.
            reward_fn: either a RewardFn or a RewardNet instance that will supply
                the rewards used for training the agent.
            venv: vectorized environment to train in.
            rng: random number generator used for exploration and for sampling.
            exploration_frac: fraction of the trajectories that will be generated
                partially randomly rather than only by the agent when sampling.
            switch_prob: the probability of switching the current policy at each
                step for the exploratory samples.
            random_prob: the probability of picking the random policy when switching
                during exploration.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.agent = agent
        self.gym = gym

        self.reward_fn = reward_fn # TODO: IMPLEMENT THE REWARD FUN
        self.exploration_frac = exploration_frac
        self.rng = rng

    def train(self, steps: int, **kwargs) -> None:
        """Train the agent using the reward function specified during instantiation.

        Args:
            steps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()
        """
        self.agent.config['train_n_episodes'] = steps
        self.agent.training(
            pfreq = 10,
            draw_freq=100,
            max_steps=100,
            save_freq = 1000,
            success_rate_decay = 0.01,
            log_dir=None)

    def sample(self, steps: int) -> Sequence[TrajectoryWithRew]:

        # TODO
        
        return trajectories


class AgentTrainer(TrajectoryGenerator):
    """Wrapper for training an SB3 algorithm on an arbitrary reward function."""

    def __init__(
        self,
        # algorithm: base_class.BaseAlgorithm,
        reward_fn: Union[RewardFn, RewardNet],
        # venv: vec_env.VecEnv,
        rng: np.random.Generator,
        exploration_frac: float = 0.0,
    ) -> None:
        """Initialize the agent trainer.

        Args:
            algorithm: the stable-baselines algorithm to use for training.
            reward_fn: either a RewardFn or a RewardNet instance that will supply
                the rewards used for training the agent.
            venv: vectorized environment to train in.
            rng: random number generator used for exploration and for sampling.
            exploration_frac: fraction of the trajectories that will be generated
                partially randomly rather than only by the agent when sampling.
            switch_prob: the probability of switching the current policy at each
                step for the exploratory samples.
            random_prob: the probability of picking the random policy when switching
                during exploration.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        # self.algorithm = algorithm # TODO

        self.reward_fn = reward_fn
        self.exploration_frac = exploration_frac
        self.rng = rng

    def train(self, steps: int, **kwargs) -> None:
        """Train the agent using the reward function specified during instantiation.

        Args:
            steps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()

        Raises:
            RuntimeError: Transitions left in `self.buffering_wrapper`; call
                `self.sample` first to clear them.
        """
        n_transitions = self.buffering_wrapper.n_transitions
        if n_transitions:
            raise RuntimeError(
                f"There are {n_transitions} transitions left in the buffer. "
                "Call AgentTrainer.sample() first to clear them.",
            )
        self.algorithm.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=self.log_callback,
            **kwargs,
        )

    def sample(self, steps: int) -> Sequence[TrajectoryWithRew]:
        agent_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
        # We typically have more trajectories than are needed.
        # In that case, we use the final trajectories because
        # they are the ones with the most relevant version of
        # the agent.
        # The easiest way to do this will be to first invert the
        # list and then later just take the first trajectories:
        agent_trajs = agent_trajs[::-1]
        avail_steps = sum(len(traj) for traj in agent_trajs)

        exploration_steps = int(self.exploration_frac * steps)
        if self.exploration_frac > 0 and exploration_steps == 0:
            self.logger.warn(
                "No exploration steps included: exploration_frac = "
                f"{self.exploration_frac} > 0 but steps={steps} is too small.",
            )
        agent_steps = steps - exploration_steps

        if avail_steps < agent_steps:
            self.logger.log(
                f"Requested {agent_steps} transitions but only {avail_steps} in buffer."
                f" Sampling {agent_steps - avail_steps} additional transitions.",
            )
            sample_until = rollout.make_sample_until(
                min_timesteps=agent_steps - avail_steps,
                min_episodes=None,
            )
            # Important note: we don't want to use the trajectories returned
            # here because 1) they might miss initial timesteps taken by the RL agent
            # and 2) their rewards are the ones provided by the reward model!
            # Instead, we collect the trajectories using the BufferingWrapper.
            algo_venv = self.algorithm.get_env()
            assert algo_venv is not None
            rollout.generate_trajectories(
                self.algorithm,
                algo_venv,
                sample_until=sample_until,
                # By setting deterministic_policy to False, we ensure that the rollouts
                # are collected from a deterministic policy only if self.algorithm is
                # deterministic. If self.algorithm is stochastic, then policy_callable
                # will also be stochastic.
                deterministic_policy=False,
                rng=self.rng,
            )
            additional_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
            agent_trajs = list(agent_trajs) + list(additional_trajs)

        agent_trajs = _get_trajectories(agent_trajs, agent_steps)

        trajectories = list(agent_trajs)

        if exploration_steps > 0:
            self.logger.log(f"Sampling {exploration_steps} exploratory transitions.")
            sample_until = rollout.make_sample_until(
                min_timesteps=exploration_steps,
                min_episodes=None,
            )
            algo_venv = self.algorithm.get_env()
            assert algo_venv is not None
            rollout.generate_trajectories(
                policy=self.exploration_wrapper,
                venv=algo_venv,
                sample_until=sample_until,
                # buffering_wrapper collects rollouts from a non-deterministic policy,
                # so we do that here as well for consistency.
                deterministic_policy=False,
                rng=self.rng,
            )
            exploration_trajs, _ = self.buffering_wrapper.pop_finished_trajectories()
            exploration_trajs = _get_trajectories(exploration_trajs, exploration_steps)
            # We call _get_trajectories separately on agent_trajs and exploration_trajs
            # and then just concatenate. This could mean we return slightly too many
            # transitions, but it gets the proportion of exploratory and agent
            # transitions roughly right.
            trajectories.extend(list(exploration_trajs))
        return trajectories


def _get_trajectories(
    trajectories: Sequence[TrajectoryWithRew],
    steps: int,
) -> Sequence[TrajectoryWithRew]:
    """Get enough trajectories to have at least `steps` transitions in total."""
    if steps == 0:
        return []

    available_steps = sum(len(traj) for traj in trajectories)
    if available_steps < steps:
        raise RuntimeError(
            f"Asked for {steps} transitions but only {available_steps} available",
        )
    # We need the cumulative sum of trajectory lengths
    # to determine how many trajectories to return:
    steps_cumsum = np.cumsum([len(traj) for traj in trajectories])
    # Now we find the first index that gives us enough
    # total steps:
    idx = int((steps_cumsum >= steps).argmax())
    # we need to include the element at position idx
    trajectories = trajectories[: idx + 1]
    # sanity check
    assert sum(len(traj) for traj in trajectories) >= steps
    return trajectories