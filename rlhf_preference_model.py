from torch import nn
import torch as th
from typing import (
    Optional,
    Sequence,
    Tuple,
    cast,
    Transitions
)

from rlhf_reward_net import RewardNet
from rlhf_reward_ensemble import RewardEnsemble


class PreferenceModel(nn.Module):
    """Class to convert two fragments' rewards into preference probability."""

    def __init__(
        self,
        model: RewardNet,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
    ) -> None:
        """Create Preference Prediction Model.

        Args:
            model: base model to compute reward.
            noise_prob: assumed probability with which the preference
                is uniformly random (used for the model of preference generation
                that is used for the loss).
            discount_factor: the model of preference generation uses a softmax
                of returns as the probability that a fragment is preferred.
                This is the discount factor used to calculate those returns.
                Default is 1, i.e. undiscounted sums of rewards (which is what
                the DRLHP paper uses).
            threshold: the preference model used to compute the loss contains
                a softmax of returns. To avoid overflows, we clip differences
                in returns that are above this threshold. This threshold
                is therefore in logspace. The default value of 50 means
                that probabilities below 2e-22 are rounded up to 2e-22.

        Raises:
            ValueError: if `RewardEnsemble` is wrapped around a class
                other than `AddSTDRewardWrapper`.
        """
        super().__init__()
        self.model = model
        self.noise_prob = noise_prob
        self.discount_factor = discount_factor
        self.threshold = threshold
        base_model = model
        self.ensemble_model = None
        # if the base model is an ensemble model, then keep the base model as
        # model to get rewards from all networks
        if isinstance(base_model, RewardEnsemble):
            # reward_model may include an AddSTDRewardWrapper for RL training; but we
            # must train directly on the base model for reward model training.
            is_base = model is base_model
            is_std_wrapper = (
                isinstance(model, reward_nets.AddSTDRewardWrapper) # TODO: get rid of wrappers
                and model.base is base_model
            )

            if not (is_base or is_std_wrapper):
                raise ValueError(
                    "RewardEnsemble can only be wrapped"
                    f" by AddSTDRewardWrapper but found {type(model).__name__}.",
                )
            self.ensemble_model = base_model
            self.member_pref_models = []
            for member in self.ensemble_model.members:
                member_pref_model = PreferenceModel(
                    cast(RewardNet, member),  # nn.ModuleList is not generic
                    self.noise_prob,
                    self.discount_factor,
                    self.threshold,
                )
                self.member_pref_models.append(member_pref_model)


    def forward(
        self,
        fragment_pairs: Sequence, #TODO: a way around TrajectoryPair
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """Computes the preference probability of the first fragment for all pairs.

        Note: This function passes the gradient through for non-ensemble models.
              For an ensemble model, this function should not be used for loss
              calculation. It can be used in case where passing the gradient is not
              required such as during active selection or inference time.
              Therefore, the EnsembleTrainer passes each member network through this
              function instead of passing the EnsembleNetwork object with the use of
              `ensemble_member_index`.

        Args:
            fragment_pairs: batch of pair of fragments.

        Returns:
            A tuple with the first element as the preference probabilities for the
            first fragment for all fragment pairs given by the network(s).
            If the ground truth rewards are available, it also returns gt preference
            probabilities in the second element of the tuple (else None).
            Reward probability shape - (num_fragment_pairs, ) for non-ensemble reward
            network and (num_fragment_pairs, num_networks) for an ensemble of networks.

        """
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        gt_reward_available = _trajectory_pair_includes_reward(fragment_pairs[0])
        if gt_reward_available:
            gt_probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = rollout.flatten_trajectories([frag1])
            trans2 = rollout.flatten_trajectories([frag2])
            rews1 = self.rewards(trans1)
            rews2 = self.rewards(trans2)
            probs[i] = self.probability(rews1, rews2)
            if gt_reward_available:
                frag1 = cast(TrajectoryWithRew, frag1)
                frag2 = cast(TrajectoryWithRew, frag2)
                gt_rews_1 = th.from_numpy(frag1.rews)
                gt_rews_2 = th.from_numpy(frag2.rews)
                gt_probs[i] = self.probability(gt_rews_1, gt_rews_2)

        return probs, (gt_probs if gt_reward_available else None)


    def rewards(self, transitions: Transitions) -> th.Tensor:
        """Computes the reward for all transitions.

        Args:
            transitions: batch of obs-act-obs-done for a fragment of a trajectory.

        Returns:
            The reward given by the network(s) for all the transitions.
            Shape - (num_transitions, ) for Single reward network and
            (num_transitions, num_networks) for ensemble of networks.
        """
        state = transitions.obs
        action = transitions.acts
        next_state = transitions.next_obs
        done = transitions.dones
        if self.ensemble_model is not None:
            rews_np = self.ensemble_model.predict_processed_all(
                state,
                action,
                next_state,
                done,
            )
            assert rews_np.shape == (len(state), self.ensemble_model.num_members)
            rews = util.safe_to_tensor(rews_np).to(self.ensemble_model.device)
        else:
            preprocessed = self.model.preprocess(state, action, next_state, done)
            rews = self.model(*preprocessed)
            assert rews.shape == (len(state),)
        return rews


    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:
        """Computes the Boltzmann rational probability the first trajectory is best.

        Args:
            rews1: array/matrix of rewards for the first trajectory fragment.
                matrix for ensemble models and array for non-ensemble models.
            rews2: array/matrix of rewards for the second trajectory fragment.
                matrix for ensemble models and array for non-ensemble models.

        Returns:
            The softmax of the difference between the (discounted) return of the
            first and second trajectory.
            Shape - (num_ensemble_members, ) for ensemble model and
            () for non-ensemble model which is a torch scalar.
        """
        # check rews has correct shape based on the model
        expected_dims = 2 if self.ensemble_model is not None else 1
        assert rews1.ndim == rews2.ndim == expected_dims
        # First, we compute the difference of the returns of
        # the two fragments. We have a special case for a discount
        # factor of 1 to avoid unnecessary computation (especially
        # since this is the default setting).
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum(axis=0)  # type: ignore[call-overload]
        else:
            device = rews1.device
            assert device == rews2.device
            discounts = self.discount_factor ** th.arange(len(rews1), device=device)
            if self.ensemble_model is not None:
                discounts = discounts.reshape(-1, 1)
            returns_diff = (discounts * (rews2 - rews1)).sum(axis=0)
        # Clip to avoid overflows (which in particular may occur
        # in the backwards pass even if they do not in the forward pass).
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        # We take the softmax of the returns. model_probability
        # is the first dimension of that softmax, representing the
        # probability that fragment 1 is preferred.
        model_probability = 1 / (1 + returns_diff.exp())
        probability = self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability
        if self.ensemble_model is not None:
            assert probability.shape == (self.model.num_members,)
        else:
            assert probability.shape == ()
        return probability
