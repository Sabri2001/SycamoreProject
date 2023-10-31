import abc
import numpy as np
from discrete_blocks import discrete_block as Block
from relative_single_agent import SACSupervisorSparse

from single_agent_gym import ReplayDiscreteGymSupervisor


class RewardComparison(abc.ABC):
    """Class for comparing rewards."""

    def __init__(self):
        """
        Initialize the reward comparison
        """

    @abc.abstractmethod
    def __call__(
        self,
        reward1,
        reward2,
    ):
        """Compare two rewards.

        Returns:
            similarity metric
        """

    @abc.abstractmethod
    def normalize(
        self,
        reward
    ):
        """Normalizes reward.

        Returns:
            normalized reward (np.array)
        """


class LinearRewardNormComparison(RewardComparison):
    """Class for comparing rewards."""

    def __init__(self):
        """
        Initialize the reward comparison
        """

    def __call__(
        self,
        reward1,
        reward2,
    ):
        """Normalise rewards before comparing them with 2-norm.

        Returns:
            similarity metric
        """
        reward1 = self.normalize(reward1)
        reward2 = self.normalize(reward2)
        print(f"Normalized reward 1: {reward1}")
        print(f"Normalized reward 2: {reward2}")
        return np.linalg.norm(reward2-reward1)


class LinearRewardPolicyComparison(RewardComparison):
    """Class for comparing rewards."""

    def __init__(self, gym1, gym2):
        """
        Initialize the reward comparison
        """
        self.gym1 = gym1
        self.gym2 = gym2

    def __call__(
        self,
        reward1,
        reward2,
        nb_traj = 50
    ):
        """Normalise rewards before comparing them with policy norm.
        To do that, starts by training two policies, one with each reward.
        Then, evaluates the difference between them by comparing the 
        empirical average return between the two policies, once when 
        giving reward1, another time when giving reward2.

        Returns:
            (similarity metric with reward1, similarity metric with reward2)
        """
        # Normalize rewards
        reward1 = self.normalize(reward1)
        reward2 = self.normalize(reward2)
        print(f"Normalized reward 1: {reward1}")
        print(f"Normalized reward 2: {reward2}")

        # Train a policy for each reward
        self.gym

        # Generate trajectories with each policy
        # TODO

        # Compare average returns of both policies with reward1
        # TODO

        # Compare average returns of both policies with reward1

        return # TODO
        
    def normalize(self, reward):
        """Substract mean, divide by standard deviation

        Args:
            reward (np.array)

        Returns:
            normalized vector
        """
        return (reward-np.mean(reward))/np.std(reward)


class RewardSuccessComparison(RewardComparison):
    """Class for comparing rewards."""

    def __init__(self):
        """
        Initialize the reward comparison
        """

    def __call__(
        self,
        reward1,
        reward2,
    ):
        """Compare the success rate of policies trained by each reward.

        Returns:
            similarity metric (distance between success rate, e.g. absolute value of difference)
        """
        pass

    def normalize(
        self,
        reward
    ):
        """Normalizes reward.

        Returns:
            normalized reward (np.array)
        """



if __name__ == '__main__':
    # Rewards to compare
    synthetic_reward = np.array([-0.2, 0.4, 1, -1, 0.1, 0])
    learned_reward = np.array([1.04315514, 0.79808593, 0.00232924,-0.11823537, 1.3424669, 1.16139051])

    # COMPARATOR WITH 2-NORM
    norm_comparator = LinearRewardNormComparison()
    distance = norm_comparator(synthetic_reward, learned_reward)
    print(f"2-norm distance between rewards: {distance}")

    # COMPARATOR IN POLICY SPACE
    # Init - one gym for each agent
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.5) 
    linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=0.5)
    target = Block([[0,0,1]])
    config = {'train_n_episodes':100,
                'train_l_buffer':200,
                'ep_batch_size':32,
                'ep_use_mask':True,
                'agent_discount_f':0.1, # 1-gamma
                'agent_last_only':True,
                'reward': 'modular',
                'torch_device':'cpu',
                'SEnc_n_channels':64,
                'SEnc_n_internal_layer':2,
                'SEnc_stride':1,
                'SEnc_order_insensitive':True,
                'SAC_n_fc_layer':3,
                'SAC_n_neurons':128,
                'SAC_batch_norm':True,
                'Q_duel':True,
                'opt_lr':1e-4,
                'opt_pol_over_val': 1,
                'opt_tau': 5e-4,
                'opt_weight_decay':0.0001,
                'opt_exploration_factor':0.001,
                'agent_exp_strat':'softmax',
                'agent_epsilon':0.05, # not needed in sac
                'opt_max_norm': 2,
                'opt_target_entropy':1.8,
                'opt_value_clip':False,
                'opt_entropy_penalty':False,
                'opt_Q_reduction': 'min',
                'V_optimistic':False,
                'reward_failure':-1,
                'reward_action':{'Ph': -0.2, 'L':-0.1},
                'reward_closer':0.4,
                'reward_nsides': 0.1,
                'reward_success':1,
                'reward_opposite_sides':0,
                'opt_lower_bound_Vt':-2,
                'gap_range':[2,3]
                }   
    gym1 = ReplayDiscreteGymSupervisor(config,
                agent_type=SACSupervisorSparse,
                use_wandb=False, # wandb set up elsewhere
                actions= ['Ph'], # place-hold only necessary action
                block_type=[hexagon],
                random_targets='random_gap', 
                targets_loc=[[2,0],[6,0]], 
                n_robots=2, 
                max_blocks = 10,
                targets=[target]*2,
                max_interfaces = 50,
                log_freq = 5, # grid size
                maxs = [9,6]
                )
    gym2 = ReplayDiscreteGymSupervisor(config,
                agent_type=SACSupervisorSparse,
                use_wandb=False, # wandb set up elsewhere
                actions= ['Ph'], # place-hold only necessary action
                block_type=[hexagon],
                random_targets='random_gap', 
                targets_loc=[[2,0],[6,0]], 
                n_robots=2, 
                max_blocks = 10,
                targets=[target]*2,
                max_interfaces = 50,
                log_freq = 5, # grid size
                maxs = [9,6]
                )

    # Comparison
    policy_comparator = LinearRewardPolicyComparison(gym1, gym2)

    # COMPARATOR OF SUCCESS RATE
    # TODO
