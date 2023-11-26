import abc
import numpy as np
import pickle
import torch as th

from discrete_blocks import discrete_block as Block
from relative_single_agent import SACSupervisorSparse
from single_agent_gym import ReplayDiscreteGymSupervisor as Gym


def create_gym(config):
    #overwrite the action choice method:
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    target = Block([[0,0,1]])
    gym = Gym(config,
              agent_type=SACSupervisorSparse,
              use_wandb=False,
              actions= ['Ph'],
              block_type=[hexagon],
              random_targets='random_gap',
              n_robots=2,
              max_blocks = 50,
              targets=[target]*2,
              targets_loc = [[2,0],[6,0]],
              max_interfaces = 50,
              log_freq = 5,
              maxs = [9,6])
    print(f"gym dtype: {type(gym)}")
    return gym

def load_agent(file,gym,explore=False):   
    # with open(file, "rb") as input_file:
    #     agent  = pickle.load(input_file)

    agent = th.load(file, map_location=th.device('cpu'), pickle_module=pickle)
    agent.model.device = 'cpu'
    agent.model.state_encoder.device = 'cpu'
    for opti in agent.optimizer.Qs:
        opti.device = 'cpu'
        opti.state_encoder.device = 'cpu'

    if not explore:
        agent.exploration_strat = 'epsilon-greedy'
        agent.eps = 0
    gym.agent = agent
    return gym


class RewardComparison(abc.ABC):
    """Class for comparing rewards."""

    def __init__(self):
        """
        Initialize the reward comparison
        """

    @abc.abstractmethod
    def __call__(
        self
    ):
        """Compare two rewards.

        Returns:
            similarity metric
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
        """Cosine distancde between rewards.

        Returns:
            similarity metric
        """
        reward1 = self.normalize(reward1)
        reward2 = self.normalize(reward2)
        print(f"Normalized reward a: {reward1}")
        print(f"Normalized reward b: {reward2} \n")
        return np.linalg.norm(reward2-reward1)
    
    def normalize(self, reward):
        """Substract mean, divide by l2-norm

        Args:
            reward (np.array)

        Returns:
            normalized vector
        """
        return (reward-np.mean(reward))/np.linalg.norm(reward)


class LinearRewardPolicyComparison(RewardComparison):
    """Class for comparing rewards."""

    def __init__(self):
        """
        Initialize the reward comparison
        """

    def __call__(
        self,
        gym,
        agent1,
        agent2
    ):
        """Compare the success rate of policies trained by each reward.

        Returns:
            similarity metric (distance between success rate, e.g. absolute value of difference)
        """
        print(f"gym: {gym.agent}")
        gym = load_agent(agent1,gym,explore=False)
        avg_return1 = gym.avg_return_agent(nb_trials=100)
        gym = load_agent(agent2,gym,explore=False)
        avg_return2 = gym.avg_return_agent(nb_trials=100)
        return avg_return1, avg_return2
        

class RewardSuccessComparison(RewardComparison):
    """Class for comparing rewards."""

    def __init__(self):
        """
        Initialize the reward comparison
        """

    def __call__(
        self,
        gym,
        agent1,
        agent2
    ):
        """Compare the success rate of policies trained by each reward.

        Returns:
            similarity metric (distance between success rate, e.g. absolute value of difference)
        """
        print(f"gym: {gym.agent}")
        gym = load_agent(agent1,gym,explore=False)
        success_rate1 = gym.evaluate_agent(nb_trials=100)
        gym = load_agent(agent2,gym,explore=False)
        success_rate2 = gym.evaluate_agent(nb_trials=100)
        return success_rate1, success_rate2


if __name__ == '__main__':
    # Init gym
    config = {'train_n_episodes':10000,
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
            # 'reward_action':{'Ph': -0.2, 'L':-0.1},
            'reward_action':{'Ph': -0.2}, # only action considered
            'reward_closer':0.4,
            'reward_nsides': 0.1,
            'reward_success':1,
            'reward_opposite_sides':0,
            'opt_lower_bound_Vt':-2,
            'gap_range': [2,3] # this way gap of 2
            # 'gap_range':[2,6]
            }
    gym = create_gym(config)

    # COSINE COMPARISON
    print("COSINE COMPARISON")
    # Norm comparator
    norm_comparator = LinearRewardNormComparison()
    # Rewards to compare
    synthetic_reward = np.array([-0.2, 0.4, 1, -1, 0.1, 0])
    learned_reward = np.array([0.033, 1.379, 0.468, -0.281, 0.110, 0.084])
    # Comparison
    distance = norm_comparator(synthetic_reward, learned_reward)
    print(f"Cosine distance: {distance}")

    
    # SUCCESS COMPARISON
    print("SUCCESS COMPARISON")
    # Norm comparator
    norm_comparator = RewardSuccessComparison()
    # Agents to compare
    synthetic_agent = "synth_test.pt"
    rlhf_agent = "test.pt"
    # Comparison
    success_rates = norm_comparator(gym, synthetic_agent, rlhf_agent)
    print(f"Success rate Gab's: {success_rates[0]} - Success rate RLHF: {success_rates[1]}")


    # POLICY COMPARISON
    print("POLICY COMPARISON")
    # Norm comparator
    norm_comparator = LinearRewardPolicyComparison()
    # Agents to compare
    synthetic_agent = "synth_test.pt"
    rlhf_agent = "test.pt"
    # Comparison
    avg_returns = norm_comparator(gym, synthetic_agent, rlhf_agent)
    print(f"Using Gab's reward in the environment")
    print(f"Average return Gab's: {avg_returns[0]} - Average return RLHF {avg_returns[1]}")
