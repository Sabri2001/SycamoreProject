"""
@author: elamrani
"""

import abc
import numpy as np
import pickle
import torch as th

from discrete_blocks import discrete_block as Block
from relative_single_agent import SACSupervisorSparse
from single_agent_gym import ReplayDiscreteGymSupervisor as Gym


def create_gym(config):
    #overwrite the action choice method:
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.7)
    target = Block([[0,0,1]])
    gym = Gym(config,
              agent_type=SACSupervisorSparse,
              use_wandb=False,
              actions= ['Ph'], # place-hold only necessary action
              block_type=[hexagon],
              random_targets='random_gap', 
              targets_loc=[[2,0],[6,0]], 
              n_robots=2, 
              max_blocks = 15,
              targets=[target]*2,
              max_interfaces = 100,
              log_freq = 5,
              maxs = [15,15]) # grid size
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
        print(f"Normalized reward b: {reward2}")
        return np.linalg.norm(reward2-reward1)
    
    def normalize(self, reward):
        """Substract mean, divide by l2-norm

        Args:
            reward (np.array)

        Returns:
            normalized vector
        """
        return (reward-np.mean(reward))/np.linalg.norm(reward)
        # return reward - np.mean(reward)
        

class ReturnPolicy(RewardComparison):
    """Class for comparing rewards."""

    def __init__(self):
        """
        Initialize the reward comparison
        """

    def __call__(
        self,
        gym,
        agent
    ):
        """Compare the success rate of policies trained by each reward.

        Returns:
            similarity metric (distance between success rate, e.g. absolute value of difference)
        """
        print(f"gym: {gym.agent}")
        gym = load_agent(agent,gym,explore=False)
        avg_return = gym.avg_return_agent(nb_trials=100)
        return avg_return
        

class RewardSuccess(RewardComparison):
    """Class for comparing rewards."""

    def __init__(self):
        """
        Initialize the reward comparison
        """

    def __call__(
        self,
        gym,
        agent
    ):
        """Compare the success rate of policies trained by each reward.

        Returns:
            similarity metric (distance between success rate, e.g. absolute value of difference)
        """
        print(f"gym: {gym.agent}")
        gym = load_agent(agent,gym,explore=False)
        success_rate = gym.evaluate_agent(nb_trials=100)
        return success_rate


if __name__ == '__main__':
    # Init gym
    config = {'train_n_episodes':20000,
            'train_l_buffer':1000000,
            'ep_batch_size':512,
            'ep_use_mask':True,
            'agent_discount_f':0.1, # 1-gamma
            'agent_last_only':True,
            'reward': 'modular',
            'torch_device': 'cpu',
            'SEnc_n_channels':64, # 64
            'SEnc_n_internal_layer':4,
            'SEnc_stride':1,
            'SEnc_order_insensitive':True,
            'SAC_n_fc_layer':3, # 3
            'SAC_n_neurons':64, # 128
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
            'opt_target_entropy':0.5,
            'opt_value_clip':False,
            'opt_entropy_penalty':False,
            'opt_Q_reduction': 'min',
            'V_optimistic':False,
            'reward_failure':-2,
            'reward_action':{'Ph': -0.2},
            'reward_closer':0.4,
            'reward_nsides': 0.05,
            'reward_success':5,
            'reward_opposite_sides':0,
            'opt_lower_bound_Vt':-2,
            'gap_range':[1,8] # so 1 to 7 actually
            }
    gym = create_gym(config)

    # COSINE COMPARISON
    print("COSINE COMPARISON")
    # Norm comparator
    norm_comparator = LinearRewardNormComparison()
    # Rewards to compare
    gab_reward = np.array([-0.2, 0.4, 5, -2, 0.05, 0])
    learned_reward_linear = np.array([-0.7607503, 0.8275733, 3.5222988, -2.7163618, -0.09642205, -0.27460566])
    learned_reward_disagreement = np.array([-0.43, 0.48, 2.25, -2.25, -0.13, -0.08])
    # Comparison
    distance = norm_comparator(gab_reward, learned_reward_linear)
    print(f"Cosine distance gab - linear: {distance} \n")
    distance = norm_comparator(gab_reward, learned_reward_disagreement)
    print(f"Cosine distance gab - disagreement: {distance}\n")

    
    # SUCCESS COMPARISON
    print("SUCCESS COMPARISON")
    # Norm comparator
    norm_comparator = RewardSuccess()
    # Agents to compare
    gab_agent = "final_trained_agents/gab.pt"
    rlhf_linear_agent = "final_trained_agents/linear.pt"
    rlhf_disagreement_agent = "final_trained_agents/disagreement.pt"
    rlhf_cnn_agent = "final_trained_agents/31_12_cnn_trained_agent.pt"
    # Comparison
    success_gab_agent = norm_comparator(gym, gab_agent)
    success_linear_agent = norm_comparator(gym, rlhf_linear_agent)
    success_disagreement_agent = norm_comparator(gym, rlhf_disagreement_agent)
    success_cnn_agent = norm_comparator(gym, rlhf_cnn_agent)
    print(f"Success rate gab: {success_gab_agent}")
    print(f"Success rate linear: {success_linear_agent}")
    print(f"Success rate disagreement: {success_disagreement_agent}")
    print(f"Success rate cnn: {success_cnn_agent}\n")


    # POLICY COMPARISON
    print("POLICY COMPARISON")
    # Norm comparator
    norm_comparator = ReturnPolicy()
    # Comparison
    return_gab_agent = norm_comparator(gym, gab_agent)
    return_linear_agent = norm_comparator(gym, rlhf_linear_agent)
    return_disagreement_agent = norm_comparator(gym, rlhf_disagreement_agent)
    return_cnn_agent = norm_comparator(gym, rlhf_cnn_agent)
    print(f"Average return gab: {return_gab_agent}")
    print(f"Average return linear: {return_linear_agent}")
    print(f"Average return disagreement: {return_disagreement_agent}")
    print(f"Average return cnn: {return_cnn_agent}")
