# IMPORTS
import copy
import time
import os
import wandb
import numpy as np
import pickle
from discrete_blocks import discrete_block as Block
from relative_single_agent import SACSupervisorSparse
from discrete_simulator import DiscreteSimulator as Sim, Transition
import discrete_graphics as gr

from single_agent_gym import ReplayDiscreteGymSupervisor
from rlhf_reward_model import RewardLinear
from rlhf_pair_generator import RandomPairGenerator
from rlhf_preference_gatherer import SyntheticPreferenceGatherer, HumanPreferenceGatherer
from rlhf_preference_model import PreferenceModel
from rlhf_reward_trainer import LinearRewardTrainer
from rlhf_preference_comparisons import PreferenceComparisons


# CONSTANTS
USE_WANDB = False
HUMAN_FEEDBACK = False
# %env "WANDB_NOTEBOOK_NAME" "rlhf_main.ipynb"

# blocks
hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5) 
linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.5) 
linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=0.5)
target = Block([[0,0,1]])

# config
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


# INIT
# Create Gym (env + agent)
gym = ReplayDiscreteGymSupervisor(config,
              agent_type=SACSupervisorSparse,
              use_wandb=USE_WANDB,
              actions= ['Ph'], # place-hold only necessary action
              block_type=[hexagon],
              random_targets='random_gap', 
              targets_loc=[[2,0],[6,0]], 
              n_robots=2, 
              max_blocks = 10,
              targets=[target]*2,
              max_interfaces = 50,
              log_freq = 5,
              maxs = [9,6]) # grid size

# Create Reward Model
reward_model = RewardLinear()

# Create Pair Generator
pair_generator = RandomPairGenerator()

# Create Preference Gatherer (human/synthetic)
if HUMAN_FEEDBACK:
    gatherer = HumanPreferenceGatherer()
else:
    coeff = np.array([
                config['reward_action']['Ph'],
                config['reward_closer'],
                config['reward_success'],
                config['reward_failure'],
                config['reward_nsides'],
                config['reward_opposite_sides']
                ])
    gatherer = SyntheticPreferenceGatherer(coeff)

# Create Preference Model
preference_model = PreferenceModel(reward_model)

# Create Reward Trainer
reward_trainer = LinearRewardTrainer(preference_model)

# Create Preference Comparisons, the main interface
if HUMAN_FEEDBACK:
    draw_freq = 1
else:
    draw_freq = 100

pref_comparisons = PreferenceComparisons(
    gym,
    reward_model,
    num_iterations=5,  # Set to 60 for better performance
    pair_generator=pair_generator,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    transition_oversampling=1,
    initial_comparison_frac=0.1,
    initial_epoch_multiplier=4,
    query_schedule="hyperbolic",
    draw_freq=draw_freq 
)

# TRAIN REWARD
pref_comparisons.train(
    total_timesteps=5_000,
    total_comparisons=200,
)

# TRAIN AGENT ON LEARNED REWARD
# TODO

# EVALUATE AGENT
# TODO
