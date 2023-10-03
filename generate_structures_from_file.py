# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:53:45 2023

@author: valla, @adapted by: elamrani

File for testing loaded agents (pickle format).
"""
from single_agent_gym import ReplayDiscreteGymSupervisor as Gym
from relative_single_agent import A2CSupervisor, A2CSupervisorDense,SACSupervisorDense,SACSupervisorSparse
from discrete_blocks import discrete_block as Block
import pickle
import discrete_graphics as gr
import numpy as np
import matplotlib.pyplot as plt

TRAINED_AGENT = "my_trained_agent.pickle"


def create_gym(config):
    #overwrite the action choice method:
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    target = Block([[0,0,1]])
    gym = Gym(config,
              agent_type=SACSupervisorSparse,
              use_wandb=True,
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
    return gym

def load_agent(file,gym,explore=False):
    
    with open(file, "rb") as input_file:
        agent  = pickle.load(input_file)
    if not explore:
        agent.exploration_strat = 'epsilon-greedy'
        agent.eps = 0
    gym.agent = agent
    return gym

if __name__ == "__main__":
    config = {'train_n_episodes':10,
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
            'gap_range':[2,6]
            }
    
    gym = create_gym(config)
    print("Gym created")
    #gym = load_agent("1miosteps.pickle",gym,explore=False)
    gym = load_agent(TRAINED_AGENT,gym,explore=False) # explore=False => greedy policy
    print("Agent loaded, policy: " + gym.agent.exploration_strat)
    # f=70
    # gym.sim.ph_mod.set_max_forces(0,Fx=[-f,f],Fy=[-f,f])
    # gym.sim.ph_mod.set_max_forces(1,Fx=[-f,f],Fy=[-f,f])
    alterations=None
    #alterations=np.array([[0,0]])
    rewards, anim = gym.exploit(gap=4, alterations=alterations, n_alter=2, h=6, draw_robots=True, auto_leave=True)
        # gap: size of the fixed gap to be tested
    print("Rewards:")
    print(rewards)
    gr.save_anim(anim,"exploit",ext='html')
    gr.save_anim(anim,"exploit",ext='gif')
    #name = 'struct8_a2'
    #plt.savefig(f'../graphics/results/experiment 3/{name}.pdf')
    #plt.savefig(f'../graphics/results/experiment 3/{name}.png')
