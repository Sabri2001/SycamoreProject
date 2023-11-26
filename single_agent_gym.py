# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:33:47 2022

@author: valla, @adapted by: elamrani

High-level file: generates gym -> training/testing from here.
"""

import copy
import time
import os
import wandb
import numpy as np
import pickle
import torch

from discrete_blocks import discrete_block as Block
from relative_single_agent import SACSupervisorSparse,generous_reward,punitive_reward,modular_reward
from discrete_simulator import DiscreteSimulator as Sim, Transition, Trajectory
import discrete_graphics as gr

# Define blocks
hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
triangle = Block([[0,0,1]],muc=0.5)
link = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5)

# Set up wandb
wandb_project = "sycamore"
wandb_entity = "sabri-elamrani"
USE_WANDB = True

# Save options
SAVE = True
TRAINED_AGENT = "26_11_trained_agent_gabriel_reward_remote.pickle"
NAME = "26_11_trained_agent_gabriel_reward_remote" # for wandb


class ReplayDiscreteGymSupervisor():
    def __init__(self,
                 config,
                 maxs = [10,10],
                 block_type = [hexagon,link],
                 random_targets = 'random',
                 targets = [triangle]*2,
                 targets_loc = [[3,0],[6,0]],
                 n_robots=2,
                 ranges = None,
                 agent_type = SACSupervisorSparse,
                 actions = ['Ph','Pl','L'], # place-hold, place-leave, leave
                 max_blocks = 30,
                 max_interfaces = 100,
                 log_freq = 100,
                 reward_fun = None,
                 use_wandb=False,
                 logger = None,
                 use_gabriel=True
            ):
        
        # wandb init
        if use_wandb:
            self.use_wandb = True
            self.run = wandb.init(project=wandb_project, entity=wandb_entity, name = NAME, config=config)
            self.config = wandb.config
        else:
            self.use_wandb = False
            self.config = config

        # logger init
        self.logger = logger
        
        # ranges (blocks)
        if ranges is None:
            ranges = np.ones((n_robots,maxs[0],maxs[1],2),dtype = bool)
        self.log_freq = log_freq
        self.n_robots = n_robots
        
        # init gap_range
        self.gap_range = config.get('gap_range')
        if self.gap_range is None:
            self.gap_range = [1,self.sim.grid.shape[0]-2]
            
        # random_gap, random_gap_center or targets directly specified
        if random_targets == 'random_gap':
            self.targets = [triangle]
            self.targets += [Block([[i,0,1] for i in range(maxs[0]-gap-1)],muc=0.5) for gap in range(self.gap_range[0],self.gap_range[1])]
        elif random_targets == 'random_gap_center':
            self.gap_range = config.get('gap_range') or [1,self.sim.grid.shape[0]-2]
            min_ground_width = int(np.floor((maxs[0]-self.gap_range[1]+1)/2))
            max_ground_width = int(np.ceil((maxs[0]-self.gap_range[0])/2))
            self.targets_gap = np.zeros((self.gap_range[1],2),dtype=int)
            self.targets_gap[self.gap_range[0]:,0]=np.ceil(np.arange(maxs[0]-self.gap_range[0],maxs[0]-self.gap_range[1],-1)/2)-min_ground_width
            self.targets_gap[self.gap_range[0]:,1]=np.floor(np.arange(maxs[0]-self.gap_range[0],maxs[0]-self.gap_range[1],-1)/2)-min_ground_width
            self.targets = [Block([[i,0,1] for i in range(w)],muc=0.5) for w in range(min_ground_width,max_ground_width+1)]
        else:
            self.targets = targets
        
        # Init simu env
        self.sim = Sim(maxs,n_robots,block_type,len(targets_loc),max_blocks,max_interfaces,ground_blocks=self.targets)
        self.random_targets = random_targets
        
        # directly specify position targets
        if random_targets == 'fixed':
            for tar,loc in zip(targets,targets_loc):
                self.sim.add_ground(tar,loc)
        self.setup = copy.deepcopy(self.sim)
        
        # Init agent
        self.agent = agent_type(n_robots,
                                block_type,
                                self.config,
                                ground_blocks = self.targets,
                                action_choice =actions,
                                grid_size=maxs,
                                use_wandb=use_wandb,
                                log_freq = self.log_freq,
                                env="norot")

        # Reward fun chosen (see def in relative_single_agent) 
        self.use_gabriel = use_gabriel

        if reward_fun is None:
            if config['reward']=='punitive':
                self.rewardf = punitive_reward
            elif config['reward']=='generous':
                self.rewardf = generous_reward
            elif config['reward']=='modular':
                self.rewardf = modular_reward
        else:
            self.rewardf = reward_fun

        # Store config as attribute -> for access elsewhere
        self.config = config
        
    def episode_restart(self,
                          max_steps,
                          draw=False,
                          transition_buffer=None,
                          trajectory_buffer=None,
                          transition_buffer_count=0,
                          trajectory_buffer_count=0,
                          auto_leave = True,
                          train = True
                          ):
        """
        Runs an episode, computes reward and updates policy.
        """
        # Some inits
        use_mask = self.config['ep_use_mask']
        batch_size = self.config['ep_batch_size']
        #if the action is not valid, stop the episode
        success = False
        failure = False
        rewards_ar = np.zeros((self.n_robots,max_steps))
        self.sim =copy.deepcopy(self.setup)
        gap=None

        # Init targets
        if self.random_targets== 'random':
            validlocs = np.ones(self.sim.grid.shape,dtype=bool)
            #dont allow the target to be all the way to the extremity of the grid
            validlocs[:2,:]=False
            validlocs[-2:,:]=False
            validlocs[:,-2:]=False
            for tar in self.targets:
                valid = np.array(np.nonzero(validlocs)).T
                idx = np.random.randint(len(valid))
                self.sim.add_ground(tar,[valid[idx,0],valid[idx,1]])
                validlocs[max(valid[idx,0]-1,0):valid[idx,0]+2,max(valid[idx,1]-1,0):valid[idx,1]+2]=False
        if self.random_targets == 'random_flat':
            validlocs = np.ones(self.sim.grid.shape[0],dtype=bool)
            #dont allow the target to be all the way to the extremity of the grid
            validlocs[1]=False
            validlocs[-1]=False
            for tar in self.targets:
                valid = np.array(np.nonzero(validlocs)).flatten()
                idx = np.random.randint(len(valid))
                self.sim.add_ground(tar,[valid[idx],0])
                
                validlocs[max(valid[idx]-2,0):valid[idx]+3]=False
        if self.random_targets == 'random_gap':
            gap = np.random.randint(self.gap_range[0],self.gap_range[1])
            self.sim.add_ground(self.targets[gap-self.gap_range[0]+1],[0,0],ground_type=gap-self.gap_range[0]+1)
            self.sim.add_ground(self.targets[0],[self.sim.grid.shape[0]-1,0],ground_type=0)
        if self.random_targets == 'random_gap_center':
            gap = np.random.randint(self.gap_range[0],self.gap_range[1])
            self.sim.add_ground(self.targets[self.targets_gap[gap,0]],[0,0],ground_type=self.targets_gap[gap,0])
            #tar = Block([[i,0,1] for i in range(self.sim.grid.shape[0]-gap-1)],muc=0.7)
            [tar.move([0,0]) for tar in self.targets]
            width = [np.max(tar.parts[:,0]) for tar in self.targets]
            self.sim.add_ground(self.targets[self.targets_gap[gap,1]],[width[self.targets_gap[gap,0]]+gap+1,0],ground_type=self.targets_gap[gap,1])
        elif self.random_targets== 'half_fixed':
            assert False, "not implemented"

        # More inits
        if draw:
            self.sim.setup_anim()
            self.sim.add_frame()
        if use_mask:
            mask = self.agent.generate_mask(self.sim,0)
        else:
            mask = None

        # Init trajectory (not needed in training mode)
        if not train:
            trajectory = Trajectory()

        # Keep track of training loss
        if train:
            total_loss = 0
            loss_nb = 0 # nb of additions to loss, to compute average

        # RUN AN EPISODE
        for step in range(max_steps):
            for idr in range(self.n_robots):
                # Mask -> always use it
                if use_mask:
                    prev_state = {'grid':copy.deepcopy(self.sim.grid),
                                  'graph': copy.deepcopy(self.sim.graph),
                                  'mask':mask.copy(),
                                  'forces':copy.deepcopy(self.sim.ph_mod),
                                  }
                else:
                    prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':None,'forces':copy.deepcopy(self.sim.ph_mod),'sim':copy.deepcopy(self.sim)}
                
                # Choose action
                action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
                
                # Take action
                valid,closer,blocktype,interfaces = self.agent.Act(self.sim,action,**action_args,draw=draw)
                if use_mask:
                    mask = self.agent.generate_mask(self.sim,(idr+1)%self.n_robots)
                if valid:
                    if np.all(self.sim.grid.min_dist < 1e-5) and (auto_leave or np.all(self.sim.grid.hold==-1)):
                        if auto_leave:
                            bids = []
                            for r in range(self.n_robots):
                                bids.append(self.sim.leave(r))
                            if self.sim.check():
                                success = True
                                if use_mask:
                                    mask[:]=False
                            else:
                                for r,bid in enumerate(bids):
                                    self.sim.hold(r,bid)
                        else:
                            success = True
                            if use_mask:
                                mask[:]=False
                else:
                    failure = True
                    #mark the state as terminal
                    if use_mask:
                        mask[:]=False
                if step == max_steps-1 and idr == self.n_robots-1:
                    #the end of the episode is reached
                    if use_mask:
                        failure = True
                        mask[:]=False
                if closer == None:
                    closer = False

                # Needed for reward computation
                if interfaces is not None:
                    sides_id,n_sides_ori = np.unique(interfaces[:,0],return_counts=True)
                    n_sides = np.zeros(6,dtype=int)
                    n_sides[sides_id.astype(int)]=n_sides_ori
                else:
                    n_sides = []
                
                # Compute reward and reward features for this robot/step
                reward_features = [int(bool(action)), int(closer), int(success), \
                                   int(failure), np.sum(n_sides), \
                                    int(not np.all(np.logical_xor(n_sides[:3],n_sides[3:])))]
                
                if self.use_gabriel:
                    reward = self.rewardf(action, valid, closer, success, failure, n_sides=n_sides, config=self.config)
                else:
                    reward = self.rewardf(reward_features)
                
                # Add reward to reward array (robot,step)
                rewards_ar[idr,step]=reward
                if self.agent.rep == 'graph':
                    transition_buffer.push(idr,prev_state['sim'],action_enc[0],self.sim,reward,terminal=success or failure)
                else:
                    current_transition = Transition(prev_state,
                                                        action_enc,
                                                        reward,
                                                        {'grid':copy.deepcopy(self.sim.grid),
                                                        'graph': copy.deepcopy(self.sim.graph),
                                                        'mask':mask,
                                                        'forces':copy.deepcopy(self.sim.ph_mod)},
                                                        reward_features)
                
                # Update policy using reward buffer if in training mode
                if train:
                    transition_buffer[(transition_buffer_count)%transition_buffer.shape[0]] = current_transition
                    transition_buffer_count +=1
                    loss = self.agent.update_policy(transition_buffer,transition_buffer_count,batch_size)
                    if loss != None:
                        total_loss += loss
                        loss_nb += 1  
                else:
                    trajectory.add_transition(current_transition)

                # Drawing and terminations
                if draw:
                    action_args.pop('rid')                   
                    self.sim.draw_act(idr,action,blocktype,prev_state,**action_args)
                    self.sim.add_frame()
          
                if success or failure:
                    break
            if success or failure:
                break
        
        if draw:
            anim = self.sim.animate()
        else:
            anim = None

        # NOTE: when implementing human fb, might have to replace block above by this one
        # if draw:
        #     anim = self.sim.frames
        # else:
        #     anim = None

        if train:
            if loss_nb == 0:
                average_loss = None
            else:
                average_loss = total_loss/loss_nb

        # Return depends on mode (training/trajectory generation)
        if train:
            return rewards_ar,step,anim,transition_buffer,transition_buffer_count,success,gap,average_loss
        else:
            trajectory.set_animation(anim)
            trajectory_buffer[trajectory_buffer_count] = trajectory
            trajectory_buffer_count += 1
            return rewards_ar,step,anim,trajectory_buffer,trajectory_buffer_count,success,gap
    
    def training(self,
                pfreq = 10,
                draw_freq=100, # leave this one, useful later
                max_steps=100,
                success_rate_decay = 0.01,
                nb_episodes = 100,
                use_wandb = False,
                log_dir=None,
                nb_traj=10
                ):
        """
        Initialises buffer, and repeatedly (train_n_episodes times) 
        calls episode_restart() to update policy.
        """
        # init success_rate for each possible gap size
        if self.random_targets == 'random_gap' or self.random_targets == 'random_gap_center':
            success_rate = np.zeros(self.gap_range[1])
            success_rate[0]=1
            gap_counts = np.zeros(self.gap_range[1])
            res_dict={}
        else:
            success_rate = 0 # fixed size => only need one

        # init transition buffer
        transition_buffer = np.empty(self.config['train_l_buffer'], dtype = object)
        buffer_count= 0

        # start training
        print("Training started")

        # run over several episodes
        for episode in range(nb_episodes):
            # run an episode
            (rewards_ep, _,
             anim, transition_buffer, 
             buffer_count, success, gap, loss) = self.episode_restart(max_steps,
                                                            #   draw = episode % draw_freq == 0, #draw_freq-1,
                                                              draw = False, # temporary
                                                              transition_buffer=transition_buffer,
                                                              transition_buffer_count=buffer_count,
                                                              auto_leave=True
                                                              )
            # update success_rate
            if self.random_targets == 'random_gap' or self.random_targets =='random_gap_center':
                if success:
                    success_rate[gap] = (1-success_rate_decay)*success_rate[gap] +success_rate_decay
                else:
                    success_rate[gap] = (1-success_rate_decay)*success_rate[gap]
            else:
                if success:
                    success_rate = (1-success_rate_decay)*success_rate +success_rate_decay
                else:
                    success_rate = (1-success_rate_decay)*success_rate
            
            # log
            if not self.use_gabriel and episode % pfreq==0:
                self.logger.info(f'episode {episode}/{nb_episodes} rewards: {np.sum(rewards_ep,axis=1)}')
                # self.logger.info(f"Success rate (gap 2): {success_rate[2]}")
                _, suc_rate = self.generate_trajectories(nb_traj)
                self.logger.info(f'Success rate: {suc_rate} - Loss: {loss}')

            # wandb
            if use_wandb and episode % self.log_freq == 0:
                if self.random_targets == 'random_gap' or self.random_targets == 'random_gap_center':
                    for i in np.arange(self.gap_range[0],self.gap_range[1]):
                        res_dict[f'success_rate_gap{i}']=success_rate[i]
                    wandb.log(res_dict)
                else:
                    wandb.log({'succes_rate':success_rate})

            # save anim
            if anim is not None:
                if self.use_wandb:
                    if success:
                        wandb.log({f'success_animation_gap_{gap}':wandb.Html(anim.to_jshtml())})
                        gr.save_anim(anim,os.path.join(log_dir, f"success_animation_gap_{i}_ep{episode}"),ext='gif')
                    else:
                        wandb.log({'animation':wandb.Html(anim.to_jshtml())})
                
        return anim
    
    def generate_trajectories(self,
                nb_traj = 100,
                draw_freq=100,
                max_steps=100,
                success_rate_decay = 0.01):
        """
        Initialises trajectory buffer, and repeatedly (n_episodes) 
        calls episode_restart(train=False) to fill buffer.
        """
        # init success_rate for each possible gap size
        if self.random_targets == 'random_gap' or self.random_targets == 'random_gap_center':
            success_rate = np.zeros(self.gap_range[1])
            success_rate[0]=1
            gap_counts = np.zeros(self.gap_range[1])
            res_dict={}
        else:
            success_rate = 0 # fixed size => only need one


        # init trajectory buffer
        trajectory_buffer = np.empty(shape=nb_traj, dtype=object)
        buffer_count = 0

        # run over several episodes
        for episode in range(nb_traj):
            # run an episode
            (_, _,
             _, trajectory_buffer, 
             buffer_count, success, gap) = self.episode_restart(max_steps,
                                                              draw = episode % draw_freq == 0, #draw_freq-1,
                                                              trajectory_buffer=trajectory_buffer,
                                                              trajectory_buffer_count=buffer_count,
                                                              auto_leave=True,
                                                              train=False
                                                              )
            # update success_rate
            if self.random_targets == 'random_gap' or self.random_targets =='random_gap_center':
                gap_counts[gap] += 1
                success_rate[gap] = success_rate[gap]*(gap_counts[gap]-1)/(gap_counts[gap]) + success/gap_counts[gap]
            else:
                success_rate = success_rate*episode/(episode+1) + success/(episode+1)
            
        return trajectory_buffer, success_rate[2:]

    def evaluate_agent(self,
                nb_trials = 100,
                draw_freq=100,
                max_steps=100,
                success_rate_decay = 0.01
                ):
        """
        Evaluates agent's current policy.
        """
        # init success_rate for each possible gap size
        if self.random_targets == 'random_gap' or self.random_targets == 'random_gap_center':
            success_rate = np.zeros(self.gap_range[1])
            success_rate[0]=1
            res_dict={}
        else:
            success_rate = 0 # gap_size 1 => scalar

        # init trajectory buffer
        trajectory_buffer = np.empty(shape=nb_trials, dtype=object)
        buffer_count = 0

        # Switch to epsilon-greedy policy (exploitation)
        self.agent.exploration_strat = 'epsilon-greedy'
        self.agent.eps = 0

        # start training
        print("Agent evaluation started")

        # run over several episodes
        for episode in range(nb_trials):
            # run an episode
            (_, _,
             _, trajectory_buffer, 
             buffer_count, success, gap) = self.episode_restart(max_steps,
                                                              draw = episode % draw_freq == 0,#draw_freq-1,
                                                              trajectory_buffer=trajectory_buffer,
                                                              trajectory_buffer_count=buffer_count,
                                                              auto_leave=True,
                                                              train=False
                                                              )
            # update success_rate
            if self.random_targets == 'random_gap' or self.random_targets =='random_gap_center':
                if success:
                    success_rate[gap] = (1-success_rate_decay)*success_rate[gap] +success_rate_decay
                else:
                    success_rate[gap] = (1-success_rate_decay)*success_rate[gap]
            else:
                if success:
                    success_rate = (1-success_rate_decay)*success_rate +success_rate_decay
                else:
                    success_rate = (1-success_rate_decay)*success_rate
        
        print("Agent evaluation finished")
        return success_rate[2] # success_rate of gap 2

    def avg_return_agent(self,
                nb_trials = 100,
                draw_freq=100,
                max_steps=100,
                success_rate_decay = 0.01
                ):
        """
        Evaluates agent's current policy using average reward return.
        """
        # init success_rate for each possible gap size
        if self.random_targets == 'random_gap' or self.random_targets == 'random_gap_center':
            success_rate = np.zeros(self.gap_range[1])
            success_rate[0]=1
            res_dict={}
        else:
            success_rate = 0 # gap_size 1 => scalar

        # init trajectory buffer
        trajectory_buffer = np.empty(shape=nb_trials, dtype=object)
        buffer_count = 0

        # Switch to epsilon-greedy policy (exploitation)
        self.agent.exploration_strat = 'epsilon-greedy'
        self.agent.eps = 0

        # start training
        print("Agent evaluation started")

        # total reward
        total_reward = 0

        # run over several episodes
        for episode in range(nb_trials):
            # run an episode
            (reward_ar, _,
             _, trajectory_buffer, 
             buffer_count, success, gap) = self.episode_restart(max_steps,
                                                              draw = episode % draw_freq == 0,#draw_freq-1,
                                                              trajectory_buffer=trajectory_buffer,
                                                              trajectory_buffer_count=buffer_count,
                                                              auto_leave=True,
                                                              train=False
                                                              )
            # update success_rate
            if self.random_targets == 'random_gap' or self.random_targets =='random_gap_center':
                if success:
                    success_rate[gap] = (1-success_rate_decay)*success_rate[gap] +success_rate_decay
                else:
                    success_rate[gap] = (1-success_rate_decay)*success_rate[gap]
            else:
                if success:
                    success_rate = (1-success_rate_decay)*success_rate +success_rate_decay
                else:
                    success_rate = (1-success_rate_decay)*success_rate
        
            # update total reward
            total_reward += np.sum(reward_ar)

        print("Agent evaluation finished")
        return total_reward/nb_trials

    def exploit(self,gap,
                alterations=None,
                max_steps=30,
                auto_leave=True,
                n_alter = 1,
                h=6,
                draw_robots=True,
                print_robot_force=True):
        
        # Init simu + agent
        time_sim=0
        time_chose=0
        time_draw=0
        use_mask = self.config['ep_use_mask']
        rewards_ar = np.zeros((self.n_robots,max_steps))
        #self.sim =copy.deepcopy(self.setup)
        gap=gap
        t0s = time.perf_counter()
        tar = Block([[i,0,1] for i in range(self.sim.grid.shape[0]-gap-1)],muc=0.7)
        self.sim.add_ground(tar,[0,0])
        self.sim.add_ground(triangle,[self.sim.grid.shape[0]-1,0])
        time_sim +=time.perf_counter()-t0s
        t0d=time.perf_counter()
        self.sim.setup_anim(h=h)
        self.sim.add_frame(draw_robots=draw_robots)
        time_draw+= time.perf_counter()-t0d
        if use_mask:
            mask = self.agent.generate_mask(self.sim,0)
        else:
            mask = None
        success = False
        failure = False

        # Start simu (only one run)
        for step in range(max_steps):
            for idr in range(self.n_robots):
                t0s = time.perf_counter()
                prev_state = {'grid':copy.deepcopy(self.sim.grid),'graph': copy.deepcopy(self.sim.graph),'mask':mask.copy(),'forces':copy.deepcopy(self.sim.ph_mod)}
                time_sim+=time.perf_counter()-t0s
                t0c=time.perf_counter()

                # Choose action
                action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
                time_chose+=time.perf_counter()-t0c
                if alterations is not None and step in alterations[:,0] and idr in alterations[:,1]:
                    for n_alter in range(n_alter):
                        mask[action_enc[0]]=False
                        action,action_args,*action_enc = self.agent.choose_action(idr,self.sim,mask=mask)
                t0s = time.perf_counter()

                # Take action
                valid,closer,blocktype,interfaces = self.agent.Act(self.sim,action,**action_args,draw=True)
                
                if use_mask:
                    mask = self.agent.generate_mask(self.sim,(idr+1)%self.n_robots)
                if valid:
                    # if print_robot_force:
                    #     for i in range(self.n_robots):
                    #         self.sim.get_force(i)
                    if np.all(self.sim.grid.min_dist < 1e-5) and (auto_leave or np.all(self.sim.grid.hold==-1)):
                        if auto_leave:
                            bids = []
                            for r in range(self.n_robots):
                                bids.append(self.sim.leave(r))
                            if self.sim.check():
                                success = True
                                if use_mask:
                                    mask[:]=False
                            else:
                                for r,bid in enumerate(bids):
                                    self.sim.hold(r,bid)
                        else:
                            success = True
                            if use_mask:
                                mask[:]=False
                else:
                    failure = True
                    #mark the state as terminal
                    mask[:]=False
                
                if step == max_steps-1 and idr == self.n_robots-1:
                    #the end of the episode is reached
                    mask[:]=False
                                  
                if interfaces is not None:
                    sides_id,n_sides_ori = np.unique(interfaces[:,0],return_counts=True)
                    n_sides = np.zeros(6,dtype=int)
                    n_sides[sides_id.astype(int)]=n_sides_ori
                else:
                    n_sides = None
                
                # Compute scalar reward for this (robot,step), add it to array
                reward = self.rewardf(action, valid, closer, success,failure,n_sides=n_sides,config=self.config)
                rewards_ar[idr,step]=reward
                
                # Draw
                action_args.pop('rid')
                time_sim += time.perf_counter()-t0s
                t0d=time.perf_counter()
                self.sim.draw_act(idr,action,blocktype,prev_state,draw_robots=draw_robots,**action_args)
                self.sim.add_frame(draw_robots=draw_robots)
                time_draw+= time.perf_counter()-t0d
                if success or failure:
                    break
            if success or failure:
                break
        t0d=time.perf_counter()
        anim = self.sim.animate()
        time_draw+= time.perf_counter()-t0d
        print(f"Time used in the simulator: {time_sim}")
        print(f"Time used to choose the actions: {time_chose}")
        print(f"Time used to draw the animation: {time_draw}")
        return rewards_ar, anim
        
    def test(self,
             draw=True):
        from relative_single_agent import int2act_norot
        self.agent.Act(self.sim,'Ph',rid=0,
                        sideblock=0,
                        sidesup = 1,
                        bid_sup = 0,
                        idconsup = 1,
                        blocktypeid = 0,
                        side_ori = 0,
                        draw= False)
        self.agent.Act(self.sim,'Ph',rid=1,
                        sideblock=0,
                        sidesup = 0,
                        bid_sup = 1,
                        idconsup = 1,
                        blocktypeid = 1,
                        side_ori = 4,
                        draw= False)
        setup = copy.deepcopy(self.sim)
        mask = self.agent.generate_mask(self.sim,0)
        #self.sim.setup_anim()
        #self.sim.add_frame()
        while mask.any():
            
            actionids,= np.nonzero(mask)
            action,action_params = int2act_norot(actionids[0],self.sim.graph.n_blocks,
                                                 self.n_robots,
                                                 self.agent.n_side_oriented,
                                                 self.agent.n_side_oriented_sup,
                                                 self.agent.last_only,
                                                 self.agent.max_blocks,
                                                 self.agent.action_choice)
            self.agent.Act(self.sim,action,**action_params,draw=True)
            self.sim.draw_state_debug()
            mask[actionids[0]]=False
            self.sim =copy.deepcopy(setup)
            #self.sim.setup_anim()
            #self.sim.add_frame()
        #anim = self.sim.animate()
        return None

    def test_gap(self,gap=None):
        if gap is None:
            if self.gap_range is not None:
                gap = np.random.randint(self.gap_range[0],self.gap_range[1])
            else:
                gap = np.random.randint(1,self.sim.grid.shape[0]-3)
        tar = Block([[i,0,1] for i in range(self.sim.grid.shape[0]-gap-1)],muc=0.7)
        self.sim.add_ground(tar,[0,0])
        self.sim.add_ground(triangle,[self.sim.grid.shape[0]-1,0])
        
        from relative_single_agent import int2act_norot
        self.agent.Act(self.sim,'Ph',rid=0,
                        sideblock=0,
                        sidesup = 0,
                        bid_sup = 0,
                        idconsup = 1,
                        blocktypeid = 0,
                        side_ori = 0,
                        draw= False)
        self.agent.Act(self.sim,'Ph',rid=1,
                        sideblock=0,
                        sidesup = 0,
                        bid_sup = 1,
                        idconsup = 0,
                        blocktypeid =0,
                        side_ori = 5,
                        draw= False)
        self.agent.Act(self.sim,'Ph',rid=0,
                        sideblock=0,
                        sidesup = 0,
                        bid_sup = 2,
                        idconsup = 0,
                        blocktypeid =0,
                        side_ori = 1,
                        draw= False)
        self.agent.Act(self.sim,'L',rid=0,
                       )
        self.agent.Act(self.sim,'L',rid=1,
                        )
        setup = copy.deepcopy(self.sim)
        self.sim.draw_state_debug()

     
if __name__ == '__main__':
    print("Start gym")
    # config
    config = {'train_n_episodes':50000,
            'train_l_buffer':200,
            'ep_batch_size':32,
            'ep_use_mask':True,
            'agent_discount_f':0.1, # 1-gamma
            'agent_last_only':True,
            'reward': 'modular',
            'torch_device': 'cuda',
            'SEnc_n_channels':32, # 64
            'SEnc_n_internal_layer':2,
            'SEnc_stride':1,
            'SEnc_order_insensitive':True,
            'SAC_n_fc_layer':2, # 3
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
            'opt_target_entropy':1.8,
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
            'gap_range':[2,6]
            }
   
    # Create various shapes from basic Block object
    hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5) 
    linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.5) 
    linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=0.5)
    #target = Block([[0,0,1],[1,0,1]])
    target = Block([[0,0,1]])

    # Create gym
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
    
    # Run training/test
    t0 = time.perf_counter()
    anim = gym.training(max_steps = 20, draw_freq = 200, pfreq =10,
                         use_wandb=USE_WANDB, nb_episodes=50000) # draw and print freq
    #gym.test_gap()
    #gr.save_anim(anim,os.path.join(".", f"test_graph"),ext='html')

    # if SAVE:
    #     with open(TRAINED_AGENT, "wb") as input_file:
    #         pickle.dump(gym.agent,input_file)

    if SAVE:
        torch.save(gym.agent, TRAINED_AGENT, pickle_module=pickle)

    t1 = time.perf_counter()
    print(f"time spent: {t1-t0}s")
    print("\nEnd gym")
