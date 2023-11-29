# IMPORTS
import numpy as np
from discrete_blocks import discrete_block as Block
from relative_single_agent import SACSupervisorSparse
import logging
import datetime
import os
import sys
import wandb
import pickle
import torch

from single_agent_gym import ReplayDiscreteGymSupervisor
from rlhf_reward_model import RewardLinear, RewardLinearEnsemble
from rlhf_pair_generator import RandomPairGenerator, DisagreementPairGenerator
from rlhf_preference_gatherer import SyntheticPreferenceGatherer, HumanPreferenceGatherer
from rlhf_preference_model import PreferenceModel
from rlhf_reward_trainer import LinearRewardTrainer, LinearRewardEnsembleTrainer
from rlhf_preference_comparisons import PreferenceComparisons


# CONSTANTS
USE_WANDB = True
HUMAN_FEEDBACK = False
LOGGING = True
REMOTE = True
SAVE_AGENT = True
LOGGING_LVL = "info"
DISAGREEMENT = False

if REMOTE:
    device = 'cuda'
else:
    device = 'cpu'

# Set up logger
class CustomFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        # Generate the log filename with the current day and month
        today = datetime.datetime.now()
        log_prefix = today.strftime("%d_%m")
        
        # Specify the directory where log files should be saved
        log_directory = "log"
        
        # Ensure the log directory exists
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        
        # Combine the directory, filename, and log_suffix
        self.filename = os.path.join(log_directory, f"{log_prefix}_{filename}")
        
        # Get the log index
        index = 1
        while os.path.exists(f"{self.filename}_{index}.log"):
            index += 1
        self.filename = f"{self.filename}_{index}.log"
        
        super().__init__(self.filename, mode, encoding, delay)

    def get_file_name(self):
        return self.filename

logger = logging.getLogger(__name__)
log_name = "human_feedback" if HUMAN_FEEDBACK else "synthetic_feedback"
if REMOTE:
    log_name += "_remote"
else:
    log_name += "_local"
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
if LOGGING_LVL == "debug":
    logger.setLevel(logging.DEBUG)
elif LOGGING_LVL == "info":
    logger.setLevel(logging.INFO)

if LOGGING:
    file_handler = CustomFileHandler(log_name)
    filename = file_handler.get_file_name()
    logger.addHandler(file_handler)

if SAVE_AGENT:
    today_str = datetime.datetime.now().strftime("%d_%m")
    location_str = "remote" if REMOTE else "local"
    base_filename = f"{today_str}_trained_agent_reward_learning_{location_str}"
    index = 1
    while os.path.exists(f"{base_filename}_{index}.pt"):
        index += 1
    TRAINED_AGENT = f"{base_filename}_{index}.pt"

if DISAGREEMENT:
    over_sampling = 2
else:
    over_sampling = 1

# blocks
hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=0.5)
linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=0.5) 
linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=0.5) 
linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=0.5)
target = Block([[0,0,1]])

# config
# config = {'train_n_episodes':100,
#             'train_l_buffer':200,
#             'ep_batch_size':32,
#             'ep_use_mask':True,
#             'agent_discount_f':0.1, # 1-gamma
#             'agent_last_only':True,
#             'reward': 'modular',
#             'torch_device': device,
#             'SEnc_n_channels':32, # 64
#             'SEnc_n_internal_layer':2,
#             'SEnc_stride':1,
#             'SEnc_order_insensitive':True,
#             'SAC_n_fc_layer':2, # 3
#             'SAC_n_neurons':64, # 128
#             'SAC_batch_norm':True,
#             'Q_duel':True,
#             'opt_lr':1e-4,
#             'opt_pol_over_val': 1,
#             'opt_tau': 5e-4,
#             'opt_weight_decay':0.0001,
#             'opt_exploration_factor':0.001,
#             'agent_exp_strat':'softmax',
#             'agent_epsilon':0.05, # not needed in sac
#             'opt_max_norm': 2,
#             'opt_target_entropy':0.5,
#             'opt_value_clip':False,
#             'opt_entropy_penalty':False,
#             'opt_Q_reduction': 'min',
#             'V_optimistic':False,
#             'reward_failure':-2,
#             'reward_action':{'Ph': -0.2},
#             'reward_closer':0.4,
#             'reward_nsides': 0.05,
#             'reward_success':5,
#             'reward_opposite_sides':0,
#             'opt_lower_bound_Vt':-2,
#             'gap_range':[2,6]
#             }

config = {'train_n_episodes':70000,
            'train_l_buffer':200,
            'ep_batch_size':256,
            'ep_use_mask':True,
            'agent_discount_f':0.1, # 1-gamma
            'agent_last_only':True,
            'reward': 'modular',
            'torch_device': device,
            'SEnc_n_channels':64, # 64
            'SEnc_n_internal_layer':6,
            'SEnc_stride':1,
            'SEnc_order_insensitive':True,
            'SAC_n_fc_layer':3, # 3
            'SAC_n_neurons':128, # 128
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
            'gap_range':[1,11]
            }

# Set up wandb
if USE_WANDB:
    wandb_project = "sycamore"
    wandb_entity = "sabri-elamrani"
    run = wandb.init(project=wandb_project, entity=wandb_entity, name=filename[4:-4] ,config=config)
    # config = wandb.config

    
# INIT
# Create Reward Model
gamma = 1-config['agent_discount_f']
nb_rewards = 3
if DISAGREEMENT:
    reward_model = RewardLinearEnsemble(gamma, nb_rewards, logger, device)
else:
    reward_model = RewardLinear(gamma, logger, device)

# Create Gym (env + agent)
# gym = ReplayDiscreteGymSupervisor(config,
            #   agent_type=SACSupervisorSparse,
            #   use_wandb=False, # wandb set up elsewhere
            #   actions= ['Ph'], # place-hold only necessary action
            #   block_type=[hexagon],
            #   random_targets='random_gap', 
            #   targets_loc=[[2,0],[6,0]], 
            #   n_robots=2, 
            #   max_blocks = 10,
            #   targets=[target]*2,
            #   max_interfaces = 50,
            #   log_freq = 5, # grid size
            #   maxs = [9,6],
            #   logger = logger,
            #   reward_fun = reward_model.reward_array_features,
            #   use_gabriel=False # not using Gabriel's reward fun
            #   )

gym = ReplayDiscreteGymSupervisor(config,
            agent_type=SACSupervisorSparse,
            use_wandb=False,
            actions= ['Ph'], # place-hold only necessary action
            block_type=[hexagon, linkh, linkr, linkl],
            random_targets='random_gap', 
            targets_loc=[[2,0],[6,0]], 
            n_robots=2, 
            max_blocks = 30,
            targets=[target]*2,
            max_interfaces = 100,
            log_freq = 5,
            maxs = [30,20], # grid size
            logger = logger,
            reward_fun = reward_model.reward_array_features,
            use_gabriel=False)

# Create Pair Generator
if DISAGREEMENT:
    pair_generator = DisagreementPairGenerator(reward_model)
else:
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
    gatherer = SyntheticPreferenceGatherer(coeff, gamma)

# Create Preference Model
preference_model = PreferenceModel(reward_model)

# Create Reward Trainer
learning_rate = 0.0001
if DISAGREEMENT:
    reward_trainer = LinearRewardEnsembleTrainer(preference_model, gamma, learning_rate, logger)
else:
    reward_trainer = LinearRewardTrainer(preference_model, gamma, learning_rate, logger)

# Create Preference Comparisons, the main interface
if HUMAN_FEEDBACK:
    draw_freq = 1
else:
    draw_freq = 100

pref_comparisons = PreferenceComparisons(
    gym,
    reward_model,
    num_iterations=30,  # Set to 60 for better performance
    pair_generator=pair_generator,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    transition_oversampling= over_sampling,
    initial_comparison_frac=0.1,
    initial_epoch_multiplier=4,
    query_schedule="hyperbolic",
    draw_freq=draw_freq,
    use_wandb=USE_WANDB,
    logger = logger,
    comparison_queue_size=50
)

# TRAIN REWARD
logger.info("#######################")
logger.info("REWARD TRAINING STARTED")
logger.info("####################### \n")
pref_comparisons.train(
    total_timesteps=5000, # 5000
    total_comparisons=300, # 200
)
logger.debug("REWARD TRAINING ENDED \n \n")

# TRAIN AGENT ON LEARNED REWARD
logger.info("\n \n ########################################")
logger.info("AGENT TRAINING ON LEARNED REWARD STARTED")
logger.info("######################################## \n")
pref_comparisons.gym.training(nb_episodes=70000, use_wandb = USE_WANDB)
logger.debug("AGENT TRAINING ON LEARNED REWARD ENDED \n \n")
if SAVE_AGENT:
    torch.save(pref_comparisons.gym.agent, TRAINED_AGENT, pickle_module=pickle)

# EVALUATE AGENT
logger.info("\n \n ########################")
logger.info("AGENT EVALUATION STARTED")
logger.info("######################## \n")
success_rate = pref_comparisons.gym.evaluate_agent(nb_trials=1000)
logger.info(f"Average success rate: {success_rate} \n \n")
logger.debug("AGENT EVALUATION ENDED")

# End wandb
if USE_WANDB:
    run.finish()
