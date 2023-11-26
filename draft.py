import pickle
import torch as th


# file = "test.pt"

# agent = th.load(file, map_location=th.device('cpu'), pickle_module=pickle)
# print(agent)

# synthetic_agent = "trained_agents/15_11_trained_agent_gabriel_reward_remote.pickle"
# with open(synthetic_agent, 'rb') as input_file:
#     agent = pickle.load(input_file)

# print(agent)

torch_file = "synth_test.pt"
agent = th.load(torch_file, map_location=th.device('cpu'), pickle_module=pickle)
print(f"Model {agent.model}")
for i, opti in enumerate(agent.optimizer.Qs):
    print(f"Optimizer {i} \n {agent.optimizer.Qs}")
