import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


coeff = nn.Parameter(torch.randn(6), requires_grad=True)

# Initialize the loss function
loss_fn = nn.BCELoss()

# Initialize the optimizer
learning_rate = 0.08
optimizer = optim.SGD([{'params': coeff, 'lr': learning_rate}])


def forward(coeff, reward_feature_pair):
    traj1 = reward_feature_pair[0]
    # print("reward feature1 : ", traj1)
    traj2 = reward_feature_pair[1]
    reward1 = reward(coeff, traj1)
    # print("reward1: ", reward1)
    reward2 = reward(coeff, traj2)
    # print("reward2: ", reward2)
    proba = probability(reward1, reward2)
    return proba
    
def reward(coeff, reward_features):
    return torch.dot(coeff, torch.tensor(reward_features, dtype=torch.float32))

def probability(rew1, rew2):
    # Stack the tensors along a new dimension (e.g., dimension 0)
    rewards = torch.stack([rew1, rew2])

    # Apply softmax to the tensor along dimension 0
    softmax_values = F.softmax(rewards, dim=0)
    return softmax_values[0]

def train_step(trajectory_pair, pref_traj1):
    optimizer.zero_grad()

    # Calculate reward values using the preference model
    proba_traj1 = forward(coeff, trajectory_pair)
    # print("proba traj1 (sotmax): ", proba_traj1)

    # Compute the cross-entropy loss
    loss = loss_fn(proba_traj1, torch.tensor(pref_traj1, dtype=torch.float32))

    loss.backward()
    optimizer.step()

    # print("loss: ", loss.item())
    return loss.item()

def train(dataset, epoch_multiplier=1.):
    num_epochs = int(epoch_multiplier * 1000)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for sample in dataset:
            trajectory_pair, pref_traj1 = sample
            # print("traj pair: ", trajectory_pair)
            # print("pref traj1: ", pref_traj1)

            loss = train_step(trajectory_pair, pref_traj1)
            total_loss += loss

        # Print the average loss for this epoch
        if epoch % 50 == 0:
            average_loss = total_loss / len(dataset)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {average_loss:.4f}")
            print(f"---> Current reward coefficients: {coeff}")


if __name__ == '__main__':
    dataset = [((np.random.rand(6), np.random.rand(6)), 1),
               ((np.random.rand(6), np.random.rand(6)), 1),
               ((np.random.rand(6), np.random.rand(6)), 0.5),
               ((np.random.rand(6), np.random.rand(6)), 0.5),
               ((np.random.rand(6), np.random.rand(6)), 0),
               ((np.random.rand(6), np.random.rand(6)), 0)
               ]
    train(dataset)
