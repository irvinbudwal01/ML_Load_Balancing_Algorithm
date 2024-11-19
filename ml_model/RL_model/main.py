import torch
import torch.optim as optim
import torch.nn as nn
from model import NetworkOptimizer
import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv('network_data.csv')
data.columns = data.columns.str.strip()  # Clean column names

# Extract input features (avg_latency, avg_packets_dropped, avg_server_utilization)
input_features = ['avg_latency', 'avg_packets_dropped', 'avg_server_utilization']
sequence_data = torch.tensor(data[input_features].values, dtype=torch.float32).unsqueeze(0)  # Shape: (1, num_servers, 3)

# Define Model
input_size = len(input_features)  # Number of features per server
hidden_size = 64  # Hidden size for the LSTM
output_size = 3  # 3 servers, so we output 3 traffic weights
model = NetworkOptimizer(input_size, hidden_size, output_size)

# Define Loss and Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Updated reward function with normalization and adjusted balance penalty
def reward_function(weights, latency, packets_dropped, server_utilization):
    # Normalize the weights to ensure the sum is 1
    weights = weights / weights.sum()

    # Reward for balanced weights (closer to equal distribution)
    balance_penalty = abs(weights[0] - 1/3) + abs(weights[1] - 1/3) + abs(weights[2] - 1/3)

    # Reward for good performance (low latency, few dropped packets, and optimal utilization)
    performance_reward = -latency.sum() - packets_dropped.sum() + server_utilization.sum()

    # Regularization term to encourage even distribution (minimizing variance)
    variance_penalty = (weights.var() - 1/9) ** 2  # 1/9 is the target variance for equal distribution

    # Total reward: higher performance with balanced traffic distribution
    total_reward = performance_reward - balance_penalty - variance_penalty
    return total_reward

# Training loop update
epochs = 100
for epoch in range(epochs):
    model.train()  # Set model to training mode

    # Forward pass
    output = model(sequence_data)  # Predict weights for servers

    # Ensure output tensor requires gradients
    output = output.squeeze().requires_grad_()  # Squeeze and ensure requires_grad=True

    # Sample server performance data for the current step (dummy data for now)
    latency = torch.tensor([50, 40, 30], dtype=torch.float32, requires_grad=True)  # Example latency values
    packets_dropped = torch.tensor([5, 3, 2], dtype=torch.float32, requires_grad=True)  # Example dropped packets
    server_utilization = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32, requires_grad=True)  # Example server utilization

    # Calculate the reward based on the model's output and performance data
    reward = reward_function(output, latency, packets_dropped, server_utilization)

    # Backward pass and optimization
    optimizer.zero_grad()
    reward.backward()  # Use reward as the gradient
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Reward: {reward.item():.4f}")

# Display final output after training
model.eval()  # Set model to evaluation mode
final_weights = model(sequence_data).detach().numpy().flatten()

print("------------------------------------------------------")
for i, weight in enumerate(final_weights):
    print(f"Server {i} Traffic Weight: {weight:.4f}")
print("------------------------------------------------------")
