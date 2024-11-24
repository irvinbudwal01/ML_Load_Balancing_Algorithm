import torch
import torch.optim as optim
import torch.nn as nn
from model import NetworkOptimizer
import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv('../dummy_data/network_data.csv')
data.columns = data.columns.str.strip()  # Clean column names

# Extract input features (avg_latency, total_packets_dropped, server_utilization)
input_features = ['avg_latency', 'total_packets_dropped', 'server_utilization']
sequence_data = torch.tensor(data[input_features].values, dtype=torch.float32).unsqueeze(0)  # Shape: (1, num_servers, 3)

# Define Model
input_size = len(input_features)  # Number of features per server
hidden_size = 64  # Hidden size for the LSTM
output_size = 3  # 3 servers, so we output 3 traffic weights
model = NetworkOptimizer(input_size, hidden_size, output_size)

# Define Loss and Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

def reward_function(weights, latency, packets_dropped, server_utilization):
    # Normalize weights to ensure they sum to 1
    weights = weights / weights.sum()

    # Stronger penalty for any weight greater than 60% or less than 10%
    imbalance_penalty = 200 * (max(weights[0], 1 - weights[0]) + max(weights[1], 1 - weights[1]) + max(weights[2], 1 - weights[2]))

    # Performance reward (negative for latency and dropped packets, positive for utilization)
    performance_reward = -latency.sum() - packets_dropped.sum() + server_utilization.sum()

    # Total reward: balance between performance and balanced traffic distribution
    total_reward = performance_reward - imbalance_penalty
    return total_reward


print("-------------------------------------")
print("RL Model")
print("-------------------------------------")
# Training loop update
epochs = 100
for epoch in range(epochs):
    model.train()  # Set model to training mode

    # Forward pass
    output = model(sequence_data)  # Predict weights for servers

    # Ensure output tensor requires gradients
    output = output.squeeze().requires_grad_()  # Squeeze and ensure requires_grad=True

    # Server performance data for the current step 
    latency = torch.tensor([200, 600, 500], dtype=torch.float32, requires_grad=True) 
    packets_dropped = torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)  
    server_utilization = torch.tensor([500, 300, 450], dtype=torch.float32, requires_grad=True)  

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

print("-------------------------------------")
for i, weight in enumerate(final_weights):
    print(f"Server {i + 1} Traffic Weight: {weight:.4f}")
print("-------------------------------------")
