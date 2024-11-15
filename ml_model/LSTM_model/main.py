import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import load_network_data
from model import NetworkOptimizer
import pandas as pd

# Load Data
data = pd.read_csv('network_data.csv')
sequence_data = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)  # (1, sequence_length, input_size)

# Set target weights (for example purposes)
#                            [packets_dropped, latency, network_utilization]
target_weights = torch.tensor([[0.8, 0.7, 0.9]], dtype=torch.float32)  # Adjusted to match the CSV column order

# Define Model
input_size = data.shape[1]
hidden_size = 64
output_size = 3  # Adjusted to match the target weights size
model = NetworkOptimizer(input_size, hidden_size, output_size)

# Define Loss and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100  # Adjust as needed

for epoch in range(epochs):
    model.train()  # Set model to training mode

    # Forward pass
    output = model(sequence_data)
    loss = criterion(output, target_weights)  # Compare output with target

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print("\nTrained Model Output:", model(sequence_data))