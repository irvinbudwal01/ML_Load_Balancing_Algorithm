import torch
import torch.optim as optim
import torch.nn as nn
from model import NetworkOptimizer
import pandas as pd

# Load and preprocess data
data = pd.read_csv('network_data.csv')
data.columns = data.columns.str.strip()  # Clean column names

# Extract input features (avg_latency, avg_packets_dropped, avg_server_utilization)
input_features = ['avg_latency', 'avg_packets_dropped', 'avg_server_utilization']
sequence_data = torch.tensor(data[input_features].values, dtype=torch.float32).unsqueeze(0)  # Shape: (1, num_servers, 3)

# Define Model
input_size = len(input_features)  # Number of features per server
hidden_size = 64  # Hidden size for the LSTM
output_size = len(data)  # Number of servers (each server gets a weight)
model = NetworkOptimizer(input_size, hidden_size, output_size)

# Define Loss and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Target weights (example for demonstration; modify based on your logic)
# In a real scenario, this could be dynamically determined or based on ground truth
target_weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)  # Adjust weights for servers

# Training loop
epochs = 100  # Number of training iterations
for epoch in range(epochs):
    model.train()  # Set model to training mode

    # Forward pass
    output = model(sequence_data)  # Predict weights for servers
    loss = criterion(output, target_weights)  # Compare predictions with target weights

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Display final output
model.eval()  # Set model to evaluation mode
final_weights = model(sequence_data).detach().numpy().flatten()
print("------------------------------------------------------")
for i, weight in enumerate(final_weights):
    print(f"Server {i} Traffic Weight: {weight:.4f}")
print("------------------------------------------------------")
