# model.py
import torch
import torch.nn as nn

# Define the LSTM Model
class NetworkOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NetworkOptimizer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.relu(lstm_out[:, -1, :]))
        return out

