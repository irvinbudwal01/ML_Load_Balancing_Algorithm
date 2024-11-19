import torch
import torch.nn as nn

# Define the LSTM Model
class NetworkOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NetworkOptimizer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)  # Ensure output sums to 1

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output
        return self.softmax(out)  # Normalize outputs to sum to 1
