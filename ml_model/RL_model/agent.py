import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class LSTMPolicyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPolicyModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        # Ensure state is 3D: (batch_size, sequence_length, input_dim)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # LSTM forward pass
        out, _ = self.lstm(state)
        out = self.fc(out[:, -1, :])
        return out


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Define the policy model (LSTM)
        self.policy_model = LSTMPolicyModel(state_size, hidden_dim=64, output_dim=action_size)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99  # Discount factor
        self.memory = []  # Experience replay buffer
        self.batch_size = 32
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration probability

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: select a random action
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_model(state)
        return torch.argmax(q_values).item()  # Exploit: select the action with max Q-value

    def train(self):
        if len(self.memory) < self.batch_size:
            return  # Skip training until sufficient data
        
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of numpy arrays to single numpy arrays first
        states = np.array(states)  # Convert list of numpy arrays to a single numpy array
        next_states = np.array(next_states)  # Do the same for next_states

        # Then convert the numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q-learning target
        with torch.no_grad():
            q_targets = rewards + self.gamma * torch.max(self.policy_model(next_states), dim=1, keepdim=True)[0] * (1 - dones)

        # Q-value predictions
        q_values = self.policy_model(states).gather(1, actions)
        
        # Compute loss
        loss = self.loss_fn(q_values, q_targets)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)