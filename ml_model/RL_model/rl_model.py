import numpy as np
import pandas as pd
import random

# Environment setup using data from the CSV
class CSVEnv:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)  # Load the CSV file
        self.state_columns = ['packets_dropped', 'latency', 'network_utilization']  # State columns from CSV
        self.state_space = self.df[self.state_columns].values.tolist()  # List of state values
        self.action_space = [0, 1, 2]  # Actions: Select one of 3 servers (0, 1, 2)
        self.current_state_idx = 0  # Start at the first state (index)
        
    def reset(self):
        # Reset the environment to the first state
        self.current_state_idx = 0
        return self.state_space[self.current_state_idx]

    def step(self, action):
        # Find the next state based on the action (assuming the action results in a transition)
        next_state_idx = (self.current_state_idx + 1) % len(self.state_space)  # Loop back to 0 if end
        next_state = self.state_space[next_state_idx]
        
        # Define reward logic here for load balancing
        reward = self.compute_reward(action, next_state)  # Get the reward based on current action and state
        
        done = (next_state_idx == len(self.state_space) - 1)  # End when we reach the last state
        self.current_state_idx = next_state_idx  # Update the current state index
        return next_state, reward, done
    
    def compute_reward(self, action, state):
        # The reward function should be designed to encourage good load balancing
        packets_dropped, latency, network_utilization = state
        
        # Reward based on how the action (server choice) impacts performance
        # Example: Penalize high latency, dropped packets, or over-utilized servers
        
        penalty = 0
        if action == 0:
            # Server 0 specific penalty (this could depend on the current utilization of server 0)
            penalty = latency * 0.1 + packets_dropped * 0.2
        elif action == 1:
            # Server 1 specific penalty (this could depend on the current utilization of server 1)
            penalty = latency * 0.15 + packets_dropped * 0.1
        elif action == 2:
            # Server 2 specific penalty (this could depend on the current utilization of server 2)
            penalty = latency * 0.2 + packets_dropped * 0.3
        
        # A lower penalty is better (negative reward)
        reward = -penalty
        return reward

# Q-learning parameters
num_episodes = 1000
epsilon = 1.0  # Exploration rate
epsilon_min = 0.0001
epsilon_decay = 0.99
alpha = 0.7  # Learning rate
gamma = 0.9  # Discount factor
max_steps_per_episode = 100  # Max steps per episode

# Read CSV file (modify the path to your actual CSV)
csv_file = 'network_data.csv'  # Modify with the correct path
env = CSVEnv(csv_file)

# Initialize Q-table with zeros
q_table = np.zeros((len(env.state_space), len(env.action_space)))

# Helper function to get state index (just for indexing)
def get_state_index(state):
    return env.state_space.index(state)

# Q-learning training loop
episode_rewards = []  # Track rewards for analysis

for episode in range(num_episodes):
    state = env.reset()  # Reset the environment
    state_idx = get_state_index(state)
    done = False
    steps = 0
    total_reward = 0  # Initialize total reward for the episode

    while not done and steps < max_steps_per_episode:
        # Exploration vs exploitation
        if np.random.rand() < epsilon:
            action = random.choice(env.action_space)  # Exploration: Random action (server choice)
        else:
            action = env.action_space[np.argmax(q_table[state_idx])]  # Exploitation: Best action based on Q-table

        # Take action, observe reward and next state
        next_state, reward, done = env.step(action)
        next_state_idx = get_state_index(next_state)

        # Update Q-value using the Q-learning formula
        q_table[state_idx, env.action_space.index(action)] = q_table[state_idx, env.action_space.index(action)] + alpha * (
            reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, env.action_space.index(action)]
        )

        # Update state
        state_idx = next_state_idx
        steps += 1
        total_reward += reward  # Track reward for the episode

    # Store total reward for this episode
    episode_rewards.append(total_reward)

    # Print Q-table after each episode (optional, might be verbose for 1000 episodes)
    print(f"Episode {episode}: Q-table:")
    print(q_table)

    # Update epsilon (decay exploration rate)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Log progress every 10 episodes
    if episode % 10 == 0:
        print(f"Episode: {episode}/{num_episodes}, Epsilon: {epsilon}")

print("Training complete!")

# Print final Q-table after training
print("Final Q-table:")
print(q_table)

