import numpy as np
import random

# Environment setup (dummy example, replace with actual environment logic)
class DummyEnv:
    def __init__(self):
        self.state_space = [(50, 110, 30), (60, 120, 40), (70, 130, 50)]  # Example states
        self.action_space = [0, 1, 2]  # Example actions
        self.current_state = self.state_space[0]

    def reset(self):
        # Reset the environment to the initial state
        self.current_state = self.state_space[0]
        return self.current_state

    def step(self, action):
        # Dummy logic for environment step
        # Here, `action` is just a random choice for simplicity
        next_state = random.choice(self.state_space)
        reward = -abs(action - 1)  # Example reward function (modify as needed)
        done = random.choice([True, False])  # End condition
        return next_state, reward, done

# Q-learning parameters
num_episodes = 1000
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.99
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
max_steps_per_episode = 100  # Max steps per episode

# Define action and state space
action_space = [0, 1, 2]  # Actions, modify as needed
state_space = [(50, 110, 30), (60, 120, 40), (70, 130, 50)]  # States, modify as needed

# Initialize Q-table with zeros
q_table = np.zeros((len(state_space), len(action_space)))

# Create the environment (replace with actual environment)
env = DummyEnv()

# Track rewards for analysis
episode_rewards = []

# Helper function to get index of state in state space
def get_state_index(state):
    return state_space.index(state)

# Q-learning training loop
for episode in range(num_episodes):
    state = env.reset()  # Reset the environment
    state_idx = get_state_index(state)
    done = False
    steps = 0
    total_reward = 0  # Initialize total reward for the episode

    while not done and steps < max_steps_per_episode:
        # Exploration vs exploitation
        if np.random.rand() < epsilon:
            action = random.choice(action_space)  # Exploration
        else:
            action = action_space[np.argmax(q_table[state_idx])]  # Exploitation

        # Take action, observe reward and next state
        next_state, reward, done = env.step(action)
        next_state_idx = get_state_index(next_state)

        # Update Q-value
        q_table[state_idx, action] = q_table[state_idx, action] + alpha * (
            reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
        )

        # Update state
        state_idx = next_state_idx
        steps += 1
        total_reward += reward  # Track reward for the episode

    # Store total reward for this episode
    episode_rewards.append(total_reward)

    # Update epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Log progress every 10 episodes
    if episode % 10 == 0:
        print(f"Episode: {episode}/{num_episodes}, Epsilon: {epsilon}")

# Analyze rewards after training
print("Training complete!")

# Print average reward per 100 episodes
print("Average reward per 100 episodes:")
print([np.mean(episode_rewards[i:i+100]) for i in range(0, len(episode_rewards), 100)])

# Optionally, print final Q-table
print("Final Q-table:")
print(q_table)
