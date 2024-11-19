from environment import DataCenterEnvironment
from agent import DQNAgent

# Initialize environment and agent
env = DataCenterEnvironment('network_data.csv')
state_size = env.states.shape[1]  # Number of features: 3 (packets_dropped, latency, network_utilization)
action_size = 3  # Number of possible actions (e.g., routing to 3 different servers)
agent = DQNAgent(state_size, action_size)

num_episodes = 50  # Number of training episodes

for episode in range(num_episodes):
    state = env.reset()  # Get the initial state
    total_reward = 0

    while True:
        # Select an action based on the current state
        action = agent.select_action(state)

        # Take the action in the environment
        next_state, reward, done, _ = env.step(action)

        # Store the experience in the agent's replay buffer
        if next_state is not None:  # Skip storing if there's no next state
            agent.store_experience(state, action, reward, next_state, done)
        
        # Train the agent
        agent.train()

        # Update state and reward
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")