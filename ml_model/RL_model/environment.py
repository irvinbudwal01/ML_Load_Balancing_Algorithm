import pandas as pd

class DataCenterEnvironment:
    def __init__(self, data_file):
        # Load the CSV data
        self.data = pd.read_csv(data_file)
        self.max_steps = len(self.data)  # Total number of rows in the CSV
        self.current_step = 0  # Start at the first row
        
        # Extract relevant columns
        self.states = self.data[['packets_dropped', 'latency', 'network_utilization']].values

    def reset(self):
        self.current_step = 0
        return self.states[self.current_step]

    def step(self, action):
        """
        Perform one step in the environment.
        """
        # Get the current state
        state = self.states[self.current_step]

        # Reward calculation
        packets_dropped, latency, network_utilization = state
        reward = -packets_dropped - latency + network_utilization

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Check if we're at the end of the data

        if not done:
            next_state = self.states[self.current_step]
        else:
            next_state = None  # No next state if done

        return next_state, reward, done