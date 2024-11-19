import pandas as pd
import numpy as np

# Number of samples (data points) and servers
num_samples = 100
num_servers = 3

# Simulate random data for three servers

# Latency: Random values between 20ms and 100ms (representing varying network conditions)
avg_latency = np.random.uniform(20, 100, num_samples)

# Packets Dropped: Random values between 0 and 50 (representing random packet drops)
avg_packets_dropped = np.random.uniform(0, 50, num_samples)

# Server Utilization: Random values between 0.5 and 1 (representing varying server load)
avg_server_utilization = np.random.uniform(0.5, 1.0, num_samples)

# Create DataFrame
data = {
    'server': np.tile([f'server_{i+1}' for i in range(num_servers)], num_samples),
    'avg_latency': np.concatenate([avg_latency for _ in range(num_servers)]),
    'avg_packets_dropped': np.concatenate([avg_packets_dropped for _ in range(num_servers)]),
    'avg_server_utilization': np.concatenate([avg_server_utilization for _ in range(num_servers)]),
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('network_data.csv', index=False)

print(df.head())
