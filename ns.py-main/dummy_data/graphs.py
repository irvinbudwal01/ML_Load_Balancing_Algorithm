import matplotlib.pyplot as plt

# Data for the bar graph
weights = ['.33/.33/.33', '0.42,0.24,0.33', '.39/.29/.32', '.36/.41/.23']
latency = [44.12, 48.82, 43.49, 42.76]
packets_dropped = [332, 333, 334, 339]
server_utilization = [1.23, 1.03, 1.17, 1.72]

# Create the bar graph
# plt.bar(categories, latency, color='skyblue')
# plt.bar(weights, packets_dropped, color='skyblue')
# plt.bar(weights, server_utilization, color='skyblue')

# Add labels and title
plt.xlabel('Weights')
# plt.ylabel('Latency (ms)')
# plt.title('Average Latencies')

# plt.ylabel('Packets Dropped')

# plt.title('Packets Dropped')

# plt.ylabel('Server Utilization (%)')

# plt.title("Server Utilization")

# Show the graph
plt.show()