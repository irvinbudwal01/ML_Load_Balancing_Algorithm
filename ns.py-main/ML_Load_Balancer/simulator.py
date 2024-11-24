import simpy
import random

from random import expovariate
from functools import partial

from ns.flow.flow import Flow

from ns.packet.dist_generator import DistPacketGenerator

from ns.packet.sink import PacketSink
from ns.port.wire import Wire

from ns.demux.random_demux import RandomDemux

from ns.port.port import Port

import torch
import torch.optim as optim
import torch.nn as nn
from model import NetworkOptimizer
import pandas as pd
import csv

def ml_model():

    # Load and preprocess data
    data = pd.read_csv('../dummy_data/network_data.csv')
    data.columns = data.columns.str.strip()  # Clean column names

    # Extract input features (avg_latency, avg_packets_dropped, avg_server_utilization)
    input_features = ['avg_latency', 'total_packets_dropped', 'server_utilization']
    sequence_data = torch.tensor(data[input_features].values, dtype=torch.float32).unsqueeze(0)  # Shape: (1, num_servers, 3)

    # Define Model
    input_size = len(input_features)  # Number of features per server
    hidden_size = 64  # Hidden size for the LSTM
    output_size = 3  # 3 servers, so we output 3 traffic weights
    model = NetworkOptimizer(input_size, hidden_size, output_size)

    # Define Loss and Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    def reward_function(weights, latency, packets_dropped, server_utilization):
        # Normalize weights to ensure they sum to 1
        weights = weights / weights.sum()

        # Stronger penalty for any weight greater than 60% or less than 10%
        imbalance_penalty = 200 * (max(weights[0], 1 - weights[0]) + max(weights[1], 1 - weights[1]) + max(weights[2], 1 - weights[2]))

        # Performance reward (negative for latency and dropped packets, positive for utilization)
        performance_reward = -latency.sum() - packets_dropped.sum() + server_utilization.sum()

        # Total reward: balance between performance and balanced traffic distribution
        total_reward = performance_reward - imbalance_penalty
        return total_reward


    print("-------------------------------------")
    print("RL Model")
    print("-------------------------------------")
    # Training loop update
    epochs = 100
    for epoch in range(epochs):
        model.train()  # Set model to training mode

        # Forward pass
        output = model(sequence_data)  # Predict weights for servers

        # Ensure output tensor requires gradients
        output = output.squeeze().requires_grad_()  # Squeeze and ensure requires_grad=True

        # Sample server performance data for the current step (dummy data for now)
        latency = torch.tensor([50, 40, 30], dtype=torch.float32, requires_grad=True)  # Example latency values
        packets_dropped = torch.tensor([5, 3, 2], dtype=torch.float32, requires_grad=True)  # Example dropped packets
        server_utilization = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32, requires_grad=True)  # Example server utilization

        # Calculate the reward based on the model's output and performance data
        reward = reward_function(output, latency, packets_dropped, server_utilization)

        # Backward pass and optimization
        optimizer.zero_grad()
        reward.backward()  # Use reward as the gradient
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Reward: {reward.item():.4f}")

    # Display final output after training
    model.eval()  # Set model to evaluation mode
    final_weights = model(sequence_data).detach().numpy().flatten()

    print("-------------------------------------")
    for i, weight in enumerate(final_weights):
        print(f"Server {i} Traffic Weight: {weight:.4f}")
    print("-------------------------------------")

    return final_weights




def packet_arrival():
    """Packets arrive with a random interval between 0 and 1 seconds."""
    return .4
    #return random.uniform(0,1)


def packet_size():
    """The packets have a constant size of 1024 bytes."""
    return 256


def delay_dist():
    """Network wires experience a constant propagation delay of 0.1 seconds."""
    return 0.1
    #return random.uniform(0,.2)

def delay_dist1():
    """Network wires experience a constant propagation delay of 0.2 seconds."""
    return 0.2
    #return random.uniform(0,.2)

def delay_dist2():
    """Network wires experience a constant propagation delay of 0.6 seconds."""
    return 0.6
    #return random.uniform(0,.6)

def delay_dist3():
    """Network wires experience a constant propagation delay of 0.5 seconds."""
    return 0.5
    #return random.uniform(0,.5)

i = 0
while(i < 80):

    final_weights = ml_model() #receive output

    env = simpy.Environment()

    flow = Flow( #define flows for each sender
        fid=0,
        src="flow 1",
        dst="flow 1",
        finish_time=10,
        arrival_dist=packet_arrival,
        size_dist=packet_size,
    )

    flow2 = Flow(
        fid=1,
        src="flow 2",
        dst="flow 2",
        finish_time=10,
        arrival_dist=packet_arrival,
        size_dist=packet_size,
    )

    flow3 = Flow(
        fid=2,
        src="flow 3",
        dst="flow 3",
        finish_time=10,
        arrival_dist=packet_arrival,
        size_dist=packet_size,
    )

    flow4 = Flow(
        fid=3,
        src="flow 4",
        dst="flow 4",
        finish_time=10,
        arrival_dist=packet_arrival,
        size_dist=packet_size,
    )

    flow5 = Flow(
        fid=4,
        src="flow 5",
        dst="flow 5",
        finish_time=10,
        arrival_dist=packet_arrival,
        size_dist=packet_size,
    )


    sender = DistPacketGenerator(env, "flow_1",packet_arrival, packet_size, flow_id=0, rec_flow=True) #UDP senders

    sender2 = DistPacketGenerator(env, "flow_2", packet_arrival, packet_size, flow_id=1, rec_flow=True)

    sender3 = DistPacketGenerator(env, "flow_3", packet_arrival, packet_size, flow_id=2, rec_flow=True)

    sender4 = DistPacketGenerator(env, "flow_4", packet_arrival, packet_size, flow_id=3, rec_flow=True)

    sender5 = DistPacketGenerator(env, "flow_5", packet_arrival, packet_size, flow_id=4, rec_flow=True)

    wire1_downstream = Wire(env, delay_dist) #wires for network
    wire2_downstream = Wire(env, delay_dist)
    wire3_downstream = Wire(env, delay_dist)
    wire4_downstream = Wire(env, delay_dist)
    wire5_downstream = Wire(env, delay_dist)
    wire6_downstream = Wire(env, delay_dist1)
    wire7_downstream = Wire(env, delay_dist2)
    wire8_downstream = Wire(env, delay_dist3)

    port1 = Port(env, rate=1200, qlimit=300) #ports to represent sink specs
    port2 = Port(env, rate=1000, qlimit=200)
    port3 = Port(env, rate=1100, qlimit=250)

    receiver = PacketSink(env, rec_waits=True, debug=False) #UDP sinks

    receiver2 = PacketSink(env, rec_waits=True, debug=False)

    receiver3 = PacketSink(env, rec_waits=True, debug=False)


    #normalize weights
    #weights = [w / sum(final_weights) for w in final_weights]

    randomMux = RandomDemux(env, final_weights) #random demux to receive weights for each sink

    sender.out = wire1_downstream #wired topology
    sender2.out = wire2_downstream
    sender3.out = wire3_downstream
    sender4.out = wire4_downstream
    sender5.out = wire5_downstream

    wire1_downstream.out = randomMux
    wire2_downstream.out = randomMux
    wire3_downstream.out = randomMux
    wire4_downstream.out = randomMux
    wire5_downstream.out = randomMux

    randomMux.outs[0] = wire6_downstream #going to sinks
    randomMux.outs[1] = wire7_downstream
    randomMux.outs[2] = wire8_downstream


    wire6_downstream.out = port1 #demux goes to ports -> sinks
    port1.out = receiver
    wire7_downstream.out = port2
    port2.out = receiver2
    wire8_downstream.out = port3
    port3.out = receiver3

    env.run(until=100)

    delay1 = "0"
    delay2 = "0"
    delay3 = "0"
    if len(receiver.waits[0]):
        delay1 = "{:.2f}".format(sum(receiver.waits[0])/len(receiver.waits[0]))

    if len(receiver2.waits[0]):
        delay2 = "{:.2f}".format(sum(receiver2.waits[0])/len(receiver2.waits[0]))

    if len(receiver3.waits[0]):
        delay3 = "{:.2f}".format(sum(receiver3.waits[0])/len(receiver3.waits[0]))

    print("Receiver 1 Average Packet Delays: ", delay1)

    print("Receiver 2 Average Packet Delays: ", delay2)

    print("Receiver 3 Average Packet Delays: ", delay3)

    packets_sent = sender.packets_sent + sender2.packets_sent + sender3.packets_sent + sender4.packets_sent + sender5.packets_sent
    print("Packets Dropped: ", port1.packets_dropped + port2.packets_dropped + port3.packets_dropped)

    print("server utilization")

    print("{:.2f}".format(port1.packets_received/ 500), 
        "{:.2f}".format(port2.packets_received/ 300), 
        "{:.2f}".format(port3.packets_received/ 450))
    
    utilization = [port1.packets_received/ 500, port2.packets_received/ 300, port3.packets_received/ 450]

    peak_util = max(utilization)
    ##data to add to .csv

    new_data = [["server_1", delay1, port1.packets_dropped, "{:.2f}".format(port1.packets_received/ 500)],
                ["server_2", delay2, port2.packets_dropped, "{:.2f}".format(port2.packets_received/ 300)],
                ["server_3", delay3, port3.packets_dropped, "{:.2f}".format(port3.packets_received/ 450)]]
    
    weight_data = [["Weights: ", final_weights[0],final_weights[1],final_weights[2]],
                   ["Stats: ", delay1 + " ", delay2 + " ", delay3 + " " ,port1.packets_dropped+port2.packets_dropped+port3.packets_dropped, " ",
                                                                          peak_util]]

    with open('../dummy_data/network_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        writer.writerows(new_data)
    
    with open('../dummy_data/stats.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        writer.writerows(weight_data)

    i += 1