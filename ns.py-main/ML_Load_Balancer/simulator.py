import simpy
import random

from random import expovariate
from functools import partial


from ns.flow.cc import TCPReno
from ns.flow.cubic import TCPCubic
from ns.flow.flow import AppType, Flow
from ns.packet.tcp_generator import TCPPacketGenerator
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.tcp_sink import TCPSink
from ns.packet.sink import PacketSink
from ns.port.wire import Wire
from ns.switch.switch import SimplePacketSwitch
#from ns.switch.switch import RandomSwitch
from ns.demux.random_demux import RandomDemux
from ns.demux.flow_demux import FlowDemux


def packet_arrival():
    """Packets arrive with a random interval between 0 and 1 seconds."""
    return random.uniform(0,1)


def packet_size():
    """The packets have a constant size of 1024 bytes."""
    return 512


def delay_dist():
    """Network wires experience a constant propagation delay of 0.1 seconds."""
    return 0.1


env = simpy.Environment()

flow = Flow(
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


sender = DistPacketGenerator(env, "flow_1", packet_arrival, packet_size, flow_id=0)

sender2 = DistPacketGenerator(env, "flow_2", packet_arrival, packet_size, flow_id=1)

sender3 = DistPacketGenerator(env, "flow_3", packet_arrival, packet_size, flow_id=2)

sender4 = DistPacketGenerator(env, "flow_4", packet_arrival, packet_size, flow_id=3)

sender5 = DistPacketGenerator(env, "flow_5", packet_arrival, packet_size, flow_id=4)

wire1_downstream = Wire(env, delay_dist)
wire2_downstream = Wire(env, delay_dist)
wire3_downstream = Wire(env, delay_dist)
wire4_downstream = Wire(env, delay_dist)
wire5_downstream = Wire(env, delay_dist)
wire6_downstream = Wire(env, delay_dist)
wire7_downstream = Wire(env, delay_dist)
wire8_downstream = Wire(env, delay_dist)


receiver = PacketSink(env, rec_waits=True, debug=True)

receiver2 = PacketSink(env, rec_waits=True, debug=True)

receiver3 = PacketSink(env, rec_waits=True, debug=True)

weights = [.25, .5, .25]
#normalize weights
weights = [w / sum(weights) for w in weights]

randomMux = RandomDemux(env, weights)

sender.out = wire1_downstream
sender2.out = wire2_downstream
sender3.out = wire3_downstream
sender4.out = wire4_downstream
sender5.out = wire5_downstream

wire1_downstream.out = randomMux
wire2_downstream.out = randomMux
wire3_downstream.out = randomMux
wire4_downstream.out = randomMux
wire5_downstream.out = randomMux

randomMux.outs[0] = wire6_downstream #going to sinks down
randomMux.outs[1] = wire7_downstream
randomMux.outs[2] = wire8_downstream


wire6_downstream.out = receiver #connect switch to sinks down
wire7_downstream.out = receiver2
wire8_downstream.out = receiver3

#receiver.out = wire6_upstream #connect sinks to switch up
#receiver2.out = wire7_upstream
#receiver3.out = wire8_upstream

#wire6_upstream.out = switch #connect sinks to switch up
#wire7_upstream.out = switch
#wire8_upstream.out = switch

#wire1_upstream.out = sender #connect to generators up
#wire2_upstream.out = sender2
#wire3_upstream.out = sender3
#wire4_upstream.out = sender4
#wire5_upstream.out = sender5

env.run(until=100) #*******************************TO FIGURE OUT: IT SEEMS FLOW DEMUX LETS YOU DECIDE WHICH OUTPUTS GO TO WHICH SINK. RANDOM DEMUX RANDOMIZES AMONG ALL PORTS?
                   #IT NEEDS TO ONLY RANDOMIZE ON OUTPUTS THAT GO TO SINKS. TO SIMPLIFY THIS PROCESS, CONVERT NETWORK BACK TO UDP INSTEAD OF TCP. FIND SOLUTION FIRST, THEN SWITCH
                   #BACK TO TCP IF POSSIBLE. ALSO, SWITCH CONTAINS WEIGHTS ARRAY.

# print(
#     "Receiver 1 packet delays: "
#     + ", ".join(["{:.2f}".format(x) for x in receiver.waits[0]])
# )
# print(
#     "Receiver 2 packet delays: "
#     + ", ".join(["{:.2f}".format(x) for x in receiver2.waits[1]])
# )

# print(
#     "Receiver 3 packet delays: "
#     + ", ".join(["{:.2f}".format(x) for x in receiver3.waits[2]])
# )

print(
    "Receiver 1 packet delays: " + ", ".join(["{:.2f}".format(x) for x in receiver.waits[0]])
)

print(
    "Receiver 2 packet delays: "
    + ", ".join(["{:.2f}".format(x) for x in receiver2.waits[1]])
)

print(
    "Receiver 3 packet delays: "
    + ", ".join(["{:.2f}".format(x) for x in receiver3.waits[2]])
)

print("packets dropped")
print("Received:" ,receiver.packets_received[0] + receiver2.packets_received[0] + receiver3.packets_received[0])
print("Sent:" , sender.packets_sent + sender2.packets_sent + sender3.packets_sent + sender4.packets_sent + sender5.packets_sent)

print("server utilization") #server 1 takes 50, server 2 takes 30, server 3 takes 40
print(receiver.packets_received[0] / 50, receiver2.packets_received[0] / 30, receiver3.packets_received[0] / 40)