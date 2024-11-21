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
#wire1_upstream = Wire(env, delay_dist)
wire2_downstream = Wire(env, delay_dist)
#wire2_upstream = Wire(env, delay_dist)
wire3_downstream = Wire(env, delay_dist)
#wire3_upstream = Wire(env, delay_dist)
wire4_downstream = Wire(env, delay_dist)
#wire4_upstream = Wire(env, delay_dist)
wire5_downstream = Wire(env, delay_dist)
#wire5_upstream = Wire(env, delay_dist)
wire6_downstream = Wire(env, delay_dist)
#wire6_upstream = Wire(env, delay_dist)
wire7_downstream = Wire(env, delay_dist)
#wire7_upstream = Wire(env, delay_dist)
wire8_downstream = Wire(env, delay_dist)
#wire8_upstream = Wire(env, delay_dist)

switch = SimplePacketSwitch(
    env,
    nports=3,
    port_rate=16134,  # in bits/second
    buffer_size=2,  # in packets
    debug=True,
)


receiver = PacketSink(env, rec_waits=True, debug=True)

receiver2 = PacketSink(env, rec_waits=True, debug=True)

receiver3 = PacketSink(env, rec_waits=True, debug=True)

#switch.weights = [0.1, 0.2, 0.3, 0.1, 0.1, 0.05, 0.05, 0.1]
switch.weights = [.25, .5, .25]
#normalize weights
switch.weights = [w / sum(switch.weights) for w in switch.weights]

switch.demux = RandomDemux(switch.ports, switch.weights)

#switch.demux = FlowDemux([wire6_downstream, wire7_downstream, wire8_downstream])

print(switch.demux.outs)

switch.demux.outs[0].out = wire6_downstream #going to sinks down
switch.demux.outs[1].out = wire7_downstream
switch.demux.outs[2].out = wire8_downstream

print(switch.demux.outs)

# switch.demux.outs[3] = wire1_upstream #going to generators up
# switch.demux.outs[4] = wire2_upstream
# switch.demux.outs[5] = wire3_upstream
# switch.demux.outs[6] = wire4_upstream
# switch.demux.outs[7] = wire5_upstream

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

print(
    "Receiver 1 packet delays: "
    + ", ".join(["{:.2f}".format(x) for x in receiver.waits[0]])
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
print(receiver.packets_received[0] + receiver2.packets_received[0] + receiver3.packets_received[0])
print(sender.packets_sent + sender2.packets_sent + sender3.packets_sent + sender4.packets_sent)

print("server utilization") #server 1 takes 50, server 2 takes 30, server 3 takes 40
print(receiver.packets_received[0] / 50, receiver2.packets_received[0] / 30, receiver3.packets_received[0] / 40)