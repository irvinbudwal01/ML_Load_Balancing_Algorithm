import simpy
import random

from random import expovariate
from functools import partial


from ns.flow.cc import TCPReno
from ns.flow.cubic import TCPCubic
from ns.flow.flow import AppType, Flow
from ns.packet.tcp_generator import TCPPacketGenerator
from ns.packet.tcp_sink import TCPSink
from ns.port.wire import Wire
from ns.switch.switch import SimplePacketSwitch
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


sender = TCPPacketGenerator(
    env, flow=flow, cc=TCPReno(), element_id=flow.src, debug=True
)

sender2 = TCPPacketGenerator(
    env, flow=flow2, cc=TCPReno(), element_id=flow2.src, debug=True
)

sender3 = TCPPacketGenerator(
    env, flow=flow3, cc=TCPReno(), element_id=flow3.src, debug=True
)

sender4 = TCPPacketGenerator(
    env, flow=flow4, cc=TCPReno(), element_id=flow4.src, debug=True
)

sender5 = TCPPacketGenerator(
    env, flow=flow5, cc=TCPReno(), element_id=flow5.src, debug=True
)

wire1_downstream = Wire(env, delay_dist)
wire1_upstream = Wire(env, delay_dist)
wire2_downstream = Wire(env, delay_dist)
wire2_upstream = Wire(env, delay_dist)
wire3_downstream = Wire(env, delay_dist)
wire3_upstream = Wire(env, delay_dist)
wire4_downstream = Wire(env, delay_dist)
wire4_upstream = Wire(env, delay_dist)
wire5_downstream = Wire(env, delay_dist)
wire5_upstream = Wire(env, delay_dist)
wire6_downstream = Wire(env, delay_dist)
wire6_upstream = Wire(env, delay_dist)
wire7_downstream = Wire(env, delay_dist)
wire7_upstream = Wire(env, delay_dist)
wire8_downstream = Wire(env, delay_dist)
wire8_upstream = Wire(env, delay_dist)

switch = SimplePacketSwitch(
    env,
    nports=8,
    port_rate=16134,  # in bits/second
    buffer_size=2,  # in packets
    debug=True,
)


receiver = TCPSink(env, rec_waits=True, debug=True)


receiver2 = TCPSink(env, rec_waits=True, debug=True)


receiver3 = TCPSink(env, rec_waits=True, debug=True)

switch.weights = [0.1, 0.2, 0.3, 0.1, 0.1, 0.05, 0.05, 0.1]

#normalize weights
switch.weights = [w / sum(switch.weights) for w in switch.weights]

switch.demux = RandomDemux(switch.ports, switch.weights)

switch.demux = FlowDemux([wire6_downstream, wire7_downstream, wire8_downstream])

print(switch.demux.outs)

switch.demux.outs[0] = wire6_downstream #going to sinks down
switch.demux.outs[1] = wire7_downstream
switch.demux.outs[2] = wire8_downstream

# switch.demux.outs[3] = wire1_upstream #going to generators up
# switch.demux.outs[4] = wire2_upstream
# switch.demux.outs[5] = wire3_upstream
# switch.demux.outs[6] = wire4_upstream
# switch.demux.outs[7] = wire5_upstream

wire6_downstream.out = receiver #connect switch to sinks down
wire7_downstream.out = receiver2
wire8_downstream.out = receiver3

receiver.out = wire6_upstream #connect sinks to switch up
receiver2.out = wire7_upstream
receiver3.out = wire8_upstream

wire6_upstream.out = switch #connect sinks to switch up
wire7_upstream.out = switch
wire8_upstream.out = switch

wire1_upstream.out = sender #connect to generators up
wire2_upstream.out = sender2
wire3_upstream.out = sender3
wire4_upstream.out = sender4
wire5_upstream.out = sender5

env.run(until=100) #*******************************TO FIGURE OUT: IT SEEMS FLOW DEMUX LETS YOU DECIDE WHICH OUTPUTS GO TO WHICH SINK. RANDOM DEMUX RANDOMIZES AMONG ALL PORTS?
                   #IT NEEDS TO ONLY RANDOMIZE ON OUTPUTS THAT GO TO SINKS. TO SIMPLIFY THIS PROCESS, CONVERT NETWORK BACK TO UDP INSTEAD OF TCP. FIND SOLUTION FIRST, THEN SWITCH
                   #BACK TO TCP IF POSSIBLE.

"""
A basic example that connects two packet generators to a network wire with
a propagation delay distribution, and then to a packet sink.
"""

from functools import partial
import random
from random import expovariate

import simpy
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.sink import PacketSink
from ns.port.wire import Wire


def arrival_1():
    """Packets arrive with a constant interval of 1.5 seconds."""
    return 1.5


def arrival_2():
    """Packets arrive with a constant interval of 2.0 seconds."""
    return 2.0


def delay_dist():
    return 0.1


def packet_size():
    return int(expovariate(0.01))


env = simpy.Environment()

ps = PacketSink(env, rec_flow_ids=False, debug=True)

pg1 = DistPacketGenerator(env, "flow_1", arrival_1, packet_size, flow_id=0)
pg2 = DistPacketGenerator(env, "flow_2", arrival_2, packet_size, flow_id=1)

wire1 = Wire(env, partial(random.gauss, 0.1, 0.02), wire_id=1, debug=False)
wire2 = Wire(env, delay_dist, wire_id=2, debug=False)

pg1.out = wire1
pg2.out = wire2
wire1.out = ps
wire2.out = ps

env.run(until=100)

print(
    "Flow 1 packet delays: "
    + ", ".join(["{:.2f}".format(x) for x in ps.waits["flow_1"]])
)
print(
    "Flow 2 packet delays: "
    + ", ".join(["{:.2f}".format(x) for x in ps.waits["flow_2"]])
)

print(
    "Packet arrival times in flow 1: "
    + ", ".join(["{:.2f}".format(x) for x in ps.arrivals["flow_1"]])
)

print(
    "Packet arrival times in flow 2: "
    + ", ".join(["{:.2f}".format(x) for x in ps.arrivals["flow_2"]])
)
