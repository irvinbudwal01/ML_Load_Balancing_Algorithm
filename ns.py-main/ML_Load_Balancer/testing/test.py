"""
Proposed Topology: 5 TCP generators sending packets to 3 different TCP sinks. Sent through 
a simple switch using FIB as a demux.
"""

import simpy
import random

from ns.flow.cc import TCPReno
from ns.flow.cubic import TCPCubic
from ns.flow.flow import AppType, Flow
from ns.packet.tcp_generator import TCPPacketGenerator
from ns.packet.tcp_sink import TCPSink
from ns.port.wire import Wire
from ns.switch.switch import SimplePacketSwitch


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

sender.out = wire1_downstream #connect generators into switch down
wire1_downstream.out = switch
sender2.out = wire2_downstream
wire2_downstream.out = switch
sender3.out = wire3_downstream
wire3_downstream.out = switch
sender4.out = wire4_downstream
wire4_downstream.out = switch
sender5.out = wire5_downstream
wire5_downstream.out = switch

fib = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 10000: 3, 10001: 4, 10002: 5, 10003: 6, 10004: 7} #configure switch

switch.demux.fib = fib
switch.demux.outs[0].out = wire6_downstream #going to sinks down
switch.demux.outs[1].out = wire7_downstream
switch.demux.outs[2].out = wire8_downstream

switch.demux.outs[3].out = wire1_upstream #going to generators up
switch.demux.outs[4].out = wire2_upstream
switch.demux.outs[5].out = wire3_upstream
switch.demux.outs[6].out = wire4_upstream
switch.demux.outs[7].out = wire5_upstream

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

env.run(until=100)

# print("delays in receiver1")
# for x in receiver.waits[0]:
#     print(x)

# print("delays in receiver2")
# for x in receiver2.waits[1]:
#     print(x)

# print("delays in receiver3")
# for x in receiver3.waits[2]:
#     print(x)

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
# print(
#     "Flow 4 packet delays: "
#     + ", ".join(["{:.2f}".format(x) for x in receiver.waits["flow 1"]])
# )