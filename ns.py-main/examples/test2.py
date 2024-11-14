"""
A basic example that showcases how TCP can be used to generate packets, and how a TCP sink
can send acknowledgment packets back to the sender in a simple two-hop network.
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

sender = TCPPacketGenerator(
    env, flow=flow, cc=TCPReno(), element_id=flow.src, debug=True
)

sender2 = TCPPacketGenerator(
    env, flow=flow2, cc=TCPReno(), element_id=flow2.src, debug=True
)

wire1_downstream = Wire(env, delay_dist)
wire1_upstream = Wire(env, delay_dist)
wire2_downstream = Wire(env, delay_dist)
wire2_upstream = Wire(env, delay_dist)
wire3_downstream = Wire(env, delay_dist)
wire3_upstream = Wire(env, delay_dist)

switch = SimplePacketSwitch(
    env,
    nports=4,
    port_rate=16384,  # in bits/second
    buffer_size=2,  # in packets
    debug=True,
)

receiver = TCPSink(env, rec_waits=True, debug=True)

sender.out = wire1_downstream #connect generators into switch
wire1_downstream.out = switch
sender2.out = wire3_downstream
wire3_downstream.out = switch

fib = {0: 0, 1: 1, 10000: 2, 10001: 3} #configure switch
switch.demux.fib = fib
switch.demux.outs[0].out = wire2_downstream
switch.demux.outs[1].out = wire2_downstream
switch.demux.outs[2].out = wire1_upstream
switch.demux.outs[3].out = wire3_upstream

wire2_downstream.out = receiver
receiver.out = wire2_upstream
wire2_upstream.out = switch

wire1_upstream.out = sender
wire3_upstream.out = sender2

env.run(until=100)