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
from ns.switch.switch import SimplePacketSwitch, FairPacketSwitch

from ns.scheduler.monitor import ServerMonitor
from ns.scheduler.wfq import WFQServer


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

switch = FairPacketSwitch(
    env,
    nports=8,
    port_rate=16134,  # in bits/second
    buffer_size=2,  # in packets
    debug=True,
)


source_rate = 4600

wfq_server = WFQServer(env, source_rate, [1, 2], debug=True)
monitor = ServerMonitor(
    env, wfq_server, partial(expovariate, 0.1), pkt_in_service_included=True
)
