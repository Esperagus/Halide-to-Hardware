import sys
import os
import argparse
import re
from .netlist_util import parse_and_pack_netlist, port_rename

from pycyclone.io import load_placement
# parse raw routing result
from canal.pnr_io import __parse_raw_routing_result
from typing import Dict, List, Tuple, Set

PE_DELAY = 500
MEM_DELAY = 0
PE_SB = 210
MEM_SB = 300
RMUX_DELAY = 18
GLB_DELAY = 1300

class PathComponents:
    def __init__(self, glbs, hops, pes, mems):
        self.glbs = glbs
        self.hops = hops
        self.pes = pes
        self.mems = mems

class Node:
    def __init__(self, blk_id: str):
        self.next: Dict[str, List[Tuple["Node", str]]] = {}
        self.parent: Dict[str, Node] = {}
        self.blk_id = blk_id

    def add_next(self, src_port: str, sink_port, node: "Node"):
        if src_port not in self.next:
            self.next[src_port] = []
        self.next[src_port].append((node, sink_port))
        node.parent[sink_port] = self

    def __repr__(self):
        return self.blk_id


class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def get_node(self, blk_id: str):
        if blk_id not in self.nodes:
            node = Node(blk_id)
            self.nodes[blk_id] = node
        return self.nodes[blk_id]

    def sort(self):
        visited = set()
        stack: List[Node] = []
        for n in self.nodes.values():
            if n.blk_id not in visited:
                self.__sort(n, stack, visited)
        return stack[::-1]

    @staticmethod
    def __sort(node: Node, stack, visited: Set[str]):
        visited.add(node.blk_id)
        for ns in node.next.values():
            for n, _ in ns:
                Graph.__sort(n, stack, visited)
        stack.append(node)


def construct_graph(netlist):
    g = Graph()
    for net in netlist.values():
        src_id, src_port = net[0]
        src_node = g.get_node(src_id)
        for sink_id, sink_port in net[1:]:
            sink_node = g.get_node(sink_id)
            src_node.add_next(src_port, sink_port, sink_node)
    return g


def get_net_id(sink_pin, netlist):
    for net_id, net in netlist.items():
        if sink_pin == net[0]:
            return net_id
    return None


def compute_delay(net, routing, placement):
    node_timing = {}
    sink_pin_timing = {}
    for segment in routing:
        segment = [tuple(e) for e in segment]
        if segment[0] not in node_timing:
            node_timing[segment[0]] = 0
        t = node_timing[segment[0]]
        for node_str in segment[1:]:
            if node_str[0] == "SB":
                # we only count output of the SB
                track, x, y, side, io_, bit_width = node_str[1:]
                if io_ == 1:
                    # PE or MEM
                    if x % 4 == 3:
                        t += MEM_SB
                    else:
                        t += PE_SB
            elif node_str[0] == "RMUX":
                t += RMUX_DELAY
        # the end of the segment is the sink pinprint(segment[0])
        node_str = segment[-1]
        # it's either a register or a port
        if node_str[0] == "PORT":
            port_name, x, y, bit_width = node_str[1:]
            sink_pin_timing[(port_name, x, y)] = t
        else:
            assert node_str[0] == "REG"
            reg_name, track, x, y, bit_width = node_str[1:]
            # we use "reg" here
            sink_pin_timing[("reg", x, y)] = t
    # compute the actual pin delay using net and placement
    result = {}
    for blk_id, port in net[1:]:
        x, y = placement[blk_id]
        t = sink_pin_timing[(port, x, y)]
        result[(blk_id, port)] = t
    return result


def get_sink_nets(netlist, blk_id):
    result = {}
    for net_id, net in netlist.items():
        if net[0][0] == blk_id:
            result[net_id] = net
    return result

def load_netlist(netlist_file):
    f = open(netlist_file, "r")
    lines = f.readlines()
 
    netlist = {}
    netlist_read = False

    for line in lines:
        if "Netlists:" in line:
            netlist_read = True
        elif "ID to Names:" in line:
            netlist_read = False
        elif netlist_read:
            if len(line.split(":")) > 1:
                edge_id = line.split(":")[0]
                connections = line.split(":")[1]

                connections = re.findall(r'\b\S+\b', connections)

                netlist[edge_id] = []
                for conn1, conn2 in zip(connections[::2], connections[1::2]):
                    netlist[edge_id].append((conn1, conn2))
    return netlist



def parse_args():
    parser = argparse.ArgumentParser("CGRA Retiming tool")
    parser.add_argument("-a", "--app", "-d", required=True, dest="application", type=str, help="Application directory")
    parser.add_argument("-f", "--min-frequency", default=200, dest="frequency", type=int,
                        help="Minimum frequency in MHz")
    args = parser.parse_args()
    # check filenames
    assert 1000 > args.frequency > 0, "Frequency must be less than 1GHz"
    dirname = os.path.join(args.application, "bin")
    netlist = os.path.join(dirname, "design.packed")
    assert os.path.exists(netlist), netlist + " does not exist"
    placement = os.path.join(dirname, "design.place")
    assert os.path.exists(placement), placement + " does not exists"
    route = os.path.join(dirname, "design.route")
    assert os.path.exists(route), route + " does not exists"
    folded = os.path.join(dirname, "design.folded")
    assert os.path.exists(folded), folded + " does not exists"
    # need to load routing files as well
    # for now we just assume RMUX exists
    return netlist, placement, route, folded


def main():
    netlist_file, placement_file, routing_file, folded_file = parse_args()

    # netlist, folded_blocks, id_to_name, changed_pe = \
    #     parse_and_pack_netlist(netlist_file, fold_reg=True)

    netlist = load_netlist(netlist_file)

    f = open(folded_file, "r")
    lines = f.readlines()


    pe_reg = set()
 
    for line in lines:
        entry = re.findall(r'\b\S+\b', line.split(":")[1])
        blk_id = entry[0]
        port = entry[-1]
        if blk_id[0] == 'p':
            pe_reg.add((blk_id, port))

        

    # breakpoint()
    # any block that folds to the PE needs to be 0
    # for entry in folded_blocks.values():
    #     blk_id = entry[0]
    #     port = entry[-1]
    #     if blk_id[0] == 'p':
    #         pe_reg.add((blk_id, port))

    # parse the placement result
    placement = load_placement(placement_file)
    route = __parse_raw_routing_result(routing_file)

    # need to parse the route
    # we need to do simple STA analysis
    graph = construct_graph(netlist)
    nodes = graph.sort()
    timing = {}
    for node in nodes:
        blk_id = node.blk_id
        # the max delay to this point
        ts = [0]
        components = []
        comp = PathComponents(0, 0, 0, 0)

        for src_port, parent in node.parent.items():

            if parent.blk_id not in timing:
                t = 0
            elif (parent.blk_id, src_port) in pe_reg:
                t = 0
            elif parent.blk_id[0] == 'r' or parent.blk_id[0] == 'm':
                t = 0  
            else:
                t = timing[parent.blk_id]
            
            if parent.blk_id[0] == 'p':
                t += PE_DELAY
                comp['pes'] += 1
            elif parent.blk_id[0] == 'm':
                t += MEM_DELAY
                comp['mems'] += 1
            elif parent.blk_id[0] == 'I' or parent.blk_id[0] == 'i':
                t += GLB_DELAY
                comp['glbs'] += 1
            
            nets = get_sink_nets(netlist, parent.blk_id)
            for net_id, net in nets.items():
                delays = compute_delay(net, route[net_id], placement)
                if (node.blk_id, src_port) in delays:
                    t += delays[(node.blk_id, src_port)]
                    comp['hops'] += delays
                    break
                
            ts.append(t)
            components.append(comp)
        timing[blk_id] = max(ts)
        #f = lambda i: components[i]
        #argmax = max(range(len(components)), key=f)

    max_delay = max(timing.values())
    max_nodes = {v:k for k,v in timing.items()}
    max_node = max_nodes[max_delay]
    clock_speed = 1.0e12 / max_delay / 1e6
    print("Maximum clock frequency:", clock_speed, "MHz")
    print("Critical Path:", max_delay, "ns")

    curr_node = graph.nodes[max_node]
    print(curr_node, timing[curr_node.blk_id])
    while(True):
        max_t = 0
        max_node = None
        for src_port, parent in curr_node.parent.items():
            if timing[parent.blk_id] > max_t:
                max_t = timing[parent.blk_id]
                max_node = parent
        if max_node == None:
            break
        curr_node = max_node
        print(curr_node, timing[curr_node.blk_id])

    print(pe_reg)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    main()



