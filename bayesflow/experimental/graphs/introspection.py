from typing import TYPE_CHECKING, TypeAlias

import networkx as nx
from numpy import invert

if TYPE_CHECKING:
    from .types import InvertedGraph


Node: TypeAlias = str
SimulationNode: TypeAlias = str
ExpandedNode: TypeAlias = str


# returns required conditions for each inference network
def network_conditions(inverted_graph: "InvertedGraph") -> dict[int, list[SimulationNode]]:
    composition = network_composition(inverted_graph)
    conditions = conditions_by_node(inverted_graph)
    networks: dict[int, list[SimulationNode]] = {}

    for network_idx, nodes in composition.items():
        networks[network_idx] = []
        for node in nodes:
            networks[network_idx].extend(conditions[node])

    return networks


# assigns nodes to be estimated by each inference network
def network_composition(inverted_graph: "InvertedGraph") -> dict[int, list[SimulationNode]]:
    conditions = conditions_by_node(inverted_graph)

    processed_nodes = set(k for k, v in conditions.items() if v == [])
    conditions = {k: v for k, v in conditions.items() if k not in processed_nodes}

    networks: dict[int, list[SimulationNode]] = {}
    network_idx = 0

    # Build inference layers iteratively: start with all nodes that require no conditions,
    # then repeatedly form the next layer by selecting nodes whose dependencies are entirely
    # covered by previous inference networks
    while conditions:
        networks[network_idx] = []
        next_nodeset = {k for k, v in conditions.items() if set(v).issubset(processed_nodes | set([k]))}

        if next_nodeset:
            processed_nodes.update(next_nodeset)

            for node in next_nodeset:
                _ = conditions.pop(node)
                networks[network_idx].extend([node])

        network_idx += 1

    for k, v in networks.items():
        networks[k] = list(set(v))

    return networks


def permutated_data_shape_order(inverted_graph: "InvertedGraph") -> list[SimulationNode]:
    shape_order = data_shape_order(inverted_graph)
    amortizable = [n for n in shape_order if allows_amortization(inverted_graph, n)]
    non_amortizable = [n for n in shape_order if not allows_amortization(inverted_graph, n)]

    # put non amortizable nodes at the end
    return amortizable + non_amortizable


# determines ordering of the data shape as defined by the user-defined simulation graph
def data_shape_order(inverted_graph: "InvertedGraph") -> list[SimulationNode]:
    # retrieve current ordering of data shape
    shape_order = []
    expanded_graph = inverted_graph.expanded_graph
    data_nodes = inverted_graph.simulation_graph.data_node()

    for node in expanded_graph.nodes:
        if data_nodes in expanded_graph.nodes[node]["previous_names"]:
            shape_order = expanded_graph.nodes[node]["split_by"]

    return shape_order


# returns a list of amortizable nodes
def amortizable_nodes(inverted_graph: "InvertedGraph") -> list[SimulationNode]:
    amortizable_nodes = []
    data_nodes = inverted_graph.simulation_graph.data_node()

    for node in inverted_graph.simulation_graph.nodes:
        if node not in data_nodes and allows_amortization(inverted_graph, node):
            amortizable_nodes.append(node)

    return amortizable_nodes


# checks if a node in the simulation graph is amortizable,
# i.e. allows independent estimation of each group
def allows_amortization(inverted_graph: "InvertedGraph", node: SimulationNode) -> bool:
    if node not in inverted_graph.simulation_graph.nodes:
        raise ValueError(f"Node {node} not found.")

    conditions = detailed_conditions_by_node(inverted_graph)
    node_names = original_node_names(inverted_graph)

    for k, v in conditions.items():
        if node_names[k] == node:
            condition_names = [node_names[x] for x in v]
            if node in condition_names:
                return False

    return True


# maps node names of inverted graph to node names in corresponding SimulationGraph
def original_node_names(inverted_graph: "InvertedGraph") -> dict[ExpandedNode, SimulationNode]:
    mapping = {}

    for node in inverted_graph.nodes:
        expanded_node = inverted_graph.expanded_graph.nodes[node]

        if expanded_node["merged_from"] != []:
            mapping[node] = expanded_node["merged_from"][0]
        elif expanded_node["previous_names"] == []:
            mapping[node] = node
        else:
            mapping[node] = expanded_node["previous_names"][0]

    return mapping


# like detailed_conditions_by_node, but uses original node names instead of
# expanded nodes
def conditions_by_node(inverted_graph: "InvertedGraph") -> dict[SimulationNode, list[SimulationNode]]:
    detailed_conditions = detailed_conditions_by_node(inverted_graph)
    node_names = original_node_names(inverted_graph)
    conditions = {}

    for node in inverted_graph.simulation_graph.nodes:
        conditions[node] = []
        for k, v in detailed_conditions.items():
            if node_names[k] == node:
                conditions[node].extend([node_names[c] for c in v])

        conditions[node] = list(set(conditions[node]))

    return conditions


# returns a dictionary with node names as keys and a list of that node's predecessors
# as values
def detailed_conditions_by_node(inverted_graph: "InvertedGraph") -> dict[ExpandedNode, list[ExpandedNode]]:
    conditions = {node: [] for node in inverted_graph.nodes}

    for node in nx.topological_sort(inverted_graph):
        conditions[node] = list(inverted_graph.predecessors(node))

    return conditions
