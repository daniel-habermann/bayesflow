from typing import TypeAlias

import networkx as nx

from .simulation_graph import SimulationGraph
from .utils import has_open_path, merge_root_nodes

Node: TypeAlias = str
SimulationNode: TypeAlias = str
ExpandedNode: TypeAlias = str


class ExpandedGraph(nx.DiGraph):
    def __init__(self, *, simulation_graph: SimulationGraph, **kwargs):
        super().__init__(**kwargs)
        self.simulation_graph = simulation_graph

    def invert(self, merge_roots: bool = True):
        from .inverted_graph import InvertedGraph

        if merge_roots:
            graph = merge_root_nodes(self.simulation_graph)
        else:
            graph = self.copy()

        undirected = graph.to_undirected()
        leaf_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

        # Sort nodes by outer nodes first. We assume that this ordering preserves
        # amortization over exchangeable nodes in most cases.
        latent_nodes = [node for node in list(nx.topological_sort(graph)) if graph.out_degree(node) != 0]

        inverse = InvertedGraph(simulation_graph=self.simulation_graph, expanded_graph=self)
        inverse.add_nodes_from(leaf_nodes)

        for x_j in latent_nodes:
            inverse.add_node(x_j)

            # Iterate over all already added nodes in inverse (shortest distance from x_j first)
            # and check if the path between that node and x_j is blocked.
            # If it is open, draw an edge from that node to x_j.
            other_nodes = [node for node in inverse.nodes() if node != x_j]
            lengths = [nx.shortest_path_length(undirected, x_j, node) for node in other_nodes]
            sorted_nodes = [node for _, node in sorted(zip(lengths, other_nodes))]

            for node in sorted_nodes:
                if has_open_path(graph, x_j, node, other_nodes):
                    inverse.add_edge(node, x_j)

        return inverse
