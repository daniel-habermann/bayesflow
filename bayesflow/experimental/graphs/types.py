from typing import TypeAlias

import networkx as nx

from .utils import merge_root_nodes, split_node, has_open_path

Node: TypeAlias = str


class SimulationGraph(nx.DiGraph):
    def __init___(self):
        super().__init__(self)

    def expand(self):
        graph = self.copy()

        for node in nx.topological_sort(graph):
            interior_node = graph.in_degree(node) != 0 and graph.out_degree(node) != 0
            if interior_node and node in graph.nodes:
                graph = split_node(graph, node)

        return ExpandedGraph(graph)

    def invert(self, merge_roots=True):
        expanded_graph = self.expand()
        inverted_graph = expanded_graph.invert(merge_roots=merge_roots)

        return inverted_graph


class ExpandedGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invert(self, merge_roots=True):
        if merge_roots:
            graph = merge_root_nodes(self)
        else:
            graph = self.copy()

        undirected = graph.to_undirected()
        leaf_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

        # Sort nodes by outer nodes first. We assume that this ordering preserves
        # amortization over exchangeable nodes in most cases.
        latent_nodes = [node for node in list(nx.topological_sort(graph)) if graph.out_degree(node) != 0]

        inverse = InvertedGraph()
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


class InvertedGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer_conditions(self):
        conditions = {node: [] for node in self.nodes}

        for node in nx.topological_sort(self):
            conditions[node] = list(self.predecessors(node))

        return conditions

    def network_composition(self):
        conditions = self.infer_conditions()
        processed_nodes = set(k for k, v in conditions.items() if v == [])
        conditions = {k: v for k, v in conditions.items() if k not in processed_nodes}

        networks = {}
        network_idx = 0

        # Build inference layers iteratively: start with all nodes that require no conditions,
        # then repeatedly form the next layer by selecting nodes whose dependencies are
        # entirely contained covered by previous inference networks.
        while conditions:
            next_nodeset = {k for k, v in conditions.items() if set(v).issubset(processed_nodes)}

            if next_nodeset:
                networks[network_idx] = list(next_nodeset)
                processed_nodes.update(next_nodeset)

            for node in next_nodeset:
                conditions.pop(node)

            network_idx += 1

        return networks
