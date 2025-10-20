import copy
import inspect
from typing import TypeAlias

import networkx as nx

from .utils import has_open_path, merge_root_nodes, split_node

Node: TypeAlias = str


class SimulationGraph(nx.DiGraph):
    def __init__(self, meta_fn=None):
        super().__init__(self)
        self.meta_fn = meta_fn

    def expand(self):
        graph = self.copy()

        for node in nx.topological_sort(graph):
            interior_node = graph.in_degree(node) != 0 and graph.out_degree(node) != 0

            if not interior_node:
                graph.nodes[node].clear()

            if interior_node and node in graph.nodes:
                graph = split_node(graph, node)

        for node in nx.topological_sort(graph):
            for key in ["split_by", "previous_names"]:
                if key not in graph.nodes[node]:
                    graph.nodes[node][key] = []

        return ExpandedGraph(graph, simulation_graph=self)

    def invert(self, merge_roots=True):
        expanded_graph = self.expand()
        inverted_graph = expanded_graph.invert(merge_roots=merge_roots)

        return inverted_graph


class ExpandedGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_graph = self.graph["simulation_graph"]

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


class InvertedGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_graph = self.graph["simulation_graph"]
        self.expanded_graph = self.graph["expanded_graph"]

    def network_composition(self):
        conditions = self._conditions()
        processed_nodes = set(k for k, v in conditions.items() if v == [])
        conditions = {k: v for k, v in conditions.items() if k not in processed_nodes}

        networks = {}
        network_idx = 0

        # Build inference layers iteratively: start with all nodes that require no conditions,
        # then repeatedly form the next layer by selecting nodes whose dependencies are
        # entirely contained covered by previous inference networks.
        while conditions:
            networks[network_idx] = []
            next_nodeset = {k for k, v in conditions.items() if set(v).issubset(processed_nodes)}

            if next_nodeset:
                processed_nodes.update(next_nodeset)

                for node in next_nodeset:
                    conditions.pop(node)
                    networks[network_idx].extend(self._original_names(node))

            network_idx += 1

        for k, v in networks.items():
            networks[k] = list(set(v))

        return networks

    def network_conditions(self):
        composition = self.network_composition()

        networks = {}

        for network_idx, orig_nodes in composition.items():
            networks[network_idx] = []
            for node in orig_nodes:
                networks[network_idx].extend(self.conditions_for_node(node))

        node_order = list(nx.topological_sort(self.simulation_graph))
        for k, v in networks.items():
            networks[k] = [n for n in node_order if n in v]

        return networks

    def variable_names(self):
        def _call_sample_fn(sample_fn, args):
            signature = inspect.signature(sample_fn)
            fn_args = signature.parameters
            accepted_args = {k: v for k, v in args.items() if k in fn_args}

            return sample_fn(**accepted_args)

        simulation_graph = copy.deepcopy(self.simulation_graph)
        meta_dict = simulation_graph.meta_fn() if simulation_graph.meta_fn else {}
        samples_by_node = {}

        for node in nx.topological_sort(simulation_graph):
            simulation_graph.nodes[node]["reps"] = 1
            parent_nodes = list(simulation_graph.predecessors(node))
            sample_fn = simulation_graph.nodes[node]["sample_fn"]

            if not parent_nodes:
                samples_by_node[node] = _call_sample_fn(sample_fn, {})
            else:
                parent_samples = [samples_by_node[p] for p in parent_nodes]
                merged_dict = {k: v for d in parent_samples for k, v in d.items()}

                sample_fn_input = merged_dict | meta_dict
                samples_by_node[node] = _call_sample_fn(sample_fn, sample_fn_input)

        variabe_dict = {k: list(v.keys()) for k, v in samples_by_node.items()}

        return variabe_dict

    def data_node(self):
        leaf_nodes = [n for n, d in self.simulation_graph.out_degree() if d == 0]

        return leaf_nodes[0]

    def data_layers(self):
        data_node = self.data_node()
        data_layers = {}

        for node in self.expanded_graph.nodes:
            expanded_node = self.expanded_graph.nodes[node]
            if data_node in expanded_node["previous_names"]:
                layer = len(expanded_node["previous_names"]) - 1
                data_layers.setdefault(layer, []).append(node)

        return data_layers

    def conditions_for_node(self, orig_node: Node):
        if orig_node not in self.simulation_graph.nodes:
            raise ValueError(f"Node {orig_node} not found.")

        node_conditions = []

        raw_conditions = self._conditions()
        for k, v in raw_conditions.items():
            if orig_node in self._original_names(k):
                node_conditions.extend(self._original_names(x) for x in v)

        node_conditions = list({x for sublist in node_conditions for x in sublist})

        return node_conditions

    def merged_nodes(self, orig_node: Node):
        if orig_node not in self.simulation_graph.nodes:
            raise ValueError(f"Node {orig_node} not found.")

        for node in self.expanded_graph.nodes:
            expanded_node = self.expanded_graph.nodes[node]
            if "merged_from" in expanded_node:
                if orig_node in expanded_node["merged_from"]:
                    return expanded_node["merged_from"]

        return None

    def is_merged(self, orig_node: Node):
        if orig_node not in self.simulation_graph.nodes:
            raise ValueError(f"Node {orig_node} not found.")

        for node in self.expanded_graph.nodes:
            expanded_node = self.expanded_graph.nodes[node]
            if "merged_from" in expanded_node:
                if orig_node in expanded_node["merged_from"]:
                    return True

        return False

    def allows_amortization(self, orig_node: Node):
        if orig_node not in self.simulation_graph.nodes:
            raise ValueError(f"Node {orig_node} not found.")

        conditions = self._conditions()
        for k, v in conditions.items():
            if orig_node in self._original_names(k):
                orig_condition_names = [self._original_names(x) for x in v]
                if orig_node in orig_condition_names:
                    return False

        return True

    def _conditions(self):
        conditions = {node: [] for node in self.nodes}

        for node in nx.topological_sort(self):
            conditions[node] = list(self.predecessors(node))

        return conditions

    def _original_names(self, node: Node):
        expanded_node = self.expanded_graph.nodes[node]

        if "merged_from" in expanded_node:
            return expanded_node["merged_from"]
        else:
            return [expanded_node["previous_names"][0]]
