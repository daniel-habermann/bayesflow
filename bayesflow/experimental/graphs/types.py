import copy
import inspect
from typing import Any, Callable, TypeAlias

import networkx as nx

import bayesflow.experimental.graphs.introspection as introspection

from .utils import has_open_path, merge_root_nodes, split_node

Node: TypeAlias = str
SimulationNode: TypeAlias = str
ExpandedNode: TypeAlias = str


class SimulationGraph(nx.DiGraph):
    def __init__(self, meta_fn: Callable | None = None):
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
            for key in ["split_by", "previous_names", "merged_from"]:
                if key not in graph.nodes[node]:
                    graph.nodes[node][key] = []

        return ExpandedGraph(graph, simulation_graph=self)

    def invert(self, merge_roots: bool = True):
        expanded_graph = self.expand()
        inverted_graph = expanded_graph.invert(merge_roots=merge_roots)

        return inverted_graph

    def variable_names(self) -> dict[SimulationNode, list[str]]:
        def _call_sample_fn(sample_fn: Callable[[], dict[str, Any]], args) -> dict[str, Any]:
            signature = inspect.signature(sample_fn)
            fn_args = signature.parameters
            accepted_args = {k: v for k, v in args.items() if k in fn_args}

            return sample_fn(**accepted_args)

        simulation_graph = copy.deepcopy(self)
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

    def data_node(self) -> SimulationNode:
        leaf_nodes = [n for n, d in self.out_degree() if d == 0]

        return leaf_nodes[0]


class ExpandedGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_graph: SimulationGraph = self.graph["simulation_graph"]

    def invert(self, merge_roots=True) -> "InvertedGraph":
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
        self.simulation_graph: SimulationGraph = self.graph["simulation_graph"]
        self.expanded_graph: ExpandedGraph = self.graph["expanded_graph"]

    def network_conditions(self) -> dict[int, list[SimulationNode]]:
        return introspection.network_conditions(self)

    def network_composition(self) -> dict[int, list[SimulationNode]]:
        return introspection.network_composition(self)

    def amortizable_nodes(self) -> list[SimulationNode]:
        return introspection.amortizable_nodes(self)

    def allows_amortization(self, node: SimulationNode) -> bool:
        return introspection.allows_amortization(self, node)

    def original_node_names(self) -> dict[ExpandedNode, SimulationNode]:
        return introspection.original_node_names(self)

    def conditions_by_node(self) -> dict[SimulationNode, list[SimulationNode]]:
        return introspection.conditions_by_node(self)

    def detailed_conditions_by_node(self) -> dict[ExpandedNode, list[ExpandedNode]]:
        return introspection.detailed_conditions_by_node(self)
