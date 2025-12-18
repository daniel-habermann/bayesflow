# TODO: add group size as conditions
import copy
import inspect
from typing import Any, Callable, TypeAlias

import networkx as nx

from .utils import split_node

Node: TypeAlias = str
SimulationNode: TypeAlias = str
ExpandedNode: TypeAlias = str


class SimulationGraph(nx.DiGraph):
    def __init__(self, *, meta_fn: Callable | None = None, **kwargs):
        super().__init__(self, **kwargs)
        self.meta_fn = meta_fn

    def expand(self):
        from .expanded_graph import ExpandedGraph

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

        return ExpandedGraph(simulation_graph=self)

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

        return {k: list(v.keys()) for k, v in samples_by_node.items()}

    def data_node(self) -> SimulationNode:
        leaf_nodes = [n for n, d in self.out_degree() if d == 0]

        return leaf_nodes[0]
