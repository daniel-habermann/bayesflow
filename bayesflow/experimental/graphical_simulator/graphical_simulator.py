import inspect
import itertools
from collections.abc import Callable
from typing import Any, Optional

import networkx as nx
import numpy as np

from bayesflow.simulators import Simulator
from bayesflow.types import Shape


class GraphicalSimulator(Simulator):
    """
    A graph-based simulator that generates samples by traversing a DAG
    and calling user-defined sampling functions at each node.

    Parameters
    ----------
    meta_fn : Optional[Callable[[], dict[str, Any]]]
        A callable that returns a dictionary of meta data.
        This meta data can be used to dynamically vary the number of sampling repetitions (`reps`)
        for nodes added via `add_node`.
    """

    def __init__(self, meta_fn: Optional[Callable[[], dict[str, Any]]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = nx.DiGraph()
        self.meta_fn = meta_fn

    def add_node(self, node: str, sampling_fn: Callable[..., dict[str, Any]], reps: int | str = 1):
        self.graph.add_node(node, sampling_fn=sampling_fn, reps=reps)

    def add_edge(self, from_node: str, to_node: str):
        self.graph.add_edge(from_node, to_node)

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """
        Generates samples by topologically traversing the DAG.
        For each node, the sampling function is called based on parent values.

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to sample. Typically, a tuple indicating the number of samples,
            but an int can also be passed.
        **kwargs
            Unused
        """
        _ = kwargs  # Simulator class requires **kwargs, which are unused here
        meta_dict = self.meta_fn() if self.meta_fn else {}

        # Initialize samples containers for each node
        for node in self.graph.nodes:
            self.graph.nodes[node]["samples"] = np.empty(batch_shape, dtype="object")

        for batch_idx in np.ndindex(batch_shape):
            for node in nx.topological_sort(self.graph):
                node_samples = []

                parent_nodes = list(self.graph.predecessors(node))
                sampling_fn = self.graph.nodes[node]["sampling_fn"]
                reps_field = self.graph.nodes[node]["reps"]
                reps = reps_field if isinstance(reps_field, int) else meta_dict[reps_field]

                if not parent_nodes:
                    # root node: generate independent samples
                    node_samples = [
                        {"__batch_idx": batch_idx, f"__{node}_idx": i} | sampling_fn() for i in range(1, reps + 1)
                    ]
                else:
                    # non-root node: depends on parent samples
                    parent_samples = [self.graph.nodes[p]["samples"][batch_idx] for p in parent_nodes]
                    merged_dicts = merge_lists_of_dicts(parent_samples)

                    for merged in merged_dicts:
                        index_entries = filter_indices(merged)
                        variable_entries = filter_variables(merged)

                        node_samples.extend(
                            [
                                index_entries | {f"__{node}_idx": i} | call_sampling_fn(sampling_fn, variable_entries)
                                for i in range(1, reps + 1)
                            ]
                        )

                self.graph.nodes[node]["samples"][batch_idx] = node_samples

        return {"a": np.zeros(3)}


def merge_lists_of_dicts(nested_list: list[list[dict]]) -> list[dict]:
    """
    Merges all combinations of dictionaries from a list of lists.
    Equivalent to a Cartesian product of dicts, then flattening.
    """

    all_combinations = itertools.product(*nested_list)
    return [{k: v for d in combo for k, v in d.items()} for combo in all_combinations]


def call_sampling_fn(sampling_fn: Callable, inputs: dict) -> dict[str, Any]:
    num_args = len(inspect.signature(sampling_fn).parameters)
    if num_args == 0:
        return sampling_fn()
    else:
        return sampling_fn(**inputs)


def filter_indices(d: dict) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k.startswith("__")}


def filter_variables(d: dict) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("__")}
