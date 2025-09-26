from collections.abc import Sequence

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.experimental.graphs.types import InvertedGraph
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.networks.standardization import Standardization


class GraphicalApproximator(Approximator):
    def __init__(
        self,
        graph: InvertedGraph,
        *,
        adapter: Adapter,
        inference_networks: list[InferenceNetwork],
        summary_networks: list[SummaryNetwork] | None = None,
        standardize: str | Sequence[str] | None = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph = graph
        self.adapter = adapter
        self.infertence_networks = inference_networks
        self.summary_networks = summary_networks

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = []
        else:
            self.standardize = standardize or []

        if self.standardize == "all":
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        summary_outputs_shape = [data_shapes["summary_variables"]]
        if self.summary_networks is not None:
            for summary_network in self.summary_networks:
                if not summary_network.built:
                    summary_network.build(summary_outputs_shape[-1])

                summary_outputs_shape.append(summary_network.compute_output_shape(summary_outputs_shape[-1]))

    # Approximator algorithm:

    # function 1: updated network composition
    # go through network composition, if nodes with same name are in different networks, combine them
    # add annotation if nodes are amortized or not

    # function 2: function that assigns each node an output shape

    # function 3: function that retrieves condition for a target node, concatenates them
