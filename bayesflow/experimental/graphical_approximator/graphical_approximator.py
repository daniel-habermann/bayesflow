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
        inference_networks: Sequence[InferenceNetwork],
        summary_networks: Sequence[SummaryNetwork] | None = None,
        standardize: str | Sequence[str] | None = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph = graph
        self.adapter = adapter
        self.inference_networks = inference_networks
        self.summary_networks = summary_networks

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = []
        else:
            self.standardize = standardize or []

        if self.standardize == "all":
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

    # pass through inference networks
    # pass through summary networks

    # summary networks first, because output needed for inference networks

    # output of graphical simulator must be assigned to all summary networks
    # chain of summary networks, data dimensionality is getting reduced in each
    # step.

    # InvertedGraph shows which output needs to be put into which summary network
    # in the chain

    # InvertedGraph also shows which output of the summary networks and which
    # parameters need to be put into which inference network

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        summary_outputs_shape = []

        for summary_network in self.summary_networks or []:
            if not summary_outputs_shape:
                input_shape = data_shapes["summary_variables"]
            else:
                input_shape = summary_outputs_shape[-1]

            if not summary_network.built:
                summary_network.build(input_shape)

            output_shape = summary_network.compute_output_shape(input_shape)
            summary_outputs_shape.append(output_shape)

        # TODO: build inference networks
        # TODO: build standardize layers

        print(summary_outputs_shape)
