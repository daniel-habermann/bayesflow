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

    def _assign_data_to_summary_networks(self):
        pass
