from collections.abc import Mapping, Sequence

import keras.utils
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.experimental.graphs.types import InvertedGraph
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.networks.standardization import Standardization
from bayesflow.simulators import Simulator


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
            self.standardize = [standardize]
        else:
            self.standardize = standardize or []

        self.standardize_layers = (
            None if standardize == "all" else {var: Standardization(trainable=False) for var in self.standardize}
        )

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        pass

    def compile(self, *args, **kwargs):
        return super(GraphicalApproximator, self).compile(*args, **kwargs)

    def compute_metrics(self, stage: str = "training", **kwargs):
        pass

    def fit(self, *, dataset: keras.utils.PyDataset | None = None, simulator: Simulator | None = None, **kwargs):
        pass

    def sample(self, *, num_samples: int, conditions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {}

    def predict(self):
        pass
