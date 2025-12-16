from collections.abc import Mapping, Sequence

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.experimental.graphical_simulator import SimulationOutput
from bayesflow.experimental.graphs.types import InvertedGraph
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.networks.standardization import Standardization
from bayesflow.types import Shape

from .utils import (
    inference_condition_shapes_by_network,
    inference_variable_shapes_by_network,
    summary_input_shapes_by_network,
    inference_conditions_by_network,
    inference_variables_by_network,
    summary_inputs_by_network,
)


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

    def build(self, data_shapes: dict[str, Shape]) -> None:
        data_shapes = {k: v for k, v in data_shapes.items() if len(v) > 0}

        # build summary networks
        input_shapes = summary_input_shapes_by_network(self, data_shapes)
        for i, summary_network in enumerate(self.summary_networks or []):
            if not summary_network.built:
                summary_network.build(input_shapes[i])

        # build inference networks
        variable_shapes = inference_variable_shapes_by_network(self, data_shapes)
        condition_shapes = inference_condition_shapes_by_network(self, data_shapes)

        for i, inference_network in enumerate(self.inference_networks or []):
            if not inference_network.built:
                inference_network.build(variable_shapes[i], condition_shapes[i])

        # build standardization layers
        if self.standardize == "all":
            # Only include variables present in data_shapes
            self.standardize = list(data_shapes.keys())
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        # Build all standardization layers
        assert self.standardize_layers is not None  # for proper type hinting

        for var in self.standardize:
            self.standardize_layers[var].build(data_shapes[var])

        self.built = True

    def compile(self, *args, **kwargs):
        return super(GraphicalApproximator, self).compile(*args, **kwargs)

    def compute_metrics(self, stage: str = "training", **kwargs):
        data = kwargs
        summary_inputs = summary_inputs_by_network(self, data)
        inference_conditions = inference_conditions_by_network(self, data)
        inference_variables = inference_variables_by_network(self, data)

        # compute summary metrics
        summary_metrics = {}

        for i, summary_network in enumerate(self.summary_networks or []):
            summary_metrics[i] = summary_network.compute_metrics(summary_inputs[i], stage=stage)
            summary_metrics[i].pop("outputs")

        # compute inference metrics
        inference_metrics = {}
        for i, inference_network in enumerate(self.inference_networks):
            inference_metrics[i] = inference_network.compute_metrics(
                inference_variables[i], conditions=inference_conditions[i], stage=stage
            )

        # combine losses and metrics
        total_loss = 0
        combined_inference_metrics = {}
        combined_summary_metrics = {}

        for i, metrics in inference_metrics.items():
            total_loss += metrics["loss"]
            for k, v in metrics.items():
                if k == "loss":
                    combined_inference_metrics[f"inference_{i}/{k}"] = v
                else:
                    combined_inference_metrics[f"inference_{i}/{k}"] = v

        for i, metrics in summary_metrics.items():
            if "loss" in metrics.keys():
                total_loss += metrics["loss"]
            for k, v in metrics.items():
                if k == "loss":
                    combined_summary_metrics[f"summary_{i}/{k}"] = v
                else:
                    combined_summary_metrics[f"summary_{i}/{k}"] = v

        metrics = {"loss": total_loss} | combined_inference_metrics

        return metrics

    def fit(self, *args, **kwargs):
        if "dataset" in kwargs.keys():
            if type(kwargs["dataset"]) is SimulationOutput:
                kwargs["dataset"] = kwargs["dataset"].data

        return super(GraphicalApproximator, self).fit(*args, **kwargs, adapter=self.adapter)

    def sample(self, *, num_samples: int, conditions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {}

    def predict(self):
        pass

    def _batch_size_from_data(self, data):
        data_shapes = self.data_shapes(data)
        batch_size = next(iter(data_shapes.values()))[0]

        return batch_size

    def data_shapes(self, adapted_data: SimulationOutput | dict):
        if isinstance(adapted_data, dict):
            return keras.tree.map_structure(keras.ops.shape, adapted_data)
        else:
            return keras.tree.map_structure(keras.ops.shape, adapted_data.data)
