from collections.abc import Mapping, Sequence
from copy import copy

import keras
import numpy as np

# TODO: add log_prob method to approximator
from ...adapters import Adapter
from ...approximators import Approximator
from ...networks import InferenceNetwork, SummaryNetwork
from ...networks.standardization import Standardization
from ...types import Shape
from ..graphical_simulator import SimulationOutput
from ..graphs import InvertedGraph
from .utils import (
    concatenate,
    inference_condition_shapes_by_network,
    inference_conditions_by_network,
    inference_variable_shapes_by_network,
    inference_variables_by_network,
    summary_input_shapes_by_network,
    summary_inputs_by_network,
    summary_outputs_by_network,
    add_sample_dimension,
    data_condition_shapes_by_network,
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
        self.data_shapes = None

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = [standardize]
        else:
            self.standardize = standardize or []

        if standardize == "all":
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

    def build(self, data_shapes: dict[str, Shape]) -> None:
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

        for var in self.standardize:
            self.standardize_layers[var].build(data_shapes[var])

        self.data_shapes = data_shapes
        self.built = True

    def compute_metrics(self, stage: str = "training", **kwargs):
        # compute summary metrics
        summary_inputs = summary_inputs_by_network(self, kwargs)
        summary_metrics = {}

        for i, summary_network in enumerate(self.summary_networks or []):
            summary_metrics[i] = summary_network.compute_metrics(summary_inputs[i], stage=stage)
            summary_metrics[i].pop("outputs")

        # compute inference metrics
        inference_conditions = inference_conditions_by_network(self, kwargs)
        inference_variables = inference_variables_by_network(self, kwargs)

        inference_metrics = {}
        for i, inference_network in enumerate(self.inference_networks):
            inference_metrics[i] = inference_network.compute_metrics(
                inference_variables[i], conditions=inference_conditions[i], stage=stage
            )

        # combine metrics
        total_loss = 0
        combined_metrics = {}

        for i, metric_type in enumerate([summary_metrics, inference_metrics]):
            prefix = "summary_metrics" if i == 0 else "infrence_metrics"
            for val, metrics in metric_type.items():
                if "loss" in metrics:
                    total_loss += metrics["loss"]
                for k, v in metrics.items():
                    combined_metrics[f"{prefix}_{val}/{k}"] = v

        return total_loss, combined_metrics

    # TODO: SimulationOutput als arbitrary iterable, keras.Dataset
    def fit(self, *args, **kwargs):
        if "dataset" in kwargs.keys():
            if type(kwargs["dataset"]) is SimulationOutput:
                kwargs["dataset"] = kwargs["dataset"].data

        return super(GraphicalApproximator, self).fit(*args, **kwargs, adapter=self.adapter)

    def sample(self, *, num_samples: int, data: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        summary_outputs = summary_outputs_by_network(self, data)
        batch_size = keras.ops.shape(summary_outputs[0])[0]
        data_node = self.graph.simulation_graph.data_node()
        variable_names = self.graph.simulation_graph.variable_names()
        network_conditions = self.graph.network_conditions()
        network_composition = self.graph.network_composition()

        sample_dict = {}
        conditions = {}

        for name in variable_names[data_node]:
            conditions[name] = add_sample_dimension(data[name], num_samples)

        for i, inference_network in enumerate(self.inference_networks):
            inference_conditions = []
            nodes_to_condition_on = set(network_conditions[i]) - {data_node} - set(network_composition[i])

            for node in nodes_to_condition_on:
                for name in variable_names[node]:
                    inference_conditions.append(conditions[name])

            if data_node in network_conditions[i]:
                required_dim = len(inference_network.base_distribution.dims) + 1
                summary_by_dim = {len(keras.ops.shape(s)): s for s in summary_outputs.values()}

                data_condition = add_sample_dimension(summary_by_dim[required_dim], num_samples)
                inference_conditions.append(data_condition)

    def _sample(self, *, num_samples: int, conditions: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        summary_outputs = summary_outputs_by_network(self, conditions)
        batch_size = keras.ops.shape(summary_outputs[0])[0]
        data_node = self.graph.simulation_graph.data_node()
        variable_names = self.graph.simulation_graph.variable_names()
        network_conditions = self.graph.network_conditions()
        network_composition = self.graph.network_composition()

        sample_dict = {}
        computed_conditions = copy.copy(conditions)

        for name in variable_names[data_node]:
            # TODO: add add_sample_dim(x, num_samples) helper
            computed_conditions[name] = keras.ops.expand_dims(computed_conditions[name], axis=1)
            computed_conditions[name] = keras.ops.broadcast_to(
                computed_conditions[name], (batch_size, num_samples, *keras.ops.shape(computed_conditions[name])[2:])
            )

        for i, inference_network in enumerate(self.inference_networks):
            inference_conditions = []
            nodes_to_condition_on = set(network_conditions[i]) - {data_node} - set(network_composition[i])

            for node in nodes_to_condition_on:
                for name in variable_names[node]:
                    inference_conditions.append(computed_conditions[name])

            if data_node in network_conditions[i]:
                required_dim = len(inference_network.base_distribution.dims) + 1
                summary_by_dim = {len(keras.ops.shape(s)): s for s in summary_outputs.values()}

                data_condition = summary_by_dim[required_dim]
                data_condition = keras.ops.expand_dims(data_condition, axis=1)
                data_condition = keras.ops.broadcast_to(
                    data_condition, (batch_size, num_samples, *keras.ops.shape(data_condition)[2:])
                )
                inference_conditions.append(data_condition)

            inference_conditions = utils.concatenate(inference_conditions)
            samples = inference_network.sample((batch_size, num_samples), conditions=inference_conditions)

            # TODO: refactor this into own method
            # variable names
            variables = []
            for node in network_composition[i]:
                for variable_name in variable_names[node]:
                    variables.append(variable_name)

            if len(variables) == keras.ops.shape(samples)[-1]:
                for variable_name, samples in zip(variables, keras.ops.unstack(samples, axis=-1)):
                    computed_conditions[variable_name] = keras.ops.expand_dims(samples, axis=-1)
                    sample_dict[variable_name] = keras.ops.expand_dims(samples, axis=-1)
            else:
                reshaped_samples = keras.ops.reshape(
                    samples,
                    (*keras.ops.shape(samples)[:-1], keras.ops.shape(samples)[-1] // len(variables), len(variables)),
                )
                for variable_name, samples in zip(variables, keras.ops.unstack(reshaped_samples, axis=-1)):
                    computed_conditions[variable_name] = samples
                    sample_dict[variable_name] = keras.ops.expand_dims(samples, axis=-1)

        return sample_dict

    def _batch_size_from_data(self, data):
        data_shapes = self.data_shapes(data)
        batch_size = next(iter(data_shapes.values()))[0]

        return batch_size

    def _data_shapes(self, adapted_data: SimulationOutput | Mapping) -> Mapping:
        if isinstance(adapted_data, dict):
            return keras.tree.map_structure(keras.ops.shape, adapted_data)
        elif isinstance(adapted_data, SimulationOutput):
            return keras.tree.map_structure(keras.ops.shape, adapted_data.data)

        return {}
