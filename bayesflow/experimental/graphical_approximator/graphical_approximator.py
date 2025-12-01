from collections.abc import Sequence

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.experimental.graphical_simulator.graphical_simulator import SimulationOutput
from bayesflow.experimental.graphs.types import InvertedGraph
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.networks.standardization import Standardization
from bayesflow.utils import concatenate_valid_shapes, concatenate_valid
import numpy as np
import keras


# TODO: add number of groups as conditions to correct networks
# TODO: allow posterior sampling with non-simulated data
# TODO: unit tests for GraphicalApproximator components
# TODO: more than one data node?
# TODO: maybe fix IRT?
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

    def sample(self, conditions, num_samples: int):
        summary_output, summary_metrics = self._compute_summary_metrics(conditions, stage="validation")
        network_composition = self.graph.network_composition()
        network_conditions = self.graph.network_conditions()
        variable_names = self.graph.variable_names()
        data_node = self.graph.data_node()
        data_conditions = self._data_conditions(conditions)
        batch_size = keras.ops.shape(summary_output[0])[0]

        sample_dict = {}

        for i, inference_network in enumerate(self.inference_networks):
            nodes_to_condition_on = network_conditions[i]
            conditions = []

            for node in nodes_to_condition_on:
                if node != data_node:
                    for variable in variable_names[node]:
                        conditions.append(sample_dict[variable])
                else:
                    conditioned_data = []
                    for conditioned_node in network_composition[i]:
                        if data_conditions[conditioned_node] is not None:
                            expanded_conditions = keras.ops.expand_dims(data_conditions[conditioned_node], axis=1)
                            expanded_conditions = keras.ops.broadcast_to(
                                expanded_conditions,
                                (batch_size, num_samples, *keras.ops.shape(expanded_conditions)[2:]),
                            )
                            conditioned_data.append(expanded_conditions)

                    unique_conditions = unique_tensors(conditioned_data)
                    conditions.extend(unique_conditions)

            concatenated_conditions = self._concatenate(conditions, batch_dims=2)
            samples = inference_network.sample((batch_size, num_samples), conditions=concatenated_conditions)

            variables = []
            for node in network_composition[i]:
                for variable_name in variable_names[node]:
                    variables.append(variable_name)

            for variable_name, samples in zip(variables, keras.ops.unstack(samples, axis=-1)):
                sample_dict[variable_name] = keras.ops.expand_dims(samples, axis=-1)

        return sample_dict

    def fit(self, *args, **kwargs):
        if "dataset" in kwargs.keys():
            if type(kwargs["dataset"]) is SimulationOutput:
                kwargs["dataset"] = kwargs["dataset"].data

        return super(GraphicalApproximator, self).fit(*args, **kwargs, adapter=self.adapter)

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        data_shapes = {k: v for k, v in data_shapes.items() if len(v) > 0}

        # build summary networks
        summary_networks = self.summary_networks or []
        summary_input_shape = self._summary_input_shape(data_shapes)
        summary_output_shapes = self._summary_output_shapes(data_shapes)

        input_shapes = [summary_input_shape] + summary_output_shapes[:-1]
        for i, summary_network in enumerate(summary_networks):
            if not summary_network.built:
                summary_network.build(input_shapes[i])

        # build inference networks
        inference_networks = self.inference_networks
        variables_shapes = self._inference_variables_shapes(data_shapes)
        conditions_shapes = self._inference_conditions_shapes(data_shapes)

        for i, inference_network in enumerate(inference_networks):
            if not inference_network.built:
                inference_network.build(variables_shapes[i], conditions_shapes[i])

        # build standardization layers
        if self.standardize == "all":
            # Only include variables present in data_shapes
            self.standardize = [var for var in data_shapes]
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        # Build all standardization layers
        for var, layer in self.standardize_layers.items():
            layer.build(data_shapes[var])

        self.built = True

    def compile(self, *args, **kwargs):
        return super(GraphicalApproximator, self).compile(*args, **kwargs)

    def compute_metrics(self, stage: str = "training", **kwargs):
        data = kwargs
        inference_conditions = self._prepare_inference_conditions(data, stage)
        inference_variables = self._prepare_inference_variables(data, stage)
        inference_metrics = {}

        for idx, inference_network in enumerate(self.inference_networks):
            inference_metrics[idx] = inference_network.compute_metrics(
                inference_variables[idx], conditions=inference_conditions[idx], stage=stage
            )

        _, summary_metrics = self._compute_summary_metrics(data, stage)

        # combine losses and metrics
        total_loss = 0
        combined_inference_metrics = {}
        combined_summary_metrics = {}

        for idx, metrics in inference_metrics.items():
            total_loss += metrics["loss"]
            for k, v in metrics.items():
                combined_inference_metrics[f"inference_{idx}/{k}"] = v

        for idx, metrics in summary_metrics.items():
            if "loss" in metrics:
                total_loss += metrics["loss"]
            for k, v in metrics.items():
                combined_summary_metrics[f"summary_{idx}/{k}"] = v

        metrics = {"loss": total_loss} | combined_inference_metrics | combined_summary_metrics

        return metrics

    def _prepare_inference_conditions(self, data: dict, stage: str = "training"):
        network_composition = self.graph.network_composition()
        network_conditions = self.graph.network_conditions()

        data_node = self.graph.data_node()
        data_conditions = self._data_conditions(data)

        inference_networks = self.inference_networks
        inference_conditions = {}
        variable_names = self.graph.variable_names()

        for i, _ in enumerate(inference_networks):
            nodes_to_condition_on = network_conditions[i]
            conditions = []

            for node in nodes_to_condition_on:
                if node != data_node:
                    for variable in variable_names[node]:
                        if variable in self.standardize:
                            conditions.append(self.standardize_layers[variable](data[variable], stage=stage))
                        else:
                            conditions.append(data[variable])
                else:
                    conditioned_data = []
                    conditioned_data_shapes = []
                    for conditioned_node in network_composition[i]:
                        if data_conditions[conditioned_node] is not None:
                            for data_condition in data_conditions[conditioned_node]:
                                data_shape = keras.ops.shape(data_condition)
                                if data_shape not in conditioned_data_shapes:
                                    conditioned_data.append(data_condition)
                                    conditioned_data_shapes.append(data_shape)

                    conditions.extend(conditioned_data)

            inference_conditions[i] = self._concatenate(conditions)

        return inference_conditions

    def _data_conditions(self, data: dict):
        data_shapes = self._data_shapes(data)
        network_composition = self.graph.network_composition()
        inference_variables_shapes = self._inference_variables_shapes(data_shapes)
        summary_outputs, _ = self._compute_summary_metrics(data)
        variable_names = self.graph.variable_names()

        data_conditions = {}

        for network_idx, variable_shape in inference_variables_shapes.items():
            required_data_dimension = len(variable_shape)
            for node in network_composition[network_idx]:
                summary_output = [
                    v for k, v in summary_outputs.items() if len(keras.ops.shape(v)) == required_data_dimension
                ]
                data_conditions[node] = summary_output

        for node in variable_names.keys():
            if node not in data_conditions.keys():
                data_conditions[node] = None

        return data_conditions

    def _prepare_inference_variables(self, data: dict, stage: str = "training"):
        network_composition = self.graph.network_composition()
        variable_names = self.graph.variable_names()

        inference_networks = self.inference_networks
        inference_variables = {}

        for i, _ in enumerate(inference_networks):
            variables = []
            for node in network_composition[i]:
                for variable in variable_names[node]:
                    if variable in self.standardize:
                        variables.append(self.standardize_layers[variable](data[variable], stage=stage))
                    else:
                        variables.append(data[variable])

            inference_variables[i] = concatenate_valid(variables, axis=-1)

        return inference_variables

    def _compute_summary_metrics(self, data: dict, stage: str = "training"):
        data_node = self.graph.data_node()
        data_keys = self.graph.variable_names()[data_node]

        summary_variables = []
        for k in data_keys:
            if k in self.standardize:
                summary_variables.append(self.standardize_layers[k](data[k], stage=stage))
            else:
                summary_variables.append(data[k])

        summary_input = concatenate_valid(summary_variables, axis=-1)
        summary_metrics = {}
        summary_outputs = {}

        summary_metrics[0] = self.summary_networks[0].compute_metrics(summary_input, stage=stage)
        summary_outputs[0] = summary_metrics[0].pop("outputs")

        for i, summary_network in enumerate(self.summary_networks[1:]):
            summary_metrics[i + 1] = summary_network.compute_metrics(summary_outputs[i], stage=stage)
            summary_outputs[i + 1] = summary_metrics[i + 1].pop("outputs")

        return summary_outputs, summary_metrics

    def _inference_variables_shapes(self, data_shapes):
        network_composition = self.graph.network_composition()
        variable_names = self.graph.variable_names()

        inference_networks = self.inference_networks
        inference_variables_shapes = {}

        for i, _ in enumerate(inference_networks):
            variable_shapes = []
            for node in network_composition[i]:
                for variable in variable_names[node]:
                    variable_shapes.append(data_shapes[variable])

            inference_variables_shapes[i] = self._concatenate_shapes(variable_shapes)

        return inference_variables_shapes

    @classmethod
    def build_adapter(
        cls,
        sample_weight: str = None,
    ) -> Adapter:
        """Create an :py:class:`~bayesflow.adapters.Adapter` suited for the approximator.

        Parameters
        ----------
        inference_variables : Sequence of str
            Names of the inference variables (to be modeled) in the data dict.
        inference_conditions : Sequence of str, optional
            Names of the inference conditions (to be used as direct conditions) in the data dict.
        summary_variables : Sequence of str, optional
            Names of the summary variables (to be passed to a summary network) in the data dict.
        sample_weight : str, optional
            Name of the sample weights
        """

        adapter = Adapter()
        adapter.to_array()
        adapter.convert_dtype("float64", "float32")

        if sample_weight is not None:
            adapter = adapter.rename(sample_weight, "sample_weight")

        return adapter

    def _inference_conditions_shapes(self, data_shapes):
        network_composition = self.graph.network_composition()
        network_conditions = self.graph.network_conditions()

        data_node = self.graph.data_node()
        data_conditions_shapes = self._data_conditions_shapes(data_shapes)

        inference_networks = self.inference_networks
        inference_conditions_shapes = {}
        variable_names = self.graph.variable_names()

        for i, _ in enumerate(inference_networks):
            nodes_to_condition_on = network_conditions[i]
            condition_shapes = []

            for node in nodes_to_condition_on:
                if node != data_node:
                    for variable in variable_names[node]:
                        condition_shapes.append(data_shapes[variable])
                else:
                    for conditioned_node in network_composition[i]:
                        if data_conditions_shapes[conditioned_node] is not None:
                            condition_shapes.append(data_conditions_shapes[conditioned_node])
                            break

            inference_conditions_shapes[i] = self._concatenate_shapes(condition_shapes)

        return inference_conditions_shapes

    def _concatenate_shapes(self, shapes):
        shape_max_rank = max(shapes, key=lambda s: len(s))
        max_rank = len(shape_max_rank)
        tiled_shapes = []

        for shape in shapes:
            tiled_shapes.append(expand_shape_rank(shape, max_rank))

        stacked_shape = tiled_shapes[0]
        for tiled_shape in tiled_shapes[1:]:
            stacked_shape = stack_shapes(stacked_shape, tiled_shape)

        return stacked_shape

    def _concatenate(self, tensors, batch_dims=1):
        max_rank = max([len(keras.ops.shape(x)) for x in tensors])
        expanded_tensors = []
        reshaped_tensors = []

        for tensor in tensors:
            flattened_shape = (-1, *keras.ops.shape(tensor)[batch_dims:])
            reshaped_tensors.append(keras.ops.reshape(tensor, flattened_shape))

        for tensor in reshaped_tensors:
            target_shape = expand_shape_rank(keras.ops.shape(tensor), max_rank)
            expanded_tensors.append(keras.ops.reshape(tensor, target_shape))

        expanded_shapes = [keras.ops.shape(x) for x in expanded_tensors]
        max_shape = [max(s) for s in zip(*expanded_shapes)]

        tiled_tensors = []
        for expanded_tensor in expanded_tensors:
            repeats = [t // s for t, s in zip(max_shape, keras.ops.shape(expanded_tensor))]
            repeats[-1] = 1  # do not repeat last dimension
            tiled_tensors.append(keras.ops.tile(expanded_tensor, repeats))

        concatenated_tensor = keras.ops.concatenate(tiled_tensors, axis=-1)
        concatenated_tensor = keras.ops.reshape(
            concatenated_tensor,
            (*keras.ops.shape(tensors[0])[:batch_dims], *keras.ops.shape(concatenated_tensor)[batch_dims:]),
        )

        return concatenated_tensor

    def _data_conditions_shapes(self, data_shapes):
        network_composition = self.graph.network_composition()
        inference_variables_shapes = self._inference_variables_shapes(data_shapes)
        summary_output_shapes = self._summary_output_shapes(data_shapes)
        variable_names = self.graph.variable_names()

        data_condition_shapes = {}

        for network_idx, variable_shape in inference_variables_shapes.items():
            required_data_dimension = len(variable_shape)
            for node in network_composition[network_idx]:
                summary_output_shape = [s for s in summary_output_shapes if len(s) == required_data_dimension][0]
                data_condition_shapes[node] = summary_output_shape

        for node in variable_names.keys():
            if node not in data_condition_shapes.keys():
                data_condition_shapes[node] = None

        return data_condition_shapes

    def _batch_size_from_data(self, data):
        data_shapes = self._data_shapes(data)
        batch_size = next(iter(data_shapes.values()))[0]

        return batch_size

    def _summary_input_shape(self, data_shapes):
        data_node = self.graph.data_node()
        data_keys = self.graph.variable_names()[data_node]

        input_shape = concatenate_valid_shapes([data_shapes[k] for k in data_keys], axis=-1)

        return input_shape

    def _summary_output_shapes(self, data_shapes):
        summary_output_shapes = []

        for summary_network in self.summary_networks or []:
            if len(summary_output_shapes) == 0:
                input_shape = self._summary_input_shape(data_shapes)
            else:
                input_shape = summary_output_shapes[-1]

            if len(input_shape) == 2:
                output_shape = summary_network.compute_output_shape(input_shape + [1])
            else:
                output_shape = summary_network.compute_output_shape(input_shape)

            summary_output_shapes.append(output_shape)

        return summary_output_shapes

    def _data_shapes(self, adapted_data: dict):
        return keras.tree.map_structure(keras.ops.shape, adapted_data)


def stack_shapes(shape_1, shape_2, axis=-1):
    rank = max(len(shape_1), len(shape_2))
    stacked_shape = []

    if axis < 0:
        axis += rank

    for i, (dim_1, dim_2) in enumerate(zip(shape_1, shape_2)):
        if i == axis:
            stacked_shape.append(dim_1 + dim_2)
        else:
            stacked_shape.append(max(dim_1, dim_2))

    return tuple(stacked_shape)


def expand_shape_rank(shape, target_rank):
    s = list(to_tuple(shape))
    while len(s) < target_rank:
        s.insert(-1, 1)

    return tuple(s)


def to_tuple(shape):
    if hasattr(shape, "as_list"):
        shape = shape.as_list()

    return tuple(shape)


def unique_tensors(tensors):
    seen = set()
    unique = []

    for t in tensors:
        # Convert to a NumPy array in a backend-agnostic way
        arr = keras.ops.convert_to_numpy(t) if hasattr(keras.ops, "convert_to_numpy") else np.array(t)

        # Build a hashable key from dtype + shape + bytes
        key = (str(arr.dtype), arr.shape, arr.tobytes())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    return unique
