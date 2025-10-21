from collections.abc import Sequence

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator
from bayesflow.experimental.graphs.types import InvertedGraph
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.networks.standardization import Standardization
from bayesflow.utils import concatenate_valid_shapes
import keras


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

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
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

    def compile(
        self,
        *args,
        **kwargs,
    ):
        return super(GraphicalApproximator, self).compile(*args, **kwargs)

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

    def _data_conditions_shapes(self, data_shapes):
        # TODO: simplify branching
        expanded_conditions = self.graph._conditions()
        summary_output_shapes = self._summary_output_shapes(data_shapes)

        data_layers = self.graph.data_layers()
        data_condition_shapes = {}

        for node, conditions in expanded_conditions.items():
            orig_node_names = self.graph._original_names(node)

            for layer, keys in data_layers.items():
                if set(keys) <= set(conditions):
                    for name in orig_node_names:
                        data_condition_shapes[name] = summary_output_shapes[layer + 1]
                elif len(set(keys) & set(conditions)) > 0:
                    for name in orig_node_names:
                        data_condition_shapes[name] = summary_output_shapes[layer]

            for name in orig_node_names:
                if name not in data_condition_shapes.keys():
                    data_condition_shapes[name] = None

        return data_condition_shapes

    def _summary_input_shape(self, data_shapes):
        data_node = self.graph.data_node()
        data_keys = self.graph.variable_names()[data_node]

        input_shape = concatenate_valid_shapes([data_shapes[k] for k in data_keys])

        return input_shape

    def _summary_output_shapes(self, data_shapes):
        summary_output_shapes = []

        for summary_network in self.summary_networks or []:
            if len(summary_output_shapes) == 0:
                input_shape = self._summary_input_shape(data_shapes)
            else:
                input_shape = summary_output_shapes[-1]

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
        s.insert(1, 1)

    return tuple(s)


def to_tuple(shape):
    if hasattr(shape, "as_list"):
        shape = shape.as_list()

    return tuple(shape)
