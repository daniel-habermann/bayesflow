from functools import reduce
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator

import numpy as np

from bayesflow.experimental.graphs.introspection import data_shape_order, permutated_data_shape_order
from bayesflow.types import Shape
from bayesflow.utils import concatenate_valid_shapes


# data input shape for first summary network
def summary_input_shape(approximator: "GraphicalApproximator", data_shapes: dict[str, Shape]) -> Shape:
    data_node = approximator.graph.simulation_graph.data_node()
    data_keys = approximator.graph.simulation_graph.variable_names()[data_node]

    input_shape = concatenate_valid_shapes([data_shapes[k] for k in data_keys], axis=-1)
    assert input_shape

    # permutate input_shape so inputs are put into summary networks in the required order
    shape_order = data_shape_order(approximator.graph)
    permutated_shape_order = permutated_data_shape_order(approximator.graph)
    indices = [shape_order.index(x) for x in permutated_shape_order]

    input_shape = (input_shape[0],) + tuple(input_shape[1:-1][idx] for idx in indices) + (input_shape[-1],)

    return input_shape


# output shape of each summary network
def summary_output_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: dict[str, Shape]):
    input_shape = summary_input_shape(approximator, data_shapes)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        shape = input_shape + (1,) if len(input_shape) == 2 else input_shape
        output_shape = summary_network.compute_output_shape(shape)
        result[i] = output_shape

        # next summary network uses previous output as input
        input_shape = output_shape

    return result


# input shape of each summary network
def summary_input_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: dict[str, Shape]):
    input_shape = summary_input_shape(approximator, data_shapes)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        shape = input_shape + (1,) if len(input_shape) == 2 else input_shape
        result[i] = input_shape

        output_shape = summary_network.compute_output_shape(shape)

        # next summary network uses previous output as input
        input_shape = output_shape

    return result


# computes shape of data conditions for each inference network
def data_condition_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: dict[str, Shape]):
    inference_shapes = inference_variable_shapes_by_network(approximator, data_shapes)
    conditions = approximator.graph.network_conditions()
    data_node = approximator.graph.simulation_graph.data_node()

    summary_shapes = summary_output_shapes_by_network(approximator, data_shapes)
    summary_by_dim = {len(s): s for s in summary_shapes.values()}

    result = {}

    for i, variable_shape in inference_shapes.items():
        # data dimension must be identical to inference variable dimension
        dim = len(variable_shape)

        # only add data conditions if data node is network conditions
        if data_node in conditions[i]:
            result[i] = summary_by_dim[dim]
        else:
            result[i] = None

    return result


# compute shapes of variables estimated by the inference networks
def inference_variable_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: dict[str, Shape]):
    network_composition = approximator.graph.network_composition()
    variable_names = approximator.graph.simulation_graph.variable_names()
    amortizable_nodes = approximator.graph.amortizable_nodes()

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        variable_shapes = []
        for node in network_composition[i]:
            for variable in variable_names[node]:
                shape = data_shapes[variable]

                # flatten group dimension if node is not amortizable
                if node not in amortizable_nodes:
                    shape = shape[:-2] + (np.sum(shape[-2:]),)

                variable_shapes.append(to_tuple(shape))

        result[i] = concatenate_shapes(variable_shapes)

    return result


# computes shapes of inference conditions for each network
def inference_condition_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: dict[str, Shape]):
    data_conditions = data_condition_shapes_by_network(approximator, data_shapes)
    network_conditions = approximator.graph.network_conditions()
    variable_names = approximator.graph.simulation_graph.variable_names()
    amortizable_nodes = approximator.graph.amortizable_nodes()
    data_node = approximator.graph.simulation_graph.data_node()

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        # collect shapes from all variables in the nodes
        condition_shapes = []
        for node in network_conditions[i]:
            if node != data_node:
                for variable in variable_names[node]:
                    shape = data_shapes[variable]

                    # flatten group dimension if node is not amortizable
                    if node not in amortizable_nodes:
                        shape = shape[:-2] + (np.sum(shape[-2:]),)

                    condition_shapes.append(to_tuple(shape))

        # add data conditions if necessary
        if data_conditions[i] is not None:
            condition_shapes.append(data_conditions[i])

        result[i] = concatenate_shapes(condition_shapes)

    return result


def concatenate(tensors, batch_dims=1):
    max_rank = max([len(keras.ops.shape(t)) for t in tensors])

    # expand tensors so each tensor has rank max_rank
    expanded = []

    for t in tensors:
        flat_shape = (-1, *keras.ops.shape(t)[batch_dims:])
        flat = keras.ops.reshape(t, flat_shape)

        expanded_shape = expand_shape_rank(keras.ops.shape(flat), max_rank)
        expanded.append(keras.ops.reshape(flat, expanded_shape))

    # compute max size along each dimension
    expanded_shapes = [keras.ops.shape(t) for t in expanded]
    max_shape_per_dim = [max(s) for s in zip(*expanded_shapes)]

    # broadcast tensors to match max_shape
    target_shapes = [(*max_shape_per_dim[:-1], keras.ops.shape(t)[-1]) for t in expanded]
    broadcasted = [keras.ops.broadcast_to(t, s) for t, s in zip(expanded, target_shapes)]

    # concatenate along last dimension
    concatenated = keras.ops.concatenate(broadcasted, axis=-1)

    # restore original batch dimensions
    original_batch_shape = keras.ops.shape(tensors[0])[:batch_dims]
    final_shape = (*original_batch_shape, *keras.ops.shape(concatenated)[1:])

    return keras.ops.reshape(concatenated, final_shape)


# concatenate shapes by expanding them to the same rank
# and then summing sizes along the last axis
def concatenate_shapes(shapes):
    max_rank = max(len(to_tuple(s)) for s in shapes)
    expanded = [expand_shape_rank(to_tuple(s), max_rank) for s in shapes]

    return reduce(stack_shapes, expanded)


# stack two shapes by summing dims on axis and max dims elsewhere
def stack_shapes(a, b, axis=-1):
    a, b = to_tuple(a), to_tuple(b)

    # make ranks equal
    rank = max(len(a), len(b))
    a = expand_shape_rank(a, rank)
    b = expand_shape_rank(b, rank)

    # normalize axis
    if axis < 0:
        axis += rank

    # stack shapes
    stacked_shape = tuple((a[i] + b[i]) if i == axis else max(a[i], b[i]) for i in range(rank))

    return stacked_shape


# insert 1's before last dimension until reaching target rank
def expand_shape_rank(shape, target_rank):
    s = list(to_tuple(shape))
    while len(s) < target_rank:
        s.insert(-1, 1)

    return tuple(s)


# convert tensorflow/torch/numpy shapes to tuples
def to_tuple(shape):
    if hasattr(shape, "as_list"):
        shape = shape.as_list()

    return tuple(shape)
