from functools import reduce

from bayesflow.experimental.graphical_approximator import GraphicalApproximator
from bayesflow.types import Shape
from bayesflow.utils import concatenate_valid_shapes


# input shape of the data for the first summary network
def summary_input_shape(approximator: GraphicalApproximator, data_shapes: dict[str, Shape]) -> Shape:
    data_node = approximator.graph.simulation_graph.data_node()
    data_keys = approximator.graph.simulation_graph.variable_names()[data_node]

    input_shape = concatenate_valid_shapes([data_shapes[k] for k in data_keys], axis=-1)
    assert input_shape is not None

    return input_shape


# output shape of each summary network
def summary_output_shapes_by_network(approximator: GraphicalApproximator, data_shapes: dict[str, Shape]):
    input_shape = summary_input_shape(approximator, data_shapes)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        shape = input_shape + (1,) if len(input_shape) == 2 else input_shape
        output_shape = summary_network.compute_output_shape(shape)
        result[i] = output_shape

        # next summary network uses previous output as input
        input_shape = output_shape

    return result


# compute shapes of variables estimated by the inference networks
def inference_variable_shapes_by_network(approximator: GraphicalApproximator, data_shapes: dict[str, Shape]):
    network_composition = approximator.graph.network_composition()
    variable_names = approximator.graph.simulation_graph.variable_names()

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        variable_shapes = []
        for node in network_composition[i]:
            for variable in variable_names[node]:
                variable_shapes.append(data_shapes[variable])

        result[i] = concatenate_shapes(variable_shapes)

    return result


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
