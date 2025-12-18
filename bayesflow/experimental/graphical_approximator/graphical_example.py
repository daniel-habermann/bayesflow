import bayesflow as bf
from bayesflow.adapters import Adapter

from ..graphical_approximator.graphical_approximator import GraphicalApproximator
from ..graphical_simulator.example_simulators import (
    crossed_design_irt_simulator,
    single_level_simulator,
    three_level_simulator,
    two_level_simulator,
)


def simulator():
    return crossed_design_irt_simulator()


def adapter():
    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    return adapter


def summary_networks():
    summary_networks = [
        bf.networks.DeepSet(summary_dim=10),
        bf.networks.DeepSet(summary_dim=10),
    ]

    return summary_networks


def inference_networks():
    inference_networks = [bf.networks.CouplingFlow(), bf.networks.CouplingFlow(), bf.networks.CouplingFlow()]

    return inference_networks


def approximator():
    inverted_graph = simulator().graph.invert()
    # TODO: check if number of inference and summary networks is correct in __init__
    approximator = GraphicalApproximator(
        inverted_graph, adapter=adapter(), inference_networks=inference_networks(), summary_networks=summary_networks()
    )

    return approximator
