import numpy as np
from .graphical_simulator import GraphicalSimulator


def twolevel_simulator():
    def sample_hypers():
        hyper_mean = np.random.normal()
        hyper_std = np.abs(np.random.normal())

        return {"hyper_mean": float(hyper_mean), "hyper_std": float(hyper_std)}

    def sample_locals(hyper_mean, hyper_std):
        local_mean = np.random.normal(hyper_mean, hyper_std)

        return {"local_mean": float(local_mean)}

    def sample_shared():
        shared_std = np.abs(np.random.normal())

        return {"shared_std": shared_std}

    def sample_y(local_mean, shared_std):
        y = np.random.normal(local_mean, shared_std)

        return {"y": float(y)}

    simulator = GraphicalSimulator()
    simulator.add_node("hypers", sampling_fn=sample_hypers, reps=1)

    simulator.add_node(
        "locals",
        sampling_fn=sample_locals,
        reps=6,
    )
    simulator.add_node("shared", sampling_fn=sample_shared, reps=1)
    simulator.add_node("y", sampling_fn=sample_y, reps=10)

    simulator.add_edge("hypers", "locals")
    simulator.add_edge("locals", "y")
    simulator.add_edge("shared", "y")

    return simulator
