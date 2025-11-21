import numpy as np
from ...simulators import make_simulator
from ...adapters import Adapter
from scipy.integrate import odeint

rng = np.random.default_rng()


def theta_prior():
    theta = np.random.uniform(-1, 1, 2)
    return dict(theta=theta)


def forward_model(theta):
    alpha = np.random.uniform(-np.pi / 2, np.pi / 2)
    r = np.random.normal(0.1, 0.01)
    x1 = -np.abs(theta[0] + theta[1]) / np.sqrt(2) + r * np.cos(alpha) + 0.25
    x2 = (-theta[0] + theta[1]) / np.sqrt(2) + r * np.sin(alpha)
    return dict(x=np.array([x1, x2]))


def simulator():
    simulator = make_simulator([theta_prior, forward_model])

    return simulator


def adapter():
    adapter = (
        Adapter()
        # convert any non-arrays to numpy arrays
        .to_array()
        # convert from numpy's default float64 to deep learning friendly float32
        .convert_dtype("float64", "float32")
        # rename the variables to match the required approximator inputs
        .rename("theta", "inference_variables")
        .rename("x", "inference_conditions")
    )

    return adapter
