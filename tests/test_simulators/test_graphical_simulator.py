import numpy as np

import bayesflow as bf


def test_single_level_simulator(single_level_simulator):
    assert isinstance(single_level_simulator, bf.experimental.graphical_simulator.GraphicalSimulator)
    assert isinstance(single_level_simulator.sample(5), dict)

    samples = single_level_simulator.sample((12,))
    expected_keys = ["N", "beta", "sigma", "x", "y"]

    assert set(samples.keys()) == set(expected_keys)
    assert 5 <= samples["N"] < 15
    assert np.shape(samples["beta"]) == (12, 2)  # num_samples, beta_dim
    assert np.shape(samples["sigma"]) == (12, 1)  # num_samples, sigma_dim
    assert np.shape(samples["x"]) == (12, samples["N"])
    assert np.shape(samples["y"]) == (12, samples["N"])
