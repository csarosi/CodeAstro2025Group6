import numpy as np
import pytest
from scipy.stats import uniform, norm, gamma
from emcee_rapper.mcmcwrapper import MCMCWrapper

# Dummy model function
def dummy_model(params, x):
    return np.zeros_like(x)

@pytest.mark.parametrize("priortype, bounds, dist_fn", [
    ("uniform", [[0, 1]], lambda s: uniform.rvs(loc=0, scale=1, size=s)),
    ("normal", [[1, 1]], lambda s: norm.rvs(loc=1, scale=1, size=s)),
    ("gamma", [[1, 1]], lambda s: gamma.rvs(a=1, scale=1, size=s)),
])
def test_sample_priors_distribution(priortype, bounds, dist_fn):
    """
    Test that sample_priors generates distributions matching expected priors.
    """
    nsamples = 100000
    data = np.zeros(10)
    x = np.zeros(10)
    parnames = ["param"]
    initial_values = [0.5]

    mcmc = MCMCWrapper(
        model_function=dummy_model,
        data=data,
        x=x,
        varnames=parnames,
        varvalues=initial_values,
        priorvars=bounds,
        priortype=priortype
    )

    samples = mcmc.sample_priors(nsamples)
    assert samples.shape == (nsamples, 1)

    sampled = samples[:, 0]
    expected = dist_fn(nsamples)

    # Test that sampled and expected distributions have similar mean and std
    np.testing.assert_allclose(np.mean(sampled), np.mean(expected), rtol=0.1)
    np.testing.assert_allclose(np.std(sampled), np.std(expected), rtol=0.1)
