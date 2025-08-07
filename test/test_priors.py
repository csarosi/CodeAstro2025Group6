import numpy as np
import pytest
from scipy.stats import uniform, norm, gamma, ks_2samp

from emcee_rapper.mcmcwrapper import MCMCWrapper

# Dummy model function
def dummy_model(params, x):
    return np.zeros_like(x)

@pytest.mark.parametrize("priortype, bounds_list, dist_fn_list", [
    ("uniform",
     [[[0, 1]], [[-1, 1]], [[2, 4]]],
     [
         lambda s: uniform.rvs(loc=0, scale=1, size=s),
         lambda s: uniform.rvs(loc=-1, scale=2, size=s),
         lambda s: uniform.rvs(loc=2, scale=2, size=s),
     ]),
    #("normal",
     #[[[0, 1]], [[-2, 2]], [[2, 3]]],
     #[
     #    lambda s: norm.rvs(loc=0, scale=1, size=s),
     #    lambda s: norm.rvs(loc=0, scale=2, size=s),
     #    lambda s: norm.rvs(loc=2, scale=3, size=s),
     #]),
    #("gamma",
     #[[[2, 2]], [[5, 1]], [[9, 0.5]]],
     #[
     #    lambda s: gamma.rvs(a=2, scale=2, size=s),
     #    lambda s: gamma.rvs(a=5, scale=1, size=s),
     #    lambda s: gamma.rvs(a=9, scale=0.5, size=s),
     #]),
])
def test_sample_priors_distribution(priortype, bounds_list, dist_fn_list):
    """
    Test that sample_priors generates distributions matching expected priors.
    KS test for distribution similarity.
    """
    nsamples = 10000
    data = np.zeros(10)
    x = np.zeros(10)
    parnames = ["param"]
    initial_values = [0.5]

    for bounds, dist_fn in zip(bounds_list, dist_fn_list):
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

        # Compare mean and std
        np.testing.assert_allclose(np.mean(sampled), np.mean(expected), rtol=0.1)
        np.testing.assert_allclose(np.std(sampled), np.std(expected), rtol=0.1)

        # KS test
        ks_stat, ks_pvalue = ks_2samp(sampled, expected)
        assert ks_pvalue > 0.05, f"KS test failed for {priortype} with bounds {bounds}"
