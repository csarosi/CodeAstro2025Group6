import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, gamma, kstest

from emcee_rapper.mcmcwrapper import MCMCWrapper


# Dummy model function
def dummy_model(params, x):
    return np.zeros_like(x)


def test_sample_priors_distribution(priortype, bounds, dist_fn, dist_name):
    nsamples = 10000
    data = np.zeros(10)
    x = np.zeros(10)
    parnames = ["param"]
    initial_values = [0.5]

    mcmc = MCMCWrapper(
        model_function=dummy_model,
        data=data,
        x=x,
        parnames=parnames,
        initial_values=initial_values,
        prior_bounds=bounds,
        priortype=priortype
    )

    samples = mcmc.sample_priors(nsamples)
    sampled = samples[:, 0]
    expected = dist_fn(nsamples)

    # Compare means and standard deviations
    print(f"\nTesting {priortype.capitalize()} Distribution")
    print(f"Sampled mean: {np.mean(sampled):.4f}, Expected mean: {np.mean(expected):.4f}")
    print(f"Sampled std : {np.std(sampled):.4f}, Expected std : {np.std(expected):.4f}")

    # Perform KS test
    ks_stat, ks_pvalue = kstest(sampled, dist_name, args=(dist_fn.keywords.get('loc', 0), dist_fn.keywords.get('scale', 1)) if hasattr(dist_fn, 'keywords') else ())
    print(f"KS Statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")


# Run tests
test_sample_priors_distribution("uniform", [[0, 1]], lambda s: uniform.rvs(loc=0, scale=1, size=s), "uniform")
test_sample_priors_distribution("normal", [[0, 1]], lambda s: norm.rvs(loc=0, scale=1, size=s), "norm")
test_sample_priors_distribution("gamma", [[2, 2]], lambda s: gamma.rvs(a=2, scale=2, size=s), "gamma")
