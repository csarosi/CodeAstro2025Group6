import emcee_rapper.mcmcwrapper as rap
import numpy as np

import matplotlib.pyplot as plt

def model(pars, x):
    a, b, c = pars
    return a * x**2 + b * x + c

true_params = [2.0, -1.0, 0.5]
x_data = np.linspace(-5, 5, 100)
y_true = model(true_params, x_data)
np.random.seed(42)
y_obs = y_true + np.random.normal(0, 1.0, size=len(x_data))

parnames = ["a", "b", "c"]
initial_guess = [1.0, 0.0, 0.0]
prior_bounds = [[-10, 10], [-10, 10], [-10, 10]]
noise_std = 1.0

wrapper = rap.MCMCWrapper(
    model_function=model,
    data=y_obs,
    x=x_data,
    varnames=parnames,
    varvalues=initial_guess,
    priortype = ['uniform']*2,
    priorvars=prior_bounds[:-1],
    sampleparams = [True, False, True],
    noise=noise_std)

wrapper.run_mcmc(nwalkers=30, nsteps=1000)
assert list(wrapper.parnames) == ['a','c']
assert list(wrapper.p0) == [1.,0.]