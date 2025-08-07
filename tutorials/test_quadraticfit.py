import emcee_rapper.mcmcwrapper as rap
import numpy as np
import matplotlib.pyplot as plt


def test_quadfit():
    """Test that the code do all the plotting."""
    def quadratic_fit(pars, x):
        a, b, c = pars
        return a * x**2 + b * x + c

    true_params = [2.0, -1.0, 0.5]
    x_data = np.linspace(-5, 5, 100)
    y_true = quadratic_fit(true_params, x_data)
    np.random.seed(42)
    y_obs = y_true + np.random.normal(0, 1.0, size=len(x_data))

    varnames = ["a", "b", "c"]
    varvalues = [1.0, 0.0, 0.0]
    priorvars = [[-10, 10], [-10, 10], [-10, 10]]
    noise_std = 1.0

    wrapper = rap.MCMCWrapper(
        model_function=quadratic_fit,
        data=y_obs,
        x=x_data,
        varnames=varnames,
        varvalues=varvalues,
        priorvars=priorvars,
        noise=noise_std
    )

    sampler = wrapper.run_mcmc(nwalkers=30, nsteps=5000)
    samples = sampler.get_chain(discard=200, thin=15, flat=True)

    medians = np.median(samples, axis=0)

    fig, ax = plt.subplots()
    ax.plot(x_data,y_obs, 'o')
    ax.plot(x_data, quadratic_fit([medians[0], medians[1], medians[2]], x_data), 'k--', lw=5)
    plt.xlabel('X', size=16)
    plt.ylabel('Y', size=16)
    plt.show()

    wrapper.walker_plot(discard=500)
    plt.show()
    wrapper.corner_plot(discard = 500)
    plt.show()

    assert sampler is not None
    pass
