import emcee_rapper.mcmcwrapper as rap
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_fit_2D():
    """Test that the code can fit a 2D gaussian with no noise"""
    def model(pars, x):
        x0, x1 = x
        a, b, c, d = pars
        return np.exp(-(x0 - a)/(b**2)) * np.exp(-(x1-c)**2/(d**2))

    true_params = np.array([1.0, 5.0, 1.0, 5.0])
    x0_data = np.linspace(-5, 5, 100)
    x1_data = np.linspace(-5, 5, 100)

    x0_coords, x1_coords = np.meshgrid(x0_data, x1_data)
    x0_coords, x1_coords = x0_coords.ravel(), x1_coords.ravel()
    coords = np.array([x0_coords, x1_coords])
    y_true = model(true_params, coords)
    np.random.seed(42)
    y_obs = y_true #+ np.random.normal(0, 10.0, size=len(coords.T))

    parnames = ["a", "b", "c", "d"]
    initial_guess = [0.0, 1.0, 0.0, 1.0]
    prior_bounds = [[-10., 10.], [0., 10.], [-10., 10.], [0., 10.]]
    # prior_bounds = [[1., 1.], [1., 1.], [1., 1.]]
    noise_std = 1.0

    wrapper = rap.MCMCWrapper(
        model_function=model,
        data=y_obs,
        x=coords,
        varnames=parnames,
        varvalues=initial_guess,
        priorvars=prior_bounds,
        noise=noise_std,
        priortype=['uniform', 'uniform', 'uniform', 'uniform']
        # priortype='normal'
    )

    

    sampler = wrapper.run_mcmc(nwalkers=30, nsteps=1000)
    samples = sampler.get_chain(discard=200, thin=15, flat=True)

    medians = np.median(samples, axis=0)
    lower = np.percentile(samples, 16, axis=0)
    upper = np.percentile(samples, 84, axis=0)
    # print(lower, true_params, upper)
    assert (np.array([lower[i] < true_params[i] < upper[i] for i in range(wrapper.npars)])).all()
    pass

test_fit_2D()
