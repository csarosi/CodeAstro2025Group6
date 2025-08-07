import numpy as np
import emcee
import matplotlib.pyplot as plt

class MCMCWrapper:
    """
    A wrapper class for performing Markov Chain Monte Carlo (MCMC) sampling 
    using the `emcee` library.

    This class allows users to estimate model parameters based on observed data, 
    given a model function, priors, and optional noise estimates.

    Parameters
    ----------
    model_function : callable
        A function that takes a parameter vector and an array of independent 
        variable(s), and returns the model output.
        
    data : array_like
        Observed data to which the model will be fit.
        
    x : array_like
        Independent variable(s) corresponding to the observed data.
        
    parnames : list of str
        Names of the parameters to be sampled.
        
    initial_values : array_like
        Initial guesses for the model parameters.
        
    prior_bounds : array_like
        List of [min, max] pairs for each parameter, specifying uniform prior bounds.
        
    noise : float or array_like, optional
        The standard deviation of the noise in the data. If a single float is provided, 
        it is broadcast to match the shape of `data`. Default is 1.0.
        
    sample : array_like of bool, optional
        Boolean array indicating which parameters to sample. Default is to sample all parameters.
    """

    def __init__(self, model_function, data, x, varnames, varvalues, priortypes,
                 priorvars, sampleparams = None, noise=1.0):
        
        self.model_function = model_function
        self.data = np.array(data)
        self.x = np.array(x)
        
        self.sampleparams = np.ones(len(varnames), dtype=bool) if sampleparams is None else np.array(sampleparams)
        assert len(varnames) == len(varvalues) == len(sampleparams), "your parameter inputs are not all the same length!"
        
        varnames =  np.array(varnames)
        self.parnames = varnames[self.sampleparams]
        
        self.parorder = np.concatenate((np.argwhere(self.sampleparams),np.argwhere(~self.sampleparams)))

        varvalues =  np.array(varvalues)
        self.fixedvals = varvalues[~self.sampleparams]
        self.p0 = varvalues[self.sampleparams]
  
        self.priortypes = priortypes
        self.priorvars = priorvars

        self.npars = len(self.parnames)
        self.noise = noise * np.ones_like(self.data)
        assert len(self.parnames) == len(self.priortypes) == len(self.priorvars), "You don't have the right number of priors your parameters!"

    def log_prior(self, params):
        """
        Computes the log-prior probability of the parameters assuming uniform priors.

        Parameters
        ----------
        params : array_like
            Array of parameter values.

        Returns
        -------
        float
            The log-prior probability. Returns -np.inf if any parameter is outside its bounds.
        """
        logP = 0
        for i in range(self.npars):
            if (self.p0[i] < self.priorvars[i][0]) | (self.p0[i] > self.priorvars[i][1]):
                logP = -np.inf
        return logP

    def log_likelihood(self, params):
        """
        Computes the log-likelihood of the data given the model parameters.

        Parameters
        ----------
        params : array_like
            Array of parameter values.

        Returns
        -------
        float
            The log-likelihood assuming Gaussian noise.
        """
        y_model = self.model_function(np.concatenate((params, self.fixedvals))[self.parorder], self.x)
        chi2 = np.sum((self.data - y_model) ** 2 / self.noise**2)
        return -0.5 * chi2

    def log_posterior(self, params):
        """
        Computes the log-posterior probability of the parameters.

        Parameters
        ----------
        params : array_like
            Array of parameter values.

        Returns
        -------
        float
            The log-posterior probability (log-prior + log-likelihood).
        """
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)

    def run_mcmc(self, nwalkers=50, nsteps=1000):
        """
        Runs the MCMC sampler using the `emcee` EnsembleSampler.

        Parameters
        ----------
        nwalkers : int, optional
            Number of walkers to use in the ensemble. Default is 50.
            
        nsteps : int, optional
            Number of MCMC steps for each walker. Default is 1000.

        Returns
        -------
        sampler : emcee.EnsembleSampler
            The MCMC sampler object containing the full chain of samples.
        """
        initial_pos = self.p0 + 1e-4 * np.random.randn(nwalkers, self.npars)
        mcmc_sampler = emcee.EnsembleSampler(nwalkers, self.npars, self.log_posterior)
        mcmc_sampler.run_mcmc(initial_pos, nsteps, progress=True)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.mcmc_sampler = mcmc_sampler
        
        return mcmc_sampler
    
    def walker_plot(self, discard = 200):
        if not hasattr(self, 'mcmc_sampler'):
            print("you can't plot your walkers yet, you still need to call run_mcmc!")
            return
        fig, axes = plt.subplots(self.npars+1, figsize=(10, 7), sharex=True)
        labels = list(self.parnames)
        labels.append("log prob")
        chain_pars = self.mcmc_sampler.get_chain(discard = discard)
        chain_log_probs = self.mcmc_sampler.get_log_prob(discard=discard)
        chain = np.dstack((chain_pars,chain_log_probs))
        for i in range(self.npars+1):
            ax = axes[i]
            ax.fill_between(range(0,len(chain[:, :, i])),
                            np.percentile(chain[:, :, i], 16, axis=1),
                            np.percentile(chain[:, :, i], 84, axis=1), color = 'k', alpha = 0.5)
            ax.plot(chain[:, :, i], alpha=0.2)
            ax.plot(np.median(chain[:, :, i], axis=1), alpha=1, color = 'k')
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        plt.show()
