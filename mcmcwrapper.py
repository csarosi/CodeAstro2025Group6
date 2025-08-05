import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma

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
        If priortype='uniform', interpreted as upper and lower bounds for each parameter
        If priortype='normal', interpreted as median and standard deviation of each parameter's gaussian prior
        If priortype='gamma', interpreted as the shape parameter and the scale parameter of a gamma distribution
        
    noise : float or array_like, optional
        The standard deviation of the noise in the data. If a single float is provided, 
        it is broadcast to match the shape of `data`. Default is 1.0.
        
    sample : array_like of bool, optional
        Boolean array indicating which parameters to sample. Default is to sample all parameters.
    
    priortype : string, optional (default='uniform')
        string denoting functional form assumed for the priors (options: 'uniform', 'normal', 'gamma')
    """
    
    def __init__(self, model_function, data, x, parnames, initial_values,
                 prior_bounds, noise=1.0, sample=None, priortype='uniform'):
        self.model_function = model_function
        self.data = np.array(data)
        self.x = np.array(x)
        self.parnames = parnames
        self.p0 = np.array(initial_values)
        self.bounds = np.array(prior_bounds)
        self.ndim = len(parnames)
        self.npars = len(parnames)
        self.noise = noise * np.ones_like(self.data)
        self.sample = np.ones(len(parnames), dtype=bool) if sample is None else sample
        self.priortype = priortype

        if self.priortype=='uniform':
            def log_prior(params):
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
                    if (params[i] < self.bounds[i][0]) or (params[i] > self.bounds[i][1]):
                        return -np.inf
                return logP
        elif self.priortype=='normal':
            def log_prior(params):
                """
                Computes the log-prior probability of the parameters assuming normal priors.
                Assumes median=bounds[0] and standard deviation=bounds[1]

                Parameters
                ----------
                params : array_like
                    Array of parameter values.

                Returns
                -------
                float
                    The log-prior probability.
                """
                logPs = [1/np.sqrt(2*np.pi*self.bounds[i][1]) * \
                         np.exp((params[i]-self.bounds[i][0])**2/(2*self.bounds[i][1]**2)) \
                            for i in range(self.ndim)]
                logP = np.sum(np.log(logPs))
                return logP
        elif self.priortype=='gamma':
            def log_prior(params):
                """
                Computes the log-prior probability of the parameters assuming normal priors.
                Gamma function prior assuming shape parameter=bounds[0] and scale parameter=bounds[1]

                Parameters
                ----------
                params : array_like
                    Array of parameter values.

                Returns
                -------
                float
                    The log-prior probability.
                """
                logPs = [1/(gamma(self.bounds[i][0]) * self.bounds[i][1]**self.bounds[i][0]) * \
                         params[i]**(self.bounds[i][0]-1) * np.exp(-params[i]/self.bounds[i][1])\
                              for i in range(self.ndim)]
                logP = np.sum(np.log(logPs))
                return logP

        else:
            raise Exception("The three options for priortype are 'uniform', 'normal', and 'gamma'")
        self.log_prior = log_prior

        assert len(self.p0) == len(self.bounds) == self.ndim, "your parameter inputs are not all the same length!"

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
        y_model = self.model_function(params, self.x)
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
        mcmc_sampler : emcee.EnsembleSampler
            The MCMC sampler object containing the full chain of samples.
        """
        initial_pos = self.p0 + 1e-4 * np.random.randn(nwalkers, self.ndim)
        mcmc_sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior)
        mcmc_sampler.run_mcmc(initial_pos, nsteps, progress=True)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.mcmc_sampler = mcmc_sampler
        
        return mcmc_sampler
    
    def walker_plot(self, discard=200):
        """
        Plot the evolution of MCMC walker chains for each parameter and the log-probability.

        This function generates a diagnostic plot showing how each parameter and the log-probability
        evolve over the course of the MCMC run. The plot includes:

        - Light-colored traces for all walkers
        - A black line showing the median value at each step
        - A shaded region between the 16th and 84th percentiles

        Parameters
        ----------
        discard : int, optional
            Number of initial steps to discard from the chain as burn-in. Default is 200.

        Notes
        -----
        This method assumes that `self.sampler` has been defined by calling `run_mcmc` beforehand.
        The number of subplots is equal to the number of model parameters plus one for the 
        log-probability. The function displays the plot using `plt.show()`.
        """

        # Create a figure with one subplot for each parameter plus one for log-probability
        fig, axes = plt.subplots(self.npars + 1, figsize=(10, 7), sharex=True)
    
        # Create a list of labels: parameter names + one for log-probability
        labels = self.parnames
        labels.append("log prob")  # This mutates self.parnames (not ideal, but kept as-is)
    
        # Extract the parameter chains, discarding the initial burn-in steps
        chain_pars = self.mcmc_sampler.get_chain(discard=discard)
    
        # Extract the log-probability chain, also discarding burn-in
        chain_log_probs = self.mcmc_sampler.get_log_prob(discard=discard)
    
        # Stack parameters and log-probability together into a 3D array
        # Shape: (steps, walkers, npars + 1)
        chain = np.dstack((chain_pars, chain_log_probs))
    
        # Loop over each parameter (and log-probability) to plot
        for i in range(self.npars + 1):
            ax = axes[i]
    
            # Plot a shaded region representing the 16th–84th percentile range
            ax.fill_between(
                range(0, len(chain[:, :, i])),
                np.percentile(chain[:, :, i], 16, axis=1),
                np.percentile(chain[:, :, i], 84, axis=1),
                color='k', alpha=0.5
            )
    
            # Plot all walker chains with low opacity
            ax.plot(chain[:, :, i], alpha=0.2)
    
            # Plot the median value of the walkers at each step
            ax.plot(np.median(chain[:, :, i], axis=1), alpha=1, color='k')
    
            # Label the y-axis with the parameter name or "log prob"
            ax.set_ylabel(labels[i])
    
        # Label the x-axis of the bottom plot
        axes[-1].set_xlabel("Step")
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
    
        # Display the figure
        plt.show()
    
        def corner_plot(self, discard=200):
            """
            Generate a corner plot of the MCMC samples after discarding initial steps (burn-in).
            
            Parameters
            ----------
            discard : int, optional
                Number of initial MCMC steps to discard as burn-in. Default is 200.
    
            Notes
            -----
            - This function flattens the MCMC chains across all walkers after discarding burn-in.
            - Requires the `corner` library.
            - Assumes that `self.mcmc_sampler` is the emcee sampler object created by `run_mcmc`.
            """
            # Flatten the chain (combine steps and walkers), discarding the burn-in
            samples = self.mcmc_sampler.get_chain(discard = discard, flat=True)
            # Create the corner plot showing marginal and joint posteriors
            figure = corner.corner(
                    samples,                       
                    quantiles=[0.16, 0.5, 0.84],    # Set the quantiles
                    labels = self.parnames,         # Label each parameter
                    show_titles=True,               # Show titles in each panel
                    title_kwargs={"fontsize": 16},  # Font size for titles
                    label_kwargs={"fontsize": 16} ) # Font size for axis labels

    
