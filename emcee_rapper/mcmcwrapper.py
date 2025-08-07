import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma
from scipy.interpolate import interp1d as interp1d
import corner

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
        
    data : array
        Observed data to which the model will be fit.
        
    x : array
        Independent variable(s) corresponding to the observed data.
        
    varnames : list of str
        Names of the parameters to be sampled.
        
    varvalues : array
        Initial guesses for the model parameters.
        
    priorvars : array
        List of [min, max] pairs for each parameter, specifying uniform prior bounds.
        If priortype='uniform', interpreted as upper and lower bounds for each parameter
        If priortype='normal', interpreted as median and standard deviation of each parameter's gaussian prior
        If priortype='gamma', interpreted as the shape parameter and the scale parameter of a gamma distribution
        
    noise : float or array, optional
        The standard deviation of the noise in the data. If a single float is provided, 
        it is broadcast to match the shape of `data`. Default is 1.0.
        
    sample : array of bool, optional
        Boolean array indicating which parameters to sample. Default is to sample all parameters.
    
    priortype : string OR list of strings, optional (default='uniform')
        string denoting functional form assumed for the priors (options: 'uniform', 'normal', 'gamma')
        if list, one of 'uniform', 'normal', and 'gamma' for each parameter.
    """
    
    def __init__(self, model_function, data, x, varnames, varvalues,
                 priorvars,priortype='uniform', noise=1.0, sampleparams=None, ):
        self.model_function = model_function
        self.data = np.array(data)
        self.x = np.array(x)
        self.sampleparams = np.ones(len(varnames), dtype=bool) if sampleparams is None else np.array(sampleparams)
        assert len(varnames) == len(varvalues) == len(self.sampleparams), "your parameter inputs are not all the same length!"
        
        varnames =  np.array(varnames)
        self.parnames = varnames[self.sampleparams]
        
        self.parorder = np.concatenate((np.argwhere(self.sampleparams),np.argwhere(~self.sampleparams)))

        varvalues =  np.array(varvalues)
        self.fixedvals = varvalues[~self.sampleparams]
        self.p0 = varvalues[self.sampleparams]
       
        self.npars = len(self.parnames)
        self.noise = noise * np.ones_like(self.data)

        self.priortype = priortype
        self.priorvars=priorvars

        if type(self.priortype) != str:
            def log_prior(params):
                """
                Returns the log of the prior probability given the parameter list.
                Priors for each parameter are set by the prior bound and the prior_type argument
                prior_type can be a single string or a list of strings with length equal to the 
                number of parameters, chosen from "uniform", "normal", and "gamma".

                Parameters
                ----------
                params : array
                    Array of parameter values.

                Returns
                -------
                float
                    The log-prior probability.
                """
                Ps = np.zeros(self.npars)
                for i in range(len(priortype)):
                    if self.priortype[i] == "uniform":
                        Ps[i] = 1.0

                        for i in range(self.npars):
                            if (params[i] <= self.priorvars[i][0]) or (params[i] >= self.priorvars[i][1]):
                                Ps[i] = 0.0
                    elif self.priortype[i] == "normal":
                        Ps[i] = 1/np.sqrt(2*np.pi*self.priorvars[i][1]) * \
                         np.exp(-(params[i]-self.priorvars[i][0])**2/(2*self.priorvars[i][1]**2))
                    elif self.priortype[i] == "gamma":
                        Ps[i] = 1/(gamma(self.priorvars[i][0]) * self.priorvars[i][1]**self.priorvars[i][0]) * \
                         params[i]**(self.priorvars[i][0]-1) * np.exp(-params[i]/self.priorvars[i][1])
                        if params[i] < 0:
                            Ps[i]=0.0
                    else:
                        raise Exception("The three options for priortype are 'uniform', 'normal', and 'gamma'")
                logP = np.sum(np.where(Ps>0,np.log(Ps),-np.inf))
                return logP, Ps
        else:
            if self.priortype=='uniform':
                def log_prior(params):
                    """
                    Computes the log-prior probability of the parameters assuming uniform priors.

                    Parameters
                    ----------
                    params : array
                        Array of parameter values.

                    Returns
                    -------
                    float
                        The log-prior probability. Returns -np.inf if any parameter is outside its bounds.
                    """
                    logPs = np.zeros(self.npars)
                    for i in range(self.npars):
                        if (params[i] <= self.priorvars[i][0]) or (params[i] >= self.priorvars[i][1]):
                            logPs[i] = -np.inf
                    logP = np.sum(logPs)
                    return logP, logPs
            elif self.priortype=='normal':
                def log_prior(params):
                    """
                    Computes the log-prior probability of the parameters assuming normal priors.
                    Assumes median=bounds[0] and standard deviation=bounds[1]

                    Parameters
                    ----------
                    params : array
                        Array of parameter values.

                    Returns
                    -------
                    float
                        The log-prior probability.
                    """
                    logPs = [1/np.sqrt(2*np.pi*self.priorvars[i][1]) * \
                            np.exp(-(params[i]-self.priorvars[i][0])**2/(2*self.priorvars[i][1]**2)) \
                                for i in range(self.npars)]
                    logP = np.sum(np.log(logPs))
                    return logP, logPs
            elif self.priortype=='gamma':
                def log_prior(params):
                    """
                    Computes the log-prior probability of the parameters assuming normal priors.
                    Gamma function prior assuming shape parameter=bounds[0] and scale parameter=bounds[1]

                    Parameters
                    ----------
                    params : array
                        Array of parameter values.

                    Returns
                    -------
                    float
                        The log-prior probability.
                    """
                    logPs = [1/(gamma(self.priorvars[i][0]) * self.priorvars[i][1]**self.priorvars[i][0]) * \
                            params[i]**(self.priorvars[i][0]-1) * np.exp(-params[i]/self.priorvars[i][1])\
                                for i in range(self.npars)]
                    for i in range(self.npars):
                        if params[i] < 0.0:
                            logPs[i] = 0.0
                    logP = np.sum(np.log(logPs))
                    return logP, logPs

            else:
                raise Exception("The three options for priortype are 'uniform', 'normal', and 'gamma'")
        self.log_prior = log_prior

        assert len(self.p0) == len(self.priorvars) == self.npars, "your parameter inputs are not all the same length!"

    def log_likelihood(self, params):
        """
        Computes the log-likelihood of the data given the model parameters.

        Parameters
        ----------
        params : array
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
        params : array
            Array of parameter values.

        Returns
        -------
        float
            The log-posterior probability (log-prior + log-likelihood).
        """
        lp = self.log_prior(params)[0]
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
        initial_pos = self.p0 + 1e-4 * np.random.randn(nwalkers, self.npars)
        mcmc_sampler = emcee.EnsembleSampler(nwalkers, self.npars, self.log_posterior)
        mcmc_sampler.run_mcmc(initial_pos, nsteps, progress=True)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.mcmc_sampler = mcmc_sampler
        samples = mcmc_sampler.get_chain(discard=200, thin=15, flat=True)

        medians = np.median(samples, axis=0)
        lower = np.percentile(samples, 16, axis=0)
        upper = np.percentile(samples, 84, axis=0)

        for i, name in enumerate(self.parnames):
            med = medians[i]
            lo = med - lower[i]
            hi = upper[i] - med
            print(f"{name}: {med:.3f} (+{hi:.3f}/-{lo:.3f})")
        
        return mcmc_sampler
    
    def walker_plot(self, discard=100):
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
        labels = list(self.parnames)
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
    
    def corner_plot(self, discard=100):
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
                title_kwargs={"fontsize": 12},  # Font size for titles
                label_kwargs={"fontsize": 16} ) # Font size for axis labels
    def sample_priors(self, nsamples):
        """Samples the prior distributions with inverse transform sampling, gives back nsamples draws
        Parameters
        ----------
        nsamples : int
            Number of samples to draw. Retuned array will have shape (nsamples, npars)
        
        Returns
        -------
        samples : array-like
            random samples from the prior probability distributions
        """
        samples = np.empty((nsamples, self.npars))
        for i in range(self.npars):
            if type(self.priortype) != str:
                priortype = self.priortype[i]
            else:
                priortype = self.priortype
            if priortype == 'uniform':
                samples[:,i] = np.random.rand(nsamples) * (self.priorvars[i][1] - self.priorvars[i][0]) + self.priorvars[i][0]
            else:
                if priortype=='normal':
                    xbnds = self.priorvars[i][0] - 5 * self.priorvars[i][1], self.priorvars[i][0] + 5 * self.priorvars[i][1]
                elif priortype=='gamma':
                    xbnds = 0, self.priorvars[i][0]*5
                x = np.linspace(xbnds[0], xbnds[1], 10000)
                Ps_of_x = []
                for j in range(len(x)):
                    params = np.zeros(self.npars)
                    params[i] = x[j]
                    P = self.log_prior(params)[1][i]
                    Ps_of_x.append(P)
                cdf = np.cumsum(Ps_of_x)
                cdf /= cdf[-1]
                interpolant = interp1d(cdf,x,bounds_error=False, fill_value=xbnds)
                for n in range(nsamples):
                    samp = np.random.rand()
                    samples[n,i] = interpolant(samp)
        return samples

