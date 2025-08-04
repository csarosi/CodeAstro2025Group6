import numpy as np
import emcee

def chi2_lnprob(data, model,sigma2):
        return -0.5 * np.sum((data - model) ** 2 / sigma2 + np.log(sigma2))
def lnprob(parameter_values,root_dir, parameter_dict, constants_dict, line_list):
    """
    Calculate the log-probability of the model given the parameters and data.

    This function evaluates the log-probability.
    It begins by applying the prior. After that, it simulates
    the model with the given parameters and calculates the chi-squared statistic
    from the observations. The log-probability is then computed as -0.5 * the summed chi-squared values.

    Parameters
    ----------
    parameter_values : list
        List of parameter values for the model step.
    parameter_names : list
        List of parameter names corresponding to the values.
    constants_dict : dict
        Dictionary containing constant values for simulation parameters.
    bounds_dict : dict
        Dictionary specifying the lower and upper bounds for each parameter.
    obs_dict : dict
        Dictionary containing Observation objects grouped by line

    Returns
    -------
    float
        The log-probability of the model given the parameters, or negative
        infinity if any parameter is out of bounds.
    """

    simulation_dict = constants_dict.copy()
    
    for i, name in enumerate(parameter_dict.keys()):
        simulation_dict[name] = parameter_values[i]
    for key in parameter_dict.keys():
        # print(key,simulation_dict[key])
        key_bound = parameter_dict[key]['bounds']
        if (simulation_dict[key] < key_bound[0]) or (simulation_dict[key] > key_bound[1]):
            #print(key + " out of bounds")
            return -np.inf

    return chi2_lnprob(data, model,sigma2)

def MCMC(write_csv, root_dir,constants_dict, param_dict, obs_list, #pool,
        nsteps=3000,nwalkers=40, restart=False, read_csv = None):

    """
    #https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
    This function performs Affine Invariant MCMC sampling using the
    emcee library to fit a disk model to observational data. It initializes
    walkers in parameter space and iteratively samples to find the best-fit
    parameters.

    Parameters:

    Returns:
    - None

    - The function requires the emcee library for MCMC sampling.
    """
    ndim = len(param_dict)
    p0 = np.random.normal(loc=[param_dict[name]['mean'] for name in param_dict],
                            size=(nwalkers, ndim),
                            scale=[param_dict[name]['stdev'] for name in param_dict])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[model])
    run = sampler.sample(p0, iterations=nsteps, skip_initial_state_check=True, store=True) #If you are using less than 15 walkers (for testing purposes, presumably),
    # set 'skip_initial_state_check' to true. This avoids an error claiming the walkers are not linearly independent.
    # It is just from the low amount of walkers used for testing; it can be turned off for testing, but probably not a good idea when running a chain.
    

    # steps=[]
    # colnames = [name for name in param_dict.keys()]
    # colnames.append('lnprobs')

    # if write_csv == read_csv:
    #     write_csv = write_csv.split('.')[0] + '_restart.csv'

    # for i, result in enumerate(run):
    #     pos, lnprobs, blob = result
    #     new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
    #     steps += new_step
    #     df = pd.DataFrame(steps)
    #     df.columns = colnames
    #     df.to_csv(root_dir + '/' +write_csv, mode='w')
    #     t2 = time.perf_counter()
    #     print('step time: ',(t2-t1), 's')
    #     # print(str(len(new_step)) + 'new steps')
    #     t1 = t2
    #     sys.stdout.write('completed step {} out of {} \r'.format(i, nsteps) )
        
    #     sys.stdout.flush()

    print("Finished MCMC.")
    print("Mean acceptance fraction: {0:3f}".format(np.mean(sampler.acceptance_fraction)))