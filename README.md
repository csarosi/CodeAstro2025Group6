### emcee_rapper

Welcome to ``emcee_rapper``! ``emcee_rapper`` is a small python package developed at code/astro 2025 that wraps the popular Markov chain Monte Carlo sampling code emcee to make fitting with it even more straightforward. It can help you set up priors and likelihood functions, pass them to emcee, and visualize the fit results.

We are pip installable! ``pip install emcee_rapper`` should do the trick

All you need to start is some data (any dimensionality works) and a function that takes data and parameters and returns synthetic data! You can specify initial guesses the parameters you want to explore, priors (uniform, normal, gamma are currently implemented), and fixed values needed for your model. 

Once you have run your models, you can call the built-in walker and corner plots 

![](https://github.com/csarosi/emcee_rapper/blob/main/readme_images/Corner_linear.png "Example Walker Plot")

![](https://github.com/csarosi/emcee_rapper/blob/main/readme_images/MarkovChains_linear.png "Example Corner Plot")


