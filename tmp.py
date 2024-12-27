import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import pprint
from scipy.optimize import minimize
import scipy.stats
import seaborn as sns
from typing import Any, Dict, List, Tuple

import src.analyze


large_language_monkeys_pass_at_k_df = src.analyze.create_or_load_large_language_monkeys_original_pass_at_k_df(
    refresh=False,
    # refresh=True,
)

large_language_monkeys_pass_at_1_df = large_language_monkeys_pass_at_k_df[
    (large_language_monkeys_pass_at_k_df["Scaling Parameter"] == 1)
    & (large_language_monkeys_pass_at_k_df["Model"] == "Pythia 12B")
    & (large_language_monkeys_pass_at_k_df["Benchmark"] == "MATH")
].copy()


def estimate_params(data, n):
    # Inputs:
    # data: an 1-D array of length number of problems
    # each entry in data should give the number of correct answers
    # n: an integer representing the number of attempts per problem

    # Outputs:
    # alpha and beta: floats representing the MLEs for alpha and beta

    # define the nll of the distribution
    def nll(params):
        # Inputs:
        # a, b: floats: alpha, beta for negative binomial distribution
        # Outputs:
        # float: the mean nll
        a, b = params
        fit_dist = scipy.stats.betabinom(n, a, b)
        pdf_values = fit_dist.logpmf(data)
        return -pdf_values.mean()

    # initialize a random guess for alpha and beta
    initial_guess = [np.random.rand(), np.random.rand()]

    # minimize the nll
    result = minimize(nll, initial_guess)
    fitted_params = result.x
    return fitted_params


def estimate_params_with_scale(data, n):
    # Inputs:
    # data: an 1-D array of length number of problems
    # each entry in data should give the number of correct answers
    # n: an integer representing the number of attempts per problem

    # Outputs:
    # alpha and beta: floats representing the MLEs for alpha and beta

    # define the nll of the distribution
    def nll(params):
        # Inputs:
        # a, b: floats: alpha, beta for negative binomial distribution
        # Outputs:
        # float: the mean nll
        a, b, scale = params

        def integrand(p):
            return scipy.stats.binom.pmf(data, n, p) * scipy.stats.beta.pdf(
                p, a, b, scale=scale
            )

        nums = scipy.integrate.quad_vec(integrand, 0, 1)[0]
        return -np.log(nums).mean()

    # initialize a random guess for alpha and beta
    initial_guess = [np.random.rand(), np.random.rand(), max(data / n) + 0.0001]

    # minimize the nll
    result = minimize(
        nll,
        initial_guess,
        bounds=((0, float("inf")), (0, float("inf")), (max(data) / n, 1)),
        method="L-BFGS-B",
        options=dict(
            maxiter=5000,
            maxls=100,
            gtol=1e-6,  # Gradient tolerance, adjust as needed),
        ),
    )
    fitted_params = result.x
    return fitted_params


data = large_language_monkeys_pass_at_1_df["Num. Samples Correct"].values.astype(int)
params = estimate_params_with_scale(data=data, n=10000)


# Compute the beta distribution using the fitted parameters.
x = np.linspace(0.0, 1.0, 100000)
p_x = scipy.stats.beta.pdf(x, params[0], params[1], scale=params[2])

print()

plt.close()
plt.plot(x, p_x)
plt.xscale("log")
plt.yscale("log")
plt.show()
