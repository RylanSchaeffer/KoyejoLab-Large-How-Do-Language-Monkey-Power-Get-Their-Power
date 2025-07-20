from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
import itertools
import joblib
import mpmath
import numpy as np
import os
import pandas as pd
import pprint
import scipy.integrate
from scipy.optimize import minimize
import scipy.stats
import scipy.stats._continuous_distns
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import src.globals


def compute_beta_binomial_two_parameters_mixture_negative_log_likelihood(
    params: dict,
    n_distr: int,
    num_samples: np.ndarray,
    num_successes: np.ndarray,
) -> float:
    to_stack = []
    params_dict = {}
    for i in range(n_distr):
        params_dict[f"alpha_{i}"] = params[i]
    for i in range(n_distr, 2 * n_distr):
        params_dict[f"beta_{i-n_distr}"] = params[i]
    for i in range(2 * n_distr, 3 * n_distr - 1):
        params_dict[f"pi_{i-2*n_distr}"] = params[i]

    remaining_mass = 1 - sum([params_dict[f"pi_{i}"] for i in range(n_distr - 1)])

    for i in range(n_distr):
        prob = (
            np.log(params_dict[f"pi_{i}"])
            if i != n_distr - 1
            else np.log(remaining_mass)
        )
        log_pmf = (
            scipy.stats.betabinom.logpmf(
                k=num_successes,
                n=num_samples,
                a=params_dict[f"alpha_{i}"],
                b=params_dict[f"beta_{i}"],
            )
            + prob
        )
        to_stack.append(log_pmf)
    stacked = np.column_stack(to_stack)
    result = scipy.special.logsumexp(stacked, axis=-1).mean()

    return -result


def fit_beta_binomial_three_parameters_to_num_samples_and_num_successes(
    num_samples_and_num_successes_df: pd.DataFrame,
    n_distr=1,
    maxiter: int = 5000,
    epsilon: Optional[float] = 1e-6,
) -> pd.Series:
    num_data = len(num_samples_and_num_successes_df)
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    print("here")
    if np.all(num_successes == 0):
        result = pd.Series(
            {
                "alpha": np.nan,
                "beta": np.nan,
                "loc": np.nan,
                "scale": np.nan,
                "neg_log_likelihood": np.nan,
                "maxiter": maxiter,
                "success": "Failure (Num Successes All Zeros)",
            }
        )
        return result

    fraction_successes = np.divide(num_successes, num_samples)
    largest_fraction_successes = np.max(fraction_successes)
    # Compute scale as (n+1) * max(fraction_successes) / n.
    # We inflate the scale to correct for bias; is this correct?
    # I think we actually want to divide by the expected value of the maximum of
    # n i.i.d. Beta(alpha, beta) random variables. But this doesn't appear to exist
    # in close form or even numerically?
    # TODO(rylan): Investigate this further.
    print(f"the scale is {largest_fraction_successes}")
    # scale = (num_data + 1.0) * largest_fraction_successes / num_data
    # Make sure that scale isn't more than 1.0 + epsilon.
    # scale = min(scale, 1.0)

    # Start with reasonable initial alpha, beta.
    try:
        alpha, beta, _, _ = scipy.stats.beta.fit(
            np.clip(
                fraction_successes, epsilon, 1.0 - epsilon
            ),  # Make sure that we remain in [0., 1.]
            floc=0.0,
            # fscale=scale,  # Force the scale to be the max scale.
        )
    except scipy.stats._continuous_distns.FitSolverError:
        alpha = 0.35
        beta = 3.5
    initial_probs = 1 / n_distr
    initial_params = (
        [alpha for _ in range(n_distr)]
        + [beta for _ in range(n_distr - 1)]
        + [initial_probs for _ in range(n_distr)]
    )
    initial_params = tuple(initial_params)
    # Create extremely generous bounds for alpha, beta.
    bounds = [(0.01, 100) for _ in range(2 * n_distr)] + [
        (0.01, 0.96) for _ in range(n_distr - 1)
    ]

    # Fit alpha, beta, scale to the scaled beta binomial
    # try:
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_beta_binomial_two_parameters_mixture_negative_log_likelihood(
            params,
            n_distr,
            # scale=scale,
            num_samples=num_samples,
            num_successes=num_successes,
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            maxiter=maxiter,
            maxls=400,
            gtol=1e-6,  # Gradient tolerance, adjust as needed),
            ftol=1e-6,
        ),
    )
    neg_log_likelihood = optimize_result.fun
    result = {
        "neg_log_likelihood": neg_log_likelihood,
        "maxiter": maxiter,
        "success": "Success" if optimize_result.success else "Failure",
    }
    for i in range(n_distr):
        result[f"alpha_{i}"] = optimize_result.x[i]
    for i in range(n_distr, 2 * n_distr):
        result[f"beta_{i-n_distr}"] = optimize_result.x[i]
    for i in range(2 * n_distr, 3 * n_distr - 1):
        result[f"pi_{i-n_distr*2}"] = optimize_result.x[i]

    result = pd.Series(result)
    return result


def plot_beta_binomial_mixture_with_histogram(
    num_samples_and_num_successes_df: pd.DataFrame,
    fitted_params: pd.Series,
    n_distr: int,
    figsize: Tuple[int, int] = (12, 8),
    bins: int = 50,
    alpha_hist: float = 0.7,
    show_components: bool = True,
) -> plt.Figure:
    """
    Plot histogram of original data alongside fitted beta-binomial mixture distribution.

    Parameters:
    -----------
    num_samples_and_num_successes_df : pd.DataFrame
        DataFrame with columns 'Num. Samples Total' and 'Num. Samples Correct'
    fitted_params : pd.Series
        Series containing fitted parameters (alpha_i, beta_i, pi_i values)
    n_distr : int
        Number of components in the mixture
    figsize : tuple
        Figure size (width, height)
    bins : int
        Number of bins for histogram
    alpha_hist : float
        Transparency of histogram bars
    show_components : bool
        Whether to show individual mixture components

    Returns:
    --------
    plt.Figure : The created figure
    """

    # Extract data
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    success_rates = num_successes / num_samples

    # Extract fitted parameters
    alphas = [fitted_params[f"alpha_{i}"] for i in range(n_distr)]
    betas = [fitted_params[f"beta_{i}"] for i in range(n_distr)]

    # Extract mixture weights (pi values)
    pis = []
    for i in range(n_distr - 1):
        pis.append(fitted_params[f"pi_{i}"])
    # Last component weight is the remaining mass
    pis.append(1 - sum(pis))

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot 1: Success rates histogram with fitted mixture PDF
    ax1.hist(
        success_rates,
        bins=bins,
        density=True,
        alpha=alpha_hist,
        color="skyblue",
        edgecolor="black",
        label="Observed Success Rates",
    )

    # Generate x values for smooth curve
    x = np.linspace(0.001, 0.999, 1000)

    # Compute mixture PDF
    mixture_pdf = np.zeros_like(x)
    colors = plt.cm.Set1(np.linspace(0, 1, n_distr))

    for i in range(n_distr):
        # Individual component PDF
        component_pdf = scipy.stats.beta.pdf(x, alphas[i], betas[i])
        mixture_pdf += pis[i] * component_pdf

        if show_components:
            ax1.plot(
                x,
                pis[i] * component_pdf,
                "--",
                color=colors[i],
                alpha=0.7,
                label=f"Component {i+1} (π={pis[i]:.3f})",
            )

    # Plot mixture PDF
    ax1.plot(x, mixture_pdf, "r-", linewidth=2, label="Fitted Mixture PDF")

    ax1.set_xlabel("Success Rate")
    ax1.set_ylabel("Density")
    ax1.set_title("Success Rates: Histogram vs Fitted Beta Mixture Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Raw counts with fitted mixture PMF
    # Create bins for raw success counts
    max_successes = num_successes.max()
    success_bins = np.arange(0, max_successes + 2) - 0.5

    ax2.hist(
        num_successes,
        bins=success_bins,
        density=True,
        alpha=alpha_hist,
        color="lightcoral",
        edgecolor="black",
        label="Observed Success Counts",
    )

    # For PMF, we need to consider the distribution of total samples
    # This is more complex for beta-binomial mixtures, so we'll approximate
    # by using the most common n value or average
    if len(np.unique(num_samples)) == 1:
        n_common = num_samples[0]
    else:
        n_common = int(np.round(np.mean(num_samples)))

    # Generate PMF for the mixture
    k_values = np.arange(0, n_common + 1)
    mixture_pmf = np.zeros_like(k_values, dtype=float)

    for i in range(n_distr):
        component_pmf = scipy.stats.betabinom.pmf(
            k_values, n_common, alphas[i], betas[i]
        )
        mixture_pmf += pis[i] * component_pmf

        if show_components and n_common <= 50:  # Only show components for reasonable n
            ax2.plot(
                k_values,
                pis[i] * component_pmf,
                "o--",
                color=colors[i],
                alpha=0.7,
                markersize=3,
                label=f"Component {i+1} PMF",
            )

    if n_common <= 50:  # Only plot PMF points for reasonable n
        ax2.plot(
            k_values,
            mixture_pmf,
            "ro-",
            linewidth=2,
            markersize=4,
            label=f"Fitted Mixture PMF (n={n_common})",
        )

    ax2.set_xlabel("Number of Successes")
    ax2.set_ylabel("Probability Mass")
    ax2.set_title(
        f"Success Counts: Histogram vs Fitted Beta-Binomial Mixture (n={n_common})"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_mixture_summary(fitted_params: pd.Series, n_distr: int) -> None:
    """Print a summary of the fitted mixture parameters."""
    print("Fitted Beta-Binomial Mixture Parameters:")
    print("=" * 45)

    # Extract mixture weights
    pis = []
    for i in range(n_distr - 1):
        pis.append(fitted_params[f"pi_{i}"])
    pis.append(1 - sum(pis))

    for i in range(n_distr):
        alpha_i = fitted_params[f"alpha_{i}"]
        beta_i = fitted_params[f"beta_{i}"]
        pi_i = pis[i]

        # Compute mean and variance of this beta component
        mean_i = alpha_i / (alpha_i + beta_i)
        var_i = (alpha_i * beta_i) / ((alpha_i + beta_i) ** 2 * (alpha_i + beta_i + 1))

        print(f"Component {i+1}:")
        print(f"  Weight (π): {pi_i:.4f}")
        print(f"  Alpha: {alpha_i:.4f}")
        print(f"  Beta: {beta_i:.4f}")
        print(f"  Mean: {mean_i:.4f}")
        print(f"  Variance: {var_i:.4f}")
        print()

    print(f"Negative Log-Likelihood: {fitted_params['neg_log_likelihood']:.4f}")
    print(f"Optimization Status: {fitted_params['success']}")
