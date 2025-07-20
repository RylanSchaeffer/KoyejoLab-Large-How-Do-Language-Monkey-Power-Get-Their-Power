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
from scipy.special import gammaln, logsumexp
from typing import Any, Dict, List, Optional, Tuple, Union


def scaled_log_sum_exp(scalars, log_probs):
    # m = max(log_probs)
    # adjusted_log_probs = np.exp(log_probs - m)
    # print(f'adjusted log probs: {adjusted_log_probs}')
    # scaled_adjusted = np.sum(scalars * adjusted_log_probs)
    # return np.log(scaled_adjusted) + m
    return logsumexp(log_probs, b=scalars)


def compute_beta_binomial_three_parameter_nll_single(
    params, num_samples, num_successes
):
    alpha, beta, theta = params

    k = num_successes
    n = num_samples
    log_prob_terms = []
    for i in range(0, n - k + 1):
        log_n_c_i = gammaln(n - k + 1) - gammaln(i + 1) - gammaln(n - k - i + 1)
        log_prob_terms.append(
            (k + i) * np.log(theta)
            + gammaln(alpha + beta)
            - gammaln(alpha + beta + i + k)
            + gammaln(i + k + alpha)
            - gammaln(alpha)
            + log_n_c_i
        )
    log_prob_terms = np.array(log_prob_terms)
    scalars = np.array([(-1) ** i for i in range(n - k + 1)])
    ret = scaled_log_sum_exp(
        scalars, log_prob_terms
    )  # + gammaln(n+1) -gammaln(k+1) - gammaln(n-k+1)
    return -ret


def compute_beta_binomial_three_parameter_nll_stable(
    params, num_samples, num_successes
):
    # print(f'params: {params}')
    nll = 0
    for n, k in zip(list(num_samples.astype(int)), list(num_successes.astype(int))):
        nll += compute_beta_binomial_three_parameter_nll_single(params, n, k)
    return nll


def fit_beta_binomial_three_parameters_stable(
    num_samples_and_num_successes_df: pd.DataFrame,
    maxiter: int = 5000,
) -> pd.Series:
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    largest = np.max(num_successes / num_samples)
    l = len(num_samples)
    initial_params = (0.5, 3.5, (l + 1) / l * largest)
    bounds = [(0.01, 100), (0.01, 100), (0.1, (1 - largest) / 2 + largest)]

    # Fit alpha, beta, scale to the scaled beta binomial
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_beta_binomial_three_parameter_nll_stable(
            num_samples=num_samples,
            num_successes=num_successes,
            params=params,
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

    result = pd.Series(
        {
            "alpha": optimize_result.x[0],
            "beta": optimize_result.x[1],
            "loc": 0.0,
            "scale": optimize_result.x[2],
            "neg_log_likelihood": optimize_result.fun,
            "aic": 2 * len(initial_params) + 2 * optimize_result.fun,
            "bic": len(initial_params) * np.log(len(num_samples_and_num_successes_df))
            + 2 * optimize_result.fun,
            "Power Law Exponent": optimize_result.x[0],
        }
    )

    return result


def compute_beta_binomial_two_parameters_nll(params, num_samples, num_successes):
    alpha, beta = params
    n = int(max(num_samples))  # Assuming fixed n

    nll = 0
    for k in range(int(max(num_successes)) + 1):  # Include max value
        eta_k = np.sum(num_successes == k)
        if eta_k == 0:
            continue

        log_prob = 0

        # Sum log(alpha + j) for j = 0 to k-1
        for j in range(k):
            log_prob += np.log(alpha + j)

        # Sum log(beta + j) for j = 0 to n-k-1
        for j in range(n - k):
            log_prob += np.log(beta + j)

        # Subtract sum log(alpha + beta + j) for j = 0 to n-1
        for j in range(n):
            log_prob -= np.log(alpha + beta + j)

        nll -= eta_k * log_prob / len(num_samples)

    return nll


def fit_mixture_distribution_stable(n_components, steps, num_samples, num_successes):
    pis = tuple([1 / n_components for _ in range(n_components)])


def fit_beta_binomial_two_parameters_stable(
    num_samples_and_num_successes_df: pd.DataFrame,
    maxiter: int = 5000,
) -> pd.Series:
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    initial_params = (0.5, 3.5)
    bounds = [
        (0.01, 100),
        (0.01, 100),
    ]

    # Fit alpha, beta, scale to the scaled beta binomial
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_beta_binomial_two_parameters_nll(
            num_samples=num_samples,
            num_successes=num_successes,
            params=params,
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

    result = pd.Series(
        {
            "alpha": optimize_result.x[0],
            "beta": optimize_result.x[1],
            "loc": 0.0,
            "scale": 1.0,
            "neg_log_likelihood": optimize_result.fun,
            "aic": 2 * len(initial_params) + 2 * optimize_result.fun,
            "bic": len(initial_params) * np.log(len(num_samples_and_num_successes_df))
            + 2 * optimize_result.fun,
            "Power Law Exponent": optimize_result.x[0],
        }
    )

    return result


def compute_beta_binomial_two_parameters_mixture_negative_log_likelihood(
    params: np.ndarray,
    n_distr: int,
    num_samples: np.ndarray,
    num_successes: np.ndarray,
) -> float:
    """
    Fixed version - the key issue was in parameter extraction.
    """
    # Extract parameters correctly
    alphas = params[:n_distr]
    betas = params[n_distr : 2 * n_distr]

    # Extract mixing weights (we optimize n_distr-1 weights, last one is computed)
    if n_distr > 1:
        raw_weights = params[2 * n_distr : 3 * n_distr - 1]
        # Ensure weights are positive and sum to < 1
        weights = np.zeros(n_distr)
        weights[:-1] = np.abs(raw_weights)  # Ensure positive
        weights[-1] = max(0.01, 1.0 - np.sum(weights[:-1]))  # Remaining weight

        # Normalize to ensure they sum to 1
        weights = weights / np.sum(weights)
    else:
        weights = np.array([1.0])

    # Add small epsilon to avoid log(0)
    weights = np.maximum(weights, 1e-8)

    # Ensure parameters are positive
    alphas = np.maximum(alphas, 0.01)
    betas = np.maximum(betas, 0.01)

    try:
        # Compute log-likelihood for each component
        log_likelihoods = []
        for i in range(n_distr):
            log_pmf = scipy.stats.betabinom.logpmf(
                k=num_successes, n=num_samples, a=alphas[i], b=betas[i]
            )
            log_likelihood_i = np.log(weights[i]) + log_pmf
            log_likelihoods.append(log_likelihood_i)

        # Stack and use logsumexp
        stacked = np.column_stack(log_likelihoods)
        log_likelihood_total = scipy.special.logsumexp(stacked, axis=1)

        # Return negative mean log-likelihood
        result = -np.mean(log_likelihood_total)

        # Check for NaN or inf
        if not np.isfinite(result):
            return 1e10

        return result

    except Exception as e:
        return 1e10  # Return large penalty for any numerical issues


def initialize_parameters_with_spread(
    num_samples: np.ndarray,
    num_successes: np.ndarray,
    n_distr: int,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, List]:
    """
    Simple initialization that spreads components across the data range.
    """
    success_rates = num_successes / num_samples
    success_rates_clipped = np.clip(success_rates, epsilon, 1.0 - epsilon)

    # Fit overall beta distribution
    try:
        overall_alpha, overall_beta, _, _ = scipy.stats.beta.fit(
            success_rates_clipped, floc=0.0
        )
    except:
        overall_alpha, overall_beta = 1.0, 1.0

    if n_distr == 1:
        initial_params = np.array([overall_alpha, overall_beta])
        bounds = [(0.01, 100.0), (0.01, 100.0)]
    else:
        # Create components spread across different parts of the distribution
        alphas = []
        betas = []

        # Create components with different means
        target_means = np.linspace(0.1, 0.9, n_distr)

        for i, target_mean in enumerate(target_means):
            # Use method of moments to get alpha, beta for this target mean
            # Start with overall variance but adjust
            target_var = min(
                target_mean * (1 - target_mean) * 0.1,
                target_mean * (1 - target_mean) * 0.9,
            )

            if target_var > 0:
                # Method of moments: mean = a/(a+b), var = ab/((a+b)^2(a+b+1))
                # Solve for a, b given mean and variance
                temp = target_mean * (1 - target_mean) / target_var - 1
                alpha = target_mean * temp
                beta = (1 - target_mean) * temp

                alpha = max(0.1, min(alpha, 10.0))  # Keep reasonable
                beta = max(0.1, min(beta, 10.0))
            else:
                # Fallback
                alpha = overall_alpha + np.random.normal(0, 0.5)
                beta = overall_beta + np.random.normal(0, 0.5)
                alpha = max(0.1, alpha)
                beta = max(0.1, beta)

            alphas.append(alpha)
            betas.append(beta)

        # Initialize weights equally
        weights = np.ones(n_distr - 1) / n_distr  # Only n_distr-1 free parameters

        initial_params = np.concatenate([alphas, betas, weights])

        bounds = (
            [(0.01, 100.0) for _ in range(n_distr)]
            + [(0.01, 100.0) for _ in range(n_distr)]  # alphas
            + [  # betas
                (0.01, 0.98) for _ in range(n_distr - 1)
            ]  # weights (must be < 1)
        )

    return initial_params, bounds


def fit_beta_binomial_three_parameters_to_num_samples_and_num_successes(
    num_samples_and_num_successes_df: pd.DataFrame,
    n_distr: int = 1,
    maxiter: int = 5000,
    epsilon: Optional[float] = 1e-6,
) -> pd.Series:
    """
    Fixed version with minimal changes to your original approach.
    Key fixes:
    1. Correct parameter extraction
    2. Better initialization to spread components
    3. Simpler weight handling
    4. Multiple restarts for robustness
    """
    num_data = len(num_samples_and_num_successes_df)
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    print(f"Fitting {n_distr}-component mixture...")

    if np.all(num_successes == 0):
        result = pd.Series(
            {
                "neg_log_likelihood": np.nan,
                "maxiter": maxiter,
                "success": "Failure (Num Successes All Zeros)",
            }
        )
        for i in range(n_distr):
            result[f"alpha_{i}"] = np.nan
            result[f"beta_{i}"] = np.nan
        for i in range(n_distr - 1):
            result[f"pi_{i}"] = np.nan
        return result

    best_result = None
    best_nll = np.inf

    # Try a few different initializations
    n_tries = min(5, max(1, n_distr))  # More tries for more components

    for attempt in range(n_tries):
        try:
            print(f"  Attempt {attempt + 1}/{n_tries}")

            # Get initial parameters
            initial_params, bounds = initialize_parameters_with_spread(
                num_samples, num_successes, n_distr, epsilon
            )

            # Add some randomness for attempts after the first
            if attempt > 0:
                noise_scale = 0.1 * (attempt + 1)  # Increase noise for later attempts
                for i in range(len(initial_params)):
                    noise = np.random.normal(0, noise_scale)
                    initial_params[i] = np.clip(
                        initial_params[i] + noise, bounds[i][0], bounds[i][1]
                    )

            # Optimize
            optimize_result = scipy.optimize.minimize(
                fun=compute_beta_binomial_two_parameters_mixture_negative_log_likelihood,
                x0=initial_params,
                args=(n_distr, num_samples, num_successes),
                bounds=bounds,
                method="L-BFGS-B",
                options={
                    "maxiter": maxiter,
                    "maxls": 400,
                    "gtol": 1e-6,
                    "ftol": 1e-6,
                },
            )

            if optimize_result.success and optimize_result.fun < best_nll:
                best_nll = optimize_result.fun
                best_result = optimize_result
                print(f"    Success! NLL = {optimize_result.fun:.6f}")
            else:
                print(f"    Failed or worse result. NLL = {optimize_result.fun:.6f}")

        except Exception as e:
            print(f"    Exception: {e}")
            continue

    if best_result is None:
        print("All optimization attempts failed!")
        result = pd.Series(
            {
                "neg_log_likelihood": np.inf,
                "maxiter": maxiter,
                "success": "Failure (No successful optimization)",
            }
        )
        for i in range(n_distr):
            result[f"alpha_{i}"] = np.nan
            result[f"beta_{i}"] = np.nan
        for i in range(n_distr - 1):
            result[f"pi_{i}"] = np.nan
        return result

    # Extract results with correct indexing
    params = best_result.x
    alphas = params[:n_distr]
    betas = params[n_distr : 2 * n_distr]

    if n_distr > 1:
        raw_weights = params[2 * n_distr : 3 * n_distr - 1]
        weights = np.zeros(n_distr)
        weights[:-1] = raw_weights
        weights[-1] = 1.0 - np.sum(weights[:-1])
        # Normalize just to be sure
        weights = np.maximum(weights, 0.001)  # Ensure positive
        weights = weights / np.sum(weights)
    else:
        weights = np.array([1.0])

    # Build result
    result_dict = {
        "neg_log_likelihood": best_result.fun,
        "maxiter": maxiter,
        "success": "Success" if best_result.success else "Partial Success",
    }

    for i in range(n_distr):
        result_dict[f"alpha_{i}"] = alphas[i]
        result_dict[f"beta_{i}"] = betas[i]

    for i in range(n_distr - 1):
        result_dict[f"pi_{i}"] = weights[i]

    print("Final parameters:")
    for i in range(n_distr):
        weight_i = weights[i]
        mean_i = alphas[i] / (alphas[i] + betas[i])
        print(
            f"  Component {i+1}: α={alphas[i]:.3f}, β={betas[i]:.3f}, π={weight_i:.3f}, mean={mean_i:.3f}"
        )

    return pd.Series(result_dict)


def compute_mixture_log_pmf(
    k: Union[int, np.ndarray],
    n: Union[int, np.ndarray],
    fitted_params: pd.Series,
    n_distr: int,
) -> Union[float, np.ndarray]:
    """
    Compute log PMF of beta-binomial mixture from fitted parameters.

    Parameters:
    -----------
    k : int or array-like
        Number of successes
    n : int or array-like
        Number of trials
    fitted_params : pd.Series
        Fitted parameters from fit_beta_binomial_three_parameters_to_num_samples_and_num_successes
    n_distr : int
        Number of mixture components

    Returns:
    --------
    float or np.ndarray
        Log probability mass function value(s)
    """
    # Convert to arrays for consistent handling
    k = np.atleast_1d(k)
    n = np.atleast_1d(n)

    # Extract parameters
    alphas = [fitted_params[f"alpha_{i}"] for i in range(n_distr)]
    betas = [fitted_params[f"beta_{i}"] for i in range(n_distr)]

    # Extract weights
    weights = []
    for i in range(n_distr - 1):
        weights.append(fitted_params[f"pi_{i}"])
    # Last weight is remaining mass
    weights.append(1.0 - sum(weights))
    weights = np.array(weights)

    # Compute log PMF for each component
    log_pmfs = []
    for i in range(n_distr):
        component_log_pmf = np.log(weights[i]) + scipy.stats.betabinom.logpmf(
            k=k, n=n, a=alphas[i], b=betas[i]
        )
        log_pmfs.append(component_log_pmf)

    # Stack and use logsumexp
    stacked = np.column_stack(log_pmfs)
    result = scipy.special.logsumexp(stacked, axis=1)

    # Return scalar if input was scalar
    if len(result) == 1:
        return result[0]
    return result


def compute_mixture_pmf(
    k: Union[int, np.ndarray],
    n: Union[int, np.ndarray],
    fitted_params: pd.Series,
    n_distr: int,
) -> Union[float, np.ndarray]:
    """
    Compute PMF (not log) of beta-binomial mixture from fitted parameters.

    Parameters:
    -----------
    k : int or array-like
        Number of successes
    n : int or array-like
        Number of trials
    fitted_params : pd.Series
        Fitted parameters from fit_beta_binomial_three_parameters_to_num_samples_and_num_successes
    n_distr : int
        Number of mixture components

    Returns:
    --------
    float or np.ndarray
        Probability mass function value(s)
    """
    log_pmf = compute_mixture_log_pmf(k, n, fitted_params, n_distr)
    return np.exp(log_pmf)


def compute_component_log_pmfs(
    k: Union[int, np.ndarray],
    n: Union[int, np.ndarray],
    fitted_params: pd.Series,
    n_distr: int,
) -> np.ndarray:
    """
    Compute log PMF for each individual component (useful for plotting components separately).

    Parameters:
    -----------
    k : int or array-like
        Number of successes
    n : int or array-like
        Number of trials
    fitted_params : pd.Series
        Fitted parameters
    n_distr : int
        Number of mixture components

    Returns:
    --------
    np.ndarray
        Array of shape (len(k), n_distr) with log PMF for each component
    """
    k = np.atleast_1d(k)
    n = np.atleast_1d(n)

    # Extract parameters
    alphas = [fitted_params[f"alpha_{i}"] for i in range(n_distr)]
    betas = [fitted_params[f"beta_{i}"] for i in range(n_distr)]
    weights = []
    for i in range(n_distr - 1):
        weights.append(fitted_params[f"pi_{i}"])
    weights.append(1.0 - sum(weights))

    # Compute weighted log PMF for each component
    component_log_pmfs = np.zeros((len(k), n_distr))
    for i in range(n_distr):
        component_log_pmfs[:, i] = np.log(weights[i]) + scipy.stats.betabinom.logpmf(
            k=k, n=n, a=alphas[i], b=betas[i]
        )

    return component_log_pmfs
