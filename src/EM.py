# EM for beta binomial
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
from src import analyze


def fit_beta_binomial_mixture_em(
    num_samples, num_successes, n_components=2, max_iter=100, tol=1e-6
):
    n = int(max(num_samples))  # Fixed sample size

    # Initialize parameters
    mixing_weights = np.ones(n_components) / n_components
    alphas = np.random.uniform(0.5, 2.0, n_components)
    betas = np.random.uniform(0.5, 2.0, n_components)

    # Create count data
    unique_k, counts = np.unique(num_successes, return_counts=True)
    max_k = int(max(num_successes))

    eta = np.zeros(max_k + 1)
    for k, count in zip(unique_k, counts):
        eta[int(k)] = count

    prev_log_likelihood = -np.inf
    converged = False

    for iteration in range(max_iter):
        # E-step: compute responsibilities
        responsibilities = np.zeros((max_k + 1, n_components))

        for k in range(max_k + 1):
            if eta[k] == 0:
                continue

            for j in range(n_components):
                log_prob = compute_beta_binomial_log_prob(k, n, alphas[j], betas[j])
                responsibilities[k, j] = mixing_weights[j] * np.exp(log_prob)

            if np.sum(responsibilities[k, :]) > 0:
                responsibilities[k, :] /= np.sum(responsibilities[k, :])

        # M-step: update parameters
        # Update mixing weights
        for j in range(n_components):
            mixing_weights[j] = np.sum(eta * responsibilities[:, j]) / np.sum(eta)

        # Update alpha and beta for each component
        for j in range(n_components):
            weighted_counts = eta * responsibilities[:, j]
            if np.sum(weighted_counts) > 0:
                alphas[j], betas[j] = update_beta_binomial_params(
                    weighted_counts, n, alphas[j], betas[j]
                )

        # Check convergence
        current_log_likelihood = compute_mixture_log_likelihood(
            eta, n, mixing_weights, alphas, betas
        )

        if abs(current_log_likelihood - prev_log_likelihood) < tol:
            converged = True
            break

        prev_log_likelihood = current_log_likelihood

    # Format output like your example
    neg_log_likelihood = -current_log_likelihood
    result = {
        "neg_log_likelihood": neg_log_likelihood,
        "maxiter": max_iter,
        "success": "Success" if converged else "Failure",
    }

    for i in range(n_components):
        result[f"alpha_{i}"] = alphas[i]

    for i in range(n_components):
        result[f"beta_{i}"] = betas[i]

    for i in range(n_components - 1):  # n-1 mixing weights (last one is constrained)
        result[f"pi_{i}"] = mixing_weights[i]

    return result


# def compute_beta_binomial_log_prob(k, n, alpha, beta):
#     """Compute log P(X = k | n, alpha, beta) for beta-binomial"""
#     log_prob = 0
#     log_prob += scipy.special.gammaln(n + 1)
#     log_prob -= scipy.special.gammaln(k + 1)
#     log_prob -= scipy.special.gammaln(n - k + 1)
#     # Sum log(alpha + j) for j = 0 to k-1
#     for j in range(k):
#         log_prob += np.log(alpha + j)

#     # Sum log(beta + j) for j = 0 to n-k-1
#     for j in range(n - k):
#         log_prob += np.log(beta + j)

#     # Subtract sum log(alpha + beta + j) for j = 0 to n-1
#     for j in range(n):
#         log_prob -= np.log(alpha + beta + j)


#     return log_prob


# not confident that this function is right
def compute_beta_binomial_log_prob(k, n, alpha, beta):
    """Compute log P(X = k | n, alpha, beta) for beta-binomial"""
    k = np.asarray(k)
    n = np.asarray(n)

    # Binomial coefficient
    log_prob = (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k + 1)
        - scipy.special.gammaln(n - k + 1)
    )

    # Vectorized computation using broadcasting
    # For each element i, we need sum(log(alpha + j)) for j in range(k[i])
    max_k = np.max(k) if k.size > 0 else 0
    max_n = np.max(n) if n.size > 0 else 0

    if max_k > 0:
        j_vals = np.arange(max_k)  # [0, 1, ..., max_k-1]
        mask = j_vals < k[:, np.newaxis]  # Shape: (len(k), max_k)
        log_prob += np.sum(np.log(alpha + j_vals) * mask, axis=1)

    if max_n > 0:
        j_vals = np.arange(max_n)  # [0, 1, ..., max_n-1]
        mask = j_vals < (n - k)[:, np.newaxis]  # Shape: (len(k), max_n)
        log_prob += np.sum(np.log(beta + j_vals) * mask, axis=1)

        mask_n = j_vals < n[:, np.newaxis]  # Shape: (len(k), max_n)
        log_prob -= np.sum(np.log(alpha + beta + j_vals) * mask_n, axis=1)

    return log_prob


def update_beta_binomial_params(weighted_counts, n, current_alpha, current_beta):
    """Update alpha, beta parameters using weighted MLE"""
    # This is the tricky part - you need to solve for alpha, beta
    # given the weighted sufficient statistics

    def weighted_nll(params):
        alpha, beta = params
        nll = 0
        for k in range(len(weighted_counts)):
            if weighted_counts[k] > 0:
                log_prob = compute_beta_binomial_log_prob(k, n, alpha, beta)
                nll -= weighted_counts[k] * log_prob
        return nll

    result = scipy.optimize.minimize(
        weighted_nll,
        x0=[current_alpha, current_beta],
        bounds=[(0.01, 100), (0.01, 100)],
        method="L-BFGS-B",
    )

    return result.x[0], result.x[1]


def compute_mixture_log_likelihood(eta, n, mixing_weights, alphas, betas):
    """Compute total log-likelihood of the mixture"""
    log_likelihood = 0

    for k in range(len(eta)):
        if eta[k] == 0:
            continue

        prob_k = 0
        for j in range(len(mixing_weights)):
            prob_k += mixing_weights[j] * np.exp(
                compute_beta_binomial_log_prob(k, n, alphas[j], betas[j])
            )

        log_likelihood += eta[k] * np.log(prob_k)

    return log_likelihood


def compute_expected_pass_at_k(beta_mixture_params, n_distr, k1=100, k2=900):
    """
    Compute E[(1-p)^k1 * (1-(1-p)^k2)] under the beta-binomial mixture
    where p ~ mixture of Beta(alpha_i, beta_i)
    """
    # Extract mixing weights
    probs = [beta_mixture_params[f"pi_{i}"] for i in range(n_distr - 1)]
    probs = np.array(probs + [1 - sum(probs)])

    total_expectation = 0

    for i in range(n_distr):
        alpha_i = beta_mixture_params[f"alpha_{i}"]
        beta_i = beta_mixture_params[f"beta_{i}"]

        # For this component, compute E[(1-p)^k1 * (1-(1-p)^k2)]
        # = E[(1-p)^k1] - E[(1-p)^(k1+k2)]

        # E[(1-p)^k] where p ~ Beta(alpha, beta) = B(beta+k, alpha) / B(beta, alpha)
        # = Γ(beta+k)Γ(alpha+beta) / [Γ(beta)Γ(alpha+beta+k)]

        # Compute E[(1-p)^k1]
        term1 = np.exp(
            scipy.special.gammaln(beta_i + k1)
            + scipy.special.gammaln(alpha_i + beta_i)
            - scipy.special.gammaln(beta_i)
            - scipy.special.gammaln(alpha_i + beta_i + k1)
        )

        # Compute E[(1-p)^(k1+k2)]
        term2 = np.exp(
            scipy.special.gammaln(beta_i + k1 + k2)
            + scipy.special.gammaln(alpha_i + beta_i)
            - scipy.special.gammaln(beta_i)
            - scipy.special.gammaln(alpha_i + beta_i + k1 + k2)
        )

        component_expectation = term1 - term2
        total_expectation += probs[i] * component_expectation

    return total_expectation


def compute_scaled_expected_pass_at_k(params, k1=100, k2=900):
    """
    Compute E[(1-p)^k1 * (1-(1-p)^k2)] under the beta-binomial mixture
    where p ~ mixture of Beta(alpha_i, beta_i)
    """
    # Extract mixing weights

    total_expectation = 0

    # For this component, compute E[(1-p)^k1 * (1-(1-p)^k2)]
    # = E[(1-p)^k1] - E[(1-p)^(k1+k2)]

    # E[(1-p)^k] where p ~ Beta(alpha, beta) = B(beta+k, alpha) / B(beta, alpha)
    # = Γ(beta+k)Γ(alpha+beta) / [Γ(beta)Γ(alpha+beta+k)]

    # Compute E[(1-p)^k1]
    term1 = np.exp(
        -1
        * analyze.compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
            (params["alpha"], params["beta"]),
            scale=params["scale"],
            num_samples=np.array([k1]),
            num_successes=np.array([0]),
        )
    )

    # Compute E[(1-p)^(k1+k2)]
    term2 = np.exp(
        -1
        * analyze.compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
            (params["alpha"], params["beta"]),
            scale=params["scale"],
            num_samples=np.array([k1 + k2]),
            num_successes=np.array([0]),
        )
    )
    # print(term1, term2)
    total_expectation = term1 - term2

    return total_expectation


def compute_estimate(params, k):
    return 1 - np.exp(
        -1
        * analyze.compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
            params=(params["alpha"], params["beta"]),
            scale=params["scale"],
            num_samples=np.array([k]),
            num_successes=np.array([0]),
        )
    )


def compute_estimate_stable(params, k):
    return 1 - np.exp(
        -1
        * bopt.compute_beta_binomial_three_parameter_nll_single(
            (params["alpha"], params["beta"], params["scale"]), k, 0
        )
    )


def compute_p_at_ks(data, ks):
    pass_at_ks = analyze.compute_pass_at_k_from_num_samples_and_num_successes_df(
        data, ks
    )
    pass_at_ks = pass_at_ks.groupby(by="Scaling Parameter").mean()
    return np.array(pass_at_ks["Score"])


def compute_estimates_better_mixture(data, params, k, n_distr):
    unsolved = 1 - compute_expected_pass_at_k(params, n_distr, 0, k)
    fail_first_true = len(data[data["Num. Samples Correct"] == 0])
    questions = len(data)
    estimate = (questions - fail_first_true * unsolved) / questions
    return estimate
