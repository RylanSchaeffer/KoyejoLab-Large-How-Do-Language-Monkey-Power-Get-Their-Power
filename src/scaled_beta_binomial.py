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
import scipy.special  # Added for logsumexp
from typing import Any, Dict, List, Optional, Tuple, Union
from src.EM import compute_beta_binomial_log_prob
from sklearn.cluster import KMeans


class scaled_beta_binom_mixture:
    def __init__(self, n_distr, num_successes, num_trials):
        self.n_distr = n_distr
        self.num_successes = np.array(num_successes)
        self.num_trials = np.array(num_trials)
        self.alphas = [0 for _ in range(n_distr)]
        self.betas = [0 for _ in range(n_distr)]
        self.probs = [0 for _ in range(n_distr)]
        n = len(num_trials)
        self.scales = [(n + 1) / n for _ in range(n_distr)]
        self.initialize_parameters()
        self.T_ij = np.stack(
            [np.ones(len(num_successes)) * self.probs[i] for i in range(n_distr)]
        )

    def initialize_parameters(self):
        kmeans = KMeans(n_clusters=self.n_distr, n_init="auto")
        kmeans.fit((self.num_successes / self.num_trials).reshape(-1, 1))
        centroids = kmeans.cluster_centers_

        labels = kmeans.labels_

        for k in range(self.n_distr):
            cluster_data = (self.num_successes / self.num_trials)[labels == k]
            mu, var = np.mean(cluster_data), np.var(cluster_data) + 0.000001
            common_term = (mu * (1 - mu) / var) - 1
            self.alphas[k] = mu * common_term
            self.betas[k] = (1 - mu) * common_term
            self.probs[k] = np.sum(labels == k) / len(labels)

    def fit_mixture(self, iters=500):
        for _ in range(iters):
            self.E_step()
            self.M_step()

        outputs = {}
        for i in range(self.n_distr):
            outputs[f"alpha_{i}"] = self.alphas[i]
            outputs[f"beta_{i}"] = self.betas[i]
            outputs[f"pi_{i}"] = self.probs[i]
        return outputs

    def E_step(self):
        log_likelihoods = []  # Changed name to clarify these are logs
        for i in range(self.n_distr):
            alpha = self.alphas[i]
            beta = self.betas[i]
            log_likelihoods.append(
                compute_beta_binomial_log_prob(
                    self.num_successes, self.num_trials, alpha, beta
                )
                + np.log(self.probs[i])
            )  # Changed: add logs instead of multiply

        log_likelihoods = np.stack(log_likelihoods)
        # use bayes' theorem with log-sum-exp for numerical stability:
        log_sum = scipy.special.logsumexp(
            log_likelihoods, axis=0
        )  # Changed: use logsumexp
        self.T_ij = np.exp(log_likelihoods - log_sum)  # Changed: proper normalization

    def M_step(self):
        # Simplified mixing probability update - no optimization needed
        for i in range(self.n_distr):
            self.probs[i] = self.T_ij[i].sum() / len(
                self.num_successes
            )  # Changed: closed form update

        for i in range(self.n_distr):
            optimize_result = scipy.optimize.minimize(
                lambda params: self.single_beta_likelihood(params=params, i=i),
                x0=(self.alphas[i], self.betas[i]),
                bounds=[(1e-3, 1000), (1e-3, 1000)],
                method="L-BFGS-B",
                options=dict(
                    maxiter=1000,
                    maxls=400,
                    gtol=1e-6,  # Gradient tolerance, adjust as needed),
                    ftol=1e-6,
                ),
            )
            self.alphas[i] = optimize_result.x[0]
            self.betas[i] = optimize_result.x[1]
            print(f"{i}: nll: {optimize_result.fun}")

    def taus_likelihood(self, taus):  # This function is no longer used
        likelihoods = self.T_ij.sum(-1)
        total = 0
        total_taus = 0
        for i in range(self.n_distr - 1):
            total += likelihoods[i] * np.log(taus[i])
            total_taus += taus[i]
        total += np.log(1 - total_taus) * likelihoods[-1]
        return total

    def single_beta_likelihood(self, params, i):
        alpha, beta = params
        log_probs = compute_beta_binomial_log_prob(
            self.num_successes, self.num_trials, alpha, beta
        )
        weighted_log_likelihood = (
            self.T_ij[i] * log_probs
        ).sum()  # Changed: sum instead of mean
        return -weighted_log_likelihood  # Changed: return negative for minimization


# first estimate T_ij
# then re-estimate Tau
# finally compute the means using weighted probabilities
