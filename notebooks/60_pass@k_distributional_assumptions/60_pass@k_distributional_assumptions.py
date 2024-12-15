import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
from scipy import integrate
import scipy.special
import seaborn as sns
from typing import Any, Dict, List, Tuple


import src.plot
import src.utils

data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

max_log_k = 9
max_k = 10**max_log_k
k_arr = np.logspace(0, max_log_k, num=250).reshape(-1, 1)

###############################################################################
# Assumption: pass@1 ~ delta(p)
###############################################################################
p_arr = np.logspace(-8, -1, num=29).reshape(1, -1)
expected_pass_at_k_arr = 1.0 - np.power(1 - p_arr, k_arr)
expected_pass_at_k_wide_df = pd.DataFrame(
    expected_pass_at_k_arr,
    index=k_arr[:, 0],
    columns=p_arr[0],
)
expected_pass_at_k_tall_df = expected_pass_at_k_wide_df.stack().reset_index()
expected_pass_at_k_tall_df.columns = ["k", "p", r"$\mathbb{E}[\text{pass@k}]$"]
expected_pass_at_k_tall_df[r"$-\log (\mathbb{E}[\text{pass@k}])$"] = -np.log(
    expected_pass_at_k_tall_df[r"$\mathbb{E}[\text{pass@k}]$"]
)

plt.close()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharex=True, sharey=False)
sns.lineplot(
    data=expected_pass_at_k_tall_df,
    x="k",
    y=r"$\mathbb{E}[\text{pass@k}]$",
    hue="p",
    hue_norm=LogNorm(),
    ax=axes[0],
    legend=False,
    palette="cool",
    linewidth=5,  # Make lines thicker.
)
axes[0].set_xscale("log")
axes[0].set_xlim(1.0, max_k)
axes[0].set_ylim(0.0, 1.0)
axes[0].set_xlabel(r"$k$")
sns.lineplot(
    data=expected_pass_at_k_tall_df,
    x="k",
    y=r"$-\log (\mathbb{E}[\text{pass@k}])$",
    hue="p",
    hue_norm=LogNorm(),
    ax=axes[1],
    legend=True,
    palette="cool",
    linewidth=5,  # Make lines thicker.
)
sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1.0, 1.0))
axes[1].set_xlim(1.0, max_k)
axes[1].set_ylim(1e-5, None)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
fig.suptitle(r"$\text{pass@1} \sim \delta(p)$")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="y=score_x=k_hue=p_col=metric_distr=constant"
)
# plt.show()


###############################################################################
# Assumption: pass@k ~ Beta(alpha, beta)
###############################################################################

# Generate arrays of alpha and beta values.
alpha_arr = np.array([0.1, 0.316, 1.0, 3.16])
beta_arr = np.array([1, 3.1623, 10])
# Create meshgrid for all combinations
alpha_mesh, beta_mesh = np.meshgrid(alpha_arr, beta_arr)
params_dict = {"alpha": alpha_mesh.flatten(), "beta": beta_mesh.flatten()}
# For each k, compute E[pass@k] using the formula we derived:
# E[pass@k] = 1 - B(alpha, k+beta)/B(alpha, beta)
# Initialize array to store results
expected_pass_at_k_arr = np.zeros((len(k_arr), alpha_arr.shape[0] * beta_arr.shape[0]))
alpha_full_arr = np.zeros(shape=(k_arr.shape[0]))
for idx, (alpha, beta) in enumerate(zip(alpha_mesh.flatten(), beta_mesh.flatten())):
    alpha_full_arr[:] = alpha
    numerator = scipy.special.beta(alpha_full_arr, k_arr[:, 0] + beta)
    denominator = scipy.special.beta(alpha_full_arr, beta)
    expected_pass_at_k_arr[:, idx] = 1.0 - np.divide(numerator, denominator)
expected_pass_at_k_wide_df = pd.DataFrame(
    expected_pass_at_k_arr,
    index=k_arr[:, 0],
    columns=pd.MultiIndex.from_arrays(
        [params_dict["alpha"], params_dict["beta"]], names=["alpha", "beta"]
    ),
)
expected_pass_at_k_tall_df = expected_pass_at_k_wide_df.stack().stack().reset_index()
expected_pass_at_k_tall_df.columns = [
    "k",
    r"$\beta$",
    r"$\alpha$",
    r"$\mathbb{E}[\text{pass@k}]$",
]
expected_pass_at_k_tall_df[r"$-\log (\mathbb{E}[\text{pass@k}])$"] = -np.log(
    expected_pass_at_k_tall_df[r"$\mathbb{E}[\text{pass@k}]$"]
)
# Create the same style plot but with different parameters
plt.close()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharex=True, sharey=False)
sns.lineplot(
    data=expected_pass_at_k_tall_df,
    x="k",
    y=r"$\mathbb{E}[\text{pass@k}]$",
    hue=r"$\alpha$",
    style=r"$\beta$",
    hue_norm=LogNorm(),
    ax=axes[0],
    legend=False,
    palette="cool",
    linewidth=5,  # Make lines thicker.
)
axes[0].set_xscale("log")
axes[0].set_xlim(1.0, max_k)
axes[0].set_ylim(0.0, 1.0)
axes[0].set_xlabel(r"$k$")
sns.lineplot(
    data=expected_pass_at_k_tall_df,
    x="k",
    y=r"$-\log (\mathbb{E}[\text{pass@k}])$",
    hue=r"$\alpha$",
    style=r"$\beta$",
    hue_norm=LogNorm(),
    ax=axes[1],
    legend=True,
    palette="cool",
    linewidth=5,  # Make lines thicker.
)
sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1.0, 1.0))
axes[1].set_xlim(1.0, max_k)
axes[1].set_ylim(1e-5, None)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
fig.suptitle(r"$\text{pass@1} \sim \text{Beta}(\alpha, \beta)$")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=k_hue=alpha_style=beta_col=metric_distr=beta",
)
# plt.show()

###############################################################################
# Assumption: pass@k ~ Kumaraswamy(alpha, beta)
###############################################################################

# Generate arrays of alpha and beta values.
alpha_arr = np.array([0.1, 0.316, 1.0, 3.16])
beta_arr = np.round(np.logspace(0, 1, num=5), 4)
# Create meshgrid for all combinations
alpha_mesh, beta_mesh = np.meshgrid(alpha_arr, beta_arr)
params_dict = {"alpha": alpha_mesh.flatten(), "beta": beta_mesh.flatten()}
# For each k, compute E[pass@k] using the formula we derived:
# E[pass@k] = 1 - B(alpha, k+beta)/B(alpha, beta)
# Initialize array to store results
expected_pass_at_k_arr = np.zeros((len(k_arr), alpha_arr.shape[0] * beta_arr.shape[0]))
alpha_full_arr = np.zeros(shape=(k_arr.shape[0]))
for idx, (alpha, beta) in enumerate(zip(alpha_mesh.flatten(), beta_mesh.flatten())):
    alpha_full_arr[:] = alpha
    expected_pass_at_k_arr[:, idx] = 1.0 - alpha * beta * scipy.special.beta(
        alpha_full_arr, alpha_full_arr + k_arr[:, 0] + beta
    )
expected_pass_at_k_wide_df = pd.DataFrame(
    expected_pass_at_k_arr,
    index=k_arr[:, 0],
    columns=pd.MultiIndex.from_arrays(
        [params_dict["alpha"], params_dict["beta"]], names=["alpha", "beta"]
    ),
)
expected_pass_at_k_tall_df = expected_pass_at_k_wide_df.stack().stack().reset_index()
expected_pass_at_k_tall_df.columns = [
    "k",
    r"$\beta$",
    r"$\alpha$",
    r"$\mathbb{E}[\text{pass@k}]$",
]
expected_pass_at_k_tall_df[r"$-\log (\mathbb{E}[\text{pass@k}])$"] = -np.log(
    expected_pass_at_k_tall_df[r"$\mathbb{E}[\text{pass@k}]$"]
)
# Create the same style plot but with different parameters
plt.close()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharex=True, sharey=False)
sns.lineplot(
    data=expected_pass_at_k_tall_df,
    x="k",
    y=r"$\mathbb{E}[\text{pass@k}]$",
    hue=r"$\alpha$",
    style=r"$\beta$",
    hue_norm=LogNorm(),
    ax=axes[0],
    legend=False,
    palette="cool",
    linewidth=5,  # Make lines thicker.
)
axes[0].set_xscale("log")
axes[0].set_xlim(1.0, max_k)
axes[0].set_ylim(0.0, 1.0)
axes[0].set_xlabel(r"$k$")
sns.lineplot(
    data=expected_pass_at_k_tall_df,
    x="k",
    y=r"$-\log (\mathbb{E}[\text{pass@k}])$",
    hue=r"$\alpha$",
    style=r"$\beta$",
    hue_norm=LogNorm(),
    ax=axes[1],
    legend=True,
    palette="cool",
    linewidth=5,  # Make lines thicker.
)
sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1.0, 1.0))
axes[1].set_xlim(1.0, max_k)
axes[1].set_ylim(1e-5, None)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
fig.suptitle(r"$\text{pass@1} \sim \text{Kumaraswamy}(\alpha, \beta)$")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=k_hue=alpha_style=beta_col=metric_distr=kumaraswamy",
)
# plt.show()


###############################################################################
# Assumption: pass@1 ~ Continuous Bernoulli(lambda)
###############################################################################
lambda_arr = np.logspace(-14, -2, num=5).reshape(1, -1)


def C_lambda(lambda_):
    """Compute normalizing constant for Continuous Bernoulli"""
    if abs(lambda_ - 0.5) < 1e-8:  # Handle numerical stability near 0.5
        return 2.0
    return 2.0 * np.arctanh(1.0 - 2.0 * lambda_) / (1.0 - 2.0 * lambda_)


def integrand(p, k, lambda_):
    """Function to integrate: (1-p)^k * lambda^p * (1-lambda)^(1-p)"""
    # Use log-space for numerical stability
    log_term = (
        k * np.log(1.0 - p) + p * np.log(lambda_) + (1.0 - p) * np.log(1.0 - lambda_)
    )
    # If log_term is too negative, return 0 to prevent underflow
    if log_term < -700:  # np.exp(-700) is approximately the smallest positive float64
        return 0.0
    return np.exp(log_term)


def expected_pass_at_k(k, lambda_):
    """Compute E[pass@k] for Continuous Bernoulli(lambda)"""
    # Compute integral
    result, _ = integrate.quad(
        integrand,
        a=0.0,
        b=1.0,  # integration limits
        args=(k, lambda_),  # additional arguments to pass to integrand
        limit=10000,  # limit on the number of subintervals
        epsabs=1e-64,  # absolute error tolerance
        epsrel=1e-64,  # relative error tolerance
    )
    return 1.0 - C_lambda(lambda_) * result


# Example usage with your lambda_arr and k_arr:
expected_pass_at_k_arr = np.zeros((len(k_arr), len(lambda_arr[0])))
for i, k in enumerate(k_arr[:, 0]):
    for j, lambda_ in enumerate(lambda_arr[0]):
        expected_pass_at_k_arr[i, j] = expected_pass_at_k(k, lambda_)
# Convert results to DataFrame
expected_pass_at_k_wide_df = pd.DataFrame(
    expected_pass_at_k_arr,
    index=k_arr[:, 0],
    columns=lambda_arr[0],
)
expected_pass_at_k_tall_df = expected_pass_at_k_wide_df.stack().reset_index()
expected_pass_at_k_tall_df.columns = ["k", r"$\lambda$", r"$\mathbb{E}[\text{pass@k}]$"]
expected_pass_at_k_tall_df[r"$-\log (\mathbb{E}[\text{pass@k}])$"] = -np.log(
    expected_pass_at_k_tall_df[r"$\mathbb{E}[\text{pass@k}]$"]
)
# Create plots
plt.close()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharex=True, sharey=False)
# Left plot: E[pass@k]
sns.lineplot(
    data=expected_pass_at_k_tall_df,
    x="k",
    y=r"$\mathbb{E}[\text{pass@k}]$",
    hue=r"$\lambda$",
    hue_norm=LogNorm(),
    ax=axes[0],
    legend=False,
    palette="cool",
    linewidth=5,  # Make lines thicker.
)
axes[0].set_xscale("log")
axes[0].set_xlim(1.0, max_k)
axes[0].set_ylim(0.0, 1.0)
axes[0].set_xlabel(r"$k$")
# Right plot: -log(E[pass@k])
sns.lineplot(
    data=expected_pass_at_k_tall_df,
    x="k",
    y=r"$-\log (\mathbb{E}[\text{pass@k}])$",
    hue=r"$\lambda$",
    hue_norm=LogNorm(),
    ax=axes[1],
    legend=True,
    palette="cool",
    linewidth=5,  # Make lines thicker.
)
sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1.0, 1.0))
axes[1].set_xlim(1.0, max_k)
axes[1].set_ylim(1e-5, None)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
fig.suptitle(r"$\text{pass@1} \sim \text{Continuous Bernoulli}(\lambda)$")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=k_hue=lambda_col=metric_distr=continuous_bernoulli",
)
# plt.show()

print("Finished 60_pass@k_distributional_assumptions!")
