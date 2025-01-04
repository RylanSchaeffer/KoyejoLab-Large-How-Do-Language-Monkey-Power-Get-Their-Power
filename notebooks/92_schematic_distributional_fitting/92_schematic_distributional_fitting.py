from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
from typing import Any, Dict, List, Tuple

import src.analyze
import src.plot
import src.utils

# Setup
data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Parameters
alpha = 0.3
beta = 3.5
scale = 1.0
pass_at_1 = np.logspace(-5, 0, 10000)
palette = sns.color_palette("cool", n_colors=len(pass_at_1))
palette_dict = dict(zip(pass_at_1, palette))

# Create the data
beta_distributution_df = pd.DataFrame.from_dict(
    {
        r"$x$": pass_at_1,
        r"$\alpha$": np.full_like(pass_at_1, fill_value=alpha),
        r"$\beta$": np.full_like(pass_at_1, fill_value=beta),
        r"$p(x)$": scipy.stats.beta.pdf(
            pass_at_1, a=alpha + 1, b=beta, loc=0.0, scale=scale
        ),
    }
)
positive_beta_distributution_df = beta_distributution_df[
    beta_distributution_df[r"$p(x)$"] > 0
]

k_values = np.unique(np.logspace(0, 4, 15).astype(int))
integral_values = np.zeros(k_values.shape)
for k_idx, k in enumerate(k_values):
    integral_values[
        k_idx
    ] = src.analyze.compute_beta_three_parameter_distribution_integral(
        k=k,
        alpha=alpha,
        beta=beta,
        scale=scale,
    )

predicted_pass_at_k_df = pd.DataFrame.from_dict(
    {
        "Scaling Parameter": k_values,
        "Neg Log Score": -np.log1p(-integral_values),
        "groupby_placeholder": ["placeholder"] * len(k_values),
    }
)

# Fit a power law to the integral values.
(
    predicted_pass_at_k_df,
    fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    predicted_pass_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=["groupby_placeholder"],
)
power_law_exponent = fitted_power_law_parameters_df["Power Law Exponent"][0]
plt.close()
fig, axes = plt.subplots(
    figsize=(18, 6),  # Slightly wider figure
    nrows=1,
    ncols=3,
    sharex=False,
    sharey=False,
)
fig.subplots_adjust(wspace=0.5)  # Increased spacing between subplots
ax0 = axes[0]
sns.scatterplot(
    data=positive_beta_distributution_df,
    x=r"$x$",
    y=r"$p(x)$",
    hue=r"$x$",
    legend=False,
    palette="cool",
    hue_norm=LogNorm(),
    linewidth=0,
    ax=ax0,
)
ax0.set(
    xscale="log",
    xlim=(pass_at_1.min(), 1.0),
    yscale="log",
    ylim=(3.162e-2, 1.5e1),
    xlabel=r"$\operatorname{pass_i@1}$",
    ylabel=r"$p_{\mathcal{D}}(\operatorname{pass_i@1})$",
    title=r"Step 1: Fit $\operatorname{pass_i@1}$ Distribution",
)
ax1 = axes[1]
sns.scatterplot(
    data=predicted_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Scaling Parameter",
    hue_norm=LogNorm(),
    legend=False,
    palette="copper",
    linewidth=0,
    s=100,
    ax=ax1,
)
ax1.set(
    xscale="log",
    yscale="log",
    xlabel=r"Number of Attempts $k$",
    ylabel=r"$\widehat{\operatorname{pass_{\mathcal{D}}@k}}$",
    title=r"Step 2: Predict $\operatorname{pass_{\mathcal{D}}@k}$",
)
ax2 = axes[2]
sns.scatterplot(
    data=predicted_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Scaling Parameter",
    hue_norm=LogNorm(),
    legend=False,
    palette="copper",
    linewidth=0,
    s=100,
    ax=ax2,
)
sns.lineplot(
    data=predicted_pass_at_k_df,
    x="Scaling Parameter",
    y="Predicted Neg Log Score",
    color="black",
    linewidth=4,
)
ax2.set(
    xscale="log",
    yscale="log",
    xlabel=r"Number of Attempts $k$",
    ylabel=r"$\widehat{\operatorname{pass_{\mathcal{D}}@k}}$",
    title=r"Step 3: Fit Power Law",
)
# Create text in axes.
fig.text(
    0.92,
    0.76,
    r"$\propto k^{-\hat{b}}$",
    fontsize=30,
    ha="center",
    va="center",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="distributional_fitting_schematic"
)
plt.show()
print("Finished0 notebooks/92_schematic_distributional_fitting")
