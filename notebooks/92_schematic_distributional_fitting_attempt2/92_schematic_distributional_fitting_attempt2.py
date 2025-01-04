from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
from typing import Any, Dict, List, Tuple

import src.analyze
import src.plot
import src.utils

# Seed to make this reproducible.
np.random.seed(0)

# Setup
data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Parameters
alpha = 0.3
beta = 3.5
loc = 0.0
scale = 0.1
max_samples = 10000
ks_list = [1, 3, 10, 32, 100, 316, 1000, 3162, 10000]
pass_at_1 = np.logspace(-5, 0, max_samples)
palette = sns.color_palette("cool", n_colors=len(pass_at_1))
palette_dict = dict(zip(pass_at_1, palette))

# Step 1: Create and plot the individual outcomes.
individual_outcomes_per_problem_df = (
    src.analyze.sample_synthetic_individual_outcomes_per_problem_df(
        num_problems=256,
        num_samples_per_problem=20000,
        distribution="beta",
        distribution_parameters={"a": alpha, "b": beta, "loc": loc, "scale": scale},
    )
)
# Make these 1-indexed rather than 0-indexed.
individual_outcomes_per_problem_df["Problem Idx"] += 1
individual_outcomes_per_problem_df["Attempt Idx"] += 1
individual_outcomes_per_problem_pivoted_df = individual_outcomes_per_problem_df.pivot(
    index="Attempt Idx",
    columns="Problem Idx",
    values="Score",
)

# Least Squares Step 2: Convert individual outcomes to pass_i@k
estimated_pass_i_at_k_df = src.analyze.compute_pass_at_k_from_individual_outcomes(
    individual_outcomes_per_problem=individual_outcomes_per_problem_pivoted_df.values.T,
    ks_list=ks_list,
)
estimated_pass_i_at_1_df = estimated_pass_i_at_k_df[
    estimated_pass_i_at_k_df["Scaling Parameter"] == 1
].copy()
estimated_pass_i_at_1_df["Log Score"] = np.log(estimated_pass_i_at_1_df["Score"])
estimated_pass_i_at_k_pivoted_df = estimated_pass_i_at_k_df.pivot(
    index="Problem Idx",
    columns="Scaling Parameter",
    values="Score",
)
# Least Squares Step 3: Compute pass_D@k and fit a power law.
estimated_pass_D_at_k_df = (
    estimated_pass_i_at_k_df.groupby("Scaling Parameter")["Score"].mean().reset_index()
)
estimated_pass_D_at_k_df["groupby_placeholder"] = "placeholder"
estimated_pass_D_at_k_df["Neg Log Score"] = -np.log(estimated_pass_D_at_k_df["Score"])
(
    estimated_pass_D_at_k_df,
    fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    estimated_pass_D_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=["groupby_placeholder"],
)
estimated_pass_D_at_k_df["Data Type"] = "Estimated"

# Create the true distribution.
beta_distributution_df = pd.DataFrame.from_dict(
    {
        r"$x$": pass_at_1,
        r"$\alpha$": np.full_like(pass_at_1, fill_value=alpha),
        r"$\beta$": np.full_like(pass_at_1, fill_value=beta),
        r"$c$": np.full_like(pass_at_1, fill_value=scale),
        r"$p(x)$": scipy.stats.beta.pdf(
            pass_at_1, a=alpha + 1, b=beta, loc=loc, scale=scale
        ),
    }
)
positive_beta_distribution_df = beta_distributution_df[
    beta_distributution_df[r"$p(x)$"] > 0
]


integral_values = np.zeros(len(ks_list))
for k_idx, k in enumerate(ks_list):
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
        "Scaling Parameter": ks_list,
        "Neg Log Score": -np.log1p(-integral_values),
        "groupby_placeholder": ["placeholder"] * len(ks_list),
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
predicted_pass_at_k_df["Data Type"] = predicted_pass_at_k_df["Scaling Parameter"].apply(
    lambda k_: "Estimated" if k_ == 1 else "Simulated"
)

plt.close()
# Create a figure with a special gridspec layout
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[0.7, 0.7, 1.2, 1.2])
# Create the merged axis for the left column
ax00 = fig.add_subplot(gs[:, 0])  # This spans both rows in the first column
ax01 = fig.add_subplot(gs[0, 1])
ax02 = fig.add_subplot(gs[0, 2])
ax03 = fig.add_subplot(gs[0, 3])
ax11 = fig.add_subplot(gs[1, 1])
ax12 = fig.add_subplot(gs[1, 2])
ax13 = fig.add_subplot(gs[1, 3])
fig.subplots_adjust(wspace=0.5)  # Increased spacing between subplots
sns.heatmap(
    data=individual_outcomes_per_problem_pivoted_df,
    ax=ax00,
    cbar=False,
)
ax00.set(
    xlabel="Problem",
    ylabel="Sample per Problem",
    title=r"Step 0: Score Samples",
)
ax01.set_axis_off()
sns.heatmap(
    data=estimated_pass_i_at_k_pivoted_df,
    ax=ax02,
    cmap="cool",
    # cbar_kws={"label": r"$\widehat{\operatorname{pass_i@k}}$"},
    norm=LogNorm(vmax=1.0),
    vmax=1.0,
)
ax02.set(
    xlabel=r"Number of Attempts $k$",
    ylabel="Problem",
    title=r"Step 1: Estimate $\operatorname{pass_i@k}$",
)
sns.lineplot(
    data=estimated_pass_D_at_k_df,
    x="Scaling Parameter",
    y="Predicted Neg Log Score",
    color="black",
    ax=ax03,
    linewidth=2,
)
sns.scatterplot(
    data=estimated_pass_D_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Scaling Parameter",
    hue_norm=LogNorm(),
    style="Data Type",
    style_order=["Estimated", "Simulated"],
    s=300,
    palette="copper",
    ax=ax03,
    legend=False,
)
# Create custom legend only for Data Type
handles = [
    plt.Line2D(
        [],
        [],
        marker="o",
        color="black",
        linestyle="None",
        markersize=10,
        label="Estimated",
    ),
]
ax03.legend(handles=handles, loc="lower left")
ax03.set(
    xscale="log",
    yscale="log",
    xlabel=r"Number of Attempts $k$",
    ylabel=r"$-\log (\operatorname{pass_{\mathcal{D}}@k})$",
    title=r"Step 2: Fit Estimated $\operatorname{pass_{\mathcal{D}}@k}$",
)
ax11.set_axis_off()
sns.scatterplot(
    data=beta_distributution_df,
    x=r"$x$",
    y=r"$p(x)$",
    hue=r"$x$",
    legend=False,
    palette="cool",
    hue_norm=LogNorm(),
    linewidth=0,
    ax=ax12,
)
ax12.set(
    xscale="log",
    xlim=(pass_at_1.min(), 1.0),
    xlabel=r"$\operatorname{pass_i@1}$",
    ylabel=r"$p_{\mathcal{D}}(\operatorname{pass_i@1})$",
    title=r"Step 1: Fit $\operatorname{pass_i@1}$ Distribution",
)
sns.lineplot(
    data=predicted_pass_at_k_df,
    x="Scaling Parameter",
    y="Predicted Neg Log Score",
    color="black",
    linewidth=2,
    ax=ax13,
)
scatter = sns.scatterplot(
    data=predicted_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Scaling Parameter",
    style="Data Type",
    style_order=["Estimated", "Simulated"],
    hue_norm=LogNorm(),
    legend=False,
    palette="copper",
    linewidth=0,
    s=300,
    ax=ax13,
)
# Create custom legend only for Data Type
handles = [
    plt.Line2D(
        [],
        [],
        marker="o",
        color="black",
        linestyle="None",
        markersize=10,
        label="Estimated",
    ),
    plt.Line2D(
        [],
        [],
        marker="X",
        color="black",
        linestyle="None",
        markersize=10,
        label="Simulated",
    ),
]
ax13.legend(handles=handles, loc="lower left")
ax13.set(
    xscale="log",
    yscale="log",
    xlabel=r"Number of Attempts $k$",
    ylabel=r"$-\log (\operatorname{pass_{\mathcal{D}}@k})$",
    title=r"Step 2: Fit Simulated $\operatorname{pass_{\mathcal{D}}@k}$",
)
fig.text(
    0.33,
    0.81,
    "Least Squares\nEstimator",
    fontsize=30,
    ha="center",
    va="center",
)
fig.text(
    0.33,
    0.33,
    "Distributional\nEstimator",
    fontsize=30,
    ha="center",
    va="center",
)
fig.text(
    0.33,
    0.73,
    r"$\longrightarrow$",
    fontsize=30,
    ha="center",
    va="center",
)
fig.text(
    0.33,
    0.25,
    r"$\longrightarrow$",
    fontsize=30,
    ha="center",
    va="center",
)

fig.text(
    0.92,
    0.85,
    r"$\approx \hat{a} \, k^{-\hat{b}}$",
    fontsize=30,
    ha="center",
    va="center",
)
fig.text(
    0.92,
    0.35,
    r"$\approx \hat{a} \, k^{-\hat{b}}$",
    fontsize=30,
    ha="center",
    va="center",
)
plt.tight_layout()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="distributional_fitting_schematic"
)
plt.show()

print("Finished0 notebooks/92_schematic_distributional_fitting")
