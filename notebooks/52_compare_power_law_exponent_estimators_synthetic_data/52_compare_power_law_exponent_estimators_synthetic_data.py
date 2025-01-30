import ast
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import pprint
import scipy.stats
import seaborn as sns

import src.analyze
import src.plot
import src.utils


data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    # "dcou6k5u",  # Synthetic.  [Resolution=1e-4] [Num Windows=5]
    "cy6ftvm7",  # Synthetic.  [Resolution=1/Num. Samples Per Problems] [Num Windows=10]
]

synthetic_cross_validated_scaling_coeff_df = src.utils.download_wandb_project_runs_configs(
    wandb_project_path="monkey-power-law-estimators",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    # refresh=False,
    refresh=True,
)


def create_full_distribution_str(row: pd.Series) -> str:
    dataset_kwargs = ast.literal_eval(row["dataset_kwargs"])
    distribution_name: str = dataset_kwargs["distribution"].capitalize()
    a: float = dataset_kwargs["a"]
    b: float = dataset_kwargs["b"]
    scale: float = dataset_kwargs["scale"]
    full_distribution_str: str = f"{distribution_name}({a}, {b}, 0, {scale})"
    return full_distribution_str


synthetic_cross_validated_scaling_coeff_df[
    "True Distribution"
] = synthetic_cross_validated_scaling_coeff_df.apply(
    create_full_distribution_str,
    axis=1,
)
synthetic_cross_validated_scaling_coeff_df[
    "True Power Law Exponent"
] = synthetic_cross_validated_scaling_coeff_df["dataset_kwargs"].map(
    lambda s: ast.literal_eval(s)["a"]
)
synthetic_cross_validated_scaling_coeff_df[
    "Num. Problems"
] = synthetic_cross_validated_scaling_coeff_df["num_problems"]
synthetic_cross_validated_scaling_coeff_df[
    "Num. Samples per Problem"
] = synthetic_cross_validated_scaling_coeff_df["num_samples_per_problem"]

fit_methods = [
    "Least Squares",
    "Beta-Binomial",
    "Discretized Beta",
    "Discretized Kumaraswamy",
    "Kumaraswamy-Binomial",
]
synthetic_cross_validated_scaling_coeff_melted_df = (
    synthetic_cross_validated_scaling_coeff_df.melt(
        id_vars=[
            "True Distribution",
            "True Power Law Exponent",
            "Num. Problems",
            "Num. Samples per Problem",
            "seed",
        ],
        value_vars=[f"{fit_method} Power Law Exponent" for fit_method in fit_methods],
        var_name="Fit Method",
        value_name="Fit Power Law Exponent",
    )
)
synthetic_cross_validated_scaling_coeff_melted_df[
    "Fit Method"
] = synthetic_cross_validated_scaling_coeff_melted_df["Fit Method"].apply(
    lambda s: s.replace(" Power Law Exponent", "")
)

synthetic_cross_validated_scaling_coeff_melted_df[
    "Asymptotic Power Law Exponent Relative Error"
] = np.abs(
    (
        synthetic_cross_validated_scaling_coeff_melted_df["Fit Power Law Exponent"]
        - synthetic_cross_validated_scaling_coeff_melted_df["True Power Law Exponent"]
    )
    / synthetic_cross_validated_scaling_coeff_melted_df["True Power Law Exponent"]
)


plt.close()
g = sns.relplot(
    data=synthetic_cross_validated_scaling_coeff_melted_df,
    kind="line",
    x="Num. Samples per Problem",
    y="Asymptotic Power Law Exponent Relative Error",
    hue="Fit Method",
    palette="husl",
    style="Num. Problems",
    col="True Distribution",
    col_wrap=8,
    facet_kws={"margin_titles": True, "sharey": True},
)
g.set(
    xscale="log",
    yscale="log",
    ylabel=r"Relative Error := $|\hat{b} - b| / b$",
)
g.set_titles(col_template="{col_name}", row_template="{row_name} Problems")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
fig = plt.gcf()
fig.text(
    0.50,
    1.0,
    "True Distribution",
    fontsize=30,
    ha="center",
    va="center",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=relative_error_least_squares_x=n_hue=distribution_params_col=distribution",
)
plt.show()


# plt.close()
# g = sns.relplot(
#     data=synthetic_cross_validated_scaling_coeff_df,
#     kind="line",
#     x="Num. Samples per Problem",
#     y="Asymptotic Relative Error",
#     hue="Fit Method",
#     palette="cool",
#     row="Num. Problems",
#     col="True Distribution",
#     facet_kws={"margin_titles": True, "sharey": False},
# )
# g.set(
#     xscale="log",
#     yscale="log",
#     ylabel="",
# )
# g.axes[int(g.axes.shape[0] // 2), 0].set_ylabel(
#     r"Relative Error := $|\hat{b} - b| / b$"
# )
# g.set_titles(col_template="{col_name}", row_template="{row_name} Problems")
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
# fig = plt.gcf()
# fig.text(
#     0.50,
#     1.0,
#     "True Distribution",
#     fontsize=30,
#     ha="center",
#     va="center",
# )
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=relative_error_asymptotic_x=n_hue=distribution_params_col=distribution",
# )
# # plt.show()

print("Finished notebooks/52_compare_power_law_exponent_estimators_synthetic_data!")
