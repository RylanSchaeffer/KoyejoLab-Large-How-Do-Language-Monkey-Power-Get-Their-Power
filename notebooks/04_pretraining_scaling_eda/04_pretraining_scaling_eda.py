import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Tuple

import src.analyze
import src.globals
import src.plot
import src.utils


data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

causal_language_modeling_probability_df = src.analyze.create_or_load_pretraining_probability_df(
    refresh=False,
    # refresh=True,
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=causal_language_modeling_probability_df,
    x="Scaling Parameter",
    y="Score",
    hue="Dataset",
    hue_order=src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER,
    style="Model Family",
)
g.set(
    title="Causal Language Modeling",
    xscale="log",
    xlabel=r"Scaling Parameter (Pretraining FLOP)",
    yscale="log",
    ylabel=r"$\mathbb{E}[p(x_t | x_{<t})]$",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=scaling_parameter_hue=model",
)
# plt.show()

pretraining_neg_log_avg_probability_df = (
    causal_language_modeling_probability_df.groupby(
        ["Model Family", "Model Nickname", "Dataset", "Scaling Parameter"]
    )["Score"]
    .mean()
    .reset_index()
)
pretraining_neg_log_avg_probability_df["Log Score"] = np.log(
    pretraining_neg_log_avg_probability_df["Score"]
)
pretraining_neg_log_avg_probability_df[
    "Neg Log Score"
] = -pretraining_neg_log_avg_probability_df["Log Score"]

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretraining_neg_log_avg_probability_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Dataset",
    hue_order=src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER,
    style="Model Family",
)
g.set(
    title="Causal Language Modeling",
    xscale="log",
    xlabel=r"Scaling Parameter (Pretraining FLOP)",
    ylabel=r"$-\log \mathbb{E}[p(x_t | x_{<t})]$",
    yscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_avg_score_x=scaling_parameter_hue=model",
)
# plt.show()


pretraining_avg_neg_log_probability_df = (
    causal_language_modeling_probability_df.groupby(
        ["Model Family", "Model Nickname", "Dataset", "Scaling Parameter"]
    )["Neg Log Score"]
    .mean()
    .reset_index()
)
plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretraining_avg_neg_log_probability_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Dataset",
    style="Model Family",
)
g.set(
    title="Causal Language Modeling",
    xscale="log",
    xlabel=r"Scaling Parameter (Pretraining FLOP)",
    ylabel=r"$\mathbb{E}[-\log p(x_t | x_{<t})]$",
    yscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=avg_neg_log_score_x=scaling_parameter_hue=model",
)
# plt.show()

# Subsample 1000 token indices per Dataset.
subsampled_causal_language_modeling_probability_dfs_list = []

# Loop through each dataset
for dataset in src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER:
    # Get dataframe subset for this dataset
    dataset_df = causal_language_modeling_probability_df[
        causal_language_modeling_probability_df["Dataset"] == dataset
    ]

    # Sample 1000 unique Token Idx values from this dataset
    sampled_tokens = dataset_df["Token Idx"].unique()
    sampled_tokens = np.random.RandomState(0).choice(
        sampled_tokens, size=min(250, len(sampled_tokens)), replace=False
    )

    # Get all rows for these token indices
    dataset_sampled = dataset_df[dataset_df["Token Idx"].isin(sampled_tokens)]
    subsampled_causal_language_modeling_probability_dfs_list.append(dataset_sampled)

# Concatenate all subsampled dataframes.
subsampled_causal_language_modeling_probability_df = pd.concat(
    subsampled_causal_language_modeling_probability_dfs_list, axis=0
)

plt.close()
g = sns.relplot(
    data=subsampled_causal_language_modeling_probability_df,
    kind="line",
    x="Scaling Parameter",
    y="Score",
    units="Token Idx",
    hue="Dataset",
    hue_order=src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER,
    col="Dataset",
    col_order=src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER,
    col_wrap=4,
    estimator=None,
    alpha=0.1,
)
g.set(
    xscale="log",
    ylabel=r"$p(x_t | x_{<t})$",
    yscale="log",
    xlabel="Scaling Parameter (Pretraining FLOP)",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Causal Language Modeling")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_vs_x=scaling_parameter_hue=dataset_col=datatset_units=token_idx",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=subsampled_causal_language_modeling_probability_df,
    kind="line",
    x="Scaling Parameter",
    y="Neg Log Score",
    units="Token Idx",
    hue="Dataset",
    hue_order=src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER,
    col="Dataset",
    col_order=src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER,
    col_wrap=4,
    estimator=None,
    alpha=0.1,
)
g.set(
    xscale="log",
    yscale="log",
    ylim=(1e-2, None),
    ylabel=r"$-\log p(x_t | x_{<t})$",
    xlabel="Scaling Parameter (Pretraining FLOP)",
)
# Increase the alpha values in the legend handles.
for legend_handle in g._legend.legend_handles:
    legend_handle.set_alpha(1.0)  # Reset transparency to full opacity
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Causal Language Modeling")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=dataset_col=dataset_units=token_idx",
)
# plt.show()


plt.close()
# Create better bins that handle zero and near-zero values
smallest_nonzero_pass_at_1 = subsampled_causal_language_modeling_probability_df[
    subsampled_causal_language_modeling_probability_df["Score"] > 0.0
]["Score"].min()
# Round smallest_nonzero_value to the nearest power of 10.
smallest_nonzero_pass_at_1 = 10.0 ** np.floor(np.log10(smallest_nonzero_pass_at_1))
log10_smallest_nonzero_pass_at_1 = np.log10(smallest_nonzero_pass_at_1)
log_bins = np.logspace(
    log10_smallest_nonzero_pass_at_1, 0, -int(log10_smallest_nonzero_pass_at_1) * 3 + 1
)
small_value_for_plotting = smallest_nonzero_pass_at_1 / 2.0
all_bins = np.concatenate(
    [[-small_value_for_plotting], [small_value_for_plotting], log_bins]
)
g = sns.displot(
    data=subsampled_causal_language_modeling_probability_df[
        subsampled_causal_language_modeling_probability_df["Scaling Parameter"]
        == subsampled_causal_language_modeling_probability_df["Scaling Parameter"].min()
    ],
    kind="hist",
    x="Score",
    hue="Dataset",
    hue_order=src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER,
    bins=all_bins,
    col="Dataset",
    col_order=src.globals.CAUSAL_LANGUAGE_MODELING_DATASETS_ORDER,
    col_wrap=4,
)
g.set(
    xscale="log",
    ylabel="Count",
    xlabel=r"$p(x_{t} | x_{<t})$",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Causal Language Modeling")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=counts_x=score_hue=model_col=dataset_bins=custom",
)
# plt.show()


print(
    "Finished notebooks/04_pretraining_causal_language_modeling_eda/04_pretraining_causal_language_modeling_eda.py!"
)
