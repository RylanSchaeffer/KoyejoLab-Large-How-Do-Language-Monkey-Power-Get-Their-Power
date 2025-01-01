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

bon_jailbreaking_pass_at_k_df = src.analyze.create_or_load_bon_jailbreaking_pass_at_k_df(
    refresh=False,
    # refresh=True,
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=bon_jailbreaking_pass_at_k_df,
    x="Scaling Parameter",
    y="Score",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    style="Modality",
)
g.set(
    title="Best-of-N Jailbreaking",
    xlabel=r"Num. Attempts per Prompt $k$",
    ylabel=r"$\operatorname{ASR_{\mathcal{D}}@k}$",
    ylim=(-0.05, 1.05),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=scaling_parameter_hue=model",
)
# plt.show()

bon_jailbreaking_neg_log_avg_pass_at_k_df = (
    bon_jailbreaking_pass_at_k_df.groupby(["Model", "Modality", "Scaling Parameter"])[
        "Score"
    ]
    .mean()
    .reset_index()
)
bon_jailbreaking_neg_log_avg_pass_at_k_df["Neg Log Score"] = -np.log(
    bon_jailbreaking_neg_log_avg_pass_at_k_df["Score"]
)

(
    bon_jailbreaking_neg_log_avg_pass_at_k_df,
    fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    bon_jailbreaking_neg_log_avg_pass_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=["Model", "Modality"],
)

print("Fitted Power Laws Parameters: ", fitted_power_law_parameters_df)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=bon_jailbreaking_neg_log_avg_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    style="Modality",
)
g = sns.lineplot(
    data=bon_jailbreaking_neg_log_avg_pass_at_k_df,
    x="Scaling Parameter",
    y="Predicted Neg Log Score",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    # style="Modality",
    legend=False,
    linestyle="--",
)
g.set(
    title="Best-of-N Jailbreaking",
    xscale="log",
    yscale="log",
    ylim=(3e-2, 8e0),
    xlabel=r"Num. Attempts per Prompt $k$",
    ylabel=r"$-\log (\operatorname{ASR_{\mathcal{D}}@k})$",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_avg_score_vs_x=scaling_parameter_hue=model",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=bon_jailbreaking_pass_at_k_df,
    kind="line",
    x="Scaling Parameter",
    y="Score",
    units="Problem Idx",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    col="Model",
    col_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    col_wrap=4,
    estimator=None,
)
g.set(
    xscale="log",
    ylabel=r"$\operatorname{ASR_i@k}$",
    xlabel=r"Num. Attempts per Prompt $k$",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Best-of-N Jailbreaking")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_vs_x=scaling_parameter_hue=model_col=model_units=problem_idx",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=bon_jailbreaking_pass_at_k_df,
    kind="line",
    x="Scaling Parameter",
    y="Neg Log Score",
    units="Problem Idx",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    col="Model",
    col_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    col_wrap=4,
    estimator=None,
)
g.set(
    xscale="log",
    yscale="log",
    ylim=(1e-2, 1e1),
    ylabel=r"$-\log(\operatorname{ASR_i@k})$",
    xlabel=r"Num. Attempts per Prompt $k$",
)
# For each subplot, plot the aggregate power law behavior in black.
for ax, model in zip(g.axes.flat, src.globals.BON_JAILBREAKING_MODELS_ORDER):
    model_df = bon_jailbreaking_neg_log_avg_pass_at_k_df[
        bon_jailbreaking_neg_log_avg_pass_at_k_df["Model"] == model
    ]
    model_df = model_df.sort_values("Scaling Parameter")
    ax.plot(
        model_df["Scaling Parameter"],
        model_df["Predicted Neg Log Score"],
        color="black",
        linewidth=6,
    )
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Best-of-N Jailbreaking")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=model_col=model_units=problem_idx",
)
# plt.show()


plt.close()
# Create better bins that handle zero and near-zero values
smallest_nonzero_pass_at_1 = bon_jailbreaking_pass_at_k_df[
    bon_jailbreaking_pass_at_k_df["Score"] > 0.0
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
    data=bon_jailbreaking_pass_at_k_df[
        bon_jailbreaking_pass_at_k_df["Scaling Parameter"] == 1
    ],
    kind="hist",
    x="Score",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    bins=all_bins,
    col="Model",
    col_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    col_wrap=4,
)
g.set(
    xscale="log",
    ylabel="Count",
    xlabel=r"$\operatorname{ASR_i@1}$",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Best-of-N Jailbreaking")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=counts_x=score_hue=model_col=model_bins=custom",
)
# plt.show()


print("Finished notebooks/00_bon_jailbreaking_eda/00_bon_jailbreaking_eda.py!")
