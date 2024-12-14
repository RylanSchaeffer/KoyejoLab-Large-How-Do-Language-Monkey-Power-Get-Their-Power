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

large_language_monkeys_original_pass_at_k_df = src.analyze.create_or_load_large_language_monkeys_original_pass_at_k_df(
    refresh=False,
    # refresh=True,
)


plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=large_language_monkeys_original_pass_at_k_df,
    x="Scaling Parameter",
    y="Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Benchmark",
)
g.set(
    title="Large Language Monkeys",
    xscale="log",
    xlabel=r"Scaling Parameter (Num. Attempts $k$)",
    ylabel=r"$\mathbb{E}[\text{Coverage}]$",
    ylim=(0.0, 1.0),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=scaling_parameter_hue=model",
)
# plt.show()

large_language_monkeys_original_neg_log_avg_pass_at_k_df = (
    large_language_monkeys_original_pass_at_k_df.groupby(
        ["Model", "Benchmark", "Scaling Parameter"]
    )["Score"]
    .mean()
    .reset_index()
)
large_language_monkeys_original_neg_log_avg_pass_at_k_df["Neg Log Score"] = -np.log(
    large_language_monkeys_original_neg_log_avg_pass_at_k_df["Score"]
)


plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=large_language_monkeys_original_neg_log_avg_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Benchmark",
)
g.set(
    title="Large Language Monkeys",
    xscale="log",
    yscale="log",
    ylim=(1e-1, None),
    xlabel=r"Scaling Parameter (Num. Attempts $k$)",
    ylabel=r"$-\log (\mathbb{E}[\text{Coverage}])$",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=model",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=large_language_monkeys_original_pass_at_k_df,
    kind="line",
    x="Scaling Parameter",
    y="Score",
    units="Problem Idx",
    hue="Model",
    col="Model",
    col_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    col_wrap=4,
    estimator=None,
)
g.set(
    xscale="log",
    ylabel=r"$\mathbb{E}[\text{Coverage}]$",
    xlabel=r"Scaling Parameter (Num. Attempts $k$)",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))  # You might need to adjust these values
g.fig.suptitle("Large Language Monkeys")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_vs_x=scaling_parameter_hue=model_col=model_units=problem_idx",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=large_language_monkeys_original_pass_at_k_df,
    kind="line",
    x="Scaling Parameter",
    y="Neg Log Score",
    units="Problem Idx",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    col="Model",
    col_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    col_wrap=4,
    estimator=None,
)
g.set(
    xscale="log",
    yscale="log",
    ylim=(1e-2, 1e1),
    ylabel=r"$-\log(\mathbb{E}[\text{Coverage}])$",
    xlabel=r"Scaling Parameter (Num. Attempts $k$)",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))  # You might need to adjust these values
g.fig.suptitle("Large Language Monkeys")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=model_col=model_units=problem_idx",
)
# plt.show()

plt.close()
# Create better bins that handle zero and near-zero values
smallest_nonzero_pass_at_1 = large_language_monkeys_original_pass_at_k_df[
    large_language_monkeys_original_pass_at_k_df["Score"] > 0.0
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
    data=large_language_monkeys_original_pass_at_k_df[
        (large_language_monkeys_original_pass_at_k_df["Scaling Parameter"] == 1)
        * (large_language_monkeys_original_pass_at_k_df["Benchmark"] == "MATH")
    ],
    kind="hist",
    x="Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    bins=all_bins,
    col="Model",
    col_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    col_wrap=4,
)
g.set(
    xscale="log",
    ylabel="Count",
    xlabel="pass@1",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Large Language Monkeys")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=counts_x=score_hue=model_col=model_bins=custom",
)
# plt.show()

print(
    "Finished notebooks/02_large_language_monkeys_original_eda/02_large_language_monkeys_original_eda.py!"
)
