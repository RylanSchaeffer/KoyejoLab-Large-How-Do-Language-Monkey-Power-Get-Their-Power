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

many_shot_icl_probability_df = src.analyze.create_or_load_many_shot_icl_probability_df(
    # refresh=False,
    refresh=True,
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=many_shot_icl_probability_df,
    x="Scaling Parameter",
    y="Score",
    hue="Dataset",
    hue_order=src.globals.MANY_SHOT_IN_CONTEXT_LEARNING_DATASET_ORDER,
    style="Model",
)
g.set(
    title="Many-Shot In-Context Learning",
    xscale="log",
    xlabel=r"Scaling Parameter (Num. Shots)",
    yscale="log",
    ylabel=r"$\mathbb{E}[p(a_t | q_t, a_{<t}, q_{<t})]$",
    # ylim=(1e-1, None),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=scaling_parameter_hue=model",
)
# plt.show()

many_shot_icl_neg_log_avg_probability_df = (
    many_shot_icl_probability_df.groupby(["Model", "Dataset", "Scaling Parameter"])[
        "Score"
    ]
    .mean()
    .reset_index()
)
many_shot_icl_neg_log_avg_probability_df["Log Score"] = np.log(
    many_shot_icl_neg_log_avg_probability_df["Score"]
)
many_shot_icl_neg_log_avg_probability_df[
    "Neg Log Score"
] = -many_shot_icl_neg_log_avg_probability_df["Log Score"]

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=many_shot_icl_neg_log_avg_probability_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Dataset",
    hue_order=src.globals.MANY_SHOT_IN_CONTEXT_LEARNING_DATASET_ORDER,
    style="Model",
)
g.set(
    title="Many-Shot In-Context Learning",
    xscale="log",
    xlabel=r"Scaling Parameter (Pretraining FLOP)",
    ylabel=r"$-\log \mathbb{E}[p(a_t | q_t, a_{<t}, q_{<t})]$",
    yscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_avg_score_x=scaling_parameter_hue=model",
)
# plt.show()


print("Finished notebooks/05_many_shot_in_context_scaling!")
