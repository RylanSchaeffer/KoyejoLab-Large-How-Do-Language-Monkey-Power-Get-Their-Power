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

pretraining_probability_df = src.analyze.create_or_load_pretraining_probability_df(
    # refresh=False,
    refresh=True,
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretraining_probability_df,
    x="Scaling Parameter",
    y="Score",
    hue="Dataset",
    style="Model Family",
)
g.set(
    title="Language Modeling Pretraining Loss",
    xscale="log",
    xlabel=r"Scaling Parameter (FLOP)",
    yscale="log",
    ylabel=r"$\mathbb{E}[p(x_t | x_{<t})$",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=scaling_parameter_hue=model",
)
plt.show()

pretraining_neg_log_avg_probability_df = (
    pretraining_probability_df.groupby(
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
    style="Model Family",
)
g.set(
    title="Language Modeling Pretraining Loss",
    xscale="log",
    xlabel=r"Scaling Parameter (FLOP)",
    ylabel=r"$-\log \mathbb{E}[p(x_t | x_{<t})$",
    yscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_avg_score_x=scaling_parameter_hue=model",
)
plt.show()


print("Finished notebooks/04_pretraining_scaling_eda/04_pretraining_scaling_eda.py!")
