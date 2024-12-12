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


gpt4_gsm8k_prob_answer_given_problem_df = src.analyze.create_or_load_gpt4_gsm8k_prob_answer_given_problem_df(
    # refresh=False,
    refresh=True,
)

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=gpt4_gsm8k_prob_answer_given_problem_df,
    x="Scaling Parameter",
    y="Score",
    hue="Model Family",
    hue_order=src.globals.GPT4_GSM8K_MODEL_FAMILY_ORDER,
)
g.set(
    xscale="log",
    xlabel=r"Scaling Parameter (Pretraining FLOP)",
    yscale="log",
    ylabel=r"$\mathbb{E}[p(\text{Answer}|\text{Problem})$",
    title="GSM8K (Pythia Models)",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_vs_x=scaling_parameter_hue=model_family",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=gpt4_gsm8k_prob_answer_given_problem_df,
    kind="line",
    x="Scaling Parameter",
    y="Neg Log Score",
    units="Problem Idx",
    hue="Model Family",
    hue_order=src.globals.GPT4_GSM8K_MODEL_FAMILY_ORDER,
    col="Model Family",
    col_order=src.globals.GPT4_GSM8K_MODEL_FAMILY_ORDER,
    # col_wrap=4,
    alpha=0.1,
    estimator=None,
)
g.set(
    xscale="log",
    xlabel=r"Scaling Parameter (Pretraining FLOP)",
    yscale="log",
    ylabel=r"$- \log p(\text{Answer}|\text{Problem})$",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))  # You might need to adjust these values
g.fig.suptitle("GSM8K (Pythia Models)")
g.fig.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=model_family_col=model_family_units=problem_idx",
)
plt.show()

gpt4_gsm8k_avg_prob_answer_given_problem_df = (
    gpt4_gsm8k_prob_answer_given_problem_df.groupby(
        ["Model", "Model Family", "Scaling Parameter"]
    )["Score"]
    .mean()
    .reset_index()
)
gpt4_gsm8k_avg_prob_answer_given_problem_df["Neg Log Score"] = -np.log(
    gpt4_gsm8k_avg_prob_answer_given_problem_df["Score"]
)

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=gpt4_gsm8k_avg_prob_answer_given_problem_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model Family",
    hue_order=src.globals.GPT4_GSM8K_MODEL_FAMILY_ORDER,
)
g.set(
    xscale="log",
    xlabel=r"Scaling Parameter (Pretraining FLOP)",
    yscale="log",
    ylabel=r"$- \log \mathbb{E}[p(\text{Answer}|\text{Problem})$",
    title="GSM8K (Pythia Models)",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=model_family",
)
plt.show()

print("Finished notebooks/03_gpt4_gsm8k_eda/03_gpt4_gsm8k_eda.py!")
