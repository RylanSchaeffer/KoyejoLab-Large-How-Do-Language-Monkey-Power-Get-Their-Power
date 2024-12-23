import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import pprint

import scipy.stats
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

bon_jailbreaking_pass_at_1_df = bon_jailbreaking_pass_at_k_df[
    (bon_jailbreaking_pass_at_k_df["Scaling Parameter"] == 1)
    & (bon_jailbreaking_pass_at_k_df["Modality"] == "Text")
].copy()
bon_jailbreaking_pass_at_1_beta_fits_df = (
    bon_jailbreaking_pass_at_1_df.groupby(["Model", "Modality", "Scaling Parameter"])
    .apply(
        lambda df: src.analyze.fit_pass_at_1_beta_distribution_parameters(
            data=df["Score"].values
        )
    )
    .reset_index()
)
print("Best-of-N Jailbreaking Beta Fit: ")
pprint.pprint(bon_jailbreaking_pass_at_1_beta_fits_df)

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
    _,
    fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    bon_jailbreaking_neg_log_avg_pass_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=["Model", "Modality"],
)

bon_jailbreaking_joint_power_law_and_distr_fit_df = pd.merge(
    bon_jailbreaking_pass_at_1_beta_fits_df[["Model", "alpha"]],
    fitted_power_law_parameters_df[["Model", "b"]],
    on=["Model"],
    how="inner",
)
plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=bon_jailbreaking_joint_power_law_and_distr_fit_df,
    x="alpha",
    y="b",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    s=150,
)
g.set(
    title="Best-of-N Jailbreaking",
    xlabel="Distribution-Predicted Exponent",
    xlim=(0.0, 0.6),
    ylim=(0.0, 0.6),
    ylabel="Scaling Law-Fit Exponent",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=scaling_law_exponent_x=distributional_fit_exponent",
)
plt.show()


print("Finished notebooks/51_pass@1_compare_power_law_with_distributional_fit!")
