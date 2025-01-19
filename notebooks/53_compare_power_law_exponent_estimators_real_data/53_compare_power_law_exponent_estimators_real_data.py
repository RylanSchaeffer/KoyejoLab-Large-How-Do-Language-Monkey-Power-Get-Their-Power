import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import pprint
import scipy.stats
import seaborn as sns

import src.analyze
import src.globals
import src.plot
import src.utils


data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


# Load backtested scaling coefficient fits.
llmonkeys_cross_validated_scaling_coeff_df = src.analyze.create_or_load_cross_validated_large_language_monkey_pythia_math_scaling_coefficient_data_df(
    refresh=False,
    # refresh=True,
)

# Convert the scaling parameters to forecasts.
ks_list = np.unique(np.logspace(0, 5, 100).astype(int))
predicted_power_law_curves_dfs_list = []
for row_idx, row in llmonkeys_cross_validated_scaling_coeff_df.iterrows():
    df = pd.DataFrame.from_dict(
        {
            "Scaling Parameter": ks_list,
            "Neg Log Score": row["Fit Power Law Prefactor"]
            * np.power(ks_list, -row["Fit Power Law Exponent"]),
            "Num. Problems": [row["Num. Problems"]] * len(ks_list),
            "Num. Samples per Problem": [row["Num. Samples per Problem"]]
            * len(ks_list),
            "Repeat Index": [row["Repeat Index"]] * len(ks_list),
            "Fit Method": [row["Fit Method"]] * len(ks_list),
            "Model": [row["Model"]] * len(ks_list),
            "Benchmark": [row["Benchmark"]] * len(ks_list),
        }
    )
    predicted_power_law_curves_dfs_list.append(df)
predicted_power_law_curves_df = pd.concat(
    predicted_power_law_curves_dfs_list, ignore_index=True
).reset_index(drop=True)

plt.close()
g = sns.relplot(
    data=predicted_power_law_curves_df[
        predicted_power_law_curves_df["Num. Problems"]
        == 128  # Plot only a slice because otherwise too many variables.
    ],
    kind="line",
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    col="Model",
    col_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    row="Num. Samples per Problem",
    style="Fit Method",
    facet_kws={"margin_titles": True},
)
g.set(
    xscale="log",
    yscale="log",
    xlabel=r"Num. Attempts per Problem $k$",
    ylabel=r"$-\log ( \operatorname{pass_{\mathcal{D}}@k})$",
)
num_samples_per_problem_values = np.sort(
    predicted_power_law_curves_df["Num. Samples per Problem"].unique()
)
for row_idx in range(g.axes.shape[0]):
    num_samples_per_problem = num_samples_per_problem_values[row_idx]
    for col_idx in range(g.axes.shape[1]):
        ax = g.axes[row_idx, col_idx]
        ax.axvline(
            x=num_samples_per_problem,
            color="k",
            linestyle="--",
        )
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_x=scaling_parameter_hue=model_col=num_samples_per_problem_row=num_problems_style=fit_method",
)
plt.show()

plt.close()
g = sns.relplot(
    data=llmonkeys_cross_validated_scaling_coeff_df,
    kind="line",
    x="Num. Samples per Problem",
    y="Fit Power Law Exponent",
    col="Model",
    col_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    row="Num. Problems",
    style="Fit Method",
    # col_wrap=4,
    facet_kws={"margin_titles": True},
)
g.set(
    xscale="log",
    yscale="log",
    xlabel="Num. Attempts per Problem",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=fit_power_law_exponent_x=num_samples_per_problem_hue=model_col=model_row=num_problems_style=fit_method",
)
# plt.show()


# Load the actual pass_D@k data to overlay.
llmonkeys_pythia_math_pass_at_k_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_pass_at_k_df(
    refresh=False,
    # refresh=True,
)

plt.close()
