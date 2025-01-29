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

bon_jailbreaking_text_cross_validated_scaling_coeff_df = src.analyze.create_or_load_cross_validated_bon_jailbreaking_text_scaling_coefficient_data_df(
    # refresh=False,
    refresh=True,
)

# Load backtested scaling coefficient fits.
llmonkeys_pythia_math_cross_validated_scaling_coeff_df = src.analyze.create_or_load_cross_validated_large_language_monkey_pythia_math_scaling_coefficient_data_df(
    # refresh=False,
    refresh=True,
)
# # Exclude this antequated method.
# llmonkeys_pythia_math_cross_validated_scaling_coeff_df = (
#     llmonkeys_pythia_math_cross_validated_scaling_coeff_df[
#         llmonkeys_pythia_math_cross_validated_scaling_coeff_df["Fit Method"]
#         != "Beta-Binomial"
#     ]
# )


# Load the actual pass_D@k data to overlay.
llmonkeys_pythia_math_pass_at_k_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_pass_at_k_df(
    refresh=False,
    # refresh=True,
)

llmonkeys_pythia_math_neg_log_avg_pass_at_k_df = (
    llmonkeys_pythia_math_pass_at_k_df.groupby(
        src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS + ["Scaling Parameter"]
    )["Score"]
    .mean()
    .reset_index()
    .rename(columns={"Score": "Avg Score"})
)
llmonkeys_pythia_math_neg_log_avg_pass_at_k_df["Neg Log Avg Score"] = -np.log(
    llmonkeys_pythia_math_neg_log_avg_pass_at_k_df["Avg Score"]
)


# Convert the scaling parameters to forecasts.
ks_list = np.unique(np.logspace(0, 4, 100).astype(int))
predicted_power_law_curves_dfs_list = []
for row_idx, row in llmonkeys_pythia_math_cross_validated_scaling_coeff_df.iterrows():
    df = pd.DataFrame.from_dict(
        {
            "Scaling Parameter": ks_list,
            "Neg Log Avg Score": row["Fit Power Law Prefactor"]
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


# Compute mean squared error between predicted and actual pass_D@k.
llmonkeys_pythia_math_neg_log_avg_pass_at_10000_df = (
    llmonkeys_pythia_math_neg_log_avg_pass_at_k_df[
        llmonkeys_pythia_math_neg_log_avg_pass_at_k_df["Scaling Parameter"] == 10000
    ]
)
predicted_power_law_curves_at_10000_df = predicted_power_law_curves_df[
    predicted_power_law_curves_df["Scaling Parameter"] == 10000
]
joint_neg_log_avg_score_at_10000_df = (
    llmonkeys_pythia_math_neg_log_avg_pass_at_10000_df.merge(
        predicted_power_law_curves_at_10000_df,
        on=src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS + ["Scaling Parameter"],
        how="outer",
        suffixes=("_Actual", "_Predicted"),
    )
)

joint_neg_log_avg_score_at_10000_df["Squared Log Error"] = np.square(
    np.log(joint_neg_log_avg_score_at_10000_df["Neg Log Avg Score_Actual"])
    - np.log(joint_neg_log_avg_score_at_10000_df["Neg Log Avg Score_Predicted"])
)
plt.close()
g = sns.relplot(
    data=joint_neg_log_avg_score_at_10000_df,
    kind="line",
    x="Num. Samples per Problem",
    y="Squared Log Error",
    col="Model",
    col_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    row="Num. Problems",
    style="Fit Method",
    facet_kws={"margin_titles": True},
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="",  # We will add this ourselves.
)
g.axes[int(g.axes.shape[0] // 2), 0].set_ylabel(
    "Squared Log Error "
    + r"$:= \Big( \log \big( \operatorname{pass_{\mathcal{D}}@10000} \big) - \log \big( \widehat{\operatorname{pass_{\mathcal{D}}@10000}} \big) \Big)^2$"
)
g.set_titles(row_template="{row_name} Problems", col_template="{col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkeys_y=mean_squared_log_error_x=num_samples_per_problem_hue=model_col=model_row=num_problems_style=fit_method",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=predicted_power_law_curves_df[
        predicted_power_law_curves_df["Num. Problems"]
        == 128  # Plot only a slice because otherwise too many variables.
    ],
    kind="line",
    x="Scaling Parameter",
    y="Neg Log Avg Score",
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
g.set_titles(row_template="{row_name} Samples per Problem")
num_samples_per_problem_values = np.sort(
    predicted_power_law_curves_df["Num. Samples per Problem"].unique()
)
for col_idx, model in enumerate(src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER):
    model_subset_df = llmonkeys_pythia_math_neg_log_avg_pass_at_k_df[
        llmonkeys_pythia_math_neg_log_avg_pass_at_k_df["Model"] == model
    ]
    for row_idx, num_samples_per_problem in enumerate(num_samples_per_problem_values):
        ax = g.axes[row_idx, col_idx]
        sns.lineplot(
            data=model_subset_df,
            x="Scaling Parameter",
            y="Neg Log Avg Score",
            # hue="Model",
            # hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
            legend=False,
            color="black",
            # alpha=0.5,
            ax=ax,
        )
        ax.axvline(
            x=num_samples_per_problem,
            color="k",
            linestyle="--",
        )
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkeys_y=neg_log_avg_score_x=scaling_parameter_hue=model_col=model_row=num_samples_per_problem_style=fit_method",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=llmonkeys_pythia_math_cross_validated_scaling_coeff_df,
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
    ylabel=r"Fit Power Law Exponent $\hat{b}$",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkeys_y=fit_power_law_exponent_x=num_samples_per_problem_hue=model_col=model_row=num_problems_style=fit_method",
)
# plt.show()

# plt.close()

print("Finished notebooks/53_compare_power_law_exponent_estimators_real_data!")
