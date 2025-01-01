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

llmonkeys_groupby_cols = ["Model", "Benchmark"]


llmonkeys_original_pass_at_k_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_pass_at_k_df(
    refresh=False,
    # refresh=True,
)

large_language_monkeys_original_neg_log_avg_pass_at_k_df = (
    llmonkeys_original_pass_at_k_df.groupby(
        llmonkeys_groupby_cols + ["Scaling Parameter"]
    )["Score"]
    .mean()
    .reset_index()
)
large_language_monkeys_original_neg_log_avg_pass_at_k_df["Neg Log Score"] = -np.log(
    large_language_monkeys_original_neg_log_avg_pass_at_k_df["Score"]
)

(
    _,
    llmonkeys_fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    large_language_monkeys_original_neg_log_avg_pass_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=llmonkeys_groupby_cols,
)
print(
    "Large Language Monkeys Least Squares Fit: ",
    llmonkeys_fitted_power_law_parameters_df,
)


llmonkeys_individual_outcomes_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
    refresh=False,
    # refresh=True,
)

llmonkeys_num_samples_and_num_successes_df = (
    src.analyze.convert_individual_outcomes_to_num_samples_and_num_successes(
        individual_outcomes_df=llmonkeys_individual_outcomes_df,
        groupby_cols=llmonkeys_groupby_cols + ["Problem Idx"],
    )
)

llmonkeys_beta_binomial_three_parameters_fits_df = (
    llmonkeys_num_samples_and_num_successes_df.groupby(llmonkeys_groupby_cols)
    .apply(
        lambda df: src.analyze.fit_beta_binomial_three_parameters_to_num_samples_and_num_successes(
            num_samples_and_num_successes_df=df
        )
    )
    .reset_index()
)

print(
    "Large Language Monkeys Beta-Binomial 3-Parameter Fit: ",
    llmonkeys_beta_binomial_three_parameters_fits_df,
)

llmonkeys_beta_binomial_two_parameters_fits_df = (
    llmonkeys_num_samples_and_num_successes_df.groupby(llmonkeys_groupby_cols)
    .apply(
        lambda df: src.analyze.fit_beta_binomial_three_parameters_to_num_samples_and_num_successes(
            num_samples_and_num_successes_df=df
        )
    )
    .reset_index()
)

print(
    "Large Language Monkeys BetaBinomial 2-Parameter Fit: ",
    llmonkeys_beta_binomial_two_parameters_fits_df,
)


llmonkeys_joint_power_law_and_distr_fit_df = pd.merge(
    llmonkeys_beta_binomial_two_parameters_fits_df[["Model", "Power Law Exponent"]],
    llmonkeys_fitted_power_law_parameters_df[["Model", "Power Law Exponent"]],
    on=["Model"],
    how="inner",
    suffixes=("_BetaBinom", "_LstSqrs"),
)


plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=llmonkeys_joint_power_law_and_distr_fit_df,
    x="Power Law Exponent_BetaBinom",
    y="Power Law Exponent_LstSqrs",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    s=150,
)
# Plot dotted identity line.
g.plot([0, 1], [0, 1], ls="--", c=".3")
g.set(
    title="Large Language Monkeys",
    xlim=(0.00, 0.6),
    ylim=(0.00, 0.6),
    xlabel=r"Power Law Exponent (Beta-Binomial)",
    ylabel="Power Law Exponent (Least Squares)",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkeys_y=scaling_law_exponent_x=distributional_fit_exponent",
)
plt.show()


bon_jailbreaking_pass_at_k_df = src.analyze.create_or_load_bon_jailbreaking_pass_at_k_df(
    # refresh=False,
    refresh=True,
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
bon_jailbreaking_pass_at_1_beta_fits_df = (
    src.analyze.compute_scaling_exponent_from_distributional_fit(
        distributional_fit_df=bon_jailbreaking_pass_at_1_beta_fits_df,
        distribution="beta_three_parameter",
        k_values=src.globals.BON_JAILBREAKING_Ks_LIST,
    )
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
    bon_jailbreaking_fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    bon_jailbreaking_neg_log_avg_pass_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=["Model", "Modality"],
)

bon_jailbreaking_joint_power_law_and_distr_fit_df = pd.merge(
    bon_jailbreaking_pass_at_1_beta_fits_df[["Model", "b"]],
    bon_jailbreaking_fitted_power_law_parameters_df[["Model", "b"]],
    on=["Model"],
    how="inner",
    suffixes=("_distribution", "_power_law"),
)
plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=bon_jailbreaking_joint_power_law_and_distr_fit_df,
    x="b_distribution",
    y="b_power_law",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_MODELS_ORDER,
    s=150,
)
g.set(
    title="Best-of-N Jailbreaking",
    xlim=(0.0, 0.6),
    ylim=(0.0, 0.6),
    xlabel=r"Power Law Exponent (Distribution-Derived)",
    ylabel="Power Law Exponent (Directly Fit)",
)
# Plot dotted identity line.
g.plot([0, 1], [0, 1], ls="--", c=".3")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="bon_jailbreaking_y=scaling_law_exponent_x=distributional_fit_exponent",
)
plt.show()


llmonkeys_original_pass_at_k_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_pass_at_k_df(
    refresh=False,
    # refresh=True,
)

llmonkeys_original_pass_at_1_df = llmonkeys_original_pass_at_k_df[
    (llmonkeys_original_pass_at_k_df["Scaling Parameter"] == 1)
    & (llmonkeys_original_pass_at_k_df["Benchmark"] == "MATH")
].copy()


llmonkeys_pass_at_1_beta_fits_df = (
    llmonkeys_original_pass_at_1_df.groupby(["Model", "Benchmark", "Scaling Parameter"])
    .apply(
        lambda df: src.analyze.fit_pass_at_1_beta_distribution_parameters(
            data=df["Score"].values
        )
    )
    .reset_index()
)
llmonkeys_pass_at_1_beta_fits_df = (
    src.analyze.compute_scaling_exponent_from_distributional_fit(
        distributional_fit_df=llmonkeys_pass_at_1_beta_fits_df,
        distribution="beta_three_parameter",
        k_values=src.globals.LARGE_LANGUAGE_MONKEYS_ORIGINAL_Ks_LIST,
    )
)
print("Large Language Monkey Beta Fit: ")
pprint.pprint(llmonkeys_pass_at_1_beta_fits_df)


print("Finished notebooks/51_pass@1_compare_power_law_with_distributional_fit!")
