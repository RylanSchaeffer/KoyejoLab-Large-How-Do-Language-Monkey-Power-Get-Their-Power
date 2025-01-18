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
)
llmonkeys_pythia_math_neg_log_avg_pass_at_k_df["Neg Log Score"] = -np.log(
    llmonkeys_pythia_math_neg_log_avg_pass_at_k_df["Score"]
)

(
    llmonkeys_pythia_math_neg_log_avg_pass_at_k_df,
    llmonkeys_lst_sqrs_fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    llmonkeys_pythia_math_neg_log_avg_pass_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS,
)
llmonkeys_pythia_math_neg_log_avg_pass_at_k_df["Data"] = "Real"
print("Large Language Monkeys Least Squares Fit: ")
pprint.pprint(llmonkeys_lst_sqrs_fitted_power_law_parameters_df)


llmonkeys_pythia_math_kumaraswamy_binomial_mle_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df(
    refresh=False,
    # refresh=True,
)
print("Large Language Monkeys ScaledKumaraswamy-Binomial 3-Parameter Fit: ")
pprint.pprint(llmonkeys_pythia_math_kumaraswamy_binomial_mle_df)

llmonkeys_pythia_math_simulated_kumaraswamy_binomial_neg_log_avg_pass_at_k = (
    src.analyze.simulate_neg_log_avg_pass_at_k_from_kumaraswamy_binomial_mle_df(
        llmonkeys_pythia_math_kumaraswamy_binomial_mle_df,
        columns_to_save=src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS,
        k_values=np.unique(np.logspace(0, 4, 30, dtype=int)),
    )
)
llmonkeys_pythia_math_simulated_kumaraswamy_binomial_neg_log_avg_pass_at_k[
    "Data"
] = "Simulated"


llmonkeys_pythia_math_neg_log_avg_combined_kumaraswamy_binomial_pass_at_k_df = (
    pd.concat(
        [
            llmonkeys_pythia_math_neg_log_avg_pass_at_k_df,
            llmonkeys_pythia_math_simulated_kumaraswamy_binomial_neg_log_avg_pass_at_k,
        ],
        ignore_index=True,
    )
)


# Plot real power laws versus distributionally-estimated power laws under Kumaraswamy-Binomial model.
plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    llmonkeys_pythia_math_neg_log_avg_combined_kumaraswamy_binomial_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Data",
    style_order=["Real", "Simulated"],
)
g.set(
    title="Large Language Monkeys (Benchmark: MATH)",
    xscale="log",
    xlabel="Scaling Parameter",
    yscale="log",
    ylabel="Neg Log Score",
    ylim=(1e-1, None),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkeys_y=neg_log_score_x=scaling_parameter_hue=model_style=real_or_syn_kumaraswamy_binomial",
)
# plt.show()


llmonkeys_joint_power_law_and_kumaraswamy_binomial_distr_fit_df = pd.merge(
    llmonkeys_pythia_math_kumaraswamy_binomial_mle_df[
        src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    llmonkeys_lst_sqrs_fitted_power_law_parameters_df[
        src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    on=src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS,
    how="inner",
    suffixes=("_KumarBinom", "_LstSqrs"),
)


plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=llmonkeys_joint_power_law_and_kumaraswamy_binomial_distr_fit_df,
    x="Power Law Exponent_KumarBinom",
    y="Power Law Exponent_LstSqrs",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Benchmark",
    s=150,
)
# Plot dotted identity line.
g.plot([0, 1], [0, 1], ls="--", c=".3")
g.set(
    title="Large Language Monkeys",
    xlim=(0.0, 0.6),
    ylim=(0.0, 0.6),
    xlabel=r"Power Law Exponent (Kumaraswamy-Binomial)",
    ylabel="Power Law Exponent (Least Squares)",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkeys_y=scaling_law_exponent_x=distributional_fit_exponent_kumaraswamy_binomial",
)
# plt.show()


llmonkeys_pythia_math_beta_binomial_mle_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_beta_binomial_mle_df(
    refresh=False,
    # refresh=True,
)
print("Large Language Monkeys ScaledBeta-Binomial 3-Parameter Fit: ")
pprint.pprint(llmonkeys_pythia_math_beta_binomial_mle_df)


llmonkeys_pythia_math_simulated_beta_binomial_neg_log_avg_pass_at_k = (
    src.analyze.simulate_neg_log_avg_pass_at_k_from_beta_binomial_mle_df(
        llmonkeys_pythia_math_beta_binomial_mle_df,
        columns_to_save=src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS,
        k_values=np.unique(np.logspace(0, 4, 30, dtype=int)),
    )
)
llmonkeys_pythia_math_simulated_beta_binomial_neg_log_avg_pass_at_k[
    "Data"
] = "Simulated"

llmonkeys_pythia_math_neg_log_avg_combined_beta_binomial_pass_at_k_df = pd.concat(
    [
        llmonkeys_pythia_math_neg_log_avg_pass_at_k_df,
        llmonkeys_pythia_math_simulated_beta_binomial_neg_log_avg_pass_at_k,
    ],
    ignore_index=True,
)


# Plot real power laws versus distributionally-estimated power laws.
plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    llmonkeys_pythia_math_neg_log_avg_combined_beta_binomial_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Data",
    style_order=["Real", "Simulated"],
)
g.set(
    title="Large Language Monkeys (Benchmark: MATH)",
    xscale="log",
    xlabel="Scaling Parameter",
    yscale="log",
    ylabel="Neg Log Score",
    ylim=(1e-1, None),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkeys_y=neg_log_score_x=scaling_parameter_hue=model_style=real_or_syn_beta_binomial",
)
# plt.show()

llmonkeys_joint_power_law_and_beta_binomial_distr_fit_df = pd.merge(
    llmonkeys_pythia_math_beta_binomial_mle_df[
        src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    llmonkeys_lst_sqrs_fitted_power_law_parameters_df[
        src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    on=src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS,
    how="inner",
    suffixes=("_BetaBinom", "_LstSqrs"),
)


plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=llmonkeys_joint_power_law_and_beta_binomial_distr_fit_df,
    x="Power Law Exponent_BetaBinom",
    y="Power Law Exponent_LstSqrs",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Benchmark",
    s=150,
)
# Plot dotted identity line.
g.plot([0, 1], [0, 1], ls="--", c=".3")
g.set(
    title="Large Language Monkeys",
    xlim=(0.0, 0.6),
    ylim=(0.0, 0.6),
    xlabel=r"Power Law Exponent (Beta-Binomial)",
    ylabel="Power Law Exponent (Least Squares)",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkeys_y=scaling_law_exponent_x=distributional_fit_exponent_beta_binomial",
)
# plt.show()


bon_jailbreaking_text_pass_at_k_df = src.analyze.create_or_load_bon_jailbreaking_text_pass_at_k_df(
    refresh=False,
    # refresh=True,
)

bon_jailbreaking_text_neg_log_avg_pass_at_k_df = (
    bon_jailbreaking_text_pass_at_k_df.groupby(
        src.globals.BON_JAILBREAKING_GROUPBY_COLS + ["Scaling Parameter"]
    )["Score"]
    .mean()
    .reset_index()
)
bon_jailbreaking_text_neg_log_avg_pass_at_k_df["Neg Log Score"] = -np.log(
    bon_jailbreaking_text_neg_log_avg_pass_at_k_df["Score"]
)
bon_jailbreaking_text_neg_log_avg_pass_at_k_df["Data"] = "Real"

(
    _,
    bon_jailbreaking_text_lst_sqrs_fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    bon_jailbreaking_text_neg_log_avg_pass_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=src.globals.BON_JAILBREAKING_GROUPBY_COLS,
)
print("Best-of-N Jailbreaking Least Squares Fit: ")
pprint.pprint(bon_jailbreaking_text_lst_sqrs_fitted_power_law_parameters_df)

bon_jailbreaking_text_kumaraswamy_binomial_mle_df = src.analyze.create_or_load_bon_jailbreaking_text_kumaraswamy_binomial_mle_df(
    refresh=False,
    # refresh=True,
)
print("Best-of-N Jailbreaking ScaledKumaraswamy-Binomial 3-Parameter Fit: ")
pprint.pprint(bon_jailbreaking_text_kumaraswamy_binomial_mle_df)

bon_jailbreaking_text_simulated_kumaraswamy_binomial_neg_log_avg_pass_at_k = (
    src.analyze.simulate_neg_log_avg_pass_at_k_from_kumaraswamy_binomial_mle_df(
        bon_jailbreaking_text_kumaraswamy_binomial_mle_df,
        columns_to_save=src.globals.BON_JAILBREAKING_GROUPBY_COLS,
        k_values=np.unique(np.logspace(0, 4, 30, dtype=int)),
    )
)
bon_jailbreaking_text_simulated_kumaraswamy_binomial_neg_log_avg_pass_at_k[
    "Data"
] = "Simulated"


bon_jailbreaking_text_neg_log_avg_combined_kumaraswamy_binomial_pass_at_k_df = (
    pd.concat(
        [
            bon_jailbreaking_text_neg_log_avg_pass_at_k_df,
            bon_jailbreaking_text_simulated_kumaraswamy_binomial_neg_log_avg_pass_at_k,
        ],
        ignore_index=True,
    )
)


# Plot real power laws versus distributionally-estimated power laws under Kumaraswamy-Binomial model.
plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    bon_jailbreaking_text_neg_log_avg_combined_kumaraswamy_binomial_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_TEXT_MODELS_ORDER,
    style="Data",
    style_order=["Real", "Simulated"],
)
g.set(
    title="Best-of-N Jailbreaking (Modality: Text)",
    xscale="log",
    xlabel="Scaling Parameter",
    yscale="log",
    ylabel="Neg Log Score",
    ylim=(3e-2, 8e0),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="bon_jailbreaking_y=neg_log_score_x=scaling_parameter_hue=model_style=real_or_syn_kumaraswamy_binomial",
)
# plt.show()


bon_jailbreaking_text_joint_power_law_and_kumaraswamy_binomial_distr_fit_df = pd.merge(
    bon_jailbreaking_text_kumaraswamy_binomial_mle_df[
        src.globals.BON_JAILBREAKING_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    bon_jailbreaking_text_lst_sqrs_fitted_power_law_parameters_df[
        src.globals.BON_JAILBREAKING_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    on=src.globals.BON_JAILBREAKING_GROUPBY_COLS,
    how="inner",
    suffixes=("_KumarBinom", "_LstSqrs"),
)


plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=bon_jailbreaking_text_joint_power_law_and_kumaraswamy_binomial_distr_fit_df,
    x="Power Law Exponent_KumarBinom",
    y="Power Law Exponent_LstSqrs",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_TEXT_MODELS_ORDER,
    style="Modality",
    s=150,
)
# Plot dotted identity line.
g.plot([0, 1], [0, 1], ls="--", c=".3")
g.set(
    title="Large Language Monkeys",
    xlim=(0.0, 0.6),
    ylim=(0.0, 0.6),
    xlabel=r"Power Law Exponent (Kumaraswamy-Binomial)",
    ylabel="Power Law Exponent (Least Squares)",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="bon_jailbreaking_y=scaling_law_exponent_x=distributional_fit_exponent_kumaraswamy_binomial",
)
# plt.show()


bon_jailbreaking_beta_binomial_mle_df = src.analyze.create_or_load_bon_jailbreaking_text_beta_binomial_mle_df(
    # refresh=False,
    refresh=True,
)
print("Best-of-N Jailbreaking ScaledBeta-Binomial 3-Parameter Fit: ")
pprint.pprint(bon_jailbreaking_beta_binomial_mle_df)

bon_jailbreaking_simulated_neg_log_avg_pass_at_k = (
    src.analyze.simulate_neg_log_avg_pass_at_k_from_beta_binomial_mle_df(
        beta_binomial_df=bon_jailbreaking_beta_binomial_mle_df,
        columns_to_save=src.globals.BON_JAILBREAKING_GROUPBY_COLS,
        k_values=np.unique(np.logspace(0, 4, 30, dtype=int)),
    )
)
bon_jailbreaking_simulated_neg_log_avg_pass_at_k["Data"] = "Simulated"

bon_jailbreaking_neg_log_avg_combined_pass_at_k_df = pd.concat(
    [
        bon_jailbreaking_text_neg_log_avg_pass_at_k_df,
        bon_jailbreaking_simulated_neg_log_avg_pass_at_k,
    ],
    ignore_index=True,
)


# Plot real power laws versus distributionally-estimated power laws.
plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    bon_jailbreaking_neg_log_avg_combined_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_TEXT_MODELS_ORDER,
    style="Data",
    style_order=["Real", "Simulated"],
)
g.set(
    title="Best-of-N Jailbreaking (Modality: Text)",
    xscale="log",
    xlabel="Scaling Parameter",
    yscale="log",
    ylabel="Neg Log Score",
    ylim=(3e-2, 8e0),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="bon_jailbreaking_y=neg_log_score_x=scaling_parameter_hue=model_style=real_or_syn_beta_binomial",
)
# plt.show()


bon_jailbreaking_joint_power_law_and_distr_fit_df = pd.merge(
    bon_jailbreaking_beta_binomial_mle_df[
        src.globals.BON_JAILBREAKING_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    bon_jailbreaking_text_lst_sqrs_fitted_power_law_parameters_df[
        src.globals.BON_JAILBREAKING_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    on=src.globals.BON_JAILBREAKING_GROUPBY_COLS,
    how="inner",
    suffixes=("_BetaBinom", "_LstSqrs"),
)
plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=bon_jailbreaking_joint_power_law_and_distr_fit_df,
    x="Power Law Exponent_BetaBinom",
    y="Power Law Exponent_LstSqrs",
    hue="Model",
    hue_order=src.globals.BON_JAILBREAKING_TEXT_MODELS_ORDER,
    style="Modality",
    s=150,
)
# Plot dotted identity line.
g.plot([0, 1], [0, 1], ls="--", c=".3")
g.set(
    title="Best-of-N Jailbreaking",
    xlim=(0.0, 0.6),
    ylim=(0.0, 0.6),
    xlabel=r"Power Law Exponent (Beta-Binomial)",
    ylabel="Power Law Exponent (Least Squares)",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="bon_jailbreaking_y=scaling_law_exponent_x=distributional_fit_exponent_beta_binomial",
)
# plt.show()


print("Finished notebooks/51_pass@1_compare_power_law_with_distributional_fit!")
