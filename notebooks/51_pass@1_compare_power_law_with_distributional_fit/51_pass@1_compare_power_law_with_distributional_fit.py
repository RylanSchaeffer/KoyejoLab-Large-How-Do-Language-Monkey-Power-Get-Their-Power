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
print("Large Language Monkeys Least Squares Fit: ")
pprint.pprint(llmonkeys_lst_sqrs_fitted_power_law_parameters_df)


llmonkeys_pythia_math_beta_binomial_mle_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_beta_binomial_mle_df(
    refresh=False,
    # refresh=True,
)


print(
    "Large Language Monkeys ScaledBeta-Binomial 3-Parameter Fit: ",
)
pprint.pprint(llmonkeys_pythia_math_beta_binomial_mle_df)

llmonkeys_joint_power_law_and_distr_fit_df = pd.merge(
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
    data=llmonkeys_joint_power_law_and_distr_fit_df,
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
    plot_filename="llmonkeys_y=scaling_law_exponent_x=distributional_fit_exponent",
)
# plt.show()


bon_jailbreaking_pass_at_k_df = src.analyze.create_or_load_bon_jailbreaking_text_pass_at_k_df(
    refresh=False,
    # refresh=True,
)

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
    bon_jailbreaking_lst_sqrs_fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    bon_jailbreaking_neg_log_avg_pass_at_k_df,
    covariate_col="Scaling Parameter",
    target_col="Neg Log Score",
    groupby_cols=src.globals.BON_JAILBREAKING_GROUPBY_COLS,
)
print(
    "Best-of-N Jailbreaking Least Squares Fit: ",
    bon_jailbreaking_lst_sqrs_fitted_power_law_parameters_df,
)

bon_jailbreaking_beta_binomial_mle_df = src.analyze.create_or_load_bon_jailbreaking_beta_binomial_mle_df(
    refresh=False,
    # refresh=True,
)

print("Best-of-N Jailbreaking ScaledBeta-Binomial 3-Parameter Fit: ")
pprint.pprint(bon_jailbreaking_beta_binomial_mle_df)


bon_jailbreaking_joint_power_law_and_distr_fit_df = pd.merge(
    bon_jailbreaking_beta_binomial_mle_df[
        src.globals.BON_JAILBREAKING_GROUPBY_COLS + ["Power Law Exponent"]
    ],
    bon_jailbreaking_lst_sqrs_fitted_power_law_parameters_df[
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
    plot_filename="bon_jailbreaking_y=scaling_law_exponent_x=distributional_fit_exponent",
)
plt.show()


print("Finished notebooks/51_pass@1_compare_power_law_with_distributional_fit!")
