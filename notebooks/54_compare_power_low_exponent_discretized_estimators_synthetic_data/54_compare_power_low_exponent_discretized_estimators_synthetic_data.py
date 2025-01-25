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
import src.plot
import src.utils


data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

synthetic_cross_validated_scaling_coeff_df = src.analyze.create_or_load_cross_validated_synthetic_scaling_coefficient_discretized_data_df(
    # refresh=False,
    refresh=True,
)


plt.close()
g = sns.relplot(
    data=synthetic_cross_validated_scaling_coeff_df,
    kind="line",
    x="Num. Samples per Problem",
    y="Full Data Least Squares Relative Error",
    hue="Fit Method",
    palette="cool",
    row="Num. Problems",
    col="True Distribution",
    facet_kws={"margin_titles": True, "sharey": False},
)
g.set(
    xscale="log",
    yscale="log",
    ylabel=r"Relative Error := $|\hat{b} - b| / b$",
)
g.set_titles(col_template="{col_name}", row_template="{row_name} Problems")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
fig = plt.gcf()
fig.text(
    0.50,
    1.0,
    "True Distribution",
    fontsize=30,
    ha="center",
    va="center",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=relative_error_least_squares_x=n_hue=distribution_params_col=distribution",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=synthetic_cross_validated_scaling_coeff_df,
    kind="line",
    x="Num. Samples per Problem",
    y="Asymptotic Relative Error",
    hue="Fit Method",
    palette="cool",
    row="Num. Problems",
    col="True Distribution",
    facet_kws={"margin_titles": True, "sharey": False},
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="",
)
g.axes[int(g.axes.shape[0] // 2), 0].set_ylabel(
    r"Relative Error := $|\hat{b} - b| / b$"
)
g.set_titles(col_template="{col_name}", row_template="{row_name} Problems")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
fig = plt.gcf()
fig.text(
    0.50,
    1.0,
    "True Distribution",
    fontsize=30,
    ha="center",
    va="center",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=relative_error_asymptotic_x=n_hue=distribution_params_col=distribution",
)
# plt.show()

print(
    "Finished notebooks/54_compare_power_low_exponent_discretized_estimators_synthetic_data!"
)
