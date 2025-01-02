import itertools
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

synthetic_scaling_exponents_df = src.analyze.create_or_load_synthetic_scaling_coefficient_data_df(
    refresh=False,
    # refresh=True,
)


plt.close()
g = sns.relplot(
    data=synthetic_scaling_exponents_df,
    kind="line",
    x=r"Num. Samples per Problem ($n$)",
    y="Relative Error",
    hue="Fit Method",
    palette="cool",
    style="Num. Problems",
    col="True Distribution",
    row="Fit Distribution",
    facet_kws={"margin_titles": True},
)
g.set(
    xscale="log",
    yscale="log",
    ylabel=r"Relative Error := $|\hat{b} - b| / b$",
)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
fig = plt.gcf()
# Add text as a "row title" to the right of the row facet titles.
fig.text(
    1.0,
    0.50,
    "Fit Distribution",
    fontsize=30,
    rotation=-90,
    ha="center",
    va="center",
)
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
    plot_filename="y=relative_error_x=n_hue=distribution_params_col=distribution",
)
plt.show()

# plt.close()
# g = sns.relplot(
#     data=synthetic_scaling_exponents_df,
#     kind="line",
#     x=r"Num. Samples per Problem ($n$)",
#     y="Fit Scaling Exponent",
#     col="True Distribution Parameters",
#     hue="Num. Problems",
#     hue_norm=LogNorm(),
#     palette="cool",
#     style="Fit Method",
#     row="True Distribution",
#     facet_kws={"margin_titles": True},
# )
# g.set(
#     xscale="log",
#     xlabel=r"Num. Samples per Problem ($n$)",
# )
# g.set_titles(col_template="{col_name}")
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=fit_scaling_exponent_x=n_hue=distribution_params_style=fit_method_col=distribution",
# )
# plt.show()
