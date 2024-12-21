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

# Load the original pass@k data on MATH.
llmonkeys_original_pass_at_k_df = (
    src.analyze.create_or_load_large_language_monkeys_original_pass_at_k_df(
        refresh=False,
    )
)

llmonkeys_original_pass_at_1_df = llmonkeys_original_pass_at_k_df[
    (llmonkeys_original_pass_at_k_df["Scaling Parameter"] == 1)
    & (llmonkeys_original_pass_at_k_df["Benchmark"] == "MATH")
].copy()


# For each model, fit a beta distribution to the pass@1 data using MLE.
#          Model Benchmark  Scaling Parameter         a         b  loc   scale
# 0   Pythia 70M      MATH                  1  0.052845  1.911908  0.0  0.0346
# 1  Pythia 160M      MATH                  1  0.145844  1.678871  0.0  0.0365
# 2  Pythia 410M      MATH                  1  0.154275  1.905497  0.0  0.0947
# 3    Pythia 1B      MATH                  1  0.186034  2.374676  0.0  0.1359
# 4  Pythia 2.8B      MATH                  1  0.224330  3.036977  0.0  0.3153
# 5  Pythia 6.9B      MATH                  1  0.241094  2.710055  0.0  0.2428
# 6   Pythia 12B      MATH                  1  0.254311  2.014420  0.0  0.2328
llmonkeys_pass_at_1_beta_fits_df = (
    llmonkeys_original_pass_at_1_df.groupby(["Model", "Benchmark", "Scaling Parameter"])
    .apply(
        lambda df: src.analyze.fit_pass_at_1_beta_distribution_parameters(
            data=df["Score"].values
        )
    )
    .reset_index()
)
pprint.pprint(llmonkeys_pass_at_1_beta_fits_df)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=llmonkeys_pass_at_1_beta_fits_df,
    x="a",
    y="b",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Benchmark",
    s=200,
)
g.set(
    xlabel=r"$\hat{\alpha}$",
    xlim=(0.0, 1.0),
    ylabel=r"$\hat{\beta}$",
    ylim=(0, 4.0),
    title="Large Language Monkeys",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.04))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkey_y=beta_hat_x=alpha_hat_hue=model_style=benchmark",
)
plt.show()

# Create better bins that handle zero and near-zero values
smallest_nonzero_pass_at_1 = llmonkeys_original_pass_at_1_df[
    llmonkeys_original_pass_at_1_df["Score"] > 0.0
]["Score"].min()
# Round smallest_nonzero_value to the nearest power of 10.
smallest_nonzero_pass_at_1 = 10.0 ** np.floor(np.log10(smallest_nonzero_pass_at_1))
log10_smallest_nonzero_pass_at_1 = np.log10(smallest_nonzero_pass_at_1)
log_bins = np.logspace(
    log10_smallest_nonzero_pass_at_1, 0, -int(log10_smallest_nonzero_pass_at_1) * 3 + 1
)
small_value_for_plotting = smallest_nonzero_pass_at_1 / 2.0
all_bins = np.concatenate(
    [[-small_value_for_plotting], [small_value_for_plotting], log_bins]
)
plt.close()
g = sns.displot(
    data=llmonkeys_original_pass_at_1_df,
    kind="hist",
    x="Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    bins=all_bins,
    col="Model",
    col_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    col_wrap=4,
)
# Add the fit Beta distributions.
for ax_idx, model_name in enumerate(
    src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER
):
    model_df = llmonkeys_pass_at_1_beta_fits_df[
        llmonkeys_pass_at_1_beta_fits_df["Model"] == model_name
    ]
    pass_at_1 = np.logspace(-5, np.log10(model_df["scale"]).values[0], 500)
    cdf = scipy.stats.beta.cdf(
        all_bins,
        a=model_df["a"].values[0],
        b=model_df["b"].values[0],
        loc=model_df["loc"].values[0],
        scale=model_df["scale"].values[0],
    )
    prob_per_bin = np.diff(cdf)
    g.axes[ax_idx].plot(
        all_bins[1:],
        128.0 * prob_per_bin,  # Transform probabilities into counts for consistency.
        color="black",
        linestyle="--",
    )
g.set(
    xscale="log",
    ylabel="Count",
    ylim=(0, 128),
    xlabel="pass@1",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Large Language Monkeys")
g.fig.subplots_adjust(top=0.9)

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="llmonkey_y=counts_x=score_hue=model_col=model_bins=custom",
)
plt.show()


print("Finished notebooks/50_pass@1_fits.py!")
