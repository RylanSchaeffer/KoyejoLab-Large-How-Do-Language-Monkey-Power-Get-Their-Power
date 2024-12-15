import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
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

(
    llmonkeys_original_pass_at_1_df,
    llmonkeys_pass_at_1_beta_fits_df,
) = src.analyze.create_or_load_large_language_monkeys_original_pass_at_1_beta_fits(
    refresh=False,
    # refresh=True,
)

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
pass_at_1 = np.logspace(-5, 0, 500)
for ax_idx, model_name in enumerate(
    src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER
):
    model_df = llmonkeys_pass_at_1_beta_fits_df[
        llmonkeys_pass_at_1_beta_fits_df["Model"] == model_name
    ]
    prob_pass_at_1 = scipy.stats.beta.pdf(
        pass_at_1,
        model_df["a"].values[0],
        model_df["b"].values[0],
        loc=model_df["loc"].values[0],
        scale=model_df["scale"].values[0],
    )
    g.axes[ax_idx].plot(
        pass_at_1,
        prob_pass_at_1,
        color="black",
        linestyle="--",
    )
g.set(
    xscale="log",
    ylabel="Count",
    xlabel="pass@1",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.25))
g.fig.suptitle("Large Language Monkeys")
g.fig.subplots_adjust(top=0.9)

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=counts_x=score_hue=model_col=model_bins=custom",
)
plt.show()


beta_distributions_pdf_df = src.analyze.create_or_load_beta_distributions_pdf_df(
    # refresh=False,
    refresh=True,
)


# We want to estimate
print()


plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=beta_distributions_pdf_df,
    x="x",
    y="p(x)",
    hue=r"$\alpha$",
    style=r"$\beta$",
    palette="viridis",
    linewidth=3,
)
g.set(
    title="Beta Distribution PDFs",
    xscale="log",
    yscale="log",
    xlim=(1e-4, 1.05e0),
    ylim=(1e-2, 1e2),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
plt.tight_layout()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=p(x)_x=x_hue=alpha_style=beta",
)
# plt.show()


print("Finished notebooks/50_pass@1_fits.py!")
