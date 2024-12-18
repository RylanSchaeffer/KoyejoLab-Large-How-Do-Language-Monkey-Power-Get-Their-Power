from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
from typing import Any, Dict, List, Tuple

import src.plot
import src.utils

# Setup
data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Parameters
alpha = 1.0
beta = 5.0
pass_at_1 = np.logspace(-5, 0, 1000)
palette = sns.color_palette("cool", n_colors=len(pass_at_1))
palette_dict = dict(zip(pass_at_1, palette))

# Create the data
beta_distributution_df = pd.DataFrame.from_dict(
    {
        r"$x$": pass_at_1,
        r"$\alpha$": np.full_like(pass_at_1, fill_value=alpha),
        r"$\beta$": np.full_like(pass_at_1, fill_value=beta),
        r"$p(x)$": scipy.stats.beta.pdf(pass_at_1, alpha, beta),
    }
)

# Setup for pass@k calculations
select_pass_at_1_values = pass_at_1[::200]
select_palette = palette[::200]
k = np.linspace(1, 10000, 10000)
pass_at_k = 1.0 - np.power(
    1.0 - select_pass_at_1_values.reshape(-1, 1), k.reshape(1, -1)
)
neg_log_pass_at_k = -np.log(pass_at_k)

# Create figure with three subplots
plt.close()
fig = plt.figure(figsize=(18, 6))  # Slightly wider figure
fig.subplots_adjust(wspace=0.5)  # Increased spacing between subplots

# First subplot - Beta distribution
ax1 = fig.add_subplot(131)
ax1.plot(k, np.power(beta / k, alpha), color="k", linewidth=5)
ax1.set(
    xscale="log",
    xlim=(1, k.max()),
    yscale="log",
    ylim=(1e-3, 3.162e1),
    xlabel=r"Number of Attempts $k$",
    ylabel=r"$-\log(\mathbb{E}[\operatorname{pass@k}])$",
)

# Second subplot - Pass@k for different pass@1 values
ax2 = fig.add_subplot(132)
for idx, select_pass_at_1 in enumerate(select_pass_at_1_values):
    ax2.plot(
        k,
        neg_log_pass_at_k[idx],
        label=f"pass@1={select_pass_at_1:.2e}",
        color=select_palette[idx],
        linewidth=5,
    )
ax2.set(
    xscale="log",
    yscale="log",
    xlim=(1, k.max()),
    ylim=(1e-3, 3.162e1),
    xlabel=r"Number of Attempts $k$",
    ylabel=r"$-\log (\operatorname{pass@k})$",
)

# Third subplot - Expected pass@k
ax3 = fig.add_subplot(133)
sns.scatterplot(
    data=beta_distributution_df,
    x=r"$x$",
    y=r"$p(x)$",
    hue=r"$x$",
    legend=False,
    palette="cool",
    hue_norm=LogNorm(),
    linewidth=0,
    ax=ax3,
)
ax3.set(
    xscale="log",
    xlim=(1e-5, 1.0),
    xlabel=r"$\operatorname{pass@1}$",
    ylim=(1e-2, None),
    ylabel=r"$p(\operatorname{pass@1})$",
)

# Add equals sign between ax1 and ax2
fig.text(0.335, 0.55, "=", fontsize=20, ha="center", va="center")

# Add plus sign between ax2 and ax3
fig.text(0.68, 0.55, "+", fontsize=20, ha="center", va="center")

# Save the combined plot
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="combined_statistical_plots"
)
plt.show()

print("Finished notebooks/90_schematic!")
