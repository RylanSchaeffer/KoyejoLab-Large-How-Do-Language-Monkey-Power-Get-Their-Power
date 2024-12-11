import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
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

large_language_monkeys_pass_at_k_df = src.analyze.create_or_load_large_language_monkeys_pass_at_k_df(
    # refresh=False,
    refresh=True,
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=large_language_monkeys_pass_at_k_df,
    x="Scaling Parameter",
    y="Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Benchmark",
)
g.set(
    title="Large Language Monkeys",
    xscale="log",
    xlabel=r"Scaling Parameter (Number of Attempts $k$)",
    ylabel=r"$\mathbb{E}[\text{pass@k}]$",
    ylim=(0.0, 1.0),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=scaling_parameter_hue=model",
)
plt.show()

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=large_language_monkeys_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    hue_order=src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_ORDER,
    style="Benchmark",
)
g.set(
    title="Large Language Monkeys",
    xscale="log",
    yscale="log",
    ylim=(1e-3, None),
    xlabel=r"Scaling Parameter (Number of Attempts $N$)",
    ylabel=r"$-\log (\mathbb{E}[\text{pass@k}])$",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=model",
)
plt.show()

print(
    "Finished notebooks/01_large_language_monkeys_eda/01_large_language_monkeys_eda.py!"
)
