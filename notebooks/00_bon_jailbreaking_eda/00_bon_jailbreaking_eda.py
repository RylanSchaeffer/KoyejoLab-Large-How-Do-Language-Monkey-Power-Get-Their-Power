import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Tuple

import src.analyze
import src.plot
import src.utils


data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

bon_jailbreaking_pass_at_k_df = src.analyze.create_or_load_bon_jailbreaking_pass_at_k_df(
    # refresh=False,
    refresh=True,
)

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=bon_jailbreaking_pass_at_k_df,
    x="Scaling Parameter",
    y="Score",
    hue="Model",
    # style="Modality",
)
g.set(
    title="Best-of-N Jailbreaking",
    xlabel=r"Scaling Parameter (Num. Attempts $N$)",
    ylabel="Attack Success Rate",
    # ylim=(-0.05, 1.05),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_x=scaling_parameter_hue=model",
)
# plt.show()

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=bon_jailbreaking_pass_at_k_df,
    x="Scaling Parameter",
    y="Neg Log Score",
    hue="Model",
    # style="Modality",
)
g.set(
    title="Best-of-N Jailbreaking",
    xscale="log",
    yscale="log",
    ylim=(1e-3, None),
    xlabel=r"Scaling Parameter (Num. Attempts $N$)",
    ylabel=r"$-\log (\text{Attack Success Rate})$",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=model",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=bon_jailbreaking_pass_at_k_df,
    kind="line",
    x="Scaling Parameter",
    y="Score",
    units="Problem Idx",
    hue="Model",
    col="Model",
    col_wrap=4,
    estimator=None,
)
g.set(
    xscale="log",
    ylabel="Attack Success Rate",
    xlabel=r"Scaling Parameter (Num. Attempts $N$)",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.27))  # You might need to adjust these values
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=score_vs_x=scaling_parameter_hue=model_col=model_units=problem_idx",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=bon_jailbreaking_pass_at_k_df,
    kind="line",
    x="Scaling Parameter",
    y="Neg Log Score",
    units="Problem Idx",
    hue="Model",
    col="Model",
    col_wrap=4,
    estimator=None,
)
g.set(
    xscale="log",
    yscale="log",
    ylim=(1e-2, 1e1),
    ylabel=r"$-\log(\text{Attack Success Rate})$",
    xlabel=r"Scaling Parameter (Num. Attempts $N$)",
)
# Move legend to the empty subplot position
g._legend.set_bbox_to_anchor((0.95, 0.27))  # You might need to adjust these values
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_score_vs_x=scaling_parameter_hue=model_col=model_units=problem_idx",
)
# plt.show()


# plt.close()
# sns.displot(
#     data=bon_jailbreaking_pass_at_k_df,
#     kind="hist",
#     x="Score",
#     hue="Model",
#     legend=False,
#     col="Scaling Parameter",
#     col_wrap=5,
# )
# # plt.show()

print("Finished notebooks/00_bon_jailbreaking_eda/00_bon_jailbreaking_eda.py!")
