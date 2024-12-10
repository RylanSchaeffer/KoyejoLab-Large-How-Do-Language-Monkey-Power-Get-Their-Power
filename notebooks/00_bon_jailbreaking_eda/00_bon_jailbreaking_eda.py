import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
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

bon_jailbreaking_pass_at_k_df = (
    src.analyze.create_or_load_bon_jailbreaking_pass_at_k_df(
        refresh=False,
    )
)

plt.close()
# sns.lineplot()
