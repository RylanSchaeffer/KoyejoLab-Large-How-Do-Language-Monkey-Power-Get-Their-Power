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


cv_llmonkeys_scaling_coeff_df = src.analyze.create_or_load_cross_validated_large_language_monkey_pythia_math_scaling_coefficient_data_df(
    refresh=False,
    # refresh=True,
)

cv_bon_jailbreaking_scaling_coeff_df = src.analyze.create_or_load_cross_validated_bon_jailbreaking_scaling_coefficient_data_df(
    refresh=False,
    # refresh=True,
)
