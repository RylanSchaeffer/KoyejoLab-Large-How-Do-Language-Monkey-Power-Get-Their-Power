import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import scipy.special
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import warnings

import src.analyze


def compute_integral(k, alpha, beta, c, debug=False):
    """
    Compute the integral using adaptive quadrature with improved error handling
    """
    if not (0 < c < 1 and alpha > 0 and beta > 0):
        return np.nan

    # Use a smaller absolute tolerance and increase max evaluations
    result, error = integrate.quad(
        beta_three_parameter_distribution_integrand,
        0.0,
        c,
        args=(k, alpha, beta, c),
        epsabs=1e-10,
        epsrel=1e-8,
        limit=500,
    )

    # Check if result is reasonable
    if not np.isfinite(result) or error > 1e-3 * abs(result):
        return np.nan

    return result


# Parameter ranges - reduced for initial testing
k_values = np.logspace(0, 5, 20, dtype=int).tolist()[::-1]  # Reduced number of points
alpha_values = np.array([0.1, 0.2, 0.3, 0.6])
beta_values = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
c_values = np.array([0.01, 0.1, 0.5])

# Create parameter combinations with debugging for first few cases
results = []
debug_first_n = 5  # Number of cases to debug
total_count = 0

for k in tqdm(k_values, desc="Processing k values"):
    for alpha in alpha_values:
        for beta in beta_values:
            for c in c_values:
                debug = total_count < debug_first_n
                integral_value = compute_integral(k, alpha, beta, c, debug=debug)
                results.append(
                    {
                        "k": k,
                        "alpha": alpha,
                        "beta": beta,
                        "c": c,
                        "integral_value": integral_value,
                    }
                )
                total_count += 1

# Create DataFrame
df = pd.DataFrame(results)

# Sort by parameters
df = df.sort_values(["k", "alpha", "beta", "c"]).reset_index(drop=False)

# Display first few rows and basic statistics
print("\nFirst few rows of the results:")
print(df.head().to_string())

print("\nSummary statistics:")
print(df["integral_value"].describe())

plt.close()
sns.relplot(
    data=df,
    kind="line",
    x="k",
    y="integral_value",
    hue="alpha",
    col="c",
    style="beta",
)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("k")
plt.ylabel("Integral Value")
# plt.show()


(
    _,
    fitted_power_law_parameters_df,
) = src.analyze.fit_power_law(
    df,
    covariate_col="k",
    target_col="integral_value",
    groupby_cols=["alpha", "beta", "c"],
)


# Count NaN values
nan_count = df["integral_value"].isna().sum()
total_count = len(df)
print(f"\nNaN values: {nan_count}/{total_count} ({100*nan_count/total_count:.1f}%)")
