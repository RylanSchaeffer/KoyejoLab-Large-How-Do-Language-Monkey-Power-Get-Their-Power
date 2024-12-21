import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import scipy.stats
import numpy as np
import scipy.stats
import scipy.integrate
import scipy.optimize
from typing import Tuple


def compute_neg_log_likelihood(
    params: Tuple[float, float],
    data: np.ndarray,
    bins: np.ndarray,
    epsilon: float = 1e-16,
):
    a, b = params
    assert not np.isnan(a)
    assert not np.isnan(b)

    # 1. Compute probability mass per bin
    cdf_values = scipy.stats.beta.cdf(bins, a, b, loc=0.0, scale=data.max())
    prob_mass = np.diff(cdf_values) + epsilon

    assert np.all(prob_mass >= 0.0)
    log_prob_mass = np.log(prob_mass)

    # 2. Bin the data
    num_data_per_bin = np.histogram(data, bins)[0]

    # 3. Compute the total log likelihood
    log_likelihood = np.mean(np.multiply(num_data_per_bin, log_prob_mass))

    # 4. Return the negative log likelihood.
    neg_log_likelihood = -log_likelihood

    print("alpha: ", a)
    print("beta: ", b)
    print("NLL: ", neg_log_likelihood)
    print("\n\n")
    assert not np.isinf(neg_log_likelihood)

    return neg_log_likelihood


data = scipy.stats.beta.rvs(1.0, 5.0, loc=0, scale=1.0, size=50000)
resolution = 1e-4

smallest_nonzero_pass_at_1 = resolution
log10_smallest_nonzero_pass_at_1 = np.log10(smallest_nonzero_pass_at_1)
num_windows_per_factor_of_10 = 5
log_bins = np.logspace(
    log10_smallest_nonzero_pass_at_1,
    0,
    -int(log10_smallest_nonzero_pass_at_1) * num_windows_per_factor_of_10 + 1,
)
small_value_for_plotting = smallest_nonzero_pass_at_1 / 2.0
bins = np.concatenate(
    [[-small_value_for_plotting], [small_value_for_plotting], log_bins]
)
bins[0] = 0.0
assert data.min() > bins[0]
assert data.max() < bins[-1]

plt.plot(1 + np.arange(len(bins)), bins)

# Initial guess for parameters
initial_params = [
    0.9,
    10.1,
]

# Constraint for parameters to ensure they are positive and realistic for a Beta distribution
bounds = [(0.01, 100), (0.01, 100)]  # Parameters must be greater than zero

# Maximize the log likelihood by minimizing its negative
result = scipy.optimize.minimize(
    lambda params: compute_neg_log_likelihood(params, data=data, bins=bins),
    x0=initial_params,
    bounds=bounds,
    method="L-BFGS-B",
    options=dict(
        maxiter=5000,
        maxls=100,
        gtol=1e-6,  # Gradient tolerance, adjust as needed),
    ),
)

pprint(result.x)
