import flint
import mpmath
import numpy as np
from typing import Tuple


def compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
    params: Tuple[float, float, float],
    num_samples: np.ndarray,
    num_successes: np.ndarray,
) -> float:
    """
    3-parameter Beta-Binomial PMF using Gauss hypergeometric function:

    P(X=x) = binom(n, x) * [c^x / B(alpha, beta)] * B(x + alpha, beta) * _2F_1(arguments).
    """

    alpha, beta, scale = params
    nll_arr = np.zeros_like(num_samples, dtype=np.float64)
    for idx, (n, x) in enumerate(zip(num_samples, num_successes)):
        if not (0 <= x <= n):
            return 0.0

        # binomial coefficient binom(n, x)
        binom_factor = mpmath.binomial(int(n), int(x))

        # c^x
        c_to_x = mpmath.power(scale, x)

        # Beta(alpha, beta)
        B_a_b = mpmath.beta(alpha, beta)

        # Beta(x+alpha, beta)
        B_xa_b = mpmath.beta(x + alpha, beta)

        # hypergeometric function
        #   _2F_1(-(n-x), x+alpha; x+alpha+beta; c)
        # using mpmath.hyp2f1
        f = flint.arb(scale).hypgeom_2f1(float(-(n - x)), x + alpha, x + alpha + beta)
        # f = mpmath.hyp2f1(-(n - x), x + alpha, x + alpha + beta, scale)

        pmf = binom_factor * c_to_x * B_xa_b * f / B_a_b
        nll = mpmath.log(pmf)
        nll_arr[idx] = -float(nll)

    nll = np.mean(nll_arr)
    return nll


# Test the function
if __name__ == "__main__":
    params = (0.5, 3.5, 0.23280001)
    num_samples = np.array([10000])
    num_successes = np.array([122])
    nll = compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
        params=params,
        num_samples=num_samples,
        num_successes=num_successes,
    )
    print("-log prob =", nll)
