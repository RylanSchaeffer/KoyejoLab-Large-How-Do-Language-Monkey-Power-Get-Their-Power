import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
from math import log
import pandas as pd
from scipy.special import beta as Beta, gamma
from scipy.optimize import curve_fit
import seaborn as sns


###############################################################################
# 1. Define the integral I(k; alpha, beta, c) numerically via mpmath.quad
###############################################################################
def I_of_k(k, alpha, beta, c):
    """
    Numerically evaluates the integral:
        I(k) = \int_{0}^{c} (1 - p)^k [p^(alpha-1)*(c - p)^(beta-1)] / [c^(alpha+beta-1)*B(alpha,beta)] dp.
    """
    # Precompute constant factors (to avoid repeating in integrand)
    denom = (c ** (alpha + beta - 1)) * Beta(alpha, beta)

    def integrand(p):
        return ((1 - p) ** k * p ** (alpha - 1) * (c - p) ** (beta - 1)) / denom

    # Use mp.quad for integration from 0 to c
    # Increase maxterms / maxsteps if needed for accuracy
    val = mp.quad(integrand, [0, c])
    return val


def I_of_k_substitution(k, alpha, beta, c):
    # For large k, we can push upper limit to, say, 10*k or 50*k if needed,
    # but typically k*c is enough if c < 1.
    upper_limit = k * c

    denom = (c ** (alpha + beta - 1)) * mp.beta(alpha, beta)

    def integrand(x):
        # x ~ O(1) region contributes; e^{-x} is the big suppressor
        # p = x / k
        # p^(alpha-1) = (x/k)^(alpha-1)
        return (
            (
                mp.e ** (-x)
                * (x ** (alpha - 1) / (k ** (alpha - 1)))
                * (c ** (beta - 1))  # approx (c - x/k)^(beta-1) ~ c^(beta-1)
            )
            / denom
            / k
        )  # from dp = dx/k

    return mp.quad(integrand, [0, upper_limit])


###############################################################################
# 2. Utility to estimate exponent by least-squares on log-log scale
###############################################################################
def estimate_exponent(k_vals, I_vals):
    """
    Given arrays k_vals and I_vals, we solve ln(I_vals) ~ m * ln(k_vals) + b
    by linear regression. Returns the fitted slope m.
    """
    logk = np.log(k_vals)
    logI = np.log(I_vals)

    # We can use numpy polyfit for a quick linear regression:
    slope, intercept = np.polyfit(logk, logI, 1)
    return slope  # m


###############################################################################
# 3. Main experiment: sweep alpha, beta, c; for each triple, compute I(k) for
#    large k-values, then fit exponent. Compare with theoretical alpha.
###############################################################################
def main_experiment():
    precision = 50

    # MPMATH precision (increase if needed)
    mp.mp.prec = precision

    # Parameter sweeps
    alpha_vals = np.linspace(0.05, 0.8, 8)  # Just a small grid example
    beta_vals = np.linspace(1.0, 5.0, 5)
    c_vals = np.linspace(0.05, 0.5, 5)

    # k-values at which we'll evaluate I(k)
    k_values = np.logspace(0, 5, 25)

    df_rows = []
    for alpha_ in alpha_vals:
        exponent_theoretical = alpha_
        for beta_ in beta_vals:
            for c_ in c_vals:
                I_data = []
                # I_substitution_data = []
                for k_ in k_values:
                    # val = float(I_of_k(k_, alpha_, beta_, c_))
                    # I_data.append(val)
                    val = float(I_of_k_substitution(k_, alpha_, beta_, c_))
                    I_data.append(val)

                # Fit exponent from data on log-log scale
                fitted_slope_integral = estimate_exponent(k_values, np.array(I_data))
                fitted_slope_neg_log_one_minus_integral = estimate_exponent(
                    k_values, -np.log1p(-np.array(I_data))
                )
                # The slope should be ~ -alpha in the asymptotic, so exponent = -slope
                # but we usually compare the magnitude.
                # If I(k) ~ 1/k^alpha => ln(I(k)) ~ -alpha ln(k).
                # => slope in log-log is -alpha.
                df_rows.append(
                    pd.Series(
                        {
                            "alpha": alpha_,
                            "beta": beta_,
                            "c": c_,
                            "numerical_exponent": -fitted_slope_integral,
                            "theoretical_exponent": exponent_theoretical,
                            "method": r"$I(k)$",
                        }
                    )
                )
                df_rows.append(
                    pd.Series(
                        {
                            "alpha": alpha_,
                            "beta": beta_,
                            "c": c_,
                            "numerical_exponent": -fitted_slope_neg_log_one_minus_integral,
                            "theoretical_exponent": exponent_theoretical,
                            "method": r"$-\log(1 - I(k))$",
                        }
                    )
                )

    df = pd.DataFrame(df_rows)

    df["relative_error"] = (
        np.abs(df["numerical_exponent"] - df["theoretical_exponent"])
        / df["theoretical_exponent"]
    )
    plt.close()
    g = sns.scatterplot(
        data=df,
        x="theoretical_exponent",
        y="numerical_exponent",
        style="method",
        hue="method",
        col="I(k) Numerical Integration",
    )
    # Plot dotted identity line.
    g.plot([0, 1], [0, 1], ls="--")
    g.set(title=f"Precision: {precision}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.04))
    plt.tight_layout()
    plt.show()
    print()


if __name__ == "__main__":
    main_experiment()
