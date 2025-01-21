from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
import itertools
import joblib
import mpmath
import numpy as np
import os
import pandas as pd
import pprint
from scipy import integrate
from scipy.optimize import minimize
import scipy.stats
from typing import Dict, List, Optional, Tuple, Union

import src.globals

# Increase precision.
mpmath.mp.prec = 100

# This helps print more columns.
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)


def compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
    params: Tuple[float, float],
    scale: float,
    num_samples: np.ndarray,
    num_successes: np.ndarray,
) -> float:
    """
    3-parameter Beta-Binomial PMF using Gauss hypergeometric function:

    P(X=x) = binom(n, x) * [c^x / B(alpha, beta)] * B(x + alpha, beta) * _2F_1(arguments).
    """

    alpha, beta = params
    # scale = np.max(np.divide(num_successes, num_samples)) + 1e-16
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
        f = mpmath.hyp2f1(
            -(n - x),
            x + alpha,
            x + alpha + beta,
            scale,
            # nmaxterms=2000000,
            # method="a+bt",
        )
        pmf = binom_factor * c_to_x * B_xa_b * f / B_a_b
        nll = -mpmath.log(pmf)
        nll_arr[idx] = float(nll)

    avg_nll: float = np.mean(nll_arr)
    return avg_nll


def compute_beta_binomial_two_parameters_negative_log_likelihood(
    params: Tuple[float, float],
    num_samples: np.ndarray,
    num_successes: np.ndarray,
) -> float:
    log_pmf = scipy.stats.betabinom.logpmf(
        k=num_successes, n=num_samples, a=params[0], b=params[1]
    ).mean()
    return -log_pmf


def compute_beta_negative_binomial_three_parameters_distribution_neg_log_likelihood(
    params: Tuple[float, float],
    scale: float,
    num_samples: np.ndarray,
    num_successes: np.ndarray,
) -> float:
    """
    3-parameter Beta-Binomial PMF using Gauss hypergeometric function:

    P(X=x) = binom(n, x) * [c^x / B(alpha, beta)] * B(x + alpha, beta) * _2F_1(arguments).
    """

    alpha, beta = params
    # scale = np.max(np.divide(num_successes, num_samples)) + 1e-16
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
        f = mpmath.hyp2f1(
            -(n - x),
            x + alpha,
            x + alpha + beta,
            scale,
            # nmaxterms=2000000,
            # method="a+bt",
        )
        pmf = binom_factor * c_to_x * B_xa_b * f / B_a_b
        nll = -mpmath.log(pmf)
        nll_arr[idx] = float(nll)

    avg_nll: float = np.mean(nll_arr)
    return avg_nll


def compute_beta_three_parameter_distribution_integrand(
    p: mpmath.mpf, k: int, alpha: mpmath.mpf, beta: mpmath.mpf, c: mpmath.mpf
) -> mpmath.mpf:
    """
    Compute the log of the integrand using mpmath for arbitrary precision.

    Args:
        p: Integration variable (must be between 0 and c)
        k: Integer parameter
        alpha: First shape parameter of the beta distribution
        beta: Second shape parameter of the beta distribution
        c: Scale parameter (upper bound)

    Returns:
        Value of the integrand at point p
    """
    if p <= 0 or p >= c:
        return mpmath.mpf("0.0")

    # Convert all inputs to mpmath types for high precision
    p = mpmath.mpf(str(p))
    k = mpmath.mpf(str(k))
    alpha = mpmath.mpf(str(alpha))
    beta = mpmath.mpf(str(beta))
    c = mpmath.mpf(str(c))

    # Compute in log space for numerical stability
    log_term1 = k * mpmath.log1p(-p)
    log_term2 = (alpha - 1.0) * mpmath.log(p)
    log_term3 = (beta - 1.0) * mpmath.log(c - p)
    log_term4 = (alpha + beta - 1) * mpmath.log(c)
    log_term5 = (
        mpmath.loggamma(alpha) + mpmath.loggamma(beta) - mpmath.loggamma(alpha + beta)
    )

    log_result = log_term1 + log_term2 + log_term3 - log_term4 - log_term5
    result = mpmath.exp(log_result)
    return result


def compute_discretized_neg_log_likelihood(
    params: Tuple[float, float],
    data: np.ndarray,
    bins: np.ndarray,
    distribution: str = "beta",
    epsilon: float = 1e-16,
):
    # 1. Compute probability mass per bin
    if distribution == "beta":
        a, b = params
        assert not np.isnan(a)
        assert not np.isnan(b)
        if data.max() == 0:
            raise ValueError("Data is all zeros.")

        cdf_values = scipy.stats.beta.cdf(bins, a, b, loc=0.0, scale=data.max())
        prob_mass_per_bin = np.diff(cdf_values) + epsilon
    elif distribution == "continuous_bernoulli":
        lam = params
        assert not np.isnan(lam)
        if lam == 0.5:
            cdf_values = bins
        else:
            cdf_values = (
                np.power(lam, bins) * np.power(1.0 - lam, 1.0 - bins) + lam - 1.0
            )
            cdf_values /= 2.0 * lam - 1

        prob_mass_per_bin = np.diff(cdf_values) + epsilon
    elif distribution == "kumaraswamy":
        a, b = params
        assert not np.isnan(a)
        assert not np.isnan(b)

        # Normally, the CDF of the Kumaraswamy distribution is:
        #   F(x) = 1 - (1 - x^a)^b
        # But we want to introduce a scale parameter, so we rescale the input to get the new CDF:
        #   F(x) = 1 - (1 - (x / scale)^a)^b
        data = data / (data.max() + epsilon)
        cdf_values = 1.0 - np.power(1.0 - np.power(bins, a), b)
        prob_mass_per_bin = np.diff(cdf_values) + epsilon

    elif distribution == "lognormal":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    assert np.all(prob_mass_per_bin >= 0.0)

    # 1.5 Compute the log of the probability mass per bin.
    log_prob_mass_per_bin = np.log(prob_mass_per_bin)

    # 2. Bin the data
    num_data_per_bin = np.histogram(data, bins)[0]

    # 3. Compute the total log likelihood
    discretized_log_likelihood = np.mean(
        np.multiply(num_data_per_bin, log_prob_mass_per_bin)
    )

    # 4. Return the negative log likelihood.
    neg_discretized_log_likelihood = -discretized_log_likelihood

    assert not np.isinf(neg_discretized_log_likelihood)

    return neg_discretized_log_likelihood


def compute_failure_rate_at_k_attempts_under_beta_three_parameter_distribution(
    k: int, alpha: float, beta: float, scale: float, dps: int = 50
) -> float:
    """
    Compute the integral using mpmath's high-precision integration.

    Here, we want to compute the integral under a scaled Beta distribution:
    \int_0^{scale} (1 - p)^k p(p; alpha, beta, scale) dp.
    """

    # Previous fit failed.
    if np.isnan(alpha) and np.isnan(beta) and np.isnan(scale):
        return np.nan

    # Parameter validation
    if not (0.0 < scale <= 1.0 and alpha > 0 and beta > 0):
        raise ValueError(
            f"Invalid parameters: alpha: {alpha}, beta: {beta}, scale: {scale}"
        )

    try:
        # Precompute constant factors (to avoid repeating in integrand)
        denom = (scale ** (alpha + beta - 1)) * mpmath.beta(alpha, beta)

        def integrand(p):
            return (
                (1.0 - p) ** k * p ** (alpha - 1.0) * (scale - p) ** (beta - 1.0)
            ) / denom

        integral = mpmath.quad(integrand, [0, scale])
        if integral < 0.0 or integral >= 1.0:
            raise ValueError(
                f"Integral invalid! Integral: {integral}, k: {k}, alpha: {alpha}, beta: {beta}, scale: {scale}"
            )
        return float(integral)
    except (ValueError, TypeError, mpmath.libmp.NoConvergence):
        return float("nan")


def compute_failure_rate_at_k_attempts_under_kumaraswamy_three_parameter_distribution(
    k: int, alpha: float, beta: float, scale: float
) -> float:
    """
    Compute the integral using mpquad's high-precision integration.

    Here, we want to compute the integral under a scaled Kumaraswamy distribution:
    \int_0^{scale} (1 - p)^k p(p; alpha, beta, scale) dp.
    """
    # Previous fit failed.
    if np.isnan(alpha) and np.isnan(beta) and np.isnan(scale):
        return np.nan

    # Parameter validation
    if not (0.0 < scale <= 1.0 and alpha > 0 and beta > 0):
        raise ValueError(
            f"Invalid parameters: alpha: {alpha}, beta: {beta}, scale: {scale}"
        )

    try:
        # Precompute constant factors (to avoid repeating in integrand)
        denom = mpmath.power(scale, alpha) / alpha / beta

        def integrand(p):
            return (
                mpmath.power(1.0 - p, k)
                * mpmath.power(p, alpha - 1.0)
                * mpmath.power(1.0 - mpmath.power(p / scale, alpha), beta - 1.0)
            ) / denom

        integral = mpmath.quad(integrand, [0, scale])
        if integral < 0.0 or integral >= 1.0:
            raise ValueError(
                f"Integral invalid! Integral: {integral}, k: {k}, alpha: {alpha}, beta: {beta}, scale: {scale}"
            )
        return float(integral)
    except (ValueError, TypeError, mpmath.libmp.NoConvergence):
        return float("nan")


def compute_kumaraswamy_binomial_three_parameters_distribution_neg_log_likelihood(
    params: Tuple[float, float],
    scale: float,
    num_samples: np.ndarray,
    num_successes: np.ndarray,
) -> float:
    """
    3-parameter Beta-Binomial PMF using Gauss hypergeometric function:

    P(X=x) = binom(n, x) * [alpha * beta / c^{alpha}] int_0^c p^{x + alpha - 1} (1 - p)^{n - x} (1 - (o/c)^alpha)^{beta - 1} dp.
    """

    alpha, beta = params
    nll_arr = np.zeros_like(num_samples, dtype=np.float64)
    for idx, (num_sample, num_success) in enumerate(zip(num_samples, num_successes)):
        if not (0 <= num_success <= num_sample):
            return 0.0

        # binomial coefficient binom(n, x)
        binom_factor = mpmath.binomial(int(num_sample), int(num_success))

        # alpha * beta / c^alpha
        alpha_beta_over_c_to_alpha = alpha * beta / mpmath.power(scale, alpha)

        def integrand(p):
            return (
                mpmath.power(p, num_success + alpha - 1.0)
                * mpmath.power(1.0 - p, num_sample - num_success)
                * mpmath.power(1.0 - mpmath.power(p / scale, alpha), beta - 1.0)
            )

        integral = mpmath.quad(integrand, [0.0, scale])
        pmf = binom_factor * alpha_beta_over_c_to_alpha * integral
        nll = -mpmath.log(pmf)
        nll_arr[idx] = float(nll)

    avg_nll: float = np.mean(nll_arr)
    return avg_nll


def compute_pass_at_k_from_individual_outcomes(
    individual_outcomes_per_problem: np.ndarray,
    ks_list: List[int],
) -> pd.DataFrame:
    num_problems, num_samples_per_problem = individual_outcomes_per_problem.shape
    # Compute the number of samples per problem and the number of successes per problem.
    # Note: For BoN jailbreaking, due to their sampling procedure, the number of samples per problem is
    # not constant and the individual_outcomes_per_problem will have many NaNs.
    # This code is written to take this into account.
    num_samples_total = np.sum(~np.isnan(individual_outcomes_per_problem), axis=1)
    num_samples_correct = np.nansum(individual_outcomes_per_problem, axis=1)
    num_samples_and_num_successes_df = pd.DataFrame.from_dict(
        {
            "Num. Samples Total": num_samples_total,
            "Num. Samples Correct": num_samples_correct,
        }
    )
    pass_at_k_df = compute_pass_at_k_from_num_samples_and_num_successes_df(
        num_samples_and_num_successes_df=num_samples_and_num_successes_df,
        ks_list=ks_list,
    )
    return pass_at_k_df


def compute_pass_at_k_from_num_samples_and_num_successes_df(
    num_samples_and_num_successes_df: pd.DataFrame,
    ks_list: List[int],
):
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    assert np.all(num_successes <= num_samples)
    num_problems = len(num_samples_and_num_successes_df)
    pass_at_k_dfs_list = []
    for k in ks_list:
        pass_at_k = src.analyze.estimate_pass_at_k(
            num_samples_total=num_samples,
            num_samples_correct=num_successes,
            k=k,
        )
        pass_at_k_df = pd.DataFrame(
            {
                "Score": pass_at_k,
                "Scaling Parameter": k,
                "Problem Idx": np.arange(num_problems),
            }
        )
        pass_at_k_dfs_list.append(pass_at_k_df)
    pass_at_k_df = pd.concat(pass_at_k_dfs_list)
    # Drop any NaN scores.
    pass_at_k_df.dropna(subset=["Score"], inplace=True)
    return pass_at_k_df


def compute_scaling_exponent_from_distributional_fit(
    distributional_fit_df: pd.DataFrame,
    distribution: str = "beta",
    k_values: Optional[Union[np.ndarray, List[float]]] = None,
) -> pd.DataFrame:
    if k_values is None:
        k_values = np.unique(np.logspace(0, 5, 20, dtype=int))
    if isinstance(k_values, list):
        k_values = np.array(k_values)

    if distribution == "beta_two_parameter":
        raise NotImplementedError
    elif distribution in {"beta_three_parameter", "kumaraswamy_three_parameter"}:
        distributional_fit_df["Log Power Law Prefactor"] = np.nan
        distributional_fit_df["Power Law Prefactor"] = np.nan
        distributional_fit_df["Power Law Exponent"] = np.nan
        integral_values = np.zeros_like(k_values, dtype=np.float64)
        for row_idx in range(len(distributional_fit_df)):
            for k_idx, k in enumerate(k_values):
                integral_values[
                    k_idx
                ] = compute_failure_rate_at_k_attempts_under_beta_three_parameter_distribution(
                    k=k,
                    alpha=distributional_fit_df["alpha"].values[row_idx],
                    beta=distributional_fit_df["beta"].values[row_idx],
                    scale=distributional_fit_df["scale"].values[row_idx],
                )

            tmp_df = pd.DataFrame.from_dict(
                {
                    "Scaling Parameter": k_values,
                    "Neg Log Score": -np.log1p(-integral_values),
                    "groupby_placeholder": ["placeholder"] * len(k_values),
                }
            )
            tmp_df["alpha Neg Log Score"] = np.exp(
                tmp_df["Neg Log Score"].values[0]
            ) * np.power(
                tmp_df["Scaling Parameter"],
                -distributional_fit_df["alpha"].values[row_idx],
            )

            # import matplotlib.pyplot as plt
            # import seaborn as sns
            #
            # plt.close()
            # g = sns.lineplot(
            #     tmp_df,
            #     x="Scaling Parameter",
            #     y="Neg Log Score",
            # )
            # g = sns.lineplot(
            #     tmp_df,
            #     x="Scaling Parameter",
            #     y="alpha Neg Log Score",
            # )
            # g.set(xscale="log", yscale="log")
            # plt.show()

            # Fit a power law to the integral values.
            (
                _,
                fitted_power_law_parameters_df,
            ) = src.analyze.fit_power_law(
                tmp_df,
                covariate_col="Scaling Parameter",
                target_col="Neg Log Score",
                groupby_cols=["groupby_placeholder"],
            )
            distributional_fit_df.loc[
                row_idx, "Log Power Law Prefactor"
            ] = fitted_power_law_parameters_df["Log Power Law Prefactor"].values[0]
            distributional_fit_df.loc[
                row_idx, "Power Law Prefactor"
            ] = fitted_power_law_parameters_df["Power Law Prefactor"].values[0]
            distributional_fit_df.loc[
                row_idx, "Power Law Exponent"
            ] = fitted_power_law_parameters_df["Power Law Exponent"].values[0]

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return distributional_fit_df


def convert_individual_outcomes_to_num_samples_and_num_successes_df(
    individual_outcomes_df: pd.DataFrame,
    groupby_cols: List[str],
) -> pd.DataFrame:
    num_samples_and_num_successes_df = (
        individual_outcomes_df.groupby(
            groupby_cols,
        )
        .agg(
            {
                "Score": ["size", "sum"],
            }
        )
        .reset_index()
    )

    # Clean up columns.
    num_samples_and_num_successes_df.columns = [
        "".join(col).strip() if isinstance(col, tuple) else col
        for col in num_samples_and_num_successes_df.columns
    ]
    num_samples_and_num_successes_df.rename(
        columns={
            "Scoresize": "Num. Samples Total",
            "Scoresum": "Num. Samples Correct",
        },
        inplace=True,
    )
    # Make sure both columns are floats.
    num_samples_and_num_successes_df[
        "Num. Samples Total"
    ] = num_samples_and_num_successes_df["Num. Samples Total"].astype(float)
    num_samples_and_num_successes_df[
        "Num. Samples Correct"
    ] = num_samples_and_num_successes_df["Num. Samples Correct"].astype(float)
    return num_samples_and_num_successes_df


def create_or_load_beta_distributions_pdf_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
):
    beta_distributions_pdf_df_path = os.path.join(
        processed_data_dir, "beta_distributions_pdf.parquet"
    )

    if refresh or not os.path.exists(beta_distributions_pdf_df_path):
        alphas = [0.5, 1.0, 1.5]
        betas = [1.5, 3, 10]

        x = np.logspace(-10, 0, 500)
        dfs_list = []
        for alpha, beta in itertools.product(alphas, betas):
            pdf = scipy.stats.beta.pdf(x, alpha, beta)
            df = pd.DataFrame(
                {
                    "x": x,
                    "p(x)": pdf,
                    r"$\alpha$": np.full_like(x, fill_value=alpha),
                    r"$\beta$": np.full_like(x, fill_value=beta),
                }
            )
            dfs_list.append(df)
        beta_distributions_pdf_df = pd.concat(dfs_list)
        beta_distributions_pdf_df.to_parquet(
            beta_distributions_pdf_df_path,
            index=False,
        )
        print(f"Wrote {beta_distributions_pdf_df_path} to disk.")
        del beta_distributions_pdf_df

    beta_distributions_pdf_df = pd.read_parquet(beta_distributions_pdf_df_path)
    print(
        "Loaded beta_distributions_pdf_df with shape: ",
        beta_distributions_pdf_df.shape,
    )
    return beta_distributions_pdf_df


def create_or_load_bon_jailbreaking_text_beta_binomial_mle_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_beta_binomial_mle_df_path = os.path.join(
        processed_data_dir,
        "bon_jailbreaking_beta_binomial_mle.parquet",
    )
    if refresh or not os.path.exists(bon_jailbreaking_beta_binomial_mle_df_path):
        print(f"Creating {bon_jailbreaking_beta_binomial_mle_df_path} anew...")

        bon_jailbreaking_individual_outcomes_df = src.analyze.create_or_load_bon_jailbreaking_text_individual_outcomes_df(
            refresh=False,
            # refresh=True,
        )
        bon_jailbreaking_num_samples_and_num_successes_df = (
            src.analyze.convert_individual_outcomes_to_num_samples_and_num_successes_df(
                individual_outcomes_df=bon_jailbreaking_individual_outcomes_df,
                groupby_cols=src.globals.BON_JAILBREAKING_GROUPBY_COLS
                + ["Problem Idx"],
            )
        )

        bon_jailbreaking_beta_binomial_mle_df = (
            bon_jailbreaking_num_samples_and_num_successes_df.groupby(
                src.globals.BON_JAILBREAKING_GROUPBY_COLS
            )
            .apply(
                lambda df: src.analyze.fit_beta_binomial_three_parameters_to_num_samples_and_num_successes(
                    num_samples_and_num_successes_df=df
                )
            )
            .reset_index()
        )

        # Add scaling exponent numerically.
        bon_jailbreaking_beta_binomial_mle_df = (
            src.analyze.compute_scaling_exponent_from_distributional_fit(
                distributional_fit_df=bon_jailbreaking_beta_binomial_mle_df,
                distribution="beta_three_parameter",
            )
        )

        bon_jailbreaking_beta_binomial_mle_df.to_parquet(
            bon_jailbreaking_beta_binomial_mle_df_path,
            index=False,
        )

        del bon_jailbreaking_beta_binomial_mle_df

    bon_jailbreaking_beta_binomial_mle_df = pd.read_parquet(
        bon_jailbreaking_beta_binomial_mle_df_path
    )
    print(
        f"Loaded {bon_jailbreaking_beta_binomial_mle_df_path} with shape: ",
        bon_jailbreaking_beta_binomial_mle_df.shape,
    )
    return bon_jailbreaking_beta_binomial_mle_df


def create_or_load_bon_jailbreaking_audio_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_individual_outcomes_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_audio_individual_outcomes.parquet"
    )

    if refresh or not os.path.exists(bon_jailbreaking_individual_outcomes_df_path):
        print(f"Creating {bon_jailbreaking_individual_outcomes_df_path} anew...")

        os.makedirs(processed_data_dir, exist_ok=True)
        bon_jailbreaking_dir = os.path.join(raw_data_dir, "best_of_n_jailbreaking")
        best_of_n_jailbreaking_dfs_list = []
        jsonl_filenames = [
            "DiVA_augs6_sigma0.25_audio_t1.0_n7200.jsonl",
            "gemini-1.5-flash-001_augs6_sigma0.25_audio_t1.0_n7200.jsonl",
            "gemini-1.5-pro-001_augs6_sigma0.25_audio_t1.0_n7200.jsonl",
            "gpt-4o-s2s_augs6_sigma0.25_audio_t1.0_n7200.jsonl",
        ]
        for jsonl_filename in jsonl_filenames:
            # for jsonl_filename in os.listdir(bon_jailbreaking_dir):
            if "audio" not in jsonl_filename:
                continue
            model_name, _, _, modality, temperature, num_samples = jsonl_filename.split(
                "_"
            )
            # Strip off the leading "t" and convert to a float.
            temperature = float(temperature[1:])
            df = pd.read_json(
                os.path.join(bon_jailbreaking_dir, jsonl_filename), lines=True
            )
            df.rename(
                columns={
                    "i": "Problem Idx",
                    "n": "Attempt Idx",
                    "flagged": "Score",
                },
                inplace=True,
            )
            # Convert Score from bool to float.
            df["Score"] = df["Score"].astype(float)
            df["Model"] = src.globals.BON_JAILBREAKING_MODELS_TO_NICE_STRINGS[
                model_name
            ]
            df["Modality"] = src.globals.BON_JAILBREAKING_MODALITY_TO_NICE_STRINGS[
                modality
            ]
            df["Temperature"] = temperature
            best_of_n_jailbreaking_dfs_list.append(df)

        best_of_n_jailbreaking_individual_outcomes_df = pd.concat(
            best_of_n_jailbreaking_dfs_list
        )
        best_of_n_jailbreaking_individual_outcomes_df = (
            best_of_n_jailbreaking_individual_outcomes_df[
                best_of_n_jailbreaking_individual_outcomes_df["Temperature"] == 1.0
            ]
        )

        best_of_n_jailbreaking_individual_outcomes_df.to_parquet(
            bon_jailbreaking_individual_outcomes_df_path,
            index=False,
        )
        print(f"Wrote {bon_jailbreaking_individual_outcomes_df_path} to disk.")
        del best_of_n_jailbreaking_individual_outcomes_df

    bon_jailbreaking_individual_outcomes_df = pd.read_parquet(
        bon_jailbreaking_individual_outcomes_df_path
    )
    print(
        f"Loaded {bon_jailbreaking_individual_outcomes_df_path} with shape: ",
        bon_jailbreaking_individual_outcomes_df.shape,
    )

    return bon_jailbreaking_individual_outcomes_df


def create_or_load_bon_jailbreaking_audio_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_audio_pass_at_k_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_audio_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(bon_jailbreaking_audio_pass_at_k_df_path):
        print(f"Creating {bon_jailbreaking_audio_pass_at_k_df_path} anew...")
        bon_jailbreaking_audio_individual_outcomes_df = (
            create_or_load_bon_jailbreaking_audio_individual_outcomes_df(
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
                refresh=refresh,
            )
        )

        bon_jailbreaking_audio_num_samples_and_num_successes_df = (
            convert_individual_outcomes_to_num_samples_and_num_successes_df(
                individual_outcomes_df=bon_jailbreaking_audio_individual_outcomes_df,
                groupby_cols=["Model", "Modality", "Temperature", "Problem Idx"],
            )
        )

        pass_at_ks_df_list = []
        for (
            model,
            modality,
            temp,
        ), subset_num_samples_and_num_successes_df in bon_jailbreaking_audio_num_samples_and_num_successes_df.groupby(
            ["Model", "Modality", "Temperature"]
        ):
            pass_at_k_df = compute_pass_at_k_from_num_samples_and_num_successes_df(
                num_samples_and_num_successes_df=subset_num_samples_and_num_successes_df,
                ks_list=src.globals.BON_JAILBREAKING_AUDIO_Ks_LIST,
            )
            pass_at_k_df["Model"] = model
            pass_at_k_df["Modality"] = modality
            pass_at_k_df["Temperature"] = temp
            pass_at_ks_df_list.append(pass_at_k_df)

        bon_jailbreaking_audio_pass_at_k_df = pd.concat(
            pass_at_ks_df_list, ignore_index=True
        )
        bon_jailbreaking_audio_pass_at_k_df["Log Score"] = np.log(
            bon_jailbreaking_audio_pass_at_k_df["Score"]
        )
        bon_jailbreaking_audio_pass_at_k_df[
            "Neg Log Score"
        ] = -bon_jailbreaking_audio_pass_at_k_df["Log Score"]
        bon_jailbreaking_audio_pass_at_k_df.to_parquet(
            bon_jailbreaking_audio_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {bon_jailbreaking_audio_pass_at_k_df_path} to disk.")
        del bon_jailbreaking_audio_pass_at_k_df

    bon_jailbreaking_audio_pass_at_k_df = pd.read_parquet(
        bon_jailbreaking_audio_pass_at_k_df_path
    )
    print(
        f"Loaded {bon_jailbreaking_audio_pass_at_k_df_path} with shape: ",
        bon_jailbreaking_audio_pass_at_k_df.shape,
    )
    return bon_jailbreaking_audio_pass_at_k_df


def create_or_load_bon_jailbreaking_text_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_individual_outcomes_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_text_individual_outcomes.parquet"
    )

    if refresh or not os.path.exists(bon_jailbreaking_individual_outcomes_df_path):
        print(f"Creating {bon_jailbreaking_individual_outcomes_df_path} anew...")

        os.makedirs(processed_data_dir, exist_ok=True)
        bon_jailbreaking_dir = os.path.join(raw_data_dir, "best_of_n_jailbreaking")
        best_of_n_jailbreaking_dfs_list = []
        jsonl_filenames = [
            "claude-3-5-sonnet-20240620_text_t1.0_n10000.jsonl",
            "claude-3-opus-20240229_text_t1.0_n10000.jsonl",
            "gemini-1.5-flash-001_text_t1.0_n10000.jsonl",
            "gemini-1.5-pro-001_text_t1.0_n10000.jsonl",
            "gpt-4o-mini_text_t1.0_n10000.jsonl",
            "gpt-4o_text_t1.0_n10000.jsonl",
            "meta-llama-Meta-Llama-3-8B-Instruct_text_t1.0_n10000.jsonl",
        ]
        for jsonl_filename in jsonl_filenames:
            # for jsonl_filename in os.listdir(bon_jailbreaking_dir):
            if "text" not in jsonl_filename:
                continue
            model_name, modality, temperature, num_samples = jsonl_filename.split("_")
            # Strip off the leading "t" and convert to a float.
            temperature = float(temperature[1:])
            df = pd.read_json(
                os.path.join(bon_jailbreaking_dir, jsonl_filename), lines=True
            )
            df.rename(
                columns={
                    "i": "Problem Idx",
                    "n": "Attempt Idx",
                    "flagged": "Score",
                },
                inplace=True,
            )
            # Convert Score from bool to float.
            df["Score"] = df["Score"].astype(float)
            df["Model"] = src.globals.BON_JAILBREAKING_MODELS_TO_NICE_STRINGS[
                model_name
            ]
            df["Modality"] = src.globals.BON_JAILBREAKING_MODALITY_TO_NICE_STRINGS[
                modality
            ]
            df["Temperature"] = temperature
            best_of_n_jailbreaking_dfs_list.append(df)

        best_of_n_jailbreaking_individual_outcomes_df = pd.concat(
            best_of_n_jailbreaking_dfs_list
        )
        best_of_n_jailbreaking_individual_outcomes_df = (
            best_of_n_jailbreaking_individual_outcomes_df[
                best_of_n_jailbreaking_individual_outcomes_df["Temperature"] == 1.0
            ]
        )

        best_of_n_jailbreaking_individual_outcomes_df.to_parquet(
            bon_jailbreaking_individual_outcomes_df_path,
            index=False,
        )
        print(f"Wrote {bon_jailbreaking_individual_outcomes_df_path} to disk.")
        del best_of_n_jailbreaking_individual_outcomes_df

    bon_jailbreaking_individual_outcomes_df = pd.read_parquet(
        bon_jailbreaking_individual_outcomes_df_path
    )
    print(
        f"Loaded {bon_jailbreaking_individual_outcomes_df_path} with shape: ",
        bon_jailbreaking_individual_outcomes_df.shape,
    )

    return bon_jailbreaking_individual_outcomes_df


def create_or_load_bon_jailbreaking_text_kumaraswamy_binomial_mle_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
):
    bon_jailbreaking_text_kumaraswamy_binomial_mle_df_path = os.path.join(
        processed_data_dir,
        "bon_jailbreaking_kumaraswamy_binomial_mle.parquet",
    )
    if refresh or not os.path.exists(
        bon_jailbreaking_text_kumaraswamy_binomial_mle_df_path
    ):
        print(
            f"Creating {bon_jailbreaking_text_kumaraswamy_binomial_mle_df_path} anew..."
        )
        bon_jailbreaking_text_individual_outcomes_df = src.analyze.create_or_load_bon_jailbreaking_text_individual_outcomes_df(
            refresh=False,
            # refresh=True,
        )
        bon_jailbreaking_text_num_samples_and_num_successes_df = (
            src.analyze.convert_individual_outcomes_to_num_samples_and_num_successes_df(
                individual_outcomes_df=bon_jailbreaking_text_individual_outcomes_df,
                groupby_cols=src.globals.BON_JAILBREAKING_GROUPBY_COLS
                + ["Problem Idx"],
            )
        )

        bon_jailbreaking_text_kumaraswamy_binomial_mle_df = (
            bon_jailbreaking_text_num_samples_and_num_successes_df.groupby(
                src.globals.BON_JAILBREAKING_GROUPBY_COLS
            )
            .apply(
                lambda df: src.analyze.fit_kumaraswamy_binomial_three_parameters_to_num_samples_and_num_successes(
                    num_samples_and_num_successes_df=df
                )
            )
            .reset_index()
        )

        # Add scaling exponent numerically.
        bon_jailbreaking_text_kumaraswamy_binomial_mle_df = (
            src.analyze.compute_scaling_exponent_from_distributional_fit(
                distributional_fit_df=bon_jailbreaking_text_kumaraswamy_binomial_mle_df,
                distribution="kumaraswamy_three_parameter",
            )
        )

        bon_jailbreaking_text_kumaraswamy_binomial_mle_df.to_parquet(
            bon_jailbreaking_text_kumaraswamy_binomial_mle_df_path,
            index=False,
        )

        del bon_jailbreaking_text_kumaraswamy_binomial_mle_df

    bon_jailbreaking_text_kumaraswamy_binomial_mle_df = pd.read_parquet(
        bon_jailbreaking_text_kumaraswamy_binomial_mle_df_path
    )
    print(
        f"Loaded {bon_jailbreaking_text_kumaraswamy_binomial_mle_df_path} with shape: ",
        bon_jailbreaking_text_kumaraswamy_binomial_mle_df.shape,
    )
    return bon_jailbreaking_text_kumaraswamy_binomial_mle_df


def create_or_load_bon_jailbreaking_text_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_text_pass_at_k_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_text_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(bon_jailbreaking_text_pass_at_k_df_path):
        print(f"Creating {bon_jailbreaking_text_pass_at_k_df_path} anew...")
        bon_jailbreaking_text_individual_outcomes_df = (
            create_or_load_bon_jailbreaking_text_individual_outcomes_df(
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
                refresh=refresh,
            )
        )

        bon_jailbreaking_text_num_samples_and_num_successes_df = (
            convert_individual_outcomes_to_num_samples_and_num_successes_df(
                individual_outcomes_df=bon_jailbreaking_text_individual_outcomes_df,
                groupby_cols=["Model", "Modality", "Temperature", "Problem Idx"],
            )
        )

        pass_at_ks_df_list = []
        for (
            model,
            modality,
            temp,
        ), subset_num_samples_and_num_successes_df in bon_jailbreaking_text_num_samples_and_num_successes_df.groupby(
            ["Model", "Modality", "Temperature"]
        ):
            pass_at_k_df = compute_pass_at_k_from_num_samples_and_num_successes_df(
                num_samples_and_num_successes_df=subset_num_samples_and_num_successes_df,
                ks_list=src.globals.BON_JAILBREAKING_TEXT_Ks_LIST,
            )
            pass_at_k_df["Model"] = model
            pass_at_k_df["Modality"] = modality
            pass_at_k_df["Temperature"] = temp
            pass_at_ks_df_list.append(pass_at_k_df)

        bon_jailbreaking_text_pass_at_k_df = pd.concat(
            pass_at_ks_df_list, ignore_index=True
        )
        bon_jailbreaking_text_pass_at_k_df["Log Score"] = np.log(
            bon_jailbreaking_text_pass_at_k_df["Score"]
        )
        bon_jailbreaking_text_pass_at_k_df[
            "Neg Log Score"
        ] = -bon_jailbreaking_text_pass_at_k_df["Log Score"]
        bon_jailbreaking_text_pass_at_k_df.to_parquet(
            bon_jailbreaking_text_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {bon_jailbreaking_text_pass_at_k_df_path} to disk.")
        del bon_jailbreaking_text_pass_at_k_df

    bon_jailbreaking_text_pass_at_k_df = pd.read_parquet(
        bon_jailbreaking_text_pass_at_k_df_path
    )
    print(
        f"Loaded {bon_jailbreaking_text_pass_at_k_df_path} with shape: ",
        bon_jailbreaking_text_pass_at_k_df.shape,
    )
    return bon_jailbreaking_text_pass_at_k_df


def create_or_load_bon_jailbreaking_vision_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_vision_individual_outcomes_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_vision_individual_outcomes.parquet"
    )

    if refresh or not os.path.exists(
        bon_jailbreaking_vision_individual_outcomes_df_path
    ):
        print(f"Creating {bon_jailbreaking_vision_individual_outcomes_df_path} anew...")

        os.makedirs(processed_data_dir, exist_ok=True)
        bon_jailbreaking_dir = os.path.join(raw_data_dir, "best_of_n_jailbreaking")
        best_of_n_jailbreaking_dfs_list = []
        jsonl_filenames = [
            "claude-3-5-sonnet-20240620_vision_t1.0_n7200.jsonl",
            "claude-3-opus-20240229_vision_t1.0_n7200.jsonl",
            "gemini-1.5-flash-001_vision_t1.0_n7200.jsonl",
            "gemini-1.5-pro-001_vision_t1.0_n7200.jsonl",
            "gpt-4o-mini_vision_t1.0_n7200.jsonl",
            "gpt-4o_vision_t1.0_n7200.jsonl",
        ]
        for jsonl_filename in jsonl_filenames:
            # for jsonl_filename in os.listdir(bon_jailbreaking_dir):
            if "vision" not in jsonl_filename:
                continue
            model_name, modality, temperature, num_samples = jsonl_filename.split("_")
            # Strip off the leading "t" and convert to a float.
            temperature = float(temperature[1:])
            df = pd.read_json(
                os.path.join(bon_jailbreaking_dir, jsonl_filename), lines=True
            )
            df.rename(
                columns={
                    "i": "Problem Idx",
                    "n": "Attempt Idx",
                    "flagged": "Score",
                },
                inplace=True,
            )
            df["Model"] = src.globals.BON_JAILBREAKING_MODELS_TO_NICE_STRINGS[
                model_name
            ]
            df["Modality"] = src.globals.BON_JAILBREAKING_MODALITY_TO_NICE_STRINGS[
                modality
            ]
            df["Temperature"] = temperature
            best_of_n_jailbreaking_dfs_list.append(df)

        best_of_n_jailbreaking_individual_outcomes_df = pd.concat(
            best_of_n_jailbreaking_dfs_list
        )
        best_of_n_jailbreaking_individual_outcomes_df = (
            best_of_n_jailbreaking_individual_outcomes_df[
                best_of_n_jailbreaking_individual_outcomes_df["Temperature"] == 1.0
            ]
        )

        best_of_n_jailbreaking_individual_outcomes_df.to_parquet(
            bon_jailbreaking_vision_individual_outcomes_df_path,
            index=False,
        )
        print(f"Wrote {bon_jailbreaking_vision_individual_outcomes_df_path} to disk.")
        del best_of_n_jailbreaking_individual_outcomes_df

    bon_jailbreaking_individual_outcomes_df = pd.read_parquet(
        bon_jailbreaking_vision_individual_outcomes_df_path
    )
    print(
        f"Loaded {bon_jailbreaking_vision_individual_outcomes_df_path} with shape: ",
        bon_jailbreaking_individual_outcomes_df.shape,
    )

    return bon_jailbreaking_individual_outcomes_df


def create_or_load_bon_jailbreaking_vision_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_vision_pass_at_k_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_vision_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(bon_jailbreaking_vision_pass_at_k_df_path):
        print(f"Creating {bon_jailbreaking_vision_pass_at_k_df_path} anew...")
        bon_jailbreaking_vision_individual_outcomes_df = (
            create_or_load_bon_jailbreaking_vision_individual_outcomes_df(
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
                refresh=refresh,
            )
        )

        bon_jailbreaking_vision_num_samples_and_num_successes_df = (
            convert_individual_outcomes_to_num_samples_and_num_successes_df(
                individual_outcomes_df=bon_jailbreaking_vision_individual_outcomes_df,
                groupby_cols=["Model", "Modality", "Temperature", "Problem Idx"],
            )
        )

        pass_at_ks_df_list = []
        for (
            model,
            modality,
            temp,
        ), subset_num_samples_and_num_successes_df in bon_jailbreaking_vision_num_samples_and_num_successes_df.groupby(
            ["Model", "Modality", "Temperature"]
        ):
            pass_at_k_df = compute_pass_at_k_from_num_samples_and_num_successes_df(
                num_samples_and_num_successes_df=subset_num_samples_and_num_successes_df,
                ks_list=src.globals.BON_JAILBREAKING_VISION_Ks_LIST,
            )
            pass_at_k_df["Model"] = model
            pass_at_k_df["Modality"] = modality
            pass_at_k_df["Temperature"] = temp
            pass_at_ks_df_list.append(pass_at_k_df)

        bon_jailbreaking_vision_pass_at_k_df = pd.concat(
            pass_at_ks_df_list, ignore_index=True
        )

        bon_jailbreaking_vision_pass_at_k_df["Log Score"] = np.log(
            bon_jailbreaking_vision_pass_at_k_df["Score"]
        )
        bon_jailbreaking_vision_pass_at_k_df[
            "Neg Log Score"
        ] = -bon_jailbreaking_vision_pass_at_k_df["Log Score"]
        bon_jailbreaking_vision_pass_at_k_df.to_parquet(
            bon_jailbreaking_vision_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {bon_jailbreaking_vision_pass_at_k_df_path} to disk.")
        del bon_jailbreaking_vision_pass_at_k_df

    bon_jailbreaking_vision_pass_at_k_df = pd.read_parquet(
        bon_jailbreaking_vision_pass_at_k_df_path
    )
    print(
        f"Loaded {bon_jailbreaking_vision_pass_at_k_df_path} with shape: ",
        bon_jailbreaking_vision_pass_at_k_df.shape,
    )
    return bon_jailbreaking_vision_pass_at_k_df


def create_or_load_cross_validated_bon_jailbreaking_text_scaling_coefficient_data_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    cv_bon_jailbreaking_text_scaling_coefficient_df_path = os.path.join(
        processed_data_dir,
        "cv_bon_jailbreaking_text_scaling_coefficient.parquet",
    )

    if refresh or not os.path.exists(
        cv_bon_jailbreaking_text_scaling_coefficient_df_path
    ):
        print(
            f"Creating {cv_bon_jailbreaking_text_scaling_coefficient_df_path} anew..."
        )
        os.makedirs(processed_data_dir, exist_ok=True)

        # Load the individual outcomes per problem.
        individual_outcomes_per_problem_df = (
            create_or_load_bon_jailbreaking_text_individual_outcomes_df(refresh=False)
        )

        def cross_validate_power_law_coefficient_estimators_from_individual_outcomes_wrapper(
            model: str, modality: str, temperature: float, subset_df: pd.DataFrame
        ):
            result_df = cross_validate_power_law_coefficient_estimators_from_individual_outcomes(
                individual_outcomes_per_problem_df=subset_df,
            )
            result_df["Model"] = model
            result_df["Modality"] = modality
            result_df["Temperature"] = temperature
            return result_df

        cv_bon_jailbreaking_text_scaling_coefficient_dfs_list = []
        for (
            model,
            modality,
            temperature,
        ), subset_df in individual_outcomes_per_problem_df.groupby(
            src.globals.BON_JAILBREAKING_GROUPBY_COLS
        ):
            print(
                f"Processing model: {model}, modality: {modality}, temperature: {temperature}"
            )
            df = cross_validate_power_law_coefficient_estimators_from_individual_outcomes_wrapper(
                model=model,
                modality=modality,
                temperature=temperature,
                subset_df=subset_df,
            )
            cv_bon_jailbreaking_text_scaling_coefficient_dfs_list.append(df)

        cv_bon_jailbreaking_text_scaling_coefficient_df = pd.concat(
            cv_bon_jailbreaking_text_scaling_coefficient_dfs_list,
            ignore_index=True,
        ).reset_index(drop=True)

        cv_bon_jailbreaking_text_scaling_coefficient_df.to_parquet(
            path=cv_bon_jailbreaking_text_scaling_coefficient_df_path
        )
        del cv_bon_jailbreaking_text_scaling_coefficient_df

    cv_bon_jailbreaking_text_scaling_coefficient_df = pd.read_parquet(
        cv_bon_jailbreaking_text_scaling_coefficient_df_path
    )

    print(
        f"Loaded {cv_bon_jailbreaking_text_scaling_coefficient_df_path} with shape: ",
        cv_bon_jailbreaking_text_scaling_coefficient_df.shape,
    )
    return cv_bon_jailbreaking_text_scaling_coefficient_df


def create_or_load_cross_validated_large_language_monkey_pythia_math_scaling_coefficient_data_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    cv_large_language_monkeys_pythia_math_scaling_coefficient_df_path = os.path.join(
        processed_data_dir,
        "cv_large_language_monkeys_pythia_math_scaling_coefficient.parquet",
    )

    if refresh or not os.path.exists(
        cv_large_language_monkeys_pythia_math_scaling_coefficient_df_path
    ):
        print(
            f"Creating {cv_large_language_monkeys_pythia_math_scaling_coefficient_df_path} anew..."
        )
        os.makedirs(processed_data_dir, exist_ok=True)

        # Load the individual outcomes per problem.
        individual_outcomes_per_problem_df = (
            create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
                refresh=False
            )
        )
        # # For prototyping, subset to Pythia 12B.
        # individual_outcomes_per_problem_df = individual_outcomes_per_problem_df[
        #     individual_outcomes_per_problem_df["Model"] == "Pythia 12B"
        # ]

        def cross_validate_power_law_coefficient_estimators_from_individual_outcomes_wrapper(
            model: str, benchmark: str, subset_df: pd.DataFrame
        ):
            result_df = cross_validate_power_law_coefficient_estimators_from_individual_outcomes(
                individual_outcomes_per_problem_df=subset_df,
            )
            result_df["Model"] = model
            result_df["Benchmark"] = benchmark
            return result_df

        cv_large_language_monkeys_pythia_math_scaling_coefficient_dfs_list = []
        for (
            model,
            benchmark,
        ), subset_df in individual_outcomes_per_problem_df.groupby(
            src.globals.LARGE_LANGUAGE_MONKEYS_GROUPBY_COLS
        ):
            print(f"Processing model: {model}, benchmark: {benchmark}")
            df = cross_validate_power_law_coefficient_estimators_from_individual_outcomes_wrapper(
                model=model, benchmark=benchmark, subset_df=subset_df
            )
            cv_large_language_monkeys_pythia_math_scaling_coefficient_dfs_list.append(
                df
            )

        cv_large_language_monkeys_pythia_math_scaling_coefficient_df = pd.concat(
            cv_large_language_monkeys_pythia_math_scaling_coefficient_dfs_list,
            ignore_index=True,
        ).reset_index(drop=True)

        cv_large_language_monkeys_pythia_math_scaling_coefficient_df.to_parquet(
            path=cv_large_language_monkeys_pythia_math_scaling_coefficient_df_path
        )
        del cv_large_language_monkeys_pythia_math_scaling_coefficient_df

    cv_large_language_monkeys_pythia_math_scaling_coefficient_df = pd.read_parquet(
        cv_large_language_monkeys_pythia_math_scaling_coefficient_df_path
    )

    print(
        f"Loaded {cv_large_language_monkeys_pythia_math_scaling_coefficient_df_path} with shape: ",
        cv_large_language_monkeys_pythia_math_scaling_coefficient_df.shape,
    )
    return cv_large_language_monkeys_pythia_math_scaling_coefficient_df


def create_or_load_cross_validated_synthetic_scaling_coefficient_data_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    synthetic_scaling_exponents_data_path = os.path.join(
        processed_data_dir, "cv_synthetic_scaling_coefficient.parquet"
    )

    if refresh or not os.path.exists(synthetic_scaling_exponents_data_path):
        print(f"Creating {synthetic_scaling_exponents_data_path} anew...")
        os.makedirs(processed_data_dir, exist_ok=True)

        # High level sketch:
        # 2. Sweep over the number of samples per problem, computing pass_i@k for many k.
        # 3. Fit the distributional parameters to the synthetic data.
        # 4. Compute the scaling exponent from the distributional fits.

        true_distribution_to_params_dict = {
            "kumaraswamy": [
                {"a": 0.05, "b": 1.5, "scale": 1.0},
                {"a": 0.05, "b": 3.5, "scale": 1.0},
                {"a": 0.2, "b": 1.5, "scale": 0.3},
                {"a": 0.2, "b": 3.5, "scale": 0.3},
            ],
            "beta": [
                {"a": 0.05, "b": 1.5, "scale": 1.0},
                {"a": 0.05, "b": 3.5, "scale": 1.0},
                {"a": 0.2, "b": 1.5, "scale": 0.3},
                {"a": 0.2, "b": 3.5, "scale": 0.3},
            ],
        }

        scaling_exponents_dfs_list = []
        for true_distribution in true_distribution_to_params_dict:
            for true_distribution_params in true_distribution_to_params_dict[
                true_distribution
            ]:
                # 1. Generate synthetic data, sweeping over multiple true distributions and distributional parameters.
                if true_distribution == "beta":
                    true_asymptotic_power_law_exponent = true_distribution_params["a"]
                    true_distribution_nice_str = f"Beta({true_distribution_params['a']}, {true_distribution_params['b']}, {true_distribution_params['scale']})"

                elif true_distribution == "continuous_bernoulli":
                    true_asymptotic_power_law_exponent = 1.0
                    true_distribution_nice_str = (
                        f"Continuous Bernoulli({true_distribution_params['lam']})"
                    )
                elif true_distribution == "kumaraswamy":
                    true_asymptotic_power_law_exponent = true_distribution_params["a"]
                    true_distribution_nice_str = f"Kumaraswamy({true_distribution_params['a']}, {true_distribution_params['b']}, {true_distribution_params['scale']})"
                else:
                    raise NotImplementedError(
                        f"Unknown distribution: {true_distribution}"
                    )

                # Prepare arguments for each simulation
                simulation_args = [
                    (
                        idx,
                        true_distribution,
                        true_distribution_params,
                        true_distribution_nice_str,
                        true_asymptotic_power_law_exponent,
                    )
                    for idx in range(5)
                ]

                # for simulation_idx in np.arange(5, dtype=int):
                #     individual_outcomes_per_problem_df = (
                #         sample_synthetic_individual_outcomes_per_problem_df(
                #             num_problems=1_000,
                #             num_samples_per_problem=100_000,
                #             distribution=true_distribution,
                #             distribution_parameters=true_distribution_params,
                #         )
                #     )
                #     df = cross_validate_power_law_coefficient_estimators_from_individual_outcomes(
                #         individual_outcomes_per_problem_df=individual_outcomes_per_problem_df,
                #         num_repeats=1,
                #     )
                #     df["True Distribution"] = true_distribution_nice_str
                #     df[
                #         "True Distribution Asymptotic Power Law Exponent"
                #     ] = true_asymptotic_power_law_exponent
                #     df["Simulation Idx"] = simulation_idx
                #     pprint.pprint(df)
                #     scaling_exponents_dfs_list.append(df)

                # Run simulations in parallel
                # Run simulations in parallel
                with ProcessPoolExecutor(max_workers=16) as executor:
                    # Map the simulation function directly to the arguments
                    results = executor.map(
                        create_or_load_cross_validated_synthetic_scaling_coefficient_data_df_helper,
                        simulation_args,
                    )
                    for df in results:
                        scaling_exponents_dfs_list.append(df)

        synthetic_scaling_exponents_df = pd.concat(
            scaling_exponents_dfs_list, ignore_index=True
        ).reset_index(drop=True)

        synthetic_scaling_exponents_df["Asymptotic Squared Error"] = 0.5 * np.square(
            synthetic_scaling_exponents_df["Fit Power Law Exponent"]
            - synthetic_scaling_exponents_df[
                "True Distribution Asymptotic Power Law Exponent"
            ]
        )
        synthetic_scaling_exponents_df["Asymptotic Relative Error"] = np.divide(
            np.abs(
                synthetic_scaling_exponents_df["Fit Power Law Exponent"]
                - synthetic_scaling_exponents_df[
                    "True Distribution Asymptotic Power Law Exponent"
                ]
            ),
            synthetic_scaling_exponents_df[
                "True Distribution Asymptotic Power Law Exponent"
            ],
        )
        synthetic_scaling_exponents_df.to_parquet(
            path=synthetic_scaling_exponents_data_path
        )
        del synthetic_scaling_exponents_df

    synthetic_scaling_exponents_df = pd.read_parquet(
        synthetic_scaling_exponents_data_path
    )

    print(
        f"Loaded {synthetic_scaling_exponents_data_path} with shape: ",
        synthetic_scaling_exponents_df.shape,
    )
    return synthetic_scaling_exponents_df


def create_or_load_cross_validated_synthetic_scaling_coefficient_data_df_helper(args):
    """Function to run a single simulation that will be parallelized"""
    # Unpack the arguments tuple
    (
        simulation_idx,
        true_distribution,
        true_distribution_params,
        true_distribution_nice_str,
        true_asymptotic_power_law_exponent,
    ) = args

    np.random.seed(simulation_idx)

    individual_outcomes_per_problem_df = (
        sample_synthetic_individual_outcomes_per_problem_df(
            num_problems=1_000,
            num_samples_per_problem=100_000,
            distribution=true_distribution,
            distribution_parameters=true_distribution_params,
        )
    )

    df = cross_validate_power_law_coefficient_estimators_from_individual_outcomes(
        individual_outcomes_per_problem_df=individual_outcomes_per_problem_df,
        num_repeats=5,
    )

    df["True Distribution"] = true_distribution_nice_str
    df[
        "True Distribution Asymptotic Power Law Exponent"
    ] = true_asymptotic_power_law_exponent
    df["Simulation Idx"] = simulation_idx
    return df


def create_or_load_large_language_monkeys_code_contests_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_code_contests_individual_outcomes_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_code_contests_individual_outcomes.parquet",
    )

    if refresh or not os.path.exists(
        large_language_monkeys_code_contests_individual_outcomes_df_path
    ):
        print(
            f"Creating {large_language_monkeys_code_contests_individual_outcomes_df_path} anew..."
        )

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_original_dfs_list = []
        subsets = [
            "CodeContests_Llama-3-8B",
            "CodeContests_Llama-3-8B-Instruct",
            "CodeContests_Llama-3-70B-Instruct",
            "CodeContests_Gemma-2B",
            "CodeContests_Gemma-7B",
            # "MiniF2F-MATH_Llama-3-8B-Instruct",
            # "MiniF2F-MATH_Llama-3-70B-Instruct",
        ]
        for subset in subsets:
            benchmark, model = subset.split("_")
            ds = load_dataset("ScalingIntelligence/monkey_business", subset)["test"]
            correct: List[List[bool]] = ds["is_corrects"]
            # Shape: (128, 10000)
            wide_df = pd.DataFrame(
                correct,
                columns=1 + np.arange(10000),
                dtype=np.float16,
            )
            # Convert to floats.
            wide_df = wide_df.astype(np.float16)
            wide_df["Problem Idx"] = ds["orig_dset_idx"]
            df = wide_df.melt(
                id_vars=["Problem Idx"],
                var_name="Attempt Idx",
                value_name="Score",
            )

            df["Benchmark"] = benchmark
            # Convert, e.g., "Pythia-1.4B" to "Pythia 1.4B".
            df["Model"] = model.replace("-", " ")
            large_language_monkeys_original_dfs_list.append(df)

        large_language_monkeys_original_individual_outcomes_df = pd.concat(
            large_language_monkeys_original_dfs_list,
        )
        large_language_monkeys_original_individual_outcomes_df[
            "Attempt Idx"
        ] = pd.to_numeric(
            large_language_monkeys_original_individual_outcomes_df["Attempt Idx"]
        )

        large_language_monkeys_original_individual_outcomes_df.to_parquet(
            large_language_monkeys_code_contests_individual_outcomes_df_path,
            index=False,
        )

        print(
            f"Wrote {large_language_monkeys_code_contests_individual_outcomes_df_path} to disk."
        )
        del large_language_monkeys_original_individual_outcomes_df

    large_language_monkeys_original_individual_outcomes_df = pd.read_parquet(
        large_language_monkeys_code_contests_individual_outcomes_df_path
    )
    print(
        f"Loaded {large_language_monkeys_code_contests_individual_outcomes_df_path} with shape: ",
        large_language_monkeys_original_individual_outcomes_df.shape,
    )
    return large_language_monkeys_original_individual_outcomes_df


def create_or_load_large_language_monkeys_code_contests_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_code_contests_pass_at_k_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_code_contests_pass_at_k.parquet",
    )

    if refresh or not os.path.exists(
        large_language_monkeys_code_contests_pass_at_k_df_path
    ):
        print(
            f"Creating {large_language_monkeys_code_contests_pass_at_k_df_path} anew..."
        )

        large_language_monkeys_code_contests_individual_outcomes_df = (
            create_or_load_large_language_monkeys_code_contests_individual_outcomes_df(
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
                refresh=refresh,
            )
        )

        large_language_monkeys_code_contests_num_samples_and_num_successes_df = convert_individual_outcomes_to_num_samples_and_num_successes_df(
            individual_outcomes_df=large_language_monkeys_code_contests_individual_outcomes_df,
            groupby_cols=["Model", "Benchmark", "Problem Idx"],
        )

        pass_at_k_dfs_list = []
        for (
            model,
            benchmark,
        ), subset_num_samples_and_num_successes_df in large_language_monkeys_code_contests_num_samples_and_num_successes_df.groupby(
            ["Model", "Benchmark"]
        ):
            pass_at_k_df = compute_pass_at_k_from_num_samples_and_num_successes_df(
                num_samples_and_num_successes_df=subset_num_samples_and_num_successes_df,
                ks_list=src.globals.LARGE_LANGUAGE_MONKEYS_ORIGINAL_Ks_LIST,
            )
            pass_at_k_df["Model"] = model
            pass_at_k_df["Benchmark"] = benchmark
            pass_at_k_dfs_list.append(pass_at_k_df)

        large_language_monkeys_code_contests_pass_at_k_df = pd.concat(
            pass_at_k_dfs_list, ignore_index=True
        )
        large_language_monkeys_code_contests_pass_at_k_df["Log Score"] = np.log(
            large_language_monkeys_code_contests_pass_at_k_df["Score"]
        )
        large_language_monkeys_code_contests_pass_at_k_df[
            "Neg Log Score"
        ] = -large_language_monkeys_code_contests_pass_at_k_df["Log Score"]
        large_language_monkeys_code_contests_pass_at_k_df.to_parquet(
            large_language_monkeys_code_contests_pass_at_k_df_path,
            index=False,
        )

        print(
            f"Wrote {large_language_monkeys_code_contests_pass_at_k_df_path} to disk."
        )
        del large_language_monkeys_code_contests_pass_at_k_df

    large_language_monkeys_code_contests_pass_at_k_df = pd.read_parquet(
        large_language_monkeys_code_contests_pass_at_k_df_path
    )
    print(
        f"Loaded {large_language_monkeys_code_contests_pass_at_k_df_path} with shape: ",
        large_language_monkeys_code_contests_pass_at_k_df.shape,
    )
    return large_language_monkeys_code_contests_pass_at_k_df


def create_or_load_large_language_monkeys_mini_f2f_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_mini_f2f_individual_outcomes_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_mini_f2f_individual_outcomes.parquet",
    )

    if refresh or not os.path.exists(
        large_language_monkeys_mini_f2f_individual_outcomes_df_path
    ):
        print(
            f"Creating {large_language_monkeys_mini_f2f_individual_outcomes_df_path} anew..."
        )

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_original_dfs_list = []
        subsets = [
            "MiniF2F-MATH_Llama-3-8B-Instruct",
            "MiniF2F-MATH_Llama-3-70B-Instruct",
        ]
        for subset in subsets:
            benchmark, model = subset.split("_")
            ds = load_dataset("ScalingIntelligence/monkey_business", subset)["test"]
            correct: List[List[bool]] = ds["is_corrects"]
            # Shape: (128, 10000)
            wide_df = pd.DataFrame(
                correct,
                columns=1 + np.arange(10000),
                dtype=np.float16,
            )
            # Convert to floats.
            wide_df = wide_df.astype(np.float16)
            wide_df["Problem Idx"] = ds["orig_dset_idx"]
            df = wide_df.melt(
                id_vars=["Problem Idx"],
                var_name="Attempt Idx",
                value_name="Score",
            )

            df["Benchmark"] = benchmark
            # Convert, e.g., "Pythia-1.4B" to "Pythia 1.4B".
            df["Model"] = model.replace("-", " ")
            large_language_monkeys_original_dfs_list.append(df)

        large_language_monkeys_original_individual_outcomes_df = pd.concat(
            large_language_monkeys_original_dfs_list,
        )
        large_language_monkeys_original_individual_outcomes_df[
            "Attempt Idx"
        ] = pd.to_numeric(
            large_language_monkeys_original_individual_outcomes_df["Attempt Idx"]
        )

        large_language_monkeys_original_individual_outcomes_df.to_parquet(
            large_language_monkeys_mini_f2f_individual_outcomes_df_path,
            index=False,
        )

        print(
            f"Wrote {large_language_monkeys_mini_f2f_individual_outcomes_df_path} to disk."
        )
        del large_language_monkeys_original_individual_outcomes_df

    large_language_monkeys_original_individual_outcomes_df = pd.read_parquet(
        large_language_monkeys_mini_f2f_individual_outcomes_df_path
    )
    print(
        f"Loaded {large_language_monkeys_mini_f2f_individual_outcomes_df_path} with shape: ",
        large_language_monkeys_original_individual_outcomes_df.shape,
    )
    return large_language_monkeys_original_individual_outcomes_df


def create_or_load_large_language_monkeys_mini_f2f_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_mini_f2f_pass_at_k_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_mini_f2f_pass_at_k.parquet",
    )

    if refresh or not os.path.exists(large_language_monkeys_mini_f2f_pass_at_k_df_path):
        print(f"Creating {large_language_monkeys_mini_f2f_pass_at_k_df_path} anew...")

        large_language_monkeys_mini_f2f_individual_outcomes_df = (
            create_or_load_large_language_monkeys_mini_f2f_individual_outcomes_df(
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
                refresh=refresh,
            )
        )

        large_language_monkeys_mini_f2f_num_samples_and_num_successes_df = convert_individual_outcomes_to_num_samples_and_num_successes_df(
            individual_outcomes_df=large_language_monkeys_mini_f2f_individual_outcomes_df,
            groupby_cols=["Model", "Benchmark", "Problem Idx"],
        )

        pass_at_k_dfs_list = []
        for (
            model,
            benchmark,
        ), subset_num_samples_and_num_successes_df in large_language_monkeys_mini_f2f_num_samples_and_num_successes_df.groupby(
            ["Model", "Benchmark"]
        ):
            pass_at_k_df = compute_pass_at_k_from_num_samples_and_num_successes_df(
                num_samples_and_num_successes_df=subset_num_samples_and_num_successes_df,
                ks_list=src.globals.LARGE_LANGUAGE_MONKEYS_ORIGINAL_Ks_LIST,
            )
            pass_at_k_df["Model"] = model
            pass_at_k_df["Benchmark"] = benchmark
            pass_at_k_dfs_list.append(pass_at_k_df)

        large_language_monkeys_mini_f2f_pass_at_k_df = pd.concat(
            pass_at_k_dfs_list, ignore_index=True
        )
        large_language_monkeys_mini_f2f_pass_at_k_df["Log Score"] = np.log(
            large_language_monkeys_mini_f2f_pass_at_k_df["Score"]
        )
        large_language_monkeys_mini_f2f_pass_at_k_df[
            "Neg Log Score"
        ] = -large_language_monkeys_mini_f2f_pass_at_k_df["Log Score"]
        large_language_monkeys_mini_f2f_pass_at_k_df.to_parquet(
            large_language_monkeys_mini_f2f_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {large_language_monkeys_mini_f2f_pass_at_k_df_path} to disk.")
        del large_language_monkeys_mini_f2f_pass_at_k_df

    large_language_monkeys_mini_f2f_pass_at_k_df = pd.read_parquet(
        large_language_monkeys_mini_f2f_pass_at_k_df_path
    )
    print(
        f"Loaded {large_language_monkeys_mini_f2f_pass_at_k_df_path} with shape: ",
        large_language_monkeys_mini_f2f_pass_at_k_df.shape,
    )
    return large_language_monkeys_mini_f2f_pass_at_k_df


def create_or_load_large_language_monkeys_pythia_math_beta_binomial_mle_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_pythia_math_beta_binomial_mle_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_pythia_math_beta_binomial_mle.parquet",
    )
    if refresh or not os.path.exists(
        large_language_monkeys_pythia_math_beta_binomial_mle_df_path
    ):
        print(
            f"Creating {large_language_monkeys_pythia_math_beta_binomial_mle_df_path} anew..."
        )
        llmonkeys_groupby_cols = ["Model", "Benchmark"]
        llmonkeys_individual_outcomes_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
            refresh=False,
            # refresh=True,
        )
        llmonkeys_num_samples_and_num_successes_df = (
            src.analyze.convert_individual_outcomes_to_num_samples_and_num_successes_df(
                individual_outcomes_df=llmonkeys_individual_outcomes_df,
                groupby_cols=llmonkeys_groupby_cols + ["Problem Idx"],
            )
        )

        large_language_monkeys_pythia_math_beta_binomial_mle_df = (
            llmonkeys_num_samples_and_num_successes_df.groupby(llmonkeys_groupby_cols)
            .apply(
                lambda df: src.analyze.fit_beta_binomial_three_parameters_to_num_samples_and_num_successes(
                    num_samples_and_num_successes_df=df
                )
            )
            .reset_index()
        )

        # Add scaling exponent numerically.
        large_language_monkeys_pythia_math_beta_binomial_mle_df = src.analyze.compute_scaling_exponent_from_distributional_fit(
            distributional_fit_df=large_language_monkeys_pythia_math_beta_binomial_mle_df,
            distribution="beta_three_parameter",
        )

        large_language_monkeys_pythia_math_beta_binomial_mle_df.to_parquet(
            large_language_monkeys_pythia_math_beta_binomial_mle_df_path,
            index=False,
        )

        del large_language_monkeys_pythia_math_beta_binomial_mle_df

    large_language_monkeys_pythia_math_beta_binomial_mle_df = pd.read_parquet(
        large_language_monkeys_pythia_math_beta_binomial_mle_df_path
    )
    print(
        f"Loaded {large_language_monkeys_pythia_math_beta_binomial_mle_df_path} with shape: ",
        large_language_monkeys_pythia_math_beta_binomial_mle_df.shape,
    )
    return large_language_monkeys_pythia_math_beta_binomial_mle_df


def create_or_load_large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_pythia_math_kumaraswamy_binomial_mle.parquet",
    )
    if refresh or not os.path.exists(
        large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df_path
    ):
        print(
            f"Creating {large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df_path} anew..."
        )
        llmonkeys_groupby_cols = ["Model", "Benchmark"]
        llmonkeys_individual_outcomes_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
            refresh=False,
            # refresh=True,
        )
        llmonkeys_num_samples_and_num_successes_df = (
            src.analyze.convert_individual_outcomes_to_num_samples_and_num_successes_df(
                individual_outcomes_df=llmonkeys_individual_outcomes_df,
                groupby_cols=llmonkeys_groupby_cols + ["Problem Idx"],
            )
        )

        large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df = (
            llmonkeys_num_samples_and_num_successes_df.groupby(llmonkeys_groupby_cols)
            .apply(
                lambda df: src.analyze.fit_kumaraswamy_binomial_three_parameters_to_num_samples_and_num_successes(
                    num_samples_and_num_successes_df=df
                )
            )
            .reset_index()
        )

        # Add scaling exponent numerically.
        large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df = src.analyze.compute_scaling_exponent_from_distributional_fit(
            distributional_fit_df=large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df,
            distribution="kumaraswamy_three_parameter",
        )

        large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df.to_parquet(
            large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df_path,
            index=False,
        )

        del large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df

    large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df = pd.read_parquet(
        large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df_path
    )
    print(
        f"Loaded {large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df_path} with shape: ",
        large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df.shape,
    )
    return large_language_monkeys_pythia_math_kumaraswamy_binomial_mle_df


def create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_pythia_math_individual_outcomes_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_pythia_math_individual_outcomes.parquet",
    )

    if refresh or not os.path.exists(
        large_language_monkeys_pythia_math_individual_outcomes_df_path
    ):
        print(
            f"Creating {large_language_monkeys_pythia_math_individual_outcomes_df_path} anew..."
        )

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_pythia_math_dfs_list = []
        subsets = [
            "MATH_Pythia-70M",
            "MATH_Pythia-160M",
            "MATH_Pythia-410M",
            "MATH_Pythia-1B",
            "MATH_Pythia-2.8B",
            "MATH_Pythia-6.9B",
            "MATH_Pythia-12B",
        ]
        for subset in subsets:
            benchmark, model = subset.split("_")
            ds = load_dataset("ScalingIntelligence/monkey_business", subset)["test"]
            correct: List[List[bool]] = ds["is_corrects"]
            # Shape: (128, 10000)
            wide_df = pd.DataFrame(
                correct,
                columns=1 + np.arange(10000),
                dtype=np.float16,
            )
            # Convert to floats.
            wide_df = wide_df.astype(np.float16)
            wide_df["Problem Idx"] = ds["orig_dset_idx"]
            df = wide_df.melt(
                id_vars=["Problem Idx"],
                var_name="Attempt Idx",
                value_name="Score",
            )

            df["Benchmark"] = benchmark
            # Convert, e.g., "Pythia-1.4B" to "Pythia 1.4B".
            df["Model"] = model.replace("-", " ")
            large_language_monkeys_pythia_math_dfs_list.append(df)

        large_language_monkeys_pythia_math_individual_outcomes_df = pd.concat(
            large_language_monkeys_pythia_math_dfs_list,
        )
        large_language_monkeys_pythia_math_individual_outcomes_df[
            "Attempt Idx"
        ] = pd.to_numeric(
            large_language_monkeys_pythia_math_individual_outcomes_df["Attempt Idx"]
        )

        large_language_monkeys_pythia_math_individual_outcomes_df.to_parquet(
            large_language_monkeys_pythia_math_individual_outcomes_df_path,
            index=False,
        )

        print(
            f"Wrote {large_language_monkeys_pythia_math_individual_outcomes_df_path} to disk."
        )
        del large_language_monkeys_pythia_math_individual_outcomes_df

    large_language_monkeys_pythia_math_individual_outcomes_df = pd.read_parquet(
        large_language_monkeys_pythia_math_individual_outcomes_df_path
    )
    print(
        f"Loaded {large_language_monkeys_pythia_math_individual_outcomes_df_path} with shape: ",
        large_language_monkeys_pythia_math_individual_outcomes_df.shape,
    )
    return large_language_monkeys_pythia_math_individual_outcomes_df


def create_or_load_large_language_monkeys_pythia_math_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_pythia_math_pass_at_k_df_path = os.path.join(
        processed_data_dir, "large_language_monkeys_pythia_math_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(
        large_language_monkeys_pythia_math_pass_at_k_df_path
    ):
        print(
            f"Creating {large_language_monkeys_pythia_math_pass_at_k_df_path} anew..."
        )
        large_language_monkeys_pythia_math_individual_outcomes_df = (
            create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
                refresh=refresh,
            )
        )

        large_language_monkeys_pythia_math_num_samples_and_num_successes_df = convert_individual_outcomes_to_num_samples_and_num_successes_df(
            individual_outcomes_df=large_language_monkeys_pythia_math_individual_outcomes_df,
            groupby_cols=["Model", "Benchmark", "Problem Idx"],
        )

        pass_at_k_dfs_list = []
        for (
            model,
            benchmark,
        ), subset_num_samples_and_num_successes_df in large_language_monkeys_pythia_math_num_samples_and_num_successes_df.groupby(
            ["Model", "Benchmark"]
        ):
            pass_at_k_df = compute_pass_at_k_from_num_samples_and_num_successes_df(
                num_samples_and_num_successes_df=subset_num_samples_and_num_successes_df,
                ks_list=src.globals.LARGE_LANGUAGE_MONKEYS_ORIGINAL_Ks_LIST,
            )
            pass_at_k_df["Model"] = model
            pass_at_k_df["Benchmark"] = benchmark
            pass_at_k_dfs_list.append(pass_at_k_df)

        large_language_monkeys_pythia_math_pass_at_k_df = pd.concat(
            pass_at_k_dfs_list, ignore_index=True
        )
        large_language_monkeys_pythia_math_pass_at_k_df["Log Score"] = np.log(
            large_language_monkeys_pythia_math_pass_at_k_df["Score"]
        )
        large_language_monkeys_pythia_math_pass_at_k_df[
            "Neg Log Score"
        ] = -large_language_monkeys_pythia_math_pass_at_k_df["Log Score"]
        large_language_monkeys_pythia_math_pass_at_k_df.to_parquet(
            large_language_monkeys_pythia_math_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {large_language_monkeys_pythia_math_pass_at_k_df_path} to disk.")
        del large_language_monkeys_pythia_math_pass_at_k_df

    large_language_monkeys_pythia_math_pass_at_k_df = pd.read_parquet(
        large_language_monkeys_pythia_math_pass_at_k_df_path
    )
    print(
        f"Loaded {large_language_monkeys_pythia_math_pass_at_k_df_path} with shape: ",
        large_language_monkeys_pythia_math_pass_at_k_df.shape,
    )
    return large_language_monkeys_pythia_math_pass_at_k_df


def create_or_load_large_language_monkeys_pythia_math_pass_at_1_beta_fits(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load the original pass@k data on MATH.
    llmonkeys_original_pass_at_k_df = (
        src.analyze.create_or_load_large_language_monkeys_pythia_math_pass_at_k_df(
            refresh=refresh,
        )
    )
    # Keep only pass@1 data on MATH.
    llmonkeys_original_pass_at_1_df = llmonkeys_original_pass_at_k_df[
        (llmonkeys_original_pass_at_k_df["Scaling Parameter"] == 1)
        & (llmonkeys_original_pass_at_k_df["Benchmark"] == "MATH")
        # & (llmonkeys_original_pass_at_k_df["Score"] > 1e-5)
    ].copy()

    large_language_monkeys_pass_at_1_beta_fits_df_path = os.path.join(
        processed_data_dir, "llmonkeys_pass_at_1_beta_fits.parquet"
    )
    if refresh or not os.path.exists(
        large_language_monkeys_pass_at_1_beta_fits_df_path
    ):
        print(f"Creating {large_language_monkeys_pass_at_1_beta_fits_df_path} anew...")

        llmonkeys_original_pass_at_1_copy_df = llmonkeys_original_pass_at_1_df.copy()
        # llmonkeys_original_pass_at_1_copy_df = llmonkeys_original_pass_at_1_copy_df[
        #     llmonkeys_original_pass_at_1_copy_df["Score"] > 0.0
        # ].copy()
        # Slightly inflate the zero values for fitting.
        llmonkeys_original_pass_at_1_copy_df["Score"][
            llmonkeys_original_pass_at_1_copy_df["Score"] == 0.0
        ] += 1e-8

        # For each model, fit a beta distribution to the pass@1 data using MLE.
        llmonkeys_pass_at_1_beta_fits_df = (
            llmonkeys_original_pass_at_1_copy_df.groupby(
                ["Model", "Benchmark", "Scaling Parameter"]
            )
            .apply(
                lambda df: pd.Series(
                    scipy.stats.beta.fit(
                        df["Score"].values, floc=0.0, fscale=1.01 * df["Score"].max()
                    ),
                    index=["a", "b", "loc", "scale"],
                )
            )
            .reset_index()
        )

        llmonkeys_pass_at_1_beta_fits_df.to_parquet(
            large_language_monkeys_pass_at_1_beta_fits_df_path,
            index=False,
        )
        del llmonkeys_pass_at_1_beta_fits_df

    llmonkeys_pass_at_1_beta_fits_df = pd.read_parquet(
        large_language_monkeys_pass_at_1_beta_fits_df_path
    )

    return llmonkeys_original_pass_at_1_df, llmonkeys_pass_at_1_beta_fits_df


def create_or_load_many_shot_icl_probability_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    many_shot_icl_probability_df_path = os.path.join(
        processed_data_dir, "many_shot_icl_probability.parquet"
    )
    if refresh or not os.path.exists(many_shot_icl_probability_df_path):
        print(f"Creating {many_shot_icl_probability_df_path} anew...")
        os.makedirs(processed_data_dir, exist_ok=True)
        dfs_list = []
        for parquet_filename in os.listdir(os.path.join(raw_data_dir, "many_shot_icl")):
            if not parquet_filename.endswith(".parquet"):
                continue
            df = pd.read_parquet(
                os.path.join(raw_data_dir, "many_shot_icl", parquet_filename)
            )
            dfs_list.append(df)

        many_shot_icl_probability_df = pd.concat(dfs_list)

        # # Create a unique token index from "Problem Idx" and "Seq Idx"
        # many_shot_icl_probability_df["Token Idx"] = (
        #     many_shot_icl_probability_df["Problem Idx"]
        #     * many_shot_icl_probability_df["Seq Idx"].max()
        #     + many_shot_icl_probability_df["Seq Idx"]
        # )

        many_shot_icl_probability_df.rename(
            columns={
                "log_probs": "Log Score",
            },
            inplace=True,
        )
        many_shot_icl_probability_df["Score"] = np.exp(
            many_shot_icl_probability_df["Log Score"]
        )
        many_shot_icl_probability_df["Neg Log Score"] = -many_shot_icl_probability_df[
            "Log Score"
        ]

        many_shot_icl_probability_df[
            "Scaling Parameter"
        ] = many_shot_icl_probability_df["Num. Shots"]

        many_shot_icl_probability_df.to_parquet(
            many_shot_icl_probability_df_path,
            index=False,
        )
        print(f"Wrote {many_shot_icl_probability_df_path} to disk.")
        del many_shot_icl_probability_df

    many_shot_icl_probability_df = pd.read_parquet(many_shot_icl_probability_df_path)
    print(
        "Loaded many_shot_icl_probability_df_path with shape: ",
        many_shot_icl_probability_df.shape,
    )
    return many_shot_icl_probability_df


def create_or_load_pretraining_math_prob_answer_given_problem_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    pretraining_gsm8k_neg_log_likelihood_df_path = os.path.join(
        processed_data_dir, "pretraining_math_neg_log_likelihood.parquet"
    )

    if refresh or not os.path.exists(pretraining_gsm8k_neg_log_likelihood_df_path):
        print("Creating pretraining_math_neg_log_likelihood_df_path anew...")

        os.makedirs(processed_data_dir, exist_ok=True)
        pretraining_gsm8k_log_likelihood_df = pd.read_csv(
            os.path.join(
                raw_data_dir, "pretraining_math", "pythia_gsm8k_log_likelihoods.csv"
            )
        )
        # pretraining_math_log_likelihood_df = pd.read_csv(
        #     os.path.join(
        #         raw_data_dir, "pretraining_math", "pythia_math_log_likelihoods.csv"
        #     )
        # )
        model_nicknames_to_keep = [
            "Pythia_14M_20B",
            "Pythia_70M_60B",
            "Pythia_160M_60B",
            "Pythia_410M_200B",
            "Pythia_1B_300B",
            "Pythia_1.4B_300B",
            "Pythia_2.8B_300B",
            "Pythia_6.9B_300B",
            "Pythia_12B_300B",
        ]
        pretraining_gsm8k_log_likelihood_df = pretraining_gsm8k_log_likelihood_df[
            pretraining_gsm8k_log_likelihood_df["Model Nickname"].isin(
                model_nicknames_to_keep
            )
        ]
        pretraining_gsm8k_log_likelihood_df["Dataset"] = "GSM8K"

        pretraining_gsm8k_summed_neg_log_likelihood_df = (
            pretraining_gsm8k_log_likelihood_df.groupby(
                ["Model Nickname", "Dataset", "prompt_idx"]
            )["Neg Log Likelihood"]
            .sum()
            .reset_index()
        )

        models_metadata_df = pd.read_csv(
            os.path.join(raw_data_dir, "pretraining_math", "models_pythia.csv")
        )
        models_metadata_df["Pretraining Compute"] = (
            6.0 * models_metadata_df["Tokens"] * models_metadata_df["Parameters"]
        )

        pretraining_gsm8k_summed_neg_log_likelihood_df = (
            pretraining_gsm8k_summed_neg_log_likelihood_df.merge(
                models_metadata_df[
                    ["Model Nickname", "Model Family", "Pretraining Compute"]
                ],
                how="inner",
                on="Model Nickname",
            )
        )
        pretraining_gsm8k_summed_neg_log_likelihood_df[
            "Log Score"
        ] = -pretraining_gsm8k_summed_neg_log_likelihood_df["Neg Log Likelihood"]

        pretraining_gsm8k_summed_neg_log_likelihood_df["Score"] = np.exp(
            pretraining_gsm8k_summed_neg_log_likelihood_df["Log Score"]
        )

        pretraining_gsm8k_summed_neg_log_likelihood_df.rename(
            columns={
                "Neg Log Likelihood": "Neg Log Score",
                "Model Nickname": "Model",
                "Pretraining Compute": "Scaling Parameter",
                "prompt_idx": "Problem Idx",
            },
            inplace=True,
        )

        pretraining_gsm8k_summed_neg_log_likelihood_df.to_parquet(
            pretraining_gsm8k_neg_log_likelihood_df_path,
            index=False,
        )
        del pretraining_gsm8k_summed_neg_log_likelihood_df

    pretraining_gsm8k_log_likelihood_df = pd.read_parquet(
        pretraining_gsm8k_neg_log_likelihood_df_path
    )
    print(
        "Loaded pretraining_gsm8k_neg_log_likelihood_df_path with shape: ",
        pretraining_gsm8k_log_likelihood_df.shape,
    )

    return pretraining_gsm8k_log_likelihood_df


def create_or_load_pretraining_probability_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    pretraining_probability_df_path = os.path.join(
        processed_data_dir, "pretraining_probability.parquet"
    )
    if refresh or not os.path.exists(pretraining_probability_df_path):
        print(f"Creating {pretraining_probability_df_path} anew...")
        os.makedirs(processed_data_dir, exist_ok=True)
        dfs_list = []
        for parquet_filename in os.listdir(
            os.path.join(raw_data_dir, "pretraining_causal_language_modeling")
        ):
            if not parquet_filename.endswith(".parquet"):
                continue
            df = pd.read_parquet(
                os.path.join(
                    raw_data_dir,
                    "pretraining_causal_language_modeling",
                    parquet_filename,
                )
            )
            dfs_list.append(df)

        pretraining_probability_df = pd.concat(dfs_list)

        # Create a unique token index from "Problem Idx" and "Seq Idx"
        pretraining_probability_df["Token Idx"] = (
            pretraining_probability_df["Problem Idx"]
            * pretraining_probability_df["Seq Idx"].max()
            + pretraining_probability_df["Seq Idx"]
        )

        pretraining_probability_df.rename(
            columns={
                "log_probs": "Log Score",
            },
            inplace=True,
        )
        pretraining_probability_df["Score"] = np.exp(
            pretraining_probability_df["Log Score"]
        )
        pretraining_probability_df["Neg Log Score"] = -pretraining_probability_df[
            "Log Score"
        ]

        models_metadata_df = pd.read_csv(
            os.path.join(
                raw_data_dir, "pretraining_causal_language_modeling", "models.csv"
            )
        )
        models_metadata_df["Scaling Parameter"] = (
            6.0 * models_metadata_df["Tokens"] * models_metadata_df["Parameters"]
        )

        pretraining_probability_df = pretraining_probability_df.merge(
            models_metadata_df[["Model Nickname", "Model Family", "Scaling Parameter"]],
            how="inner",
            on="Model Nickname",
        )

        pretraining_probability_df.to_parquet(
            pretraining_probability_df_path,
            index=False,
        )
        print(f"Wrote {pretraining_probability_df_path} to disk.")
        del pretraining_probability_df

    pretraining_probability_df = pd.read_parquet(pretraining_probability_df_path)
    print(
        "Loaded pretraining_probability_df_path with shape: ",
        pretraining_probability_df.shape,
    )
    return pretraining_probability_df


def cross_validate_power_law_coefficient_estimators_from_individual_outcomes(
    individual_outcomes_per_problem_df: pd.DataFrame,
    num_repeats: int = 5,
    num_problems_list: Optional[List[int]] = None,
    num_samples_per_problem_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    unique_problem_indices = individual_outcomes_per_problem_df["Problem Idx"].unique()
    unique_attempt_indices = individual_outcomes_per_problem_df["Attempt Idx"].unique()

    if num_problems_list is None:
        num_problems_list = [
            64,
            96,
            128,
        ]
        assert max(num_problems_list) <= len(unique_problem_indices)
    if num_samples_per_problem_list is None:
        num_samples_per_problem_list = [
            10,
            32,
            100,
            316,
            1000,
            3162,
            10000,
        ]
        assert max(num_samples_per_problem_list) <= len(unique_attempt_indices)

    max_num_samples_per_problem = max(num_samples_per_problem_list)

    individual_outcomes_per_problem: np.ndarray = (
        individual_outcomes_per_problem_df.pivot(
            index="Problem Idx",
            columns="Attempt Idx",
            values="Score",
        ).values
    )

    ks_list: List[int] = np.unique(
        np.logspace(
            0,
            np.log10(max_num_samples_per_problem),
            100,  # Fit using 100 uniformly spaced samples.
        ).astype(int)
    ).tolist()

    # Step 1: Compute the "true" power law exponent using least squares fitting on all the data.
    pass_at_k_df = src.analyze.compute_pass_at_k_from_individual_outcomes(
        individual_outcomes_per_problem=individual_outcomes_per_problem,
        ks_list=ks_list,
    )
    avg_pass_at_k_df = (
        pass_at_k_df.groupby("Scaling Parameter")["Score"].mean().reset_index()
    )
    avg_pass_at_k_df["Neg Log Score"] = -np.log(avg_pass_at_k_df["Score"])
    avg_pass_at_k_df["Placeholder"] = "Placeholder"
    (
        _,
        least_sqrs_fitted_power_law_parameters_df,
    ) = src.analyze.fit_power_law(
        df=avg_pass_at_k_df,
        covariate_col="Scaling Parameter",
        target_col="Neg Log Score",
        groupby_cols=["Placeholder"],
    )

    # Note: These aren't actually the true parameters but we will treat them as true for our analysis.
    full_data_least_squares_power_law_prefactor = (
        least_sqrs_fitted_power_law_parameters_df["Power Law Prefactor"].values[0]
    )
    full_data_least_squares_power_law_exponent = (
        least_sqrs_fitted_power_law_parameters_df["Power Law Exponent"].values[0]
    )

    # Clean up.
    del pass_at_k_df, avg_pass_at_k_df, least_sqrs_fitted_power_law_parameters_df

    # Step 2: Take subsets of problems and samples per problem and repeats.
    all_combos = itertools.product(
        num_problems_list, num_samples_per_problem_list, range(num_repeats)
    )
    # Implementation 1 (Serial).
    cross_validated_power_law_coefficient_estimators_dfs_list = []
    for num_problems, num_samples_per_problem, repeat_idx in all_combos:
        df = cross_validate_power_law_coefficient_estimators_from_individual_outcomes_helper(
            num_problems,
            num_samples_per_problem,
            repeat_idx,
            individual_outcomes_per_problem,
            individual_outcomes_per_problem_df,
            unique_problem_indices,
            unique_attempt_indices,
        )
        cross_validated_power_law_coefficient_estimators_dfs_list.append(df)
    # # Implementation 2 (Parallel).
    # cross_validated_power_law_coefficient_estimators_dfs_list = joblib.Parallel(
    #     n_jobs=10, backend="threading"
    # )(
    #     joblib.delayed(
    #         cross_validate_power_law_coefficient_estimators_from_individual_outcomes_helper
    #     )(
    #         num_problems,
    #         num_samples_per_problem,
    #         repeat_idx,
    #         individual_outcomes_per_problem,
    #         individual_outcomes_per_problem_df,
    #         unique_problem_indices,
    #         unique_attempt_indices,
    #     )
    #     for num_problems, num_samples_per_problem, repeat_idx in all_combos
    # )

    cross_validated_power_law_coefficient_estimators_df = pd.concat(
        cross_validated_power_law_coefficient_estimators_dfs_list, ignore_index=True
    ).reset_index(drop=True)

    cross_validated_power_law_coefficient_estimators_df[
        "Full Data Least Squares Power Law Exponent"
    ] = full_data_least_squares_power_law_prefactor
    cross_validated_power_law_coefficient_estimators_df[
        "Full Data Least Squares Power Law Exponent"
    ] = full_data_least_squares_power_law_exponent
    cross_validated_power_law_coefficient_estimators_df[
        "Full Data Least Squares Squared Error"
    ] = np.square(
        cross_validated_power_law_coefficient_estimators_df["Fit Power Law Exponent"]
        - cross_validated_power_law_coefficient_estimators_df[
            "Full Data Least Squares Power Law Exponent"
        ]
    )
    cross_validated_power_law_coefficient_estimators_df[
        "Full Data Least Squares Relative Error"
    ] = np.divide(
        np.abs(
            cross_validated_power_law_coefficient_estimators_df[
                "Fit Power Law Exponent"
            ]
            - cross_validated_power_law_coefficient_estimators_df[
                "Full Data Least Squares Power Law Exponent"
            ]
        ),
        cross_validated_power_law_coefficient_estimators_df[
            "Full Data Least Squares Power Law Exponent"
        ],
    )
    return cross_validated_power_law_coefficient_estimators_df


def cross_validate_power_law_coefficient_estimators_from_individual_outcomes_helper(
    num_problems: int,
    num_samples_per_problem: int,
    repeat_idx: int,
    individual_outcomes_per_problem: np.ndarray,
    individual_outcomes_per_problem_df: pd.DataFrame,
    unique_problem_indices: np.ndarray,
    unique_attempt_indices: np.ndarray,
) -> pd.DataFrame:
    problems_subset_indices = np.random.choice(
        individual_outcomes_per_problem.shape[0],
        size=num_problems,
        replace=False,
    ).astype(int)
    samples_subset_indices = np.random.choice(
        individual_outcomes_per_problem.shape[1],
        size=num_samples_per_problem,
        replace=False,
    ).astype(int)
    subset_individual_outcomes_per_problem = individual_outcomes_per_problem[
        np.ix_(problems_subset_indices, samples_subset_indices)
    ]

    ks_list: List[int] = np.unique(
        np.logspace(
            0,
            np.log10(num_samples_per_problem),
            100,
        ).astype(int)
    ).tolist()

    subset_pass_at_k_df = src.analyze.compute_pass_at_k_from_individual_outcomes(
        individual_outcomes_per_problem=subset_individual_outcomes_per_problem,
        ks_list=ks_list,
    )

    # Method 1: Least-squares fit.
    subset_avg_pass_at_k_df = (
        subset_pass_at_k_df.groupby("Scaling Parameter")["Score"].mean().reset_index()
    )
    subset_avg_pass_at_k_df["Neg Log Score"] = -np.log(subset_avg_pass_at_k_df["Score"])
    subset_avg_pass_at_k_df["Placeholder"] = "Placeholder"
    (
        subset_avg_pass_at_k_with_predictions_df,
        subset_least_sqrs_fitted_power_law_parameters_df,
    ) = src.analyze.fit_power_law(
        df=subset_avg_pass_at_k_df,
        covariate_col="Scaling Parameter",
        target_col="Neg Log Score",
        groupby_cols=["Placeholder"],
    )

    df_lst_sqrs_fit = pd.DataFrame(
        {
            "Num. Problems": num_problems,
            "Num. Samples per Problem": [num_samples_per_problem],
            "Fit Power Law Prefactor": [
                subset_least_sqrs_fitted_power_law_parameters_df[
                    "Power Law Prefactor"
                ].values[0]
            ],
            "Fit Power Law Exponent": [
                subset_least_sqrs_fitted_power_law_parameters_df[
                    "Power Law Exponent"
                ].values[0]
            ],
            "Fit Method": "Least Squares",
            "Repeat Index": [repeat_idx],
        }
    )

    # Method 2: Distributional fit to pass_i@1.
    subset_individual_outcomes_per_problem_df = individual_outcomes_per_problem_df[
        (
            individual_outcomes_per_problem_df["Problem Idx"].isin(
                unique_problem_indices[problems_subset_indices]
            )
        )
        & (
            individual_outcomes_per_problem_df["Attempt Idx"].isin(
                unique_attempt_indices[samples_subset_indices]
            )
        )
    ]
    subset_num_samples_and_num_successes_df = (
        src.analyze.convert_individual_outcomes_to_num_samples_and_num_successes_df(
            individual_outcomes_df=subset_individual_outcomes_per_problem_df,
            groupby_cols=["Problem Idx"],
        )
    )

    subset_kumaraswamy_binomial_mle_df = pd.DataFrame(
        src.analyze.fit_kumaraswamy_binomial_three_parameters_to_num_samples_and_num_successes(
            num_samples_and_num_successes_df=subset_num_samples_and_num_successes_df
        )
    ).T

    # Add power law exponent numerically.
    subset_kumaraswamy_binomial_mle_df = (
        src.analyze.compute_scaling_exponent_from_distributional_fit(
            distributional_fit_df=subset_kumaraswamy_binomial_mle_df,
            distribution="beta_three_parameter",
        )
    )

    df_kumaraswamy_binomial_fit = pd.DataFrame(
        {
            "Num. Problems": num_problems,
            "Num. Samples per Problem": [num_samples_per_problem],
            "Fit Power Law Prefactor": [
                subset_kumaraswamy_binomial_mle_df["Power Law Prefactor"].values[0]
            ],
            "Fit Power Law Exponent": [
                subset_kumaraswamy_binomial_mle_df["Power Law Exponent"].values[0]
            ],
            "Fit Method": "Kumaraswamy-Binomial",
            "Repeat Index": [repeat_idx],
        }
    )

    subset_beta_binomial_mle_df = pd.DataFrame(
        src.analyze.fit_beta_binomial_three_parameters_to_num_samples_and_num_successes(
            num_samples_and_num_successes_df=subset_num_samples_and_num_successes_df
        )
    ).T

    # Add power law exponent numerically.
    subset_beta_binomial_mle_df = (
        src.analyze.compute_scaling_exponent_from_distributional_fit(
            distributional_fit_df=subset_beta_binomial_mle_df,
            distribution="beta_three_parameter",
        )
    )

    df_beta_binomial_fit = pd.DataFrame(
        {
            "Num. Problems": num_problems,
            "Num. Samples per Problem": [num_samples_per_problem],
            "Fit Power Law Prefactor": [
                subset_beta_binomial_mle_df["Power Law Prefactor"].values[0]
            ],
            "Fit Power Law Exponent": [
                subset_beta_binomial_mle_df["Power Law Exponent"].values[0]
            ],
            "Fit Method": "Beta-Binomial",
            "Repeat Index": [repeat_idx],
        }
    )

    # # Debugging.
    # import matplotlib.pyplot as plt
    #
    # plt.close()
    # plt.plot(
    #     subset_avg_pass_at_k_df["Scaling Parameter"],
    #     subset_avg_pass_at_k_df["Neg Log Score"],
    #     label="Real",
    # )
    # plt.plot(
    #     subset_avg_pass_at_k_with_predictions_df["Scaling Parameter"],
    #     subset_avg_pass_at_k_with_predictions_df["Predicted Neg Log Score"],
    #     label="Lst Sqr",
    # )
    # plt.plot(
    #     ks_list,
    #     df_kumaraswamy_binomial_fit["Fit Power Law Prefactor"][0]
    #     * np.power(
    #         np.array(ks_list), -df_kumaraswamy_binomial_fit["Fit Power Law Exponent"][0]
    #     ),
    #     label="KumarBin",
    # )
    # plt.plot(
    #     ks_list,
    #     df_beta_binomial_fit["Fit Power Law Prefactor"][0]
    #     * np.power(
    #         np.array(ks_list), -df_beta_binomial_fit["Fit Power Law Exponent"][0]
    #     ),
    #     label="BetaBin",
    # )
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Columns: Num. Problems, Num. Samples per Problem, Fit Power Law Prefactor, Fit Power Law Exponent, Fit Method, Repeat Index
    # 3 Rows: Least Squares, Kumaraswamy-Binomial, Beta-Binomial
    result_df = pd.concat(
        [df_lst_sqrs_fit, df_kumaraswamy_binomial_fit, df_beta_binomial_fit],
        ignore_index=True,
    )
    return result_df


def estimate_pass_at_k(
    num_samples_total: Union[int, List[int], np.ndarray],
    num_samples_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).

        The OpenAI function, as initially written, assumes $n >= k$.
        Technically, the BoN sampling methodology violates this assumption. I'm not sure
        what the fix should be.
        """
        if (n - c) < k:
            # Every subset of size $k$ will have at least 1 success.
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples_total, int):
        num_samples_total = np.full_like(
            num_samples_correct, fill_value=num_samples_total
        )
    else:
        assert len(num_samples_total) == len(num_samples_correct)

    pass_at_k = np.array(
        [estimator(n, c, k) for n, c in zip(num_samples_total, num_samples_correct)]
    )
    return pass_at_k


def fit_beta_binomial_three_parameters_to_num_samples_and_num_successes(
    num_samples_and_num_successes_df: pd.DataFrame,
    maxiter: int = 25,
    # epsilon: Optional[float] = None,
    epsilon: Optional[float] = 1e-6,
) -> pd.Series:
    num_data = len(num_samples_and_num_successes_df)
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    if np.all(num_successes == 0):
        result = pd.Series(
            {
                "alpha": np.nan,
                "beta": np.nan,
                "loc": np.nan,
                "scale": np.nan,
                "neg_log_likelihood": np.nan,
                "maxiter": maxiter,
                "success": "Failure (Num Successes All Zeros)",
            }
        )
        return result

    fraction_successes = np.divide(num_successes, num_samples)
    largest_fraction_successes = np.max(fraction_successes)
    # Compute scale as (n+1) * max(fraction_successes) / n.
    # We inflate the scale to correct for bias; is this correct?
    # I think we actually want to divide by the expected value of the maximum of
    # n i.i.d. Beta(alpha, beta) random variables. But this doesn't appear to exist
    # in close form or even numerically?
    # TODO(rylan): Investigate this further.
    scale = (num_data + 1.0) * largest_fraction_successes / num_data
    # Make sure that scale isn't more than 1.0 + epsilon.
    scale = min(scale, 1.0)

    # Start with reasonable initial alpha, beta.
    alpha, beta, _, _ = scipy.stats.beta.fit(
        np.clip(
            fraction_successes, epsilon, 1.0 - epsilon
        ),  # Make sure that we remain in [0., 1.]
        floc=0.0,
        fscale=scale,  # Force the scale to be the max scale.
    )
    initial_params = (alpha, beta)
    # Create extremely generous bounds for alpha, beta.
    bounds = [
        (0.01, 100),
        (0.01, 100),
    ]

    # Fit alpha, beta, scale to the scaled beta binomial
    # try:
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
            params,
            scale=scale,
            num_samples=num_samples,
            num_successes=num_successes,
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            maxiter=maxiter,
            maxls=200,
            gtol=1e-4,  # Gradient tolerance, adjust as needed),
            ftol=1e-4,
        ),
    )
    alpha = optimize_result.x[0]
    beta = optimize_result.x[1]
    neg_log_likelihood = optimize_result.fun

    result = pd.Series(
        {
            "alpha": alpha,
            "beta": beta,
            "loc": 0.0,
            "scale": scale,
            "neg_log_likelihood": neg_log_likelihood,
            "maxiter": maxiter,
            "success": "Success" if optimize_result.success else "Failure",
        }
    )
    return result


def fit_beta_binomial_two_parameters_to_num_samples_and_num_successes(
    num_samples_and_num_successes_df: pd.DataFrame,
) -> pd.Series:
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    initial_params = (0.5, 3.5)
    bounds = [
        (0.01, 100),
        (0.01, 100),
    ]

    # Fit alpha, beta, scale to the scaled beta binomial
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_beta_binomial_two_parameters_negative_log_likelihood(
            num_samples=num_samples,
            num_successes=num_successes,
            params=params,
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            maxiter=5000,
            maxls=100,
            gtol=1e-6,  # Gradient tolerance, adjust as needed),
        ),
    )

    result = pd.Series(
        {
            "alpha": optimize_result.x[0],
            "beta": optimize_result.x[1],
            "loc": 0.0,
            "scale": 1.0,
            "neg_log_likelihood": optimize_result.fun,
            "aic": 2 * len(initial_params) + 2 * optimize_result.fun,
            "bic": len(initial_params) * np.log(len(num_samples_and_num_successes_df))
            + 2 * optimize_result.fun,
            "Power Law Exponent": optimize_result.x[0],
        }
    )

    return result


def fit_beta_negative_binomial_three_parameters_to_num_samples_and_num_successes(
    num_samples_and_num_successes_df: pd.DataFrame,
    maxiter: int = 25,
    # epsilon: Optional[float] = None,
    epsilon: Optional[float] = 1e-6,
) -> pd.Series:
    num_data = len(num_samples_and_num_successes_df)
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    if np.all(num_successes == 0):
        result = pd.Series(
            {
                "alpha": np.nan,
                "beta": np.nan,
                "loc": np.nan,
                "scale": np.nan,
                "neg_log_likelihood": np.nan,
                "maxiter": maxiter,
                "success": "Failure (Num Successes All Zeros)",
            }
        )
        return result

    fraction_successes = np.divide(num_successes, num_samples)
    largest_fraction_successes = np.max(fraction_successes)
    # Compute scale as (n+1) * max(fraction_successes) / n.
    # We inflate the scale to correct for bias; is this correct?
    # I think we actually want to divide by the expected value of the maximum of
    # n i.i.d. Beta(alpha, beta) random variables. But this doesn't appear to exist
    # in close form or even numerically?
    # TODO(rylan): Investigate this further.
    scale = (num_data + 1.0) * largest_fraction_successes / num_data
    # Make sure that scale isn't more than 1.0 + epsilon.
    scale = min(scale, 1.0)

    # Start with reasonable initial alpha, beta.
    alpha, beta, _, _ = scipy.stats.beta.fit(
        np.clip(
            fraction_successes, epsilon, 1.0 - epsilon
        ),  # Make sure that we remain in [0., 1.]
        floc=0.0,
        fscale=scale,  # Force the scale to be the max scale.
    )
    initial_params = (alpha, beta)
    # Create extremely generous bounds for alpha, beta.
    bounds = [
        (0.01, 100),
        (0.01, 100),
    ]

    # Fit alpha, beta, scale to the scaled beta binomial
    # try:
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
            params,
            scale=scale,
            num_samples=num_samples,
            num_successes=num_successes,
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            maxiter=maxiter,
            maxls=200,
            gtol=1e-4,  # Gradient tolerance, adjust as needed),
            ftol=1e-4,
        ),
    )
    alpha = optimize_result.x[0]
    beta = optimize_result.x[1]
    neg_log_likelihood = optimize_result.fun

    result = pd.Series(
        {
            "alpha": alpha,
            "beta": beta,
            "loc": 0.0,
            "scale": scale,
            "neg_log_likelihood": neg_log_likelihood,
            "maxiter": maxiter,
            "success": "Success" if optimize_result.success else "Failure",
        }
    )
    return result


def fit_kumaraswamy_binomial_three_parameters_to_num_samples_and_num_successes(
    num_samples_and_num_successes_df: pd.DataFrame,
    maxiter: int = 25,
    # epsilon: Optional[float] = None,
    epsilon: Optional[float] = 1e-6,
) -> pd.Series:
    num_data = len(num_samples_and_num_successes_df)
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    if np.all(num_successes == 0):
        result = pd.Series(
            {
                "alpha": np.nan,
                "beta": np.nan,
                "loc": np.nan,
                "scale": np.nan,
                "neg_log_likelihood": np.nan,
                "maxiter": maxiter,
                "success": "Failure (Num Successes All Zeros)",
            }
        )
        return result

    fraction_successes = np.divide(num_successes, num_samples)
    largest_fraction_successes = np.max(fraction_successes)
    # Compute scale as (n+1) * max(fraction_successes) / n.
    # We inflate the scale to correct for bias; is this correct?
    # I think we actually want to divide by the expected value of the maximum of
    # n i.i.d. Beta(alpha, beta) random variables. But this doesn't appear to exist
    # in close form or even numerically?
    # TODO(rylan): Investigate this further.
    scale = (num_data + 1.0) * largest_fraction_successes / num_data
    # Make sure that scale isn't more than 1.0 + epsilon.
    scale = min(scale, 1.0)

    # Start with reasonable initial alpha, beta.
    alpha, beta, _, _ = scipy.stats.beta.fit(
        np.clip(
            fraction_successes, epsilon, 1.0 - epsilon
        ),  # Make sure that we remain in [0., 1.]
        floc=0.0,
        fscale=scale,  # Force the scale to be the max scale.
    )
    initial_params = (alpha, beta)
    # Create extremely generous bounds for alpha, beta.
    bounds = [
        (0.01, 100),
        (0.01, 100),
    ]

    # Fit alpha, beta, scale to the scaled beta binomial
    # try:
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_kumaraswamy_binomial_three_parameters_distribution_neg_log_likelihood(
            params,
            scale=scale,
            num_samples=num_samples,
            num_successes=num_successes,
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            maxiter=maxiter,
            maxls=200,
            gtol=1e-4,  # Gradient tolerance, adjust as needed),
            ftol=1e-4,
        ),
    )
    alpha = optimize_result.x[0]
    beta = optimize_result.x[1]
    neg_log_likelihood = optimize_result.fun

    result = pd.Series(
        {
            "alpha": alpha,
            "beta": beta,
            "loc": 0.0,
            "scale": scale,
            "neg_log_likelihood": neg_log_likelihood,
            "maxiter": maxiter,
            "success": "Success" if optimize_result.success else "Failure",
        }
    )
    return result


def fit_pass_at_1_continuous_bernoulli_distribution_parameters(
    data: np.ndarray,
    resolution: float = 1e-4,
    initial_params: Tuple[float, float] = (0.9, 5.1),
    bounds: Tuple[Tuple[float, float]] = ((0.01, 100), (0.01, 100)),
    num_windows_per_factor_of_10: int = 10,
) -> pd.Series:
    smallest_nonzero_pass_at_1 = resolution
    log10_smallest_nonzero_pass_at_1 = np.log10(smallest_nonzero_pass_at_1)
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
    assert data.min() >= bins[0]
    assert data.max() < bins[-1]

    # Maximize the log likelihood by minimizing its negative
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_discretized_neg_log_likelihood(
            params, data=data, bins=bins, distribution="beta"
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            maxiter=5000,
            maxls=100,
            gtol=1e-6,  # Gradient tolerance, adjust as needed),
        ),
    )

    result = pd.Series(
        {
            "a": optimize_result.x[0],
            "b": optimize_result.x[1],
            "loc": 0.0,
            "scale": data.max(),
            "neg_log_likelihood": optimize_result.fun,
            "aic": 2 * len(initial_params) + 2 * optimize_result.fun,
            "bic": len(initial_params) * np.log(len(data)) + 2 * optimize_result.fun,
        }
    )

    return result


def fit_pass_at_1_discretized_beta_distribution_parameters(
    data: np.ndarray,
    resolution: float = 1e-4,
    initial_params: Tuple[float, float] = (0.9, 5.1),
    bounds: Tuple[Tuple[float, float]] = ((0.01, 100), (0.01, 100)),
    num_windows_per_factor_of_10: int = 10,
) -> pd.Series:
    smallest_nonzero_pass_at_1 = resolution
    log10_smallest_nonzero_pass_at_1 = np.log10(smallest_nonzero_pass_at_1)
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
    assert data.min() >= bins[0]
    assert (data.max() < bins[-1]) or data.max() == 1.0

    # Maximize the log likelihood by minimizing its negative
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_discretized_neg_log_likelihood(
            params, data=data, bins=bins, distribution="beta"
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            maxiter=5000,
            maxls=100,
            gtol=1e-6,  # Gradient tolerance, adjust as needed),
        ),
    )

    result = pd.Series(
        {
            "alpha": optimize_result.x[0],
            "beta": optimize_result.x[1],
            "loc": 0.0,
            "scale": data.max(),
            "neg_log_likelihood": optimize_result.fun,
            "aic": 2 * len(initial_params) + 2 * optimize_result.fun,
            "bic": len(initial_params) * np.log(len(data)) + 2 * optimize_result.fun,
        }
    )

    return result


def fit_pass_at_1_discretized_kumaraswamy_distribution_parameters(
    pass_i_at_1_data: np.ndarray,  # Shape: (num of problems,)
    resolution: float = 1e-4,
    initial_params: Tuple[float, float] = (0.9, 5.1),
    bounds: Tuple[Tuple[float, float]] = ((0.01, 100), (0.01, 100)),
    num_windows_per_factor_of_10: int = 10,
) -> pd.Series:
    smallest_nonzero_pass_at_1 = resolution
    log10_smallest_nonzero_pass_at_1 = np.log10(smallest_nonzero_pass_at_1)
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
    assert pass_i_at_1_data.min() >= bins[0]
    assert pass_i_at_1_data.max() < bins[-1]

    # Maximize the log likelihood by minimizing its negative
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_discretized_neg_log_likelihood(
            params, data=pass_i_at_1_data, bins=bins, distribution="kumaraswamy"
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            maxiter=5000,
            maxls=100,
            gtol=1e-6,  # Gradient tolerance, adjust as needed),
        ),
    )

    result = pd.Series(
        {
            "a": optimize_result.x[0],
            "b": optimize_result.x[1],
            "loc": 0.0,
            "scale": pass_i_at_1_data.max(),
            "neg_log_likelihood": optimize_result.fun,
            "aic": 2 * len(initial_params) + 2 * optimize_result.fun,
            "bic": len(initial_params) * np.log(len(pass_i_at_1_data))
            + 2 * optimize_result.fun,
        }
    )

    return result


def fit_power_law(
    df: pd.DataFrame,
    covariate_col: str,
    target_col: str,
    groupby_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits a power law relationship between covariate and target columns within each group.
    The relationship is of the form: log(target) = a * log(covariate) + b
    which implies: target = exp(b) * covariate^a

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the data
    covariate_col : str
        Name of the column containing the independent variable
    target_col : str
        Name of the column containing the dependent variable
    groupby_cols : List[str]
        List of column names to group by before fitting

    Returns:
    --------
    pd.Series
        Multi-indexed Series containing the fitted parameters 'a' and 'b' for each group
    """

    def objective_function(
        params: Tuple[float, float], x: np.ndarray, y: np.ndarray
    ) -> float:
        """Calculate sum of squared errors for current parameters"""
        a, b = params
        predicted = a - b * x
        return np.sum(np.power(y - predicted, 2.0))

    def fit_group(group_df):
        x = group_df[covariate_col]
        # Exclude any np.inf or np.nan values
        which_x_finite = np.isfinite(x)
        if np.all(~which_x_finite):
            return pd.Series(
                {
                    "Log Power Law Prefactor": np.nan,
                    "Power Law Prefactor": np.nan,
                    "Power Law Exponent": np.nan,
                    "Status": f"Failure (All NaN {covariate_col})",
                }
            )
        y = group_df[target_col]
        which_y_finite = np.isfinite(y)
        if np.all(~which_y_finite):
            return pd.Series(
                {
                    "Log Power Law Prefactor": np.nan,
                    "Power Law Prefactor": np.nan,
                    "Power Law Exponent": np.nan,
                    "Status": f"Failure (All NaN {target_col})",
                }
            )
        mask = which_x_finite & which_y_finite
        if np.all(~mask):
            raise ValueError(
                f"No valid data points to fit the power law model.\nFraction x finite: {which_x_finite.mean()}\nFraction y finite: {which_y_finite.mean()}"
            )

        # If we have remaining data, we can proceed.
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            raise ValueError("No valid data points to fit the power law model")

        # Log transform the data
        log_x = np.log(x)
        log_y = np.log(y)

        # Initial guess using linear regression
        x_mean = log_x.mean()
        y_mean = log_y.mean()
        b_init = -np.sum((log_x - x_mean) * (log_y - y_mean)) / np.sum(
            (log_x - x_mean) ** 2
        )
        a_init = -(y_mean - b_init * x_mean)

        # Optimize parameters
        result = minimize(
            objective_function,
            x0=[a_init, b_init],
            args=(log_x, log_y),
            method="Nelder-Mead",
        )

        # Assumed form: a * (scaling factor)^(-b)
        return pd.Series(
            {
                "Log Power Law Prefactor": result.x[0],
                "Power Law Prefactor": np.exp(result.x[0]),
                "Power Law Exponent": result.x[1],
                "Status": "Success" if result.success else "Failure (Fitting)",
            }
        )

    # Group the data and apply the fitting function
    fitted_power_law_parameters_df = df.groupby(groupby_cols).apply(fit_group)

    # Create a copy of the input dataframe to store predictions
    df_with_predictions = df.copy()

    # Calculate predicted values for each group
    for group_idx, params in fitted_power_law_parameters_df.iterrows():
        # Convert group_idx to tuple if it's a single value
        group_idx = (group_idx,) if not isinstance(group_idx, tuple) else group_idx

        # Create boolean mask for the current group
        mask = True
        for col, val in zip(groupby_cols, group_idx):
            mask = mask & (df_with_predictions[col] == val)

        # Calculate predictions using the power law relationship
        x_values = df_with_predictions.loc[mask, covariate_col]
        predicted_values = params["Power Law Prefactor"] * np.power(
            x_values, -params["Power Law Exponent"]
        )

        # Add predictions to the dataframe
        df_with_predictions.loc[mask, f"Predicted {target_col}"] = predicted_values

    fitted_power_law_parameters_df.reset_index(inplace=True)

    return df_with_predictions, fitted_power_law_parameters_df


def sample_synthetic_individual_outcomes_per_problem_df(
    num_problems: int,
    num_samples_per_problem: int,
    distribution: str,
    distribution_parameters: Dict[str, float],
) -> pd.DataFrame:
    if distribution == "beta":
        true_pass_at_1_per_problem = scipy.stats.beta.rvs(
            a=distribution_parameters["a"],
            b=distribution_parameters["b"],
            loc=distribution_parameters.get("loc", 0.0),
            scale=distribution_parameters.get("scale", 1.0),
            size=(num_problems,),
        )
        # Shape: (num_problems, num_samples_per_problem)
        individual_outcomes = scipy.stats.bernoulli.rvs(
            p=true_pass_at_1_per_problem,
            size=(num_samples_per_problem, num_problems),
        ).T
    elif distribution == "kumaraswamy":
        # Generate samples from Kumaraswamy distribution
        a = distribution_parameters["a"]
        b = distribution_parameters["b"]
        scale = distribution_parameters.get("scale", 1.0)
        # Generate uniform random variables
        u = np.random.uniform(0.0, 1.0, size=(num_problems,))
        # Transform to Kumaraswamy using inverse CDF.
        true_pass_at_1_per_problem = np.power(1.0 - np.power(1.0 - u, 1.0 / b), 1.0 / a)
        true_pass_at_1_per_problem *= scale
        assert np.all(
            (0.0 <= true_pass_at_1_per_problem) & (true_pass_at_1_per_problem <= 1.0)
        )
        # Shape: (num_problems, num_samples_per_problem)
        individual_outcomes = scipy.stats.bernoulli.rvs(
            p=true_pass_at_1_per_problem,
            size=(num_samples_per_problem, num_problems),
        ).T
    else:
        raise NotImplementedError

    # Convert to a DataFrame with columns "Problem Idx" and "Attempt Idx".
    problem_idx = np.repeat(np.arange(num_problems), num_samples_per_problem)
    attempt_idx = np.tile(np.arange(num_samples_per_problem), num_problems)
    individual_outcomes_per_problem_df = pd.DataFrame(
        {
            "Problem Idx": problem_idx,
            "Attempt Idx": attempt_idx,
            "Score": individual_outcomes.flatten(),
        }
    )
    return individual_outcomes_per_problem_df


def simulate_neg_log_avg_pass_at_k_from_beta_binomial_mle_df(
    beta_binomial_df: pd.DataFrame,
    columns_to_save: List[str],
    k_values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if k_values is None:
        k_values = np.unique(np.logspace(0, 5, 20, dtype=int))

    simulated_pass_at_k_dfs_list = []
    for _, row in beta_binomial_df.iterrows():
        integral_values = np.zeros_like(k_values, dtype=np.float64)
        for k_idx, k in enumerate(k_values):
            integral_values[
                k_idx
            ] = src.analyze.compute_failure_rate_at_k_attempts_under_beta_three_parameter_distribution(
                k=k,
                alpha=row["alpha"],
                beta=row["beta"],
                scale=row["scale"],
            )
        data_dict = {
            "Scaling Parameter": k_values,
            "Neg Log Score": -np.log1p(-integral_values),
        }
        for column in columns_to_save:
            data_dict[column] = [row[column]] * len(k_values)
        simulated_pass_at_k_dfs_list.append(pd.DataFrame.from_dict(data_dict))
    simulated_pass_at_k_dfs = pd.concat(simulated_pass_at_k_dfs_list, ignore_index=True)
    return simulated_pass_at_k_dfs


def simulate_neg_log_avg_pass_at_k_from_beta_negative_binomial_mle_df(
    beta_negative_binomial_df: pd.DataFrame,
    columns_to_save: List[str],
    k_values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if k_values is None:
        k_values = np.unique(np.logspace(0, 5, 20, dtype=int))

    simulated_pass_at_k_dfs_list = []
    for _, row in beta_negative_binomial_df.iterrows():
        integral_values = np.zeros_like(k_values, dtype=np.float64)
        for k_idx, k in enumerate(k_values):
            integral_values[
                k_idx
            ] = src.analyze.compute_failure_rate_at_k_attempts_under_beta_three_parameter_distribution(
                k=k,
                alpha=row["alpha"],
                beta=row["beta"],
                scale=row["scale"],
            )
        data_dict = {
            "Scaling Parameter": k_values,
            "Neg Log Score": -np.log1p(-integral_values),
        }
        for column in columns_to_save:
            data_dict[column] = [row[column]] * len(k_values)
        simulated_pass_at_k_dfs_list.append(pd.DataFrame.from_dict(data_dict))
    simulated_pass_at_k_dfs = pd.concat(simulated_pass_at_k_dfs_list, ignore_index=True)
    return simulated_pass_at_k_dfs


def simulate_neg_log_avg_pass_at_k_from_kumaraswamy_binomial_mle_df(
    kumaraswamy_binomial_df: pd.DataFrame,
    columns_to_save: List[str],
    k_values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if k_values is None:
        k_values = np.unique(np.logspace(0, 5, 20, dtype=int))

    simulated_pass_at_k_dfs_list = []
    for _, row in kumaraswamy_binomial_df.iterrows():
        integral_values = np.zeros_like(k_values, dtype=np.float64)
        for k_idx, k in enumerate(k_values):
            integral_values[
                k_idx
            ] = src.analyze.compute_failure_rate_at_k_attempts_under_kumaraswamy_three_parameter_distribution(
                k=k,
                alpha=row["alpha"],
                beta=row["beta"],
                scale=row["scale"],
            )
        data_dict = {
            "Scaling Parameter": k_values,
            "Neg Log Score": -np.log1p(-integral_values),
        }
        for column in columns_to_save:
            data_dict[column] = [row[column]] * len(k_values)
        simulated_pass_at_k_dfs_list.append(pd.DataFrame.from_dict(data_dict))
    simulated_pass_at_k_dfs = pd.concat(simulated_pass_at_k_dfs_list, ignore_index=True)
    return simulated_pass_at_k_dfs
