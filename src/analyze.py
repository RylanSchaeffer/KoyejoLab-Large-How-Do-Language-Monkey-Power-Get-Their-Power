from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
import flint
import itertools
import mpmath
from math import comb
from scipy.special import betaln
import numpy as np
import os
import pandas as pd
from scipy import integrate
from scipy.optimize import minimize
import scipy.stats
from typing import Dict, List, Optional, Tuple, Union

import src.globals

# Increase precision.
mpmath.mp.prec = 1000
flint.ctx.prec = 1000

# This helps print more columns.
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)


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
    # scale = np.max(np.divide(num_successes, num_samples)) + 1e-16
    nll_arr = np.zeros_like(num_samples, dtype=np.float64)
    for idx, (n, x) in enumerate(zip(num_samples, num_successes)):
        if not (0 <= x <= n):
            return 0.0

        # def integrand(z):
        #     return (
        #         z ** (x + alpha - 1.0)
        #         * (1.0 - z) ** (beta - 1.0)
        #         * (1.0 - scale * z) ** (n - x)
        #     )

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
        # f = mpmath.hyp2f1(
        #     -(n - x),
        #     x + alpha,
        #     x + alpha + beta,
        #     scale,
        #     # nmaxterms=2000000,
        #     # method="a+bt",
        # )
        f = flint.arb(scale).hypgeom_2f1(float(-(n - x)), x + alpha, x + alpha + beta)
        pmf = binom_factor * c_to_x * B_xa_b * f / B_a_b
        # val = mpmath.quad(integrand, [0, 1])
        # pmf = binom_factor * c_to_x * val / B_a_b
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


def compute_beta_three_parameter_distribution_integrand(
    p: float, k: int, alpha: float, beta: float, c: float
) -> float:
    """
    Log of the integrand to improve numerical stability
    """
    if p <= 0 or p >= c:
        return 0.0

    # Compute in log space to avoid overflow/underflow
    log_term1 = k * np.log1p(-p)
    log_term2 = (alpha - 1) * np.log(p)
    log_term3 = (beta - 1) * np.log(c - p)
    log_term4 = -(alpha + beta - 1) * np.log(c)
    log_term5 = -scipy.special.betaln(alpha, beta)
    log_result = log_term1 + log_term2 + log_term3 + log_term4 + log_term5
    return np.exp(log_result)


def compute_beta_three_parameter_distribution_integral(
    k: int, alpha: float, beta: float, scale: float
) -> float:
    """
    Compute the integral using adaptive quadrature with improved error handling
    """
    if not (0 < scale < 1 and alpha > 0 and beta > 0):
        return np.nan

    # Use a smaller absolute tolerance and increase max evaluations
    result, error = integrate.quad(
        compute_beta_three_parameter_distribution_integrand,
        0.0,
        scale,
        args=(k, alpha, beta, scale),
        epsabs=1e-12,
        epsrel=1e-10,
        limit=5000,
    )

    # Check if result is reasonable
    if not np.isfinite(result) or error > 1e-3 * abs(result):
        return np.nan

    return result


def compute_kumaraswamy_three_parameter_distribution_integrand(
    p: float, k: int, alpha: float, beta: float, scale: float
) -> float:
    raise NotImplementedError


def compute_kumaraswamy_three_parameter_distribution_integral(
    k: int, alpha: float, beta: float, scale: float
) -> float:
    """
    Compute the integral using adaptive quadrature with improved error handling
    """
    if not (0 < scale < 1 and alpha > 0 and beta > 0):
        return np.nan

    # Use a smaller absolute tolerance and increase max evaluations
    result, error = integrate.quad(
        compute_kumaraswamy_three_parameter_distribution_integrand,
        0.0,
        scale,
        args=(k, alpha, beta, scale),
        epsabs=1e-12,
        epsrel=1e-10,
        limit=5000,
    )

    # Check if result is reasonable
    if not np.isfinite(result) or error > 1e-3 * abs(result):
        return np.nan

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


def compute_pass_at_k_from_individual_outcomes(
    individual_outcomes_per_problem: np.ndarray,
    ks_list: List[int],
) -> pd.DataFrame:
    num_problems, num_samples_per_problem = individual_outcomes_per_problem.shape
    pass_at_k_dfs_list = []
    num_samples_total = np.full(num_problems, fill_value=num_samples_per_problem)
    num_samples_correct = individual_outcomes_per_problem.sum(axis=1)
    for k in ks_list:
        pass_at_k = src.analyze.estimate_pass_at_k(
            num_samples_total=num_samples_total,
            num_samples_correct=num_samples_correct,
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


def compute_signed_logsumexp(log_terms: np.ndarray, signs: np.ndarray) -> float:
    """
    Compute log( sum_{i} [ signs[i] * exp(log_terms[i]) ] )
    in a numerically stable manner.

    Parameters
    ----------
    log_terms : array_like
        The logarithms of the absolute values of the terms to be summed.
    signs : array_like
        +1 or -1 for each term.

    Returns
    -------
    float
        log of the signed sum.  If the final sum is <= 0 due to numerical
        cancellation, returns -np.inf.
    """
    # 1) Find the maximum log_term to factor out
    max_log = np.max(log_terms)

    # 2) Accumulate sum of signed exponentials (relative to max_log)
    total = 0.0
    for lt, s in zip(log_terms, signs):
        total += s * np.exp(lt - max_log)

    # 3) Check the sign of the result
    if total <= 0.0:
        # Numerically, the sum should be positive for a valid probability,
        # but catastrophic cancellation could cause negative or zero.
        # We return -inf so the final log PMF is well-defined (log(0) = -inf).
        return -np.inf

    # 4) Otherwise, return log of the magnitude + max_log
    return np.log(total) + max_log


def compute_scaling_exponent_from_distributional_fit(
    distributional_fit_df: pd.DataFrame,
    distribution: str = "beta",
    k_values: Optional[Union[np.ndarray, List[float]]] = None,
) -> pd.DataFrame:
    if k_values is None:
        k_values = np.logspace(0, 5, 25, dtype=int)
    if isinstance(k_values, list):
        k_values = np.array(k_values)

    if distribution == "beta_two_parameter":
        raise NotImplementedError
    elif distribution == "beta_three_parameter":
        distributional_fit_df["a"] = np.nan
        distributional_fit_df["b"] = np.nan
        integral_values = np.zeros_like(k_values, dtype=np.float64)
        for row_idx in range(len(distributional_fit_df)):
            for k_idx, k in enumerate(k_values):
                integral_values[
                    k_idx
                ] = compute_beta_three_parameter_distribution_integral(
                    k=k,
                    alpha=distributional_fit_df["alpha"].values[row_idx],
                    beta=distributional_fit_df["beta"].values[row_idx],
                    scale=distributional_fit_df["scale"].values[row_idx],
                )

            tmp_df = pd.DataFrame.from_dict(
                {
                    "Scaling Parameter": k_values,
                    "Neg Log Score": -np.log(1.0 - integral_values),
                    "groupby_placeholder": ["placeholder"] * len(k_values),
                }
            )

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
            distributional_fit_df.loc[row_idx, "a"] = fitted_power_law_parameters_df[
                "a"
            ].values[0]
            distributional_fit_df.loc[row_idx, "b"] = fitted_power_law_parameters_df[
                "b"
            ].values[0]

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return distributional_fit_df


def convert_individual_outcomes_to_num_samples_and_num_successes(
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


def create_or_load_bon_jailbreaking_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_individual_outcomes_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_individual_outcomes.parquet"
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


def create_or_load_bon_jailbreaking_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_pass_at_k_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(bon_jailbreaking_pass_at_k_df_path):
        print(f"Creating {bon_jailbreaking_pass_at_k_df_path} anew...")
        bon_jailbreaking_individual_outcomes_df = (
            create_or_load_bon_jailbreaking_individual_outcomes_df(
                raw_data_dir=raw_data_dir,
                processed_data_dir=processed_data_dir,
                refresh=refresh,
            )
        )

        bon_jailbreaking_pass_at_k_df = (
            bon_jailbreaking_individual_outcomes_df.groupby(
                ["Model", "Modality", "Temperature", "Problem Idx"]
            )
            .agg(
                {
                    "Score": ["size", "sum"],
                }
            )
            .reset_index()
        )

        bon_jailbreaking_pass_at_k_df.columns = [
            "".join(col).strip() if isinstance(col, tuple) else col
            for col in bon_jailbreaking_pass_at_k_df.columns
        ]
        bon_jailbreaking_pass_at_k_df.rename(
            columns={
                "Scoresize": "Num. Samples Total",
                "Scoresum": "Num. Samples Correct",
            },
            inplace=True,
        )

        bon_jailbreaking_pass_at_k_df = compute_pass_at_k_from_individual_outcomes(
            bon_jailbreaking_pass_at_k_df,
            ks_list=src.globals.BON_JAILBREAKING_Ks_LIST,
        )
        bon_jailbreaking_pass_at_k_df.pivot(
            id_vars=["Model", "Modality", "Temperature"],
        )

        bon_jailbreaking_pass_at_k_df["Log Score"] = np.log(
            bon_jailbreaking_pass_at_k_df["Score"]
        )
        bon_jailbreaking_pass_at_k_df["Neg Log Score"] = -bon_jailbreaking_pass_at_k_df[
            "Log Score"
        ]
        bon_jailbreaking_pass_at_k_df.to_parquet(
            bon_jailbreaking_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {bon_jailbreaking_pass_at_k_df_path} to disk.")
        del bon_jailbreaking_pass_at_k_df

    models_gsm8k_pass_at_k_df = pd.read_parquet(bon_jailbreaking_pass_at_k_df_path)
    print(
        f"Loaded {bon_jailbreaking_pass_at_k_df_path} with shape: ",
        models_gsm8k_pass_at_k_df.shape,
    )
    return models_gsm8k_pass_at_k_df


def create_or_load_large_language_monkeys_llama_code_contests_individual_outcomes_df(
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
    large_language_monkeys_original_pass_at_k_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_llama_code_contests_pass_at_k.parquet",
    )

    if refresh or not os.path.exists(large_language_monkeys_original_pass_at_k_df_path):
        print(f"Creating {large_language_monkeys_original_pass_at_k_df_path} anew...")

        large_language_monkeys_llama_code_contests_scores_df = create_or_load_large_language_monkeys_llama_code_contests_individual_outcomes_df(
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            refresh=refresh,
        )
        large_language_monkeys_code_contests_pass_at_k_df = (
            large_language_monkeys_llama_code_contests_scores_df.groupby(
                ["Model", "Benchmark", "Problem Idx"]
            )
            .agg(
                {
                    "Score": ["size", "sum"],
                }
            )
            .reset_index()
        )

        large_language_monkeys_code_contests_pass_at_k_df.columns = [
            "".join(col).strip() if isinstance(col, tuple) else col
            for col in large_language_monkeys_code_contests_pass_at_k_df.columns
        ]
        large_language_monkeys_code_contests_pass_at_k_df.rename(
            columns={
                "Scoresize": "Num. Samples Total",
                "Scoresum": "Num. Samples Correct",
            },
            inplace=True,
        )

        large_language_monkeys_code_contests_pass_at_k_dfs_list = []
        for k in src.globals.LARGE_LANGUAGE_MONKEYS_ORIGINAL_Ks_LIST:
            large_language_monkeys_original_pass_at_k_df_copy = (
                large_language_monkeys_code_contests_pass_at_k_df.copy()
            )
            large_language_monkeys_original_pass_at_k_df_copy["Scaling Parameter"] = k
            large_language_monkeys_original_pass_at_k_df_copy[
                "Score"
            ] = estimate_pass_at_k(
                num_samples_total=large_language_monkeys_code_contests_pass_at_k_df[
                    "Num. Samples Total"
                ].values,
                num_samples_correct=large_language_monkeys_code_contests_pass_at_k_df[
                    "Num. Samples Correct"
                ].values,
                k=k,
            )
            large_language_monkeys_code_contests_pass_at_k_dfs_list.append(
                large_language_monkeys_original_pass_at_k_df_copy
            )
        large_language_monkeys_code_contests_pass_at_k_df = pd.concat(
            large_language_monkeys_code_contests_pass_at_k_dfs_list
        )
        large_language_monkeys_code_contests_pass_at_k_df["Log Score"] = np.log(
            large_language_monkeys_code_contests_pass_at_k_df["Score"]
        )
        large_language_monkeys_code_contests_pass_at_k_df[
            "Neg Log Score"
        ] = -large_language_monkeys_code_contests_pass_at_k_df["Log Score"]

        large_language_monkeys_code_contests_pass_at_k_df.to_parquet(
            large_language_monkeys_original_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {large_language_monkeys_original_pass_at_k_df_path} to disk.")
        del large_language_monkeys_code_contests_pass_at_k_df

    large_language_monkeys_code_contests_pass_at_k_df = pd.read_parquet(
        large_language_monkeys_original_pass_at_k_df_path
    )
    print(
        "Loaded large_language_monkeys_original_pass_at_k_df_path with shape: ",
        large_language_monkeys_code_contests_pass_at_k_df.shape,
    )
    return large_language_monkeys_code_contests_pass_at_k_df


def create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_original_individual_outcomes_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_original_individual_outcomes.parquet",
    )

    if refresh or not os.path.exists(
        large_language_monkeys_original_individual_outcomes_df_path
    ):
        print(
            f"Creating {large_language_monkeys_original_individual_outcomes_df_path} anew..."
        )

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_original_dfs_list = []
        subsets = [
            "MATH_Pythia-70M",
            "MATH_Pythia-160M",
            "MATH_Pythia-410M",
            "MATH_Pythia-1B",
            # "MATH_Pythia-1.4B",  # Exclude to have 7 to plot.
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
            large_language_monkeys_original_individual_outcomes_df_path,
            index=False,
        )

        print(
            f"Wrote {large_language_monkeys_original_individual_outcomes_df_path} to disk."
        )
        del large_language_monkeys_original_individual_outcomes_df

    large_language_monkeys_original_individual_outcomes_df = pd.read_parquet(
        large_language_monkeys_original_individual_outcomes_df_path
    )
    print(
        f"Loaded {large_language_monkeys_original_individual_outcomes_df_path} with shape: ",
        large_language_monkeys_original_individual_outcomes_df.shape,
    )
    return large_language_monkeys_original_individual_outcomes_df


def create_or_load_large_language_monkeys_pythia_math_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_original_pass_at_k_df_path = os.path.join(
        processed_data_dir, "large_language_monkeys_original_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(large_language_monkeys_original_pass_at_k_df_path):
        print(f"Creating {large_language_monkeys_original_pass_at_k_df_path} anew...")

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_dir = os.path.join(
            raw_data_dir, "large_language_monkeys_original"
        )
        large_language_monkeys_original_dfs_list = []
        subsets = [
            "MATH_Pythia-70M",
            "MATH_Pythia-160M",
            "MATH_Pythia-410M",
            "MATH_Pythia-1B",
            # "MATH_Pythia-1.4B",
            "MATH_Pythia-2.8B",
            "MATH_Pythia-6.9B",
            "MATH_Pythia-12B",
        ]
        for subset in subsets:
            benchmark, model = subset.split("_")
            # df = pd.read_json(
            #     f"hf://datasets/ScalingIntelligence/monkey_business/{subset}"
            # )
            ds = load_dataset("ScalingIntelligence/monkey_business", subset)["test"]
            correct: List[List[bool]] = ds["is_corrects"]
            # Shape: (128, 10000)
            wide_df = pd.DataFrame(
                correct,
                columns=1 + np.arange(10000),
                dtype=np.float16,
            )
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

        large_language_monkeys_original_scores_df = pd.concat(
            large_language_monkeys_original_dfs_list,
        )

        large_language_monkeys_original_pass_at_k_df = (
            large_language_monkeys_original_scores_df.groupby(
                ["Model", "Benchmark", "Problem Idx"]
            )
            .agg(
                {
                    "Score": ["size", "sum"],
                }
            )
            .reset_index()
        )

        large_language_monkeys_original_pass_at_k_df.columns = [
            "".join(col).strip() if isinstance(col, tuple) else col
            for col in large_language_monkeys_original_pass_at_k_df.columns
        ]
        large_language_monkeys_original_pass_at_k_df.rename(
            columns={
                "Scoresize": "Num. Samples Total",
                "Scoresum": "Num. Samples Correct",
            },
            inplace=True,
        )

        large_language_monkeys_original_pass_at_k_dfs_list = []
        for k in src.globals.LARGE_LANGUAGE_MONKEYS_ORIGINAL_Ks_LIST:
            large_language_monkeys_original_pass_at_k_df_copy = (
                large_language_monkeys_original_pass_at_k_df.copy()
            )
            large_language_monkeys_original_pass_at_k_df_copy["Scaling Parameter"] = k
            large_language_monkeys_original_pass_at_k_df_copy[
                "Score"
            ] = estimate_pass_at_k(
                num_samples_total=large_language_monkeys_original_pass_at_k_df[
                    "Num. Samples Total"
                ].values,
                num_samples_correct=large_language_monkeys_original_pass_at_k_df[
                    "Num. Samples Correct"
                ].values,
                k=k,
            )
            large_language_monkeys_original_pass_at_k_dfs_list.append(
                large_language_monkeys_original_pass_at_k_df_copy
            )
        large_language_monkeys_original_pass_at_k_df = pd.concat(
            large_language_monkeys_original_pass_at_k_dfs_list
        )
        large_language_monkeys_original_pass_at_k_df["Log Score"] = np.log(
            large_language_monkeys_original_pass_at_k_df["Score"]
        )
        large_language_monkeys_original_pass_at_k_df[
            "Neg Log Score"
        ] = -large_language_monkeys_original_pass_at_k_df["Log Score"]

        large_language_monkeys_original_pass_at_k_df.to_parquet(
            large_language_monkeys_original_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {large_language_monkeys_original_pass_at_k_df_path} to disk.")
        del large_language_monkeys_original_pass_at_k_df

    large_language_monkeys_original_pass_at_k_df = pd.read_parquet(
        large_language_monkeys_original_pass_at_k_df_path
    )
    print(
        "Loaded large_language_monkeys_original_pass_at_k_df_path with shape: ",
        large_language_monkeys_original_pass_at_k_df.shape,
    )
    return large_language_monkeys_original_pass_at_k_df


def create_or_load_large_language_monkeys_original_pass_at_1_beta_fits(
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


def create_or_load_synthetic_scaling_coefficient_data_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    synthetic_scaling_exponents_data_path = os.path.join(
        processed_data_dir, "synthetic_scaling_coefficient.parquet"
    )

    if refresh or not os.path.exists(synthetic_scaling_exponents_data_path):
        print(f"Creating {synthetic_scaling_exponents_data_path} anew...")
        os.makedirs(processed_data_dir, exist_ok=True)

        # High level sketch:
        # 1. Generate synthetic data, sweeping over multiple true distributions and distributional parameters.
        # 2. Sweep over the number of samples per problem, computing pass_i@k for many k.
        # 3. Fit the distributional parameters to the synthetic data.
        # 4. Compute the scaling exponent from the distributional fits.

        true_distribution_to_params_dict = {
            "beta": [
                {"a": 0.05, "b": 1.5},
                {"a": 0.05, "b": 5.0},
                {"a": 0.1, "b": 1.5},
                {"a": 0.1, "b": 5.0},
            ],
            # "scaled_beta": [
            #     {"a": 0.5, "b": 1.0, "scale": 0.05},
            #     {"a": 0.5, "b": 1.0, "scale": 0.1},
            #     {"a": 0.5, "b": 5.0, "scale": 0.1},
            #     {"a": 0.5, "b": 1.0, "scale": 0.05},
            #     {"a": 0.5, "b": 1.0, "scale": 0.5},
            #     {"a": 0.5, "b": 5.0, "scale": 0.5},
            # ],
        }
        num_problems_list: List[int] = [
            64,
            128,
            256,
        ]
        num_samples_per_problem_list: List[int] = [
            100,
            316,
            1000,
            3162,
            10000,
            31623,
        ]
        max_num_samples_per_problem = max(num_samples_per_problem_list)
        num_repeats = 30

        scaling_exponents_dfs_list = []
        for distribution in true_distribution_to_params_dict:
            for distribution_params in true_distribution_to_params_dict[distribution]:
                if distribution == "beta":
                    theoretical_scaling_exponent = distribution_params["a"]
                else:
                    raise NotImplementedError(f"Unknown distribution: {distribution}")

                for num_problems, repeat_idx in itertools.product(
                    num_problems_list, range(num_repeats)
                ):
                    # Shape: (num_problems, max num samples per problem)
                    individual_outcomes_per_problem = (
                        src.analyze.sample_synthetic_individual_outcomes_per_problem(
                            num_problems=num_problems,
                            num_samples_per_problem=max_num_samples_per_problem,
                            distribution=distribution,
                            distribution_parameters=distribution_params,
                        )
                    )

                    for num_samples_per_problem in num_samples_per_problem_list:
                        subset_individual_outcomes_per_problem = (
                            individual_outcomes_per_problem[:, :num_samples_per_problem]
                        )

                        pass_at_k_df = src.analyze.compute_pass_at_k_from_individual_outcomes(
                            individual_outcomes_per_problem=subset_individual_outcomes_per_problem,
                            ks_list=src.globals.BON_JAILBREAKING_Ks_LIST,
                        )

                        # Extract pass_i@1 and then fit the distributional parameters.
                        pass_at_1_df = pass_at_k_df[
                            pass_at_k_df["Scaling Parameter"] == 1
                        ]
                        beta_fitted_power_law_parameters_df = (
                            src.analyze.fit_pass_at_1_beta_distribution_parameters(
                                data=pass_at_1_df["Score"].values,
                                resolution=1.0 / num_samples_per_problem,
                            )
                        )

                        avg_pass_at_k_df = (
                            pass_at_k_df.groupby("Scaling Parameter")["Score"]
                            .mean()
                            .reset_index()
                        )
                        avg_pass_at_k_df["Neg Log Score"] = -np.log(
                            avg_pass_at_k_df["Score"]
                        )
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

                        scaling_exponents_dfs_list.append(
                            pd.DataFrame(
                                {
                                    "Distribution": [distribution],
                                    "Distribution Parameters": [
                                        str(distribution_params)
                                    ],  # Can't hash a dict, so convert it to a string.
                                    "Num. Problems": num_problems,
                                    r"Num. Samples per Problem ($n$)": [
                                        num_samples_per_problem
                                    ],
                                    "Fit Scaling Exponent": [
                                        least_sqrs_fitted_power_law_parameters_df[
                                            "b"
                                        ].values[0]
                                    ],
                                    "Fit Method": "Least Squares",
                                    "Theoretical Scaling Exponent": [
                                        theoretical_scaling_exponent
                                    ],
                                    "Repeat Index": [repeat_idx],
                                }
                            )
                        )

                        scaling_exponents_dfs_list.append(
                            pd.DataFrame(
                                {
                                    "Distribution": [distribution],
                                    "Distribution Parameters": [
                                        str(distribution_params)
                                    ],  # Can't hash a dict, so convert it to a string.
                                    "Num. Problems": num_problems,
                                    r"Num. Samples per Problem ($n$)": [
                                        num_samples_per_problem
                                    ],
                                    "Fit Scaling Exponent": [
                                        beta_fitted_power_law_parameters_df["alpha"]
                                    ],
                                    "Fit Method": "Distribution",
                                    "Theoretical Scaling Exponent": [
                                        theoretical_scaling_exponent
                                    ],
                                    "Repeat Index": [repeat_idx],
                                }
                            )
                        )

        synthetic_scaling_exponents_df = pd.concat(
            scaling_exponents_dfs_list, ignore_index=True
        ).reset_index(drop=True)
        synthetic_scaling_exponents_df[r"$(\beta - \hat{\beta})^2$"] = 0.5 * np.square(
            synthetic_scaling_exponents_df["Fit Scaling Exponent"]
            - synthetic_scaling_exponents_df["Theoretical Scaling Exponent"]
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
        """
        assert n >= c
        if n < k:
            return np.nan
        elif (n - c) < k:
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
) -> pd.Series:
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    largest_fraction_successes = np.max(np.divide(num_successes, num_samples))
    initial_params = (0.5, 3.5, largest_fraction_successes)
    bounds = [
        (0.01, 100),
        (0.01, 100),
        (
            largest_fraction_successes
            + 1e-8,  # Scale can't be smaller than the largest fraction of successes.
            1.0,
        ),
    ]

    # Fit alpha, beta, scale to the scaled beta binomial
    optimize_result = scipy.optimize.minimize(
        lambda params: compute_beta_binomial_three_parameters_distribution_neg_log_likelihood(
            params,
            num_samples=num_samples,
            num_successes=num_successes,
        ),
        x0=initial_params,
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(
            # maxiter=5000,
            maxiter=10,
            maxls=100,
            gtol=1e-6,  # Gradient tolerance, adjust as needed),
        ),
    )

    result = pd.Series(
        {
            "alpha": optimize_result.x[0],
            "beta": optimize_result.x[1],
            "loc": 0.0,
            "scale": optimize_result.x[2],
            "neg_log_likelihood": optimize_result.fun,
            "aic": 2 * len(initial_params) + 2 * optimize_result.fun,
            "bic": len(initial_params) * np.log(len(num_samples_and_num_successes_df))
            + 2 * optimize_result.fun,
        }
    )
    print(result)

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


def fit_pass_at_1_beta_distribution_parameters(
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


def fit_pass_at_1_kumaraswamy_distribution_parameters(
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


def fit_pass_at_1_log_normal_distribution_parameters(
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
            params, data=data, bins=bins, distribution="log_normal"
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


def fit_pass_at_1_log_uniform_distribution_parameters(
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
            params, data=data, bins=bins, distribution="log_uniform"
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
        y = group_df[target_col]

        # Exclude any np.inf or np.nan values
        mask = np.isfinite(x) & np.isfinite(y)
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


def sample_synthetic_individual_outcomes_per_problem(
    num_problems: int,
    num_samples_per_problem: int,
    distribution: str,
    distribution_parameters: Dict[str, float],
) -> np.ndarray:
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
        raise NotImplementedError
    else:
        raise NotImplementedError

    return individual_outcomes
