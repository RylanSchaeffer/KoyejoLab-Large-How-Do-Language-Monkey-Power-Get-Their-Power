from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
from functools import partial
import itertools
import numpy as np
import os
import pandas as pd
import scipy.stats
import pyarrow
from typing import Dict, List, Optional, Tuple, Union

import src.globals


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
            pdf = scipy.stats.beta.prob_pass_at_1(x, alpha, beta)
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


def create_or_load_bon_jailbreaking_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    bon_jailbreaking_pass_at_k_df_path = os.path.join(
        processed_data_dir, "bon_jailbreaking_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(bon_jailbreaking_pass_at_k_df_path):
        print("Creating bon_jailbreaking_pass_at_k_df_path anew...")

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

        best_of_n_jailbreaking_df = pd.concat(best_of_n_jailbreaking_dfs_list)
        best_of_n_jailbreaking_df = best_of_n_jailbreaking_df[
            best_of_n_jailbreaking_df["Temperature"] == 1.0
        ]

        bon_jailbreaking_pass_at_k_df = (
            best_of_n_jailbreaking_df.groupby(
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

        models_gsm8k_pass_at_k_dfs_list = []
        for k in src.globals.BON_JAILBREAKING_Ks_LIST:
            models_math_scores_df_copy = bon_jailbreaking_pass_at_k_df.copy()
            models_math_scores_df_copy["Scaling Parameter"] = k
            models_math_scores_df_copy["Score"] = estimate_pass_at_k(
                num_samples_total=bon_jailbreaking_pass_at_k_df[
                    "Num. Samples Total"
                ].values,
                num_samples_correct=bon_jailbreaking_pass_at_k_df[
                    "Num. Samples Correct"
                ].values,
                k=k,
            )
            models_gsm8k_pass_at_k_dfs_list.append(models_math_scores_df_copy)
        bon_jailbreaking_pass_at_k_df = pd.concat(models_gsm8k_pass_at_k_dfs_list)
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
        "Loaded bon_jailbreaking_pass_at_k_df_path with shape: ",
        models_gsm8k_pass_at_k_df.shape,
    )
    return models_gsm8k_pass_at_k_df


def create_or_load_gpt4_gsm8k_prob_answer_given_problem_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    gpt4_gsm8k_neg_log_likelihood_df_path = os.path.join(
        processed_data_dir, "gpt4_gsm8k_neg_log_likelihood.parquet"
    )

    if refresh or not os.path.exists(gpt4_gsm8k_neg_log_likelihood_df_path):
        print("Creating gpt4_gsm8k_neg_log_likelihood_df_path anew...")

        os.makedirs(processed_data_dir, exist_ok=True)
        gpt4_gsm8k_neg_log_likelihood_df = pd.read_csv(
            os.path.join(raw_data_dir, "gpt4_gsm8k", "pythia_gsm8k_log_likelihoods.csv")
        )
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
        gpt4_gsm8k_neg_log_likelihood_df = gpt4_gsm8k_neg_log_likelihood_df[
            gpt4_gsm8k_neg_log_likelihood_df["Model Nickname"].isin(
                model_nicknames_to_keep
            )
        ]

        gpt4_gsm8k_summed_neg_log_likelihood_df = (
            gpt4_gsm8k_neg_log_likelihood_df.groupby(["Model Nickname", "prompt_idx"])[
                "Neg Log Likelihood"
            ]
            .sum()
            .reset_index()
        )

        models_metadata_df = pd.read_csv(
            os.path.join(raw_data_dir, "gpt4_gsm8k", "models_pythia.csv")
        )
        models_metadata_df["Pretraining Compute"] = (
            6.0 * models_metadata_df["Tokens"] * models_metadata_df["Parameters"]
        )

        gpt4_gsm8k_summed_neg_log_likelihood_df = (
            gpt4_gsm8k_summed_neg_log_likelihood_df.merge(
                models_metadata_df[
                    ["Model Nickname", "Model Family", "Pretraining Compute"]
                ],
                how="inner",
                on="Model Nickname",
            )
        )
        gpt4_gsm8k_summed_neg_log_likelihood_df["Score"] = np.exp(
            -gpt4_gsm8k_summed_neg_log_likelihood_df["Neg Log Likelihood"]
        )
        gpt4_gsm8k_summed_neg_log_likelihood_df[
            "Log Score"
        ] = -gpt4_gsm8k_summed_neg_log_likelihood_df["Neg Log Likelihood"]
        gpt4_gsm8k_summed_neg_log_likelihood_df.rename(
            columns={
                "Neg Log Likelihood": "Neg Log Score",
                "Model Nickname": "Model",
                "Pretraining Compute": "Scaling Parameter",
                "prompt_idx": "Problem Idx",
            },
            inplace=True,
        )

        gpt4_gsm8k_summed_neg_log_likelihood_df.to_parquet(
            gpt4_gsm8k_neg_log_likelihood_df_path,
            index=False,
        )
        del gpt4_gsm8k_summed_neg_log_likelihood_df

    gpt4_gsm8k_neg_log_likelihood_df = pd.read_parquet(
        gpt4_gsm8k_neg_log_likelihood_df_path
    )
    print(
        "Loaded gpt4_gsm8k_neg_log_likelihood_df_path with shape: ",
        gpt4_gsm8k_neg_log_likelihood_df.shape,
    )

    return gpt4_gsm8k_neg_log_likelihood_df


def create_or_load_large_language_monkeys_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_pass_at_k_df_path = os.path.join(
        processed_data_dir, "large_language_monkeys_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(large_language_monkeys_pass_at_k_df_path):
        print("Creating large_language_monkeys_pass_at_k_df_path anew...")

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_dir = os.path.join(
            raw_data_dir, "large_language_monkeys"
        )
        large_language_monkeys_dfs_list = []
        parquet_filenames = [
            "gsm8k_pass_at_k.parquet",
            "math_pass_at_k.parquet",
        ]
        for parquet_filename in parquet_filenames:
            benchmark = parquet_filename.split("_")[0]
            df = pd.read_parquet(
                os.path.join(large_language_monkeys_dir, parquet_filename)
            )
            df[
                "Benchmark"
            ] = src.globals.LARGE_LANGUAGE_MONKEYS_BENCHMARKS_TO_NICE_STRINGS[benchmark]
            large_language_monkeys_dfs_list.append(df)
        large_language_monkeys_pass_at_k_df = pd.concat(
            large_language_monkeys_dfs_list,
        )

        # Only keep large & final checkpoints.
        models_to_keep = [
            "Pythia_70M_300B",
            "Pythia_160M_300B",
            "Pythia_410M_300B",
            "Pythia_1B_300B",
            "Pythia_2.8B_300B",
            "Pythia_6.9B_300B",
            "Pythia_12B_300B",
        ]
        large_language_monkeys_pass_at_k_df = large_language_monkeys_pass_at_k_df[
            large_language_monkeys_pass_at_k_df["Model Nickname"].isin(models_to_keep)
        ]
        large_language_monkeys_pass_at_k_df[
            "Model Nickname"
        ] = large_language_monkeys_pass_at_k_df["Model Nickname"].map(
            src.globals.LARGE_LANGUAGE_MONKEYS_PYTHIA_MODELS_TO_NICE_STRINGS
        )

        large_language_monkeys_pass_at_k_df.drop(
            columns=["Inference Compute", "neg_log_pass@k"],
            inplace=True,
        )

        large_language_monkeys_pass_at_k_df.rename(
            columns={
                "pass@k": "Score",
                "k": "Scaling Parameter",
                "prompt_idx": "Problem Idx",
                "Model Nickname": "Model",
            },
            inplace=True,
        )

        large_language_monkeys_pass_at_k_df["Log Score"] = np.log(
            large_language_monkeys_pass_at_k_df["Score"]
        )
        large_language_monkeys_pass_at_k_df[
            "Neg Log Score"
        ] = -large_language_monkeys_pass_at_k_df["Log Score"]
        large_language_monkeys_pass_at_k_df.to_parquet(
            large_language_monkeys_pass_at_k_df_path,
            index=False,
        )

        print(f"Wrote {large_language_monkeys_pass_at_k_df_path} to disk.")
        del large_language_monkeys_pass_at_k_df

    large_language_monkeys_pass_at_k_df = pd.read_parquet(
        large_language_monkeys_pass_at_k_df_path
    )
    print(
        "Loaded large_language_monkeys_pass_at_k_df_path with shape: ",
        large_language_monkeys_pass_at_k_df.shape,
    )
    return large_language_monkeys_pass_at_k_df


def create_or_load_large_language_monkeys_original_pass_at_k_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_original_pass_at_k_df_path = os.path.join(
        processed_data_dir, "large_language_monkeys_original_pass_at_k.parquet"
    )

    if refresh or not os.path.exists(large_language_monkeys_original_pass_at_k_df_path):
        print("Creating large_language_monkeys_original_pass_at_k_df_path anew...")

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
        src.analyze.create_or_load_large_language_monkeys_original_pass_at_k_df(
            refresh=refresh,
        )
    )
    # Keep only pass@1 data on MATH.
    llmonkeys_original_pass_at_1_df = llmonkeys_original_pass_at_k_df[
        (llmonkeys_original_pass_at_k_df["Scaling Parameter"] == 1)
        & (llmonkeys_original_pass_at_k_df["Benchmark"] == "MATH")
        & (llmonkeys_original_pass_at_k_df["Score"] > 1e-5)
    ].copy()

    large_language_monkeys_pass_at_1_beta_fits_df_path = os.path.join(
        processed_data_dir, "llmonkeys_pass_at_1_beta_fits.parquet"
    )
    if refresh or not os.path.exists(
        large_language_monkeys_pass_at_1_beta_fits_df_path
    ):
        print(f"Creating {large_language_monkeys_pass_at_1_beta_fits_df_path} anew...")

        # For each model, fit a beta distribution to the pass@1 data using MLE.
        llmonkeys_pass_at_1_beta_fits_df = (
            llmonkeys_original_pass_at_1_df.groupby(
                ["Model", "Benchmark", "Scaling Parameter"]
            )
            .apply(
                lambda df: pd.Series(
                    scipy.stats.beta.fit(df["Score"].values),
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
            os.path.join(raw_data_dir, "pretraining_scaling")
        ):
            if not parquet_filename.endswith(".parquet"):
                continue
            df = pd.read_parquet(
                os.path.join(raw_data_dir, "pretraining_scaling", parquet_filename)
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
            os.path.join(raw_data_dir, "pretraining_scaling", "models.csv")
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
        if (n - c) < k:
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
