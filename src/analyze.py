from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import itertools
import numpy as np
import os
import pandas as pd
import pyarrow
from typing import List, Optional, Union

import src.globals


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
            df["Model"] = src.globals.MODELS_TO_NICE_STRINGS[model_name]
            df["Modality"] = src.globals.MODALITY_TO_NICE_STRINGS[modality]
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
                    "Attempt Idx": ["mean"],
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
                "Scoresize": "Num. Samples",
                "Scoresum": "Num. Samples Correct",
            },
            inplace=True,
        )

        models_gsm8k_pass_at_k_dfs_list = []
        for k in src.globals.BON_JAILBREAKING_Ks_LIST:
            models_math_scores_df_copy = bon_jailbreaking_pass_at_k_df.copy()
            models_math_scores_df_copy["Scaling Parameter"] = k
            models_math_scores_df_copy["Score"] = estimate_pass_at_k(
                num_samples=bon_jailbreaking_pass_at_k_df["Num. Samples"],
                num_correct=bon_jailbreaking_pass_at_k_df["Num. Samples Correct"],
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


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    pass_at_k = np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )
    return pass_at_k
