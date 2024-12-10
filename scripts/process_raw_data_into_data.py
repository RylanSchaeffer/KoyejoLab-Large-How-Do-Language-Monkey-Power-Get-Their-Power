import json
import numpy as np
import os
import pandas as pd


raw_data_dir = f"{os.getcwd()}/data/raw_data"
processed_data_dir = f"{os.getcwd()}/data/processed_data"
os.makedirs(processed_data_dir, exist_ok=True)


def process_raw_bon_jailbreaking_data_into_data():
    # Process Best-of-N Jailbreaking.
    bon_jailbreaking_dir = os.path.join(raw_data_dir, "best_of_n_jailbreaking")
    best_of_n_jailbreaking_dfs_list = []
    for jsonl_filename in os.listdir(bon_jailbreaking_dir):
        model_name, modality, temperature, num_samples = jsonl_filename.split("_")
        df = pd.read_json(
            os.path.join(bon_jailbreaking_dir, jsonl_filename), lines=True
        )
        df.rename(
            columns={
                "i": "Problem Idx",
                "n": "Scaling Parameter",
                "flagged": "Score",
            },
            inplace=True,
        )
        df["Model"] = model_name
        df["Modality"] = modality
        df["Temperature"] = temperature
        best_of_n_jailbreaking_dfs_list.append(df)
    best_of_n_jailbreaking_df = pd.concat(best_of_n_jailbreaking_dfs_list)
    best_of_n_jailbreaking_df.to_parquet(
        os.path.join(processed_data_dir, "best_of_n_jailbreaking.parquet"), index=False
    )


def process_raw_many_shot_learning_data_into_data():
    raise NotImplementedError


if __name__ == "__main__":
    process_raw_bon_jailbreaking_data_into_data()
    process_raw_many_shot_learning_data_into_data()
