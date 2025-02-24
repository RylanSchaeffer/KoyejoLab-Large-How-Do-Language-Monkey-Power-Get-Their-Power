import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import Any, Dict
import pprint

import src.globals
import src.analyze


def evaluate_sampling_strategy():
    run = wandb.init(
        project="monkey-power-law-sampling-strategies",
        config=src.globals.EVALUATE_SAMPLING_STRATEGY_DEFAULT_CONFIG,
        entity=wandb.api.default_entity,
    )

    config: Dict[str, Any] = dict(wandb.config)
    pprint.pprint(config)

    np.random.seed(config["seed"])

    # Load the individual outcomes per problem.
    if config["dataset_name"] == "bon_jailbreaking_text":
        assert "Model" in config["dataset_kwargs"]
        assert "Modality" in config["dataset_kwargs"]
        assert "Temperature" in config["dataset_kwargs"]
        individual_outcomes_per_problem_df = (
            src.analyze.create_or_load_bon_jailbreaking_text_individual_outcomes_df(
                refresh=False
            )
        )
        # Subset to only this model
        individual_outcomes_per_problem_df = individual_outcomes_per_problem_df[
            individual_outcomes_per_problem_df["Model"]
            == config["dataset_kwargs"]["Model"]
        ]
        # Subset to only this modality.
        individual_outcomes_per_problem_df = individual_outcomes_per_problem_df[
            individual_outcomes_per_problem_df["Modality"]
            == config["dataset_kwargs"]["Modality"]
        ]
    elif config["dataset_name"] == "large_language_monkeys_pythia_math":
        assert "Model" in config["dataset_kwargs"]
        assert "Benchmark" in config["dataset_kwargs"]
        individual_outcomes_per_problem_df = src.analyze.create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
            refresh=False
        )
        individual_outcomes_per_problem_df = individual_outcomes_per_problem_df[
            individual_outcomes_per_problem_df["Model"]
            == config["dataset_kwargs"]["Model"]
        ]
        individual_outcomes_per_problem_df = individual_outcomes_per_problem_df[
            individual_outcomes_per_problem_df["Benchmark"]
            == config["dataset_kwargs"]["Benchmark"]
        ]
    elif config["dataset_name"] == "synthetic":
        individual_outcomes_per_problem_df = (
            src.analyze.sample_synthetic_individual_outcomes_per_problem_df(
                num_problems=1_000,
                num_samples_per_problem=100_000,
                distribution=config["dataset_kwargs"]["distribution"],
                distribution_parameters={
                    "a": config["dataset_kwargs"]["a"],
                    "b": config["dataset_kwargs"]["b"],
                    "loc": 0.0,
                    "scale": config["dataset_kwargs"]["scale"],
                },
            )
        )
    else:
        raise NotImplementedError

    sampling_strategy_results: Dict[
        str, float
    ] = src.analyze.evaluate_sampling_strategies_for_power_law_estimators_from_individual_outcomes(
        individual_outcomes_per_problem_df=individual_outcomes_per_problem_df,
        sampling_strategy=config["sampling_strategy"],
        sampling_strategy_kwargs=config["sampling_strategy_kwargs"],
        num_problems_to_sample_from=config["num_problems_to_sample_from"],
        total_samples_budget=config["total_samples_budget"],
    )

    wandb.log(sampling_strategy_results, step=1)
    wandb.finish()


if __name__ == "__main__":
    evaluate_sampling_strategy()
    print("Finished fit_and_score_estimators!")
