import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import Any, Dict
import pprint

import src.globals
import src.analyze


def fit_and_score_estimators():
    run = wandb.init(
        project="monkey-power-law-estimators",
        config=src.globals.FIT_AND_SCORE_ESTIMATORS_DEFAULT_CONFIG,
        entity=wandb.api.default_entity,
    )

    config: Dict[str, Any] = dict(wandb.config)
    pprint.pprint(config)

    np.random.seed(config["seed"])

    # Load the individual outcomes per problem.
    if config["dataset_name"] == "bon_jailbreaking_text_scaling":
        assert "Model" in config["dataset_kwargs"]
        assert "Modality" in config["dataset_kwargs"]
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
        # Subset to only this dataset.
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
        raise NotImplementedError
    else:
        raise NotImplementedError

    cv_power_law_parameter_estimates_df = src.analyze.cross_validate_power_law_coefficient_estimators_from_individual_outcomes(
        individual_outcomes_per_problem_df=individual_outcomes_per_problem_df,
        num_problems_list=[config["num_problems"]],
        num_samples_per_problem_list=[config["num_samples_per_problem"]],
        repeat_indices_list=[config["seed"]],
    )

    # Log cross validated power law parameter estimates.
    data_to_log = {
        "Full Data Least Squares Power Law Exponent": float(
            cv_power_law_parameter_estimates_df[
                "Full Data Least Squares Power Law Exponent"
            ].values[0]
        ),
        "Full Data Least Squares Power Law Prefactor": float(
            cv_power_law_parameter_estimates_df[
                "Full Data Least Squares Power Law Prefactor"
            ].values[0]
        ),
    }
    for row_idx, row in cv_power_law_parameter_estimates_df.iterrows():
        fit_method = row["Fit Method"]
        data_to_log[f"{fit_method} Power Law Prefactor"] = float(
            row["Fit Power Law Prefactor"]
        )
        data_to_log[f"{fit_method} Power Law Exponent"] = float(
            row["Fit Power Law Exponent"]
        )
        data_to_log[f"{fit_method} Full Data Least Squares Squared Error"] = float(
            row["Full Data Least Squares Squared Error"]
        )
        data_to_log[f"{fit_method} Full Data Least Squares Relative Error"] = float(
            row["Full Data Least Squares Relative Error"]
        )
    wandb.log(data_to_log, step=1)

    # Plot the data and the fits.
    plt.close()
    plt.figure(figsize=(12, 10))
    ks = np.unique(np.logspace(0, 4))
    # plt.plot(
    #     subset_avg_pass_at_k_df["Scaling Parameter"],
    #     subset_avg_pass_at_k_df["Neg Log Score"],
    #     label="Real",
    # )
    plt.plot(
        ks,
        data_to_log["Full Data Least Squares Power Law Prefactor"]
        * np.power(
            ks,
            -data_to_log["Full Data Least Squares Power Law Exponent"],
        ),
        label="Lst Sqr (All) (a={:0.3f}, b={:0.3f})".format(
            data_to_log["Full Data Least Squares Power Law Prefactor"],
            data_to_log["Full Data Least Squares Power Law Exponent"],
        ),
    )
    plt.plot(
        ks,
        data_to_log["Least Squares Power Law Prefactor"]
        * np.power(
            ks,
            -data_to_log["Least Squares Power Law Exponent"],
        ),
        label="Lst Sqr (a={:0.3f}, b={:0.3f})".format(
            data_to_log["Least Squares Power Law Prefactor"],
            data_to_log["Least Squares Power Law Exponent"],
        ),
    )
    plt.plot(
        ks,
        data_to_log["Beta-Binomial Power Law Prefactor"]
        * np.power(
            ks,
            -data_to_log["Beta-Binomial Power Law Exponent"],
        ),
        label="BetaBin (a={:0.3f}, b={:0.3f})".format(
            data_to_log["Beta-Binomial Power Law Prefactor"],
            data_to_log["Beta-Binomial Power Law Exponent"],
        ),
    )
    plt.plot(
        ks,
        data_to_log["Discretized Beta Power Law Prefactor"]
        * np.power(
            ks,
            -data_to_log["Discretized Beta Power Law Exponent"],
        ),
        label="DiscrBeta (a={:0.3f}, b={:0.3f})".format(
            data_to_log["Discretized Beta Power Law Prefactor"],
            data_to_log["Discretized Beta Power Law Exponent"],
        ),
    )
    plt.plot(
        ks,
        data_to_log["Kumaraswamy-Binomial Power Law Prefactor"]
        * np.power(
            ks,
            -data_to_log["Kumaraswamy-Binomial Power Law Exponent"],
        ),
        label="KumarBin (a={:0.3f}, b={:0.3f})".format(
            data_to_log["Kumaraswamy-Binomial Power Law Prefactor"],
            data_to_log["Kumaraswamy-Binomial Power Law Exponent"],
        ),
    )
    plt.plot(
        ks,
        data_to_log["Discretized Kumaraswamy Power Law Prefactor"]
        * np.power(
            ks,
            -data_to_log["Discretized Kumaraswamy Power Law Exponent"],
        ),
        label="DiscrKumar (a={:0.3f}, b={:0.3f})".format(
            data_to_log["Discretized Kumaraswamy Power Law Prefactor"],
            data_to_log["Discretized Kumaraswamy Power Law Exponent"],
        ),
    )
    plt.axvline(config["num_samples_per_problem"], linestyle="--", color="k")
    plt.xlabel("Num. Attempts per Problem $k$")
    plt.ylabel("$-\log (\operatorname{pass_{\mathcal{D}}@k})$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    wandb_image = wandb.Image(plt.gcf())
    wandb.log({"estimators_y=neg_log_pass_D_at_k_x=k": wandb_image}, step=1)
    # plt.show()
    plt.close()
    wandb.finish()


def fit_proportions_from_observations(
    observations: np.ndarray, num_outcomes: int
) -> np.ndarray:
    """
    Estimate the distribution over O outcomes given 'data'
    where each entry in 'data' is an integer in [0, O-1].
    Returns a vector of length O with empirical frequencies.
    """
    counts = np.bincount(observations, minlength=num_outcomes)
    total = counts.sum()
    assert total > 0, "No data to fit distribution!"
    return counts / total


if __name__ == "__main__":
    fit_and_score_estimators()
    print("Finished discrete distribution simulation!")
