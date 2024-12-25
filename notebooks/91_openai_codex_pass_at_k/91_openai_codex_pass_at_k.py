import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.plot

np.random.seed(0)


def compute_biased_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Empirical (biased) estimator: 1-(1-phat)^k
    where phat is the empirical pass@1 rate (c/n)
    """
    p_hat = c / n
    return 1.0 - np.power(1.0 - p_hat, k)


def compute_unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for sampling without replacement"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# Parameters for the simulation
ns = [
    10.0,
    32,
    100.0,
    316,
    1000.0,
    3162,
    10000.0,
    31623.0,
    100000.0,
]
true_probs = [0.001, 0.01, 0.1]
ks = [
    5,
    10,
    20,
    100,
]
n_trials = 5000  # Number of trials for each sample size

dfs_list = []
for true_prob in true_probs:
    for i, n in enumerate(ns):
        for trial in range(n_trials):
            # Generate random successes based on true probability
            successes = np.random.binomial(n, true_prob)
            biased_pass_at_k = np.zeros(len(ks))
            unbiased_pass_at_k = np.zeros(len(ks))
            for k_idx, k in enumerate(ks):
                biased_pass_at_k[k_idx] = compute_biased_pass_at_k(n, successes, k)
                unbiased_pass_at_k[k_idx] = compute_unbiased_pass_at_k(n, successes, k)
            dfs_list.append(
                pd.DataFrame.from_dict(
                    {
                        "n": [n] * (2 * len(ks)),
                        "true_prob": [true_prob] * (2 * len(ks)),
                        "trial": [trial] * (2 * len(ks)),
                        "k": (2 * ks),
                        "type": [r"$1 - (1-\hat{p})^k$"] * len(ks)
                        + [r"$1 - \binom{n-c}{k} / \binom{n}{k}$"] * len(ks),
                        "pass_at_k": biased_pass_at_k.tolist()
                        + unbiased_pass_at_k.tolist(),
                    }
                )
            )

df = pd.concat(dfs_list, ignore_index=True).reset_index(drop=True)
# Exclude any trials where ks >= n.
df = df[df["k"] < df["n"]]


plt.close()
g = sns.relplot(
    data=df,
    kind="line",
    x="n",
    y="pass_at_k",
    hue="k",
    style="type",
    col="true_prob",
    estimator="mean",
    errorbar="ci",
    palette="tab10",
    facet_kws={"sharey": False},
)
g.set(xscale="log")
g.set_titles(col_template=r"True $p$: {col_name}")
plt.show()


print()
