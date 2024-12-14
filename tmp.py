import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def main():
    """
    Plot Beta distribution PDFs for various alpha, beta parameters

    Args:
        param_sets: List of dictionaries with 'alpha', 'beta' values
        num_points: Number of points to plot for each distribution
    """

    # Define sets of parameters to plot
    param_sets = [
        # {"alpha": 1e-5, "beta": 1},  # U-shaped
        # {"alpha": 1e-3, "beta": 1},  # U-shaped
        # {"alpha": 1e-1, "beta": 1},  # U-shaped
        {"alpha": 1, "beta": 1},  # Uniform
        {"alpha": 1, "beta": 3},  # Uniform
        {"alpha": 1, "beta": 10},  # Uniform
    ]

    # Create figure
    plt.close()
    plt.figure(figsize=(12, 6))

    # Create x values from 0 to 1
    x = np.logspace(-20, 0, 100)

    # Plot each distribution
    for params in param_sets:
        alpha = params["alpha"]
        beta = params["beta"]

        # Calculate PDF values
        pdf = stats.beta.pdf(x, alpha, beta)

        # Plot with label showing parameters
        label = f"α={alpha}, β={beta}"
        plt.plot(x, pdf, label=label)

    # Customize plot
    plt.xlabel("x")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Probability Density")
    plt.title("Beta Distribution PDFs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print()


if __name__ == "__main__":
    main()
