import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stats
import scipy
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
from typing import Tuple, List
from scipy.special import logsumexp

def plot_histo_and_beta(series, beta_params):
    hist = plt.hist(series, bins=30, density = True, alpha =0.6, color = 'skyblue', label = 'P@1 Distribution')
    x = np.linspace(0, 1, 1000)
    beta_pdf = scipy.stats.beta.pdf(x, beta_params['alpha'], beta_params['beta'], loc=beta_params['loc'], scale=beta_params['scale'])
    plt.plot(x, beta_pdf, 'r-', lw=2, label=f'Beta({beta_params['alpha']:.6f}, {beta_params['beta']:.6f})')

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram with Beta Distribution Overlay')
    plt.legend()
    plt.grid(alpha=0.3)


def compute_efron_estimator(data, predicted_samples):
    largest_success_rate = data['Num. Samples Correct'].max()
    total = 0
    for i in range(1,largest_success_rate):
        term = predicted_samples**i * len(data[data['Num. Samples Correct']==i]) * (-1)**(i+1)
        total+= term
    
    total += len(data[data['Num. Samples Correct'] != 0])
    total /= len(data)
    return total


def compute_beta_binomial_mixture_negative_log_likelihood(
    params: np.ndarray,
    num_samples: np.ndarray,
    num_successes: np.ndarray,
    n_components: int,) -> float:
    """
    Compute negative log-likelihood for a mixture of beta-binomial distributions.
    
    Args:
        params: Flattened array of parameters [weights, alphas, betas]
                - weights: mixing weights (n_components-1, last weight is 1-sum(others))
                - alphas: alpha parameters for each component (n_components)
                - betas: beta parameters for each component (n_components)
        num_samples: Array of sample sizes
        num_successes: Array of number of successes
        n_components: Number of mixture components
    
    Returns:
        Negative log-likelihood
    """
    # Parse parameters
    n_weights = n_components - 1  # We fix the last weight as 1 - sum(others)
    weights_raw = params[:n_weights]
    alphas = params[n_weights:n_weights + n_components]
    betas = params[n_weights + n_components:n_weights + 2 * n_components]
    
    # Convert to proper mixing weights (ensure they sum to 1)
    if n_components == 1:
        weights = np.array([1.0])
    else:
        weights = np.zeros(n_components)
        weights[:n_weights] = weights_raw
        weights[-1] = 1.0 - np.sum(weights_raw)
        
        # Ensure all weights are positive
        if np.any(weights <= 0) or np.any(weights >= 1):
            return np.inf
    
    # Compute log probabilities for each component
    log_probs = np.zeros((len(num_samples), n_components))
    
    for i in range(n_components):
        try:
            log_probs[:, i] = scipy.stats.betabinom.logpmf(
                k=num_successes, 
                n=num_samples, 
                a=alphas[i], 
                b=betas[i]
            )
        except:
            return np.inf
    
    # Add log weights
    log_weights = np.log(weights)
    log_probs_weighted = log_probs + log_weights[np.newaxis, :]
    
    # Compute log-likelihood using logsumexp for numerical stability
    log_likelihood = np.mean(logsumexp(log_probs_weighted, axis=1))
    print(log_likelihood)
    return -log_likelihood

def fit_beta_binomial_mixture_to_num_samples_and_num_successes(
    num_samples_and_num_successes_df: pd.DataFrame,
    n_components: int = 2,
    maxiter: int = 5000,
    n_random_starts: int = 5) -> pd.Series:
    """
    Fit a mixture of beta-binomial distributions to the data.
    
    Args:
        num_samples_and_num_successes_df: DataFrame with columns 'Num. Samples Total' and 'Num. Samples Correct'
        n_components: Number of mixture components
        maxiter: Maximum number of optimization iterations
        n_random_starts: Number of random initializations to try
    
    Returns:
        Series with fitted parameters and model statistics
    """
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    
    best_result = None
    best_neg_log_likelihood = np.inf
    
    # Try multiple random initializations
    for start_idx in range(n_random_starts):
        # Initialize parameters
        np.random.seed(start_idx)  # For reproducibility
        
        if n_components == 1:
            # Single component case
            initial_weights = np.array([])
            initial_alphas = np.array([0.5])
            initial_betas = np.array([3.5])
        else:
            # Multiple components case
            initial_weights = np.random.dirichlet(np.ones(n_components))  # Remove last weight
            initial_alphas = np.array([0.5 for _ in range(n_components)])
            initial_betas = np.array([3.5 for _ in range(n_components)])
        
        initial_params = np.concatenate([initial_weights, initial_alphas, initial_betas])
        
        # Set up bounds
        bounds = []
        # Bounds for weights (0 < weight < 1)
        for _ in range(n_components - 1):
            bounds.append((0, 1))
        # Bounds for alphas and betas
        for _ in range(2 * n_components):
            bounds.append((0.01, 100))
        
        try:
            # Optimize
            optimize_result = scipy.optimize.minimize(
                lambda params: compute_beta_binomial_mixture_negative_log_likelihood(
                    params=params,
                    num_samples=num_samples,
                    num_successes=num_successes,
                    n_components=n_components,
                ),
                x0=initial_params,
                bounds=bounds,
                method="L-BFGS-B",
                options=dict(
                    maxiter=maxiter,
                    maxls=400,
                    gtol=1e-6,
                    ftol=1e-6,
                ),
            )
            
            if optimize_result.success and optimize_result.fun < best_neg_log_likelihood:
                best_result = optimize_result
                best_neg_log_likelihood = optimize_result.fun
                
        except Exception as e:
            print(f"Optimization failed for start {start_idx}: {e}")
            continue
    
    if best_result is None:
        raise ValueError("All optimization attempts failed")
    
    # Parse the best result
    n_weights = n_components - 1
    params = best_result.x
    
    if n_components == 1:
        weights = np.array([1.0])
        alphas = params[0:1]
        betas = params[1:2]
    else:
        weights_raw = params[:n_weights]
        weights = np.zeros(n_components)
        weights[:n_weights] = weights_raw
        weights[-1] = 1.0 - np.sum(weights_raw)
        alphas = params[n_weights:n_weights + n_components]
        betas = params[n_weights + n_components:n_weights + 2 * n_components]
    
    # Calculate model statistics
    n_params = len(initial_params)
    n_data = len(num_samples_and_num_successes_df)
    
    result_dict = {
        "n_components": n_components,
        "neg_log_likelihood": best_result.fun,
        "aic": 2 * n_params + 2 * best_result.fun,
        "bic": n_params * np.log(n_data) + 2 * best_result.fun,
        "n_parameters": n_params,
    }
    
    # Add component-specific parameters
    for i in range(n_components):
        result_dict[f"weight_{i}"] = weights[i]
        result_dict[f"alpha_{i}"] = alphas[i]
        result_dict[f"beta_{i}"] = betas[i]
    
    # For backwards compatibility, if single component, also provide the original format
    if n_components == 1:
        result_dict.update({
            "alpha": alphas[0],
            "beta": betas[0],
            "loc": 0.0,
            "scale": 1.0,
            "Power Law Exponent": alphas[0],
        })
    
    return pd.Series(result_dict)

def predict_mixture_probabilities(
    num_samples: np.ndarray,
    num_successes: np.ndarray,
    fitted_params: pd.Series,
) -> np.ndarray:
    """
    Predict probabilities for new data using fitted mixture model.
    
    Args:
        num_samples: Array of sample sizes
        num_successes: Array of number of successes
        fitted_params: Series with fitted parameters from fit_beta_binomial_mixture_*
    
    Returns:
        Array of predicted probabilities
    """
    n_components = int(fitted_params["n_components"])
    
    # Extract parameters
    weights = np.array([fitted_params[f"weight_{i}"] for i in range(n_components)])
    alphas = np.array([fitted_params[f"alpha_{i}"] for i in range(n_components)])
    betas = np.array([fitted_params[f"beta_{i}"] for i in range(n_components)])
    
    # Compute probabilities for each component
    probs = np.zeros((len(num_samples), n_components))
    
    for i in range(n_components):
        probs[:, i] = scipy.stats.betabinom.pmf(
            k=num_successes, 
            n=num_samples, 
            a=alphas[i], 
            b=betas[i]
        )
    
    # Weight by mixture weights
    weighted_probs = probs * weights[np.newaxis, :]
    
    return np.sum(weighted_probs, axis=1)

def predict_mixture_probabilities(
    num_samples: np.ndarray,
    num_successes: np.ndarray,
    fitted_params: pd.Series,
) -> np.ndarray:
    """
    Predict probabilities for new data using fitted mixture model.
    
    Args:
        num_samples: Array of sample sizes
        num_successes: Array of number of successes
        fitted_params: Series with fitted parameters from fit_beta_binomial_mixture_*
    
    Returns:
        Array of predicted probabilities
    """
    n_components = int(fitted_params["n_components"])
    
    # Extract parameters
    weights = np.array([fitted_params[f"weight_{i}"] for i in range(n_components)])
    alphas = np.array([fitted_params[f"alpha_{i}"] for i in range(n_components)])
    betas = np.array([fitted_params[f"beta_{i}"] for i in range(n_components)])
    
    # Compute probabilities for each component
    probs = np.zeros((len(num_samples), n_components))
    
    for i in range(n_components):
        probs[:, i] = scipy.stats.betabinom.pmf(
            k=num_successes, 
            n=num_samples, 
            a=alphas[i], 
            b=betas[i]
        )
    
    # Weight by mixture weights
    weighted_probs = probs * weights[np.newaxis, :]
    
    return np.sum(weighted_probs, axis=1)

def plot_mixture_results(
    num_samples_and_num_successes_df: pd.DataFrame,
    fitted_params: pd.Series,
    figsize: Tuple[int, int] = (15, 10),
    bins: int = 50,
    show_components: bool = True,
    alpha: float = 0.7,
) -> None:
    """
    Plot histogram of data with fitted mixture model overlay.
    
    Args:
        num_samples_and_num_successes_df: Original data DataFrame
        fitted_params: Results from fit_beta_binomial_mixture_*
        figsize: Figure size (width, height)
        bins: Number of histogram bins
        show_components: Whether to show individual mixture components
        alpha: Transparency for component plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract data
    num_samples = num_samples_and_num_successes_df["Num. Samples Total"].values
    num_successes = num_samples_and_num_successes_df["Num. Samples Correct"].values
    success_rates = num_successes / num_samples
    
    # Extract fitted parameters
    n_components = int(fitted_params["n_components"])
    weights = np.array([fitted_params[f"weight_{i}"] for i in range(n_components)])
    alphas = np.array([fitted_params[f"alpha_{i}"] for i in range(n_components)])
    betas = np.array([fitted_params[f"beta_{i}"] for i in range(n_components)])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Beta-Binomial Mixture Model Results (n_components={n_components})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Success Rate Histogram with Fitted PDF
    ax1 = axes[0, 0]
    
    # Plot histogram of success rates
    n_hist, bins_hist, _ = ax1.hist(success_rates, bins=bins, density=True, 
                                   alpha=0.6, color='lightblue', edgecolor='black', 
                                   label='Data')
    
    # Create fine-grained x values for smooth curves
    x_fine = np.linspace(0, 1, 1000)
    
    # Plot overall mixture PDF
    mixture_pdf = np.zeros_like(x_fine)
    for i in range(n_components):
        component_pdf = scipy.stats.beta.pdf(x_fine, alphas[i], betas[i])
        mixture_pdf += weights[i] * component_pdf
        
        # Plot individual components if requested
        if show_components and n_components > 1:
            ax1.plot(x_fine, weights[i] * component_pdf, '--', 
                    alpha=alpha, linewidth=2, 
                    label=f'Component {i+1} (w={weights[i]:.3f})')
    
    # Plot overall mixture
    ax1.plot(x_fine, mixture_pdf, 'r-', linewidth=3, label='Mixture PDF')
    
    ax1.set_xlabel('Success Rate')
    ax1.set_ylabel('Density')
    ax1.set_title('Success Rate Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Raw Count Histogram
    ax2 = axes[0, 1]
    
    # Create 2D histogram for num_samples vs num_successes
    scatter = ax2.scatter(num_samples, num_successes, alpha=0.6, c=success_rates, 
                         cmap='viridis', s=30)
    
    # Add diagonal lines for reference success rates
    max_samples = num_samples.max()
    for rate in [0.1, 0.25, 0.5, 0.75, 0.9]:
        x_line = np.linspace(0, max_samples, 100)
        y_line = rate * x_line
        ax2.plot(x_line, y_line, '--', alpha=0.4, color='gray')
        ax2.text(max_samples * 0.8, rate * max_samples * 0.8, f'{rate:.1f}', 
                alpha=0.6, fontsize=8)
    
    plt.colorbar(scatter, ax=ax2, label='Success Rate')
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Number of Successes')
    ax2.set_title('Sample Size vs Success Count')
    ax2.grid(True, alpha=0.3)
    
    # 3. Model Comparison and Statistics
    ax3 = axes[1, 0]
    ax3.axis('off')  # Turn off axis for text display
    
    # Display model statistics
    stats_text = f"""Model Statistics:
    
Number of Components: {n_components}
Negative Log-Likelihood: {fitted_params['neg_log_likelihood']:.4f}
AIC: {fitted_params['aic']:.4f}
BIC: {fitted_params['bic']:.4f}
Number of Parameters: {fitted_params['n_parameters']}

Component Parameters:"""
    
    for i in range(n_components):
        stats_text += f"""
Component {i+1}:
  Weight: {weights[i]:.4f}
  Alpha:  {alphas[i]:.4f}
  Beta:   {betas[i]:.4f}
  Mean:   {alphas[i]/(alphas[i]+betas[i]):.4f}"""
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 4. Q-Q Plot or Residual Analysis
    ax4 = axes[1, 1]
    
    # Calculate predicted probabilities for each data point
    predicted_probs = np.zeros(len(num_samples))
    for i in range(n_components):
        component_probs = scipy.stats.betabinom.pmf(
            k=num_successes, n=num_samples, a=alphas[i], b=betas[i]
        )
        predicted_probs += weights[i] * component_probs
    
    # Plot observed vs predicted (on log scale to handle small probabilities)
    log_predicted = np.log10(predicted_probs + 1e-10)  # Add small value to avoid log(0)
    
    # Calculate empirical probabilities (frequency in bins)
    success_rate_bins = np.linspace(0, 1, 21)  # 20 bins
    bin_centers = (success_rate_bins[:-1] + success_rate_bins[1:]) / 2
    hist_counts, _ = np.histogram(success_rates, bins=success_rate_bins)
    empirical_probs = hist_counts / len(success_rates)
    
    # Calculate mixture probabilities at bin centers
    mixture_probs_at_centers = np.zeros_like(bin_centers)
    for i in range(n_components):
        component_probs_at_centers = scipy.stats.beta.pdf(bin_centers, alphas[i], betas[i])
        mixture_probs_at_centers += weights[i] * component_probs_at_centers
    
    # Normalize mixture probabilities to match empirical scale
    bin_width = success_rate_bins[1] - success_rate_bins[0]
    mixture_probs_normalized = mixture_probs_at_centers * bin_width
    
    # Plot comparison
    ax4.scatter(mixture_probs_normalized, empirical_probs, alpha=0.7, s=50)
    
    # Add diagonal line for perfect fit
    max_prob = max(mixture_probs_normalized.max(), empirical_probs.max())
    ax4.plot([0, max_prob], [0, max_prob], 'r--', alpha=0.7, label='Perfect Fit')
    
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Empirical Probability')
    ax4.set_title('Model Fit: Predicted vs Empirical')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


    