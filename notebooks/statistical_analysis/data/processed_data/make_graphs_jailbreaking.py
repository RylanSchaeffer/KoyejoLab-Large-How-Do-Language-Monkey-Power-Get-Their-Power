import sys
import os
import src.analyze as analyze 
import src.stats_utils as stats_utils
import src.mixtures as mixtures
import src.better_optimiation as bopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import src.EM as EM
from src.EM import compute_estimate, compute_estimate_stable, compute_p_at_ks, compute_estimates_better_mixture
import src.better_em as BEM
import heapq
import src.bem_geometric as bemg
from src.bem_geometric import compute_estimates_better_three_param_geometric, compute_estimates_three_param
import random
from multiprocessing import Pool, cpu_count
import functools

def compute_error(pass_at_ks, estimates):
    return np.mean((pass_at_ks - estimates)**2)    

def make_single_graph_and_get_loss(model_name, n, ks, data, individual_data, shuffle=True):
    figure_name = f'notebooks/statistical_analysis/data/processed_data/graphs_shuffled/{model_name}_jailbreaking_{n}.svg'

    samples = n
    
    #define the ks that we will try to predict

    
    #label the number of total samples and compute the number of correct attempts for each problem
    data['Num. Samples Total'] = data['Scaling Parameter'].max()
    data['Num. Samples Correct'] = data['Score']*data['Num. Samples Total']
    data = data[data['Scaling Parameter'] == 1]
    pythia12_math = data[(data['Model'] == model_name)]
    model_ind = individual_data[(individual_data['Model'] == model_name)]
    # If you want to shuffle within each Problem Idx group
    if shuffle:
        def shuffle_group(group):
            group = group.copy()
            group['Score'] = np.random.permutation(group['Score'])
            return group

        model_ind = model_ind.groupby('Problem Idx').apply(shuffle_group).reset_index(drop=True)
    model_ind = model_ind[model_ind['Attempt Idx'] <= samples]

    model_ind['Num. Samples Correct'] = model_ind.groupby('Problem Idx')['Score'].transform('sum')
    model_ind['Num. Samples Total'] = samples
    smaller_pythia12_math = model_ind
    smaller_beta_3_discretized_params = analyze.fit_discretized_beta_three_parameters_to_num_samples_and_num_successes(smaller_pythia12_math)
    #original estimator
    ks_fit = np.array([i for i in range(1, samples)])
    pass_at_ks = analyze.compute_pass_at_k_from_num_samples_and_num_successes_df(smaller_pythia12_math, ks_fit)
    pass_at_ks = pass_at_ks.groupby('Scaling Parameter')['Score'].mean()
    model = LinearRegression(fit_intercept=True)
    model.fit(np.log(ks_fit).reshape(-1,1), -np.log(pass_at_ks))

    individual_data_model = individual_data[(individual_data['Model'] == model_name)]

    heap = []
    heapq.heapify(heap)
    budget = samples*len(pythia12_math)
    results = []
    for ele in individual_data_model['Problem Idx'].unique():
        heapq.heappush(heap, (0, ele))
    total_samples = 0
    while total_samples < budget:
        total_samples += 1
        attempts, index = heapq.heappop(heap)
        attempt_index = attempts + 1
        
        # Check if this attempt exists
        filtered_data = individual_data_model[
            (individual_data_model['Problem Idx'] == index) & 
            (individual_data_model['Attempt Idx'] == attempt_index)
        ]
        
        if filtered_data.empty:
            continue  # Skip if no data for this attempt
            
        score = filtered_data['Score'].iloc[0]
        attempts += 1
        
        if score == 0:
            heapq.heappush(heap, (attempts, index))  # Fixed: use 'index' not 'ele'
        else:
            results.append({'Problem Idx': index, 'Num. Samples Total': attempts, 'Num. Samples Correct': 1})
    while heap:
        attempts, index = heapq.heappop(heap)
        results.append({'Problem Idx': index, 'Num. Samples Total': attempts, 'Num. Samples Correct': 0})
    efficient_data = pd.DataFrame(results)

    #beta 2 geometric
    n_distr = 1
    geom_mix = bemg.beta_geometric_mixture(n_distr=n_distr, num_successes = efficient_data['Num. Samples Correct'], num_trials = efficient_data['Num. Samples Total'])
    geom_params = geom_mix.fit_mixture()
    #beta 3 geometric
    # smaller_beta_3_params_geometric_stable = bopt.fit_beta_binomial_three_parameters_stable(efficient_data)
    # print(ks)
    pass_at_ks = compute_p_at_ks(pythia12_math, ks)
    # print(len(pass_at_ks))

    #openai regression predictions
    X = ks.reshape(-1,1)
    regression_predictions = np.exp(-model.predict(np.log(X)))
    #beta discretized estimates
    beta_estimates = [compute_estimate(smaller_beta_3_discretized_params, k) for k in ks] 
    #2-param binomial mixture
    # mixture_estimates = [compute_estimates_better_mixture(smaller_pythia12_math, beta_mixture_params, k, n_distr) for k in ks]
    #3-param binomial 
    # beta_3_stable_estimates_better = [compute_estimates_three_param(smaller_pythia12_math, smaller_beta_3_params_stable, k) for k in ks] 
    #3-param geometric
    # three_param_geom_estimates = [bemg.compute_estimates_better_three_param_geometric(efficient_data, smaller_beta_3_params_geometric_stable, k) for k in ks]
    #2-param geometric
    geom_correct_estimates = [bemg.compute_estimates_better_mixture_geometric(efficient_data, geom_params, k, n_distr) for k in ks]

    plt.plot(ks, beta_estimates, label = 'Discretized Beta')
    # plt.plot(ks, mixture_estimates, label = 'Beta-Binomial')
    plt.plot(ks, np.clip(regression_predictions, 0, 1), label = "Regression (clipped at 1)")
    plt.plot(ks, np.clip(geom_correct_estimates, 0, 1), label = "Beta w Dynamic Sampling (Ours)", linewidth=4, color='red', linestyle='-', markeredgecolor='darkred')
    # plt.plot(ks, beta_3_stable_estimates_better, label = 'Scaled Beta-Binomial')
    # plt.plot(ks, three_param_geom_estimates, label = 'Scaled Beta w Dynamic Sampling')
    plt.plot(ks, pass_at_ks, label = 'Pass@k Estimate w 10k Samples', linewidth=4, color='black', linestyle='dashed')
    plt.title(f'Estimates of Pass@k for {model_name}')
    plt.ylabel('Pass@k')
    plt.xlabel('log(k)')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(figure_name, bbox_inches='tight')
    plt.clf() 
    discretized_loss = compute_error(beta_estimates, pass_at_ks)
    our_method_loss = compute_error(geom_correct_estimates, pass_at_ks)
    regression_loss = compute_error(np.clip(regression_predictions, 0, 1), pass_at_ks)
    discretized_losses = beta_estimates - pass_at_ks
    our_method_losses = geom_correct_estimates - pass_at_ks
    regression_losses = np.clip(regression_predictions, 0, 1) - pass_at_ks

    return discretized_loss, our_method_loss, regression_loss, discretized_losses, our_method_losses, regression_losses

def process_single_experiment(args):
    """Process a single combination of trial, model, and sample"""
    trial, model, sample, ks, data, individual_data = args
    
    # Set random seed for reproducibility based on trial number
    np.random.seed(trial + hash(model + str(sample)) % 1000)
    random.seed(trial + hash(model + str(sample)) % 1000)
    
    try:
        discretized_loss, our_method_loss, regression_loss, discretized_losses, our_method_losses, regression_losses = make_single_graph_and_get_loss(
            model, sample, ks, data, individual_data
        )
        
        # Format results as lines for CSV
        lines = []
        
        line = f'discretized,{model},{sample},{discretized_loss}'
        for loss in discretized_losses:
            line += f',{loss}'
        lines.append(line)
        
        line = f'our_method,{model},{sample},{our_method_loss}'
        for loss in our_method_losses:
            line += f',{loss}'
        lines.append(line)
        
        line = f'regression,{model},{sample},{regression_loss}'
        for loss in regression_losses:
            line += f',{loss}'
        lines.append(line)
        
        return lines
        
    except Exception as e:
        print(f"Error processing trial {trial}, model {model}, sample {sample}: {e}")
        return []

def main():
    # Get data for the number of math problems solved
    data = analyze.create_or_load_bon_jailbreaking_text_pass_at_k_df()
    
    # This tells us whether each attempt was a success or failure
    individual_data = analyze.create_or_load_bon_jailbreaking_text_individual_outcomes_df()
    
    ks = np.concatenate([np.arange(1, 10), np.array(np.logspace(np.log10(10), np.log10(10000), num=50)).astype(int)])
    models = ['Claude 3.5 Opus', 'Claude 3.5 Sonnet', 'GPT4o', 'GPT4o Mini', 'Gemini 1.5 Flash', 'Gemini 1.5 Pro', 'Llama 3 8B IT']
    samples = [i for i in range(5, 101, 5)]
    trials = 10
    
    # Create header line
    header_line = 'method,model,per_problem_budget,squared_error'
    for k in ks:
        header_line += f',{k}'
    header_line += '\n'
    
    # Write header to file
    with open('notebooks/statistical_analysis/data/processed_data/jailbreaking_shuffled.csv', 'w') as f:
        f.write(header_line)
    
    # Create argument list for all combinations
    args_list = []
    for trial in range(trials):
        for model in models:
            for sample in samples:
                args_list.append((trial, model, sample, ks, data, individual_data))


    
    print(f"Total experiments to run: {len(args_list)}")
    print(f"Using {cpu_count()} CPU cores")
    
    # Process experiments in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_single_experiment, args_list)
    
    # Write all results to file
    with open('notebooks/statistical_analysis/data/processed_data/jailbreaking_shuffled.csv', 'a') as f:
        for result_lines in results:
            for line in result_lines:
                f.write(line + '\n')
    
    print("All experiments completed!")

if __name__ == '__main__':
    main()