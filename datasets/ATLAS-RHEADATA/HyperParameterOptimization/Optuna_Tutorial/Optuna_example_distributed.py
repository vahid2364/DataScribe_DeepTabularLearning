#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:39:12 2024

@author: attari.v
"""

import os
import multiprocessing
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
#from optuna.visualization import plot_parallel_coordinate

# Dictionary to track core usage
core_usage = defaultdict(int)

def objective(trial):
    # Get the core/process ID
    core_id = multiprocessing.current_process().pid
    
    # Increment the usage count for this core
    core_usage[core_id] += 1
    
    # Example of an optimization process (replace with your logic)
    x = trial.suggest_float('x', 0, 10)
    result = x ** 2

    print(f"Trial on core {core_id}, x = {x}, result = {result}")
    
    return result

if __name__ == '__main__':
    
    #multiprocessing.set_start_method('fork')  # Try 'fork' for multiprocessing on macOS
    
    # Create or load the study
    try:
        # Attempt to load the study
        study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
    except KeyError:
        # If the study doesn't exist, create a new one
        study = optuna.create_study(study_name='distributed-example', storage='sqlite:///example.db')
    
    #optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study.optimize(objective, n_trials=100, n_jobs=3)
    
    # %%
    
    # Print the core usage statistics
    print("\nCore usage statistics:")
    for core, count in core_usage.items():
        print(f"Core {core} ran {count} trials")
        
    # %%
    
    # Extract trials and parameter values
    trials = study.trials
    params = [trial.params for trial in trials]
    
    # Convert to DataFrame for easier manipulation and plotting
    df_params = pd.DataFrame(params)
    
    print('****************')
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)

    # Plot optimization history
    plt.figure(figsize=(10, 6))
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax.set_facecolor('none')
    for line in ax.get_lines():
        line.set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    plt.savefig('optimization_history-FDN.jpg')
    plt.show()
    
    # %%
    
    best_trial = min(study.trials, key=lambda t: t.value)
    
    # Optimum value of the parameter 'x' for the best trial
    optimum_x = best_trial.params['x']
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df_params['x'], bins=80, color='blue', edgecolor='black')
    plt.axvline(optimum_x, color='red', linestyle='--', linewidth=2, label=f'Optimum x: {optimum_x:.2f}')  # Optimum point
    
    # Add labels and title
    plt.xlabel('Parameter x')
    plt.ylabel('Frequency')
    plt.title('Distribution of Parameter x')
    
    # Add a legend for the optimum line
    plt.legend()
    
