import os
import multiprocessing
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Function to run trials in parallel and track core usage per process
def run_optimization(num_trials):
    # Dictionary to track core usage in this process
    core_usage = defaultdict(int)

    def objective(trial):
        core_id = multiprocessing.current_process().pid
        core_usage[core_id] += 1
        
        x = trial.suggest_float('x', 0, 10)
        time.sleep(0.1)  # Simulate a heavier workload
        result = x ** 2

        print(f"Trial on core {core_id}, x = {x}, result = {result}")
        return result

    study = optuna.create_study(
        study_name='distributed-example2', 
        storage='sqlite:///example2.db', 
        load_if_exists=True  # Load the study if it already exists
    )
    study.optimize(objective, n_trials=num_trials)

    # Return the study and the core usage for this process
    return study, core_usage

if __name__ == '__main__':
    
    start_time = time.time()
    
    # Split the number of trials across multiple processes
    num_processes = 5
    num_trials_per_process = 100 // num_processes  # 100 total trials, split evenly

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Distribute the optimization across multiple processes
        results = pool.map(run_optimization, [num_trials_per_process] * num_processes)

    # Unpack the studies and core_usage from each process
    studies = [result[0] for result in results]
    core_usages = [result[1] for result in results]

    # Aggregate core usage across all processes
    aggregated_core_usage = defaultdict(int)
    for usage in core_usages:
        for core_id, count in usage.items():
            aggregated_core_usage[core_id] += count

    # Print aggregated core usage statistics
    print("\nAggregated Core usage statistics:")
    for core, count in aggregated_core_usage.items():
        print(f"Core {core} ran {count} trials")

    # Combine results of all trials from each study
    all_trials = []
    for study in studies:
        all_trials.extend(study.trials)

    # Plot optimization history (combining trials)
    plt.figure(figsize=(10, 6))
    ax = optuna.visualization.matplotlib.plot_optimization_history(studies[0])
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
    
    # End timer and print the total runtime
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")