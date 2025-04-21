#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:41:53 2024

@author: attari.v
"""

import optuna
from optuna.storages import RDBStorage

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import warnings

import multiprocessing
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import time


# Set PyTorch device to CPU
device = torch.device('cpu')

warnings.filterwarnings("ignore")

# Import the function from data_preprocessing.py
from data_preprocessing import process_and_split_data


# Function to run trials in parallel and track core usage per process
def run_optimization(study_name, num_trials, X_train, X_test, y_train, y_test):
    # Dictionary to track core usage in this process
    core_usage = defaultdict(int)
    
    # Initialize progress bar for the current core
    core_id = multiprocessing.current_process().pid
    progress_bar = tqdm(total=num_trials, desc=f"Trials on core {core_id}", position=core_id)

    # Function to create the TabNet model
    def create_model(trial):
        """
        Function to create and return a TabNetRegressor model with specified parameters.
        """
        n_d = trial.suggest_int('num_feature_dim (n_d)', 12, 96, step=4)
        n_a = trial.suggest_int('num_feature_att (n_a)', 8, 48, step=4)
        n_steps = trial.suggest_int('num_dec_steps (n_steps)', 3, 7)
        gamma = trial.suggest_float('strength_feature_penalty (gamma)', 0.8, 1.5)
        lambda_sparse = trial.suggest_loguniform('sparsity reg (lambda_sparse)', 1e-4, 1e-2)
        learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 9e-3)
        # Optimizer selection
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
        
        # Set the optimizer function based on the selected optimizer
        if optimizer_name == 'Adam':
            optimizer_fn = torch.optim.Adam
        elif optimizer_name == 'SGD':
            optimizer_fn = torch.optim.SGD
        elif optimizer_name == 'RMSprop':
            optimizer_fn = torch.optim.RMSprop
        
        tabnet_model = TabNetRegressor(
            optimizer_fn=optimizer_fn,
            optimizer_params=dict(lr=learning_rate),
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse
        )
        
        return tabnet_model

    # Define the objective function for Optuna
    def objective(trial):
        
        core_id = multiprocessing.current_process().pid
        core_usage[core_id] += 1
        
        # Create TabNet model
        model = create_model(trial)
    
        # Set hyperparameters
        batch_size = trial.suggest_int('batch_size', 32, 128, step=8)
        virtual_batch_size = trial.suggest_int('virtual_batch_size', 32, 128, step=8)
        epochs = trial.suggest_int('epochs', 10, 50, step=20)
            
        # Fit the model using TabNet's built-in method
        history = model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=epochs,
            patience=20,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False
        )
        
        # Get the best training loss
        best_train_loss = min(model.history['loss'])
    
        # Predict on test data (unseen data)
        y_pred = model.predict(X_test)
    
        # Calculate test loss (MSE)
        test_loss = mean_squared_error(y_test, y_pred)
    
        # Calculate relative performance deterioration
        if best_train_loss > 0:  # Avoid division by zero
            relative_deterioration = (best_train_loss - test_loss) / best_train_loss
        else:
            relative_deterioration = 0
        
        # Return the test loss as the objective, but store the relative deterioration
        trial.set_user_attr('relative_deterioration', relative_deterioration)

        return test_loss
    
    # Load the study (don't create new tables in the worker processes)
    storage = RDBStorage(
        url="sqlite:///distributed-TabNet-Optuna.db"
    )
    study = optuna.load_study(
        study_name=study_name, 
        storage=storage
    )
    
    study.optimize(objective, n_trials=num_trials)
    
    # Close the progress bar once the optimization is finished
    progress_bar.close()

    # Return the study and the core usage for this process
    return study, core_usage

# Load and split the data only once
def load_and_split_data():
    df = pd.read_csv('../../input_data/v2/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv')
    input_columns = ['Nb', 'Cr', 'V', 'W', 'Zr']
    output_columns = ['Kou Criteria']
    X_train, X_test, y_train, y_test = process_and_split_data(
        df, 
        input_columns, 
        output_columns, 
        threshold=1e-9, 
        apply_sc=True, scaling_method='minmax', 
        apply_qt=True, qt_method='uniform', 
        apply_log1p=False, 
        apply_sqrt=False, 
        test_size=0.10, 
        random_state=42
    )
    return X_train, X_test, y_train, y_test

# Save the study results
def save_study_results(study, filename='optuna_trials.csv'):
    df = study.trials_dataframe()
    df.to_csv(filename, index=False)
    print(f"Trials saved to {filename}")
    
    return df

if __name__ == "__main__":
    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data()

    start_time = time.time()
    
    # Initialize database and create table once before multiprocessing
    storage = RDBStorage(
        url="sqlite:///distributed-TabNet-Optuna.db"
    )
    
    study_name = 'distributed-TabNet-Optuna'
    
    try:
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage,
            load_if_exists=True  # Create or load the study
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
    
    # Split the number of trials across multiple processes
    total_trials  = 20
    num_processes = 5
    num_trials_per_process = total_trials // num_processes  # Adjust this based on desired total trials

    # Pass data to each process for multiprocessing
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(run_optimization, [(study_name, num_trials_per_process, X_train, X_test, y_train, y_test)] * num_processes)

    # Unpack the studies and core_usage from each process
    studies = [result[0] for result in results]
    core_usages = [result[1] for result in results]

    aggregated_core_usage = defaultdict(int)
    for usage in core_usages:
        for core_id, count in usage.items():
            aggregated_core_usage[core_id] += count
    
    # %% Track and calculate the average relative deterioration.
    
    # Calculate average relative performance deterioration
    relative_deteriorations = []
    for trial in studies[0].trials:
        if 'relative_deterioration' in trial.user_attrs:
            relative_deteriorations.append(trial.user_attrs['relative_deterioration'])
    
    # Calculate the average relative deterioration
    if relative_deteriorations:
        avg_relative_deterioration = sum(relative_deteriorations) / len(relative_deteriorations)
        print(f"Average Relative Performance Deterioration: {avg_relative_deterioration:.4f}")
    else:
        print("No relative deterioration data found.")

    # %%
    print("\nAggregated Core usage statistics:")
    for core, count in aggregated_core_usage.items():
        print(f"Core {core} ran {count} trials")

    print("Best parameters: ", studies[0].best_params)
    print("Best validation loss: ", studies[0].best_value)
    
    
    # %%
        
    df = save_study_results(studies[0])
    
    # %% Plots
    
    # 
    #plt.figure(figsize=(10, 6))
    #ax = optuna.visualization.plot_hypervolume_history(studies[0])
    #plt.savefig('optimization_hypervolume_history-FDN.jpg')
    #plt.show()
    
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
    ax.set_yscale('log')
    plt.savefig('optimization_history-TabNet.jpg',dpi=300)
    plt.show()

    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    ax.set_facecolor('none')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    plt.savefig('hyperparameter_importance-TabNet.jpg')
    plt.show()  
    
    plt.figure(figsize=(10, 6))
    ax = optuna.visualization.matplotlib.plot_contour(study, params=["batch_size", "learning_rate"])
    plt.tight_layout()
    plt.savefig('optimization_contour_XY-TabNet.jpg')
    plt.show()
    
    # only supports 2 or 3 objective studies
    # plt.figure(figsize=(10, 6))
    # ax = optuna.visualization.plot_pareto_front(studies[0])
    # plt.tight_layout()
    # plt.savefig('optimization_pareto_front-FDN.jpg')
    # plt.show()
    
    # %%
    
    # Define the list of parameters you want to visualize in the contour plot
    params = ['num_feature_dim (n_d)',
              'num_feature_att (n_a)',
              'num_dec_steps (n_steps)',
              'strength_feature_penalty (gamma)',
              'sparsity reg (lambda_sparse)',
              'learning_rate',
              'batch_size',
              'virtual_batch_size',
              'epochs']
    
    
    # Generate the contour plot (returns an Axes object)
    # Generate the contour plot
    axes = optuna.visualization.matplotlib.plot_contour(study, params=params)
    
    # Access the figure object and set the figure size
    fig = plt.gcf()
    fig.set_size_inches(28, 24)  # Set the desired figure size
    
    # Remove the title
    fig.suptitle("")  # Set to empty string to remove the title
    
      # Customize each subplot in the axes array
    for ax in axes.flatten():
        # Customize the spines (axes box) to be black
        for spine in ax.spines.values():
            spine.set_color('black')
    
        # Customize ticks and labels to black
        ax.tick_params(axis='both', colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
    
        # Remove grid lines inside the plot
        ax.grid(False)
    
    # Save the customized contour plot to a file
    plt.savefig('customized_contour_plot_DNF.jpg')
    
    # Show the plot
    plt.show()    
    # %%
    
    # End timer
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")