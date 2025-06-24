#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:49:57 2024

@author: attari.v
"""

import optuna
from optuna.storages import RDBStorage
#import psycopg2

import multiprocessing
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
#import tensorflow as tf
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input
#import xgboost as xgb
 
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from collections import defaultdict
import time

# Import the function from data_preprocessing.py (ensure this is correct)
from data_preprocessing import process_and_split_data

# Import the functions from xgboost_model.py
from xgboost_model import xgboost_model  # Import the function from the file where it's defined

# Function to run trials in parallel and track core usage per process
def run_optimization(study_name, num_trials, X_train, X_test, y_train, y_test):
    # Dictionary to track core usage in this process
    core_usage = defaultdict(int)
    
    # Initialize progress bar for the current core
    core_id = multiprocessing.current_process().pid
    progress_bar = tqdm(total=num_trials, desc=f"Trials on core {core_id}", position=core_id)

    # Function to create and optimize the XGBoost model using Optuna
    def create_XGBoost_model(trial):
        """
        Function to create and return an XGBoost model with parameters specified via Optuna trial.
        """
    
        # Suggesting hyperparameters using Optuna's trial
        n_estimators = trial.suggest_int('n_estimators', 10, 50, step=20)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)  # Updated to use suggest_float as per the warning
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        
        # Create the XGBoost model using the provided function and hyperparameters from Optuna
        model = xgboost_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="rmse"  # Define the evaluation metric in the constructor
        )
    
        # Train the model (Assuming X_train, y_train, X_val, y_val are already defined)
        # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
        return model
    
    # Define the objective function for Optuna
    def objective(trial):
        
        core_id = multiprocessing.current_process().pid
        core_usage[core_id] += 1
        
        # Create the model
        model = create_XGBoost_model(trial)
                            
        # Prepare evaluation sets to monitor training and test loss
        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        # Fit the model with evaluation metric
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,  # Use early stopping after 10 rounds of no improvement
            verbose=True
        )
        
        # Get the best iteration (corresponding to the best score on validation set)
        best_iteration = model.best_iteration
        best_train_loss = model.evals_result()['validation_0']['rmse'][best_iteration]
    
        # Predict on test data (unseen data)
        y_pred = model.predict(X_test)
        
        # Reshape predictions if necessary
        y_pred = y_pred.reshape(-1, 1)
                
        # Calculate test loss (MSE)
        test_loss = mean_squared_error(y_test, y_pred)
        test_r2   = r2_score(y_test, y_pred)
    
        # Calculate relative performance deterioration
        if best_train_loss > 0:  # Avoid division by zero
            relative_deterioration = (best_train_loss - test_loss) / best_train_loss
        else:
            relative_deterioration = 0
        
        # Return the test loss as the objective, but store the relative deterioration
        trial.set_user_attr('relative_deterioration', relative_deterioration)
        trial.set_user_attr('test_r2', test_r2)

        return test_loss
    
    # Load the study (don't create new tables in the worker processes)
    storage = RDBStorage(
        url="sqlite:///distributed-XGBoost-Optuna.db"
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
        url="sqlite:///distributed-XGBoost-Optuna.db"
    )
    
    study_name = 'distributed-XGBoost-Optuna'
    
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
    total_trials  = 30
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
    #plt.savefig('optimization_hypervolume_history-XGBoost.jpg')
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
    plt.savefig('optimization_history-XGBoost.jpg')
    plt.show()

    # Plot parameter importances
    #plt.figure(figsize=(10, 6))
    fig = plt.figure(figsize=(10, 6))
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    ax.set_facecolor('none')
    # Access the current axis
    ax = plt.gca()
    
    # Remove title and labels
    #fig.suptitle('')  # Remove the overall figure title
    #for text in ax.texts:
    #    text.set_visible(False)
    
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('hyperparameter_importance-XGBoost.jpg')
    plt.show()  
    
    
    plt.figure(figsize=(10, 6)) 
    ax = optuna.visualization.matplotlib.plot_contour(study, params=["n_estimators", "max_depth"])
    plt.tight_layout()
    plt.savefig('optimization_contour_XY-XGBoost.jpg')
    plt.show()
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plots
    
    # Set the background to white
    plt.style.use('default')
    
    # Extract the data from the Optuna study
    trials = study.trials_dataframe()
    
    # Filter out complete trials and get parameters of interest
    trials = trials[trials['state'] == 'COMPLETE']  # Filter out incomplete trials
    latent_dim = trials['params_learning_rate']
    batch_size = trials['params_n_estimators']
    drop_out_rate = trials['params_max_depth']
    objective_value = trials['value']  # The objective value (e.g., loss)
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale the marker size based on the objective value (loss)
    marker_size = (objective_value - objective_value.min() + 1) * 50  # Adjust scaling factor as needed
    
    # Plot with latent_dim, batch_size, and drop_out_rate as the axes
    sc = ax.scatter(latent_dim, batch_size, drop_out_rate, c=objective_value, cmap='viridis', s=marker_size)
    
    # Set axis labels
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('N Estimators')
    ax.set_zlabel('Max Depth')
    
    # Set the learning rate axis to a logarithmic scale
    ax.set_zscale('log')
    
    # Add a color bar to show the objective values (e.g., loss)
    cbar = plt.colorbar(sc)
    cbar.set_label('Objective Value (Loss)')
    
    plt.tight_layout()
    plt.savefig('optimization_3d_XGBoost.jpg')
    plt.show()
    
    # only supports 2 or 3 objective studies
    # plt.figure(figsize=(10, 6))
    # ax = optuna.visualization.plot_pareto_front(studies[0])
    # plt.tight_layout()
    # plt.savefig('optimization_pareto_front-XGBoost.jpg')
    # plt.show()
    
    # %%
    
    # Define the list of parameters you want to visualize in the contour plot
    params = ['n_estimators',
              'max_depth',
              'learning_rate',
              'subsample',
              'colsample_bytree'
              ]
    
    
    # Generate the contour plot (returns an Axes object)
    # Generate the contour plot
    axes = optuna.visualization.matplotlib.plot_contour(study, params=params)
    
    # Access the figure object and set the figure size
    fig = plt.gcf()
    fig.set_size_inches(14, 12)  # Set the desired figure size
    
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
    plt.savefig('customized_contour_plot_XGBoost.jpg')
    
    # Show the plot
    plt.show()
    
    # %%
    
    # End timer
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")