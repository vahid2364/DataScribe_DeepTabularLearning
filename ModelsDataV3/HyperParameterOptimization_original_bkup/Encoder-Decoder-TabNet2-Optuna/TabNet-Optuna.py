#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:59:54 2024

@author: attari.v
"""

import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import warnings

# Set PyTorch device to CPU
device = torch.device('cpu')

warnings.filterwarnings("ignore")

# Import the function from data_preprocessing.py
from data_preprocessing import process_and_split_data


# Function to create the TabNet model
def create_model(trial):
    """
    Function to create and return a TabNetRegressor model with specified parameters.
    """
    n_d = trial.suggest_int('num_feature_transformer_dim (n_d)', 16, 48, step=16)
    n_a = trial.suggest_int('num_feature_transformer_att (n_a)', 16, 48, step=16)
    n_steps = trial.suggest_int('num_dec_steps (n_steps)', 3, 7)
    gamma = trial.suggest_float('strength_feature_reuse_penalty (gamma)', 1.0, 1.5)
    lambda_sparse = trial.suggest_loguniform('sparsity regularization parameter (lambda_sparse)', 1e-4, 1e-2)
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-4, 5e-3)

    tabnet_model = TabNetRegressor(
        optimizer_fn=torch.optim.Adam,
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
    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Create model
    model = create_model(trial)

    # Set hyperparameters
    batch_size = trial.suggest_int('batch_size', 32, 128, step=16)
    epochs = trial.suggest_int('epochs', 10, 50, step=20)

    # Fit the model using TabNet's built-in method
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=epochs,
        patience=20,
        batch_size=batch_size,
        virtual_batch_size=32,
        num_workers=0,
        drop_last=False
    )
    
    # Get validation loss (mean squared error)
    y_pred = model.predict(X_test)

    # Calculate MSE and R²
    val_loss = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Optuna minimizes the objective, so return validation loss
    return val_loss

# Load and split the data only once
def load_and_split_data():
#    df = pd.read_csv('../input_data/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv')
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
        
if __name__ == "__main__":
    # Set up Optuna study for minimizing the objective
    study = optuna.create_study(direction="minimize")

    # Run the optimization with parallel jobs
    study.optimize(objective, n_trials=5)  # n_jobs specifies parallelism

    # Print best hyperparameters
    print("Best parameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)
        
    # %%

    # Save the results to a CSV file
    save_study_results(study)
    
    # %%

    # First plot: Optimization history
    
    # Set figure size (width, height)
    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    
    # Generate the optimization history plot (returns an Axes object)
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    
    # Remove face color (transparent background)
    ax.set_facecolor('none')
    
    # Make all lines black
    for line in ax.get_lines():
        line.set_color('black')
    
    # Customize the spines (axes box) to be black
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # Customize ticks and labels to black
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    # Keep the axis and the plot box
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Remove the title
    ax.get_figure().suptitle("")
    
    # Bring the legend inside the box, make the legend black
    legend = ax.legend(loc='upper right', frameon=True)
    for text in legend.get_texts():
        text.set_color('black')
    
    plt.savefig('optimization_history-TabNet.jpg')
    plt.show()
    
    # %%
    
    # Second plot: Parameter importances
    
   # Set figure size (width, height)
    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    
    # Generate the parameter importances plot (returns an Axes object)
    ax = optuna.visualization.matplotlib.plot_param_importances(study)
    
    # Remove face color (transparent background)
    ax.set_facecolor('none')
    
    # Customize the spines (axes box) to be black
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # Customize ticks and labels to black
    ax.tick_params(axis='both', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    # Keep the axis and the plot box
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Remove the title
    ax.get_figure().suptitle("")
    
    # Bring the legend inside the box, make the legend black (if applicable)
    legend = ax.get_legend()
    if legend:
        legend.set_frame_on(True)
        for text in legend.get_texts():
            text.set_color('black')
    
    plt.savefig('hyperparameter_importance-TabNet.jpg')
    plt.show()
    
    # %%
    
    # Define the list of parameters you want to visualize in the contour plot
    params = ['learning_rate',
              'n_d',
              'n_a',
              'n_steps',
              'lambda_sparse',
              'gamma']
    
    
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
    plt.savefig('customized_contour_plot.jpg')
    
    # Show the plot
    plt.show()


    
    
    