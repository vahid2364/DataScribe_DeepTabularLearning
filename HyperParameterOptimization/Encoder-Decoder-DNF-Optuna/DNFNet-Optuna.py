#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:49:57 2024

@author: attari.v
"""

import optuna
from optuna.storages import RDBStorage
import psycopg2

import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from collections import defaultdict
import time

# Import the function from data_preprocessing.py (ensure this is correct)
from data_preprocessing import process_and_split_data

# Import the functions from dnf_model.py
from dnf_model import create_dnnf_encoder, create_dnnf_decoder

# Function to run trials in parallel and track core usage per process
def run_optimization(study_name, num_trials, X_train, X_test, y_train, y_test):
    # Dictionary to track core usage in this process
    core_usage = defaultdict(int)
    
    # Initialize progress bar for the current core
    core_id = multiprocessing.current_process().pid
    progress_bar = tqdm(total=num_trials, desc=f"Trials on core {core_id}", position=core_id)

    # Function to create the FDN model
    def create_dnf_model(trial):
        """
        Function to create and return an DNF model with specified parameters.
        """
        latent_dim = trial.suggest_int('latent_dim', 32, 192, step=32)
        num_conjunctions = trial.suggest_int('num_conjunctions', 5, 25, step=5)
        rate = trial.suggest_float('drop_out_rate', 0.1, 0.3, step=0.1)
        conjunction_units = trial.suggest_int('conjunction_units', 16, 208, step=16)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adadelta'])

        # Get input and output dimensions
        input_dim = X_train.shape[1]  # Number of input features
        output_dim = y_train.shape[1]  # Number of output features
                
        # Create DNF encoder and decoder
        encoder = create_dnnf_encoder(input_dim=input_dim, latent_dim=latent_dim, num_conjunctions=num_conjunctions, conjunction_units=conjunction_units, dropout_rate=rate)
        decoder = create_dnnf_decoder(output_dim=output_dim, latent_dim=latent_dim, num_conjunctions=num_conjunctions, conjunction_units=conjunction_units, dropout_rate=rate)
        
        # Create the autoencoder model
        autoencoder_input = Input(shape=(input_dim,)) 
        encoded = encoder(autoencoder_input)
        decoded = decoder(encoded)
        DNFNet_EncoderDecoder = Model(inputs=autoencoder_input, outputs=decoded)
        
        # Choose the optimizer based on the suggestion
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        
        DNFNet_EncoderDecoder.compile(optimizer=optimizer, loss='mse')
        return DNFNet_EncoderDecoder
    
    # Define the objective function for Optuna
    def objective(trial):
        
        core_id = multiprocessing.current_process().pid
        core_usage[core_id] += 1
        
        # Create the model
        model = create_dnf_model(trial)
    
        batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
        epochs = trial.suggest_int('epochs', 10, 50, step=20)
    
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
        #model_checkpoint = ModelCheckpoint(f'best_fdn_model_core_{core_id}.keras', save_best_only=True, monitor='val_loss')
        
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))      # Add channel dimension
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))      # Add channel dimension
            
        # Fit the model
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_test_reshaped, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )

        # Get the best training loss
        best_train_loss = min(history.history['loss'])
    
        # Predict on test data (unseen data)
        y_pred = model.predict(X_test_reshaped)
        
        # Ensure y_test is also 2D
        #y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
        y_pred = y_pred.reshape(-1,1)        
        
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
        url="sqlite:///distributed-DNF-Optuna.db"
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
        url="sqlite:///distributed-DNF-Optuna.db"
    )
    
    study_name = 'distributed-DNF-Optuna'
    
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
    total_trials  = 10
    num_processes = 1
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
    plt.savefig('optimization_history-DNF.jpg')
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
    plt.savefig('hyperparameter_importance-DNF.jpg')
    plt.show()  
    
    plt.figure(figsize=(10, 6)) 
    ax = optuna.visualization.matplotlib.plot_contour(study, params=["latent_dim", "batch_size"])
    plt.tight_layout()
    plt.savefig('optimization_contour_XY-DNF.jpg')
    plt.show()
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plots
    
    # Set the background to white
    plt.style.use('default')
    
    # Extract the data from the Optuna study
    trials = study.trials_dataframe()
    
    # Filter out complete trials and get parameters of interest
    trials = trials[trials['state'] == 'COMPLETE']  # Filter out incomplete trials
    latent_dim = trials['params_latent_dim']
    batch_size = trials['params_batch_size']
    drop_out_rate = trials['params_learning_rate']
    objective_value = trials['value']  # The objective value (e.g., loss)
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale the marker size based on the objective value (loss)
    marker_size = (objective_value - objective_value.min() + 1) * 50  # Adjust scaling factor as needed
    
    # Plot with latent_dim, batch_size, and drop_out_rate as the axes
    sc = ax.scatter(latent_dim, batch_size, drop_out_rate, c=objective_value, cmap='viridis', s=marker_size)
    
    # Set axis labels
    ax.set_xlabel('Latent Dim')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Learning Rate')
    
    # Set the learning rate axis to a logarithmic scale
    ax.set_zscale('log')
    
    # Add a color bar to show the objective values (e.g., loss)
    cbar = plt.colorbar(sc)
    cbar.set_label('Objective Value (Loss)')
    
    plt.tight_layout()
    plt.savefig('optimization_3d_DNF.jpg')
    plt.show()
    
    # only supports 2 or 3 objective studies
    # plt.figure(figsize=(10, 6))
    # ax = optuna.visualization.plot_pareto_front(studies[0])
    # plt.tight_layout()
    # plt.savefig('optimization_pareto_front-FDN.jpg')
    # plt.show()
    
    # %%
    
    # Define the list of parameters you want to visualize in the contour plot
    params = ['latent_dim',
              'batch_size',
              'learning_rate',
              'num_conjunctions',
              'conjunction_units',
              'drop_out_rate',
              'epochs']
    
    
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
    plt.savefig('customized_contour_plot_DNF.jpg')
    
    # Show the plot
    plt.show()
    
    # %%
    
    # End timer
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")