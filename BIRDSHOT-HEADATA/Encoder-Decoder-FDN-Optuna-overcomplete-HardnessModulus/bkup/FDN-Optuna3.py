#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:03:46 2024

@author: attari.v
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:59:54 2024

@author: attari.v
"""

import optuna
from optuna.storages import RDBStorage
import psycopg2

import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from collections import defaultdict
import time

# Import the function from data_preprocessing.py (ensure this is correct)
from data_preprocessing import process_and_split_data

# Import your FDN model functions from FullyDense_Model.py
from FullyDense_Model import create_complex_encoder, create_complex_decoder

# Function to run trials in parallel and track core usage per process
def run_optimization(study_name, num_trials, X_train, X_test, y_train, y_test):
    # Dictionary to track core usage in this process
    core_usage = defaultdict(int)
    
    # Initialize progress bar for the current core
    core_id = multiprocessing.current_process().pid
    progress_bar = tqdm(total=num_trials, desc=f"Trials on core {core_id}", position=core_id)

    # Function to create the FDN model
    def create_fdn_model(trial):
        """
        Function to create and return an FDN model with specified parameters.
        """
        latent_dim = trial.suggest_int('latent_dim', 64, 224, step=16)
        lamb = trial.suggest_float('lambda', 1e-6, 1e-3, log=True)
        rate = trial.suggest_float('drop_out_rate', 0.1, 0.4, step=0.1)
        alp = trial.suggest_float('alpha', 0.01, 0.2, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adadelta'])
        
        input_dim = X_train.shape[1]  # Number of input features
        output_dim = y_train.shape[1]  # Number of output features
    
        # Sample the number of layers for encoder and decoder
        num_layers_encoder = trial.suggest_int('num_layers_encoder', 2, 4)  # Dynamically choose the number of layers for encoder
        num_layers_decoder = trial.suggest_int('num_layers_decoder', 2, 4)  # Dynamically choose the number of layers for decoder
        
        # Dynamically generate the number of neurons per layer for encoder and decoder
        neurons_per_layer_encoder = [
            trial.suggest_int(f'encoder_neurons_layer_{i}', 256 * (2 ** (num_layers_encoder - i - 1)), 2056, step=128) 
            for i in range(num_layers_encoder)
        ]
        neurons_per_layer_decoder = [
            trial.suggest_int(f'decoder_neurons_layer_{i}', 256 * (2 ** i), 2056, step=128)
            for i in range(num_layers_decoder)
        ]
        
        # Dynamically generate the number of neurons per layer for encoder
        neurons_per_layer_encoder = [
            trial.suggest_int(f'encoder_neurons_layer_{i}', max(256, 256 * (2 ** (num_layers_encoder - i - 1))), 2056, step=128) 
            for i in range(num_layers_encoder)
        ]
        
        # Dynamically generate the number of neurons per layer for decoder
        neurons_per_layer_decoder = [
            trial.suggest_int(f'decoder_neurons_layer_{i}', max(256, 256 * (2 ** i)), 2056, step=128) 
            for i in range(num_layers_decoder)
        ]
        
        # Create the encoder and decoder models
        encoder = create_complex_encoder(input_dim, latent_dim, num_layers=num_layers_encoder, neurons_per_layer=neurons_per_layer_encoder, lamb=lamb, rate=rate, alp=alp)
        decoder = create_complex_decoder(output_dim, latent_dim, num_layers=num_layers_decoder, neurons_per_layer=neurons_per_layer_decoder, lamb=lamb, rate=rate, alp=alp)


        # Create the autoencoder model
        autoencoder_input = tf.keras.Input(shape=(input_dim,))
        encoded = encoder(autoencoder_input)
        decoded = decoder(encoded)
        FD_EncoderDecoder = tf.keras.Model(inputs=autoencoder_input, outputs=decoded)
    
        # Choose the optimizer based on the suggestion
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        
        FD_EncoderDecoder.compile(optimizer=optimizer, loss='mse')
        return FD_EncoderDecoder
    
    # Define the objective function for Optuna
    def objective(trial):
        
        core_id = multiprocessing.current_process().pid
        core_usage[core_id] += 1
        
        # Create the model
        model = create_fdn_model(trial)
    
        batch_size = trial.suggest_int('batch_size', 64, 128, step=16)
        epochs = trial.suggest_int('epochs', 10, 50, step=20)
    
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
        #model_checkpoint = ModelCheckpoint(f'best_fdn_model_core_{core_id}.keras', save_best_only=True, monitor='val_loss')
    
        # Fit the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
    
        # # Predict on validation data
        # y_pred = model.predict(X_test)
    
        # # Calculate validation loss (MSE)
        # val_loss = mean_squared_error(y_test, y_pred)
        
        # # Update progress bar
        # progress_bar.update(1)
    
        # return val_loss

        # Get the best training loss
        best_train_loss = min(history.history['loss'])
    
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
        url="sqlite:///distributed-FDN-Optuna.db"
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
        url="sqlite:///distributed-FDN-Optuna.db"
    )
    
    study_name = 'distributed-FDN-Optuna'
    
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
    total_trials  = 18
    num_processes = 6
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
    plt.savefig('optimization_history-FDN.jpg')
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
    plt.savefig('hyperparameter_importance-FDN.jpg')
    plt.show()  
    
    plt.figure(figsize=(10, 6))
    ax = optuna.visualization.matplotlib.plot_contour(study, params=["latent_dim", "batch_size"])
    plt.tight_layout()
    plt.savefig('optimization_contour_XY-FDN.jpg')
    plt.show()
    
    # only supports 2 or 3 objective studies
    # plt.figure(figsize=(10, 6))
    # ax = optuna.visualization.plot_pareto_front(studies[0])
    # plt.tight_layout()
    # plt.savefig('optimization_pareto_front-FDN.jpg')
    # plt.show()
    
    # %%
    
    # End timer
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")