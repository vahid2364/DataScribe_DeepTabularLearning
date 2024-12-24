#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:59:54 2024

@author: attari.v
"""

import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

# Import the function from data_preprocessing.py
from data_preprocessing import process_and_split_data

# Import your FDN model functions from the previous code
from FullyDense_Model import create_complex_encoder, create_complex_decoder

# Function to create the FDN model
def create_fdn_model(trial):
    """
    Function to create and return an FDN model with specified parameters.
    """
    # Suggest hyperparameters for latent dimension, optimizer, and learning rate
    latent_dim = trial.suggest_int('latent_dim', 64, 192, step=64)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'sgd'])
    
    # Get input and output dimensions (you need to define these from your data)
    input_dim = X_train.shape[1]  # Number of input features
    output_dim = y_train.shape[1]  # Number of output features

    # Create the encoder and decoder models
    encoder = create_complex_encoder(input_dim, latent_dim)
    decoder = create_complex_decoder(output_dim, latent_dim)

    # Create the autoencoder model
    autoencoder_input = tf.keras.Input(shape=(input_dim,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    FD_EncoderDecoder = tf.keras.Model(inputs=autoencoder_input, outputs=decoded)

    # Choose the optimizer based on the suggestion
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # Compile the model
    FD_EncoderDecoder.compile(optimizer=optimizer, loss='mse')
    
    return FD_EncoderDecoder

# Define the objective function for Optuna
def objective(trial):
    # Create the model
    model = create_fdn_model(trial)

    # Set hyperparameters for batch size and epochs
    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    epochs = trial.suggest_int('epochs', 50, 150, step=50)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_fdn_model.keras', save_best_only=True, monitor='val_loss')

    # Fit the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0
    )

    # Predict on validation data
    y_pred = model.predict(X_test)

    # Calculate validation loss (MSE) and R² score
    val_loss = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log the R² score (for tracking purposes, not for optimization)
    #trial.report(r2, step=0)

    # Return only the validation loss to be minimized
    return val_loss

# Manually create a progress bar
def optimize_with_progress_bar(study, n_trials, n_jobs):
    with tqdm(total=n_trials) as pbar:
        for _ in range(n_trials):
            study.optimize(objective, n_trials=1, n_jobs=n_jobs, callbacks=[lambda study, trial: pbar.update()])

# Load and split the data only once
def load_and_split_data():
    df = pd.read_csv('../input_data/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv')
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
    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data()


    # Set up Optuna study for minimizing the objective
    study = optuna.create_study(direction="minimize", study_name='distributed-example', storage='sqlite:///example.db')

    optimize_with_progress_bar(study, n_trials=16, n_jobs=8)

    # Print the best hyperparameters
    print("Best parameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)
        
    # Save the results to a CSV file
    save_study_results(study)
    
    # %%

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