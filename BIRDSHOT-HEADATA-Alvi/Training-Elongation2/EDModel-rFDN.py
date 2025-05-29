#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:10:51 2025

@author: attari.v
"""


import os
import optuna
from optuna.storages import RDBStorage
import psycopg2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tqdm import tqdm
from collections import defaultdict
import time
import io
from contextlib import redirect_stdout

import multiprocessing

from optuna.storages import RDBStorage
import optuna

import random
# Seed values
SEED = 42

# Set seeds for all libraries
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic behavior for TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)

# Optional: Disable GPU for full determinism
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import the function from data_preprocessing.py (ensure this is correct)
#from data_preprocessing import process_and_split_data

# Import your FDN model functions from FullyDense_Model.py
from FullyDense_Model import create_complex_encoder, create_complex_decoder


# %%
# %%

# Function to collect objective values and optionally sort them
def collect_objective_values(study, sort_values=True):
    objective_values = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            objective_values.append(trial.value)
    
    if sort_values:
        # Sort objective values in descending order
        sorted_objective_values = sorted(objective_values, reverse=True)
        # Generate new trial numbers (1, 2, 3, ...)
        new_trial_numbers = list(range(1, len(sorted_objective_values) + 1))
        return new_trial_numbers, sorted_objective_values
    else:
        # Generate trial numbers in their original order
        trial_numbers = list(range(1, len(objective_values) + 1))
        return trial_numbers, objective_values


# Function to create the FDN model
def create_fdn_model(trial):
    """
    Function to create and return an FDN model with specified parameters.
    """
    lamb = trial.suggest_float('lambda', 1e-6, 1e-3, log=True)
    rate = trial.suggest_float('drop_out_rate', 0.1, 0.4, step=0.1)
    alp = trial.suggest_float('alpha', 0.01, 0.2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adadelta'])
    
    input_dim = 8 #X_train.shape[1]  # Number of input features
    output_dim = 1 #y_train.shape[1]  # Number of output features

    # Sample the number of layers for encoder and decoder
    num_layers_encoder = trial.suggest_int('num_layers_encoder', 2, 5)  # Dynamically choose the number of layers for encoder
    num_layers_decoder = trial.suggest_int('num_layers_decoder', 2, 5)  # Dynamically choose the number of layers for decoder
    
    # Sample the number of neurons in the first encoder layer with constraint
    fln = trial.suggest_int(
        'encoder_neurons_layer_1', 
        4,      # Minimum neurons in the first layer - overcomplete
        256,    # Maximum constrained by latent_dim - overcomplete
        #256,   # Minimum neurons in the first layer - undercomplete
        #1024,  # Maximum constrained by latent_dim - undercomplete
        step=4
    )
                            
    # Dynamically generate the number of neurons per layer for encoder and decoder
    neurons_per_layer_encoder = [
        int(fln) * (2 ** i) for i in range(num_layers_encoder)             # overcomplete
        ##int ( int(fln) / (2 ** i) ) for i in range(num_layers_encoder)   # undercomplete
    ]
            
    # Reverse the encoder layers for the decoder to ensure symmetry
    # neurons_per_layer_decoder = neurons_per_layer_encoder[::-1]
    neurons_per_layer_decoder = [
        max(8, neurons_per_layer_encoder[-1] // (2 ** i)) for i in range(num_layers_decoder) # overcomplete
        ## max(8, neurons_per_layer_encoder[-1] * (2 ** i)) for i in range(num_layers_decoder) # undercomplete

    ]
    
    # Debugging prints (optional)
    #print('fln',fln,num_layers_encoder,num_layers_decoder)
    #print('encoder', neurons_per_layer_encoder)
    #print('decoder', neurons_per_layer_decoder)

    #latent_dim = trial.suggest_int('latent_dim', neurons_per_layer_encoder[-1]+32, 512, step=16)
    #latent_dim = trial.suggest_int('latent_dim', min(neurons_per_layer_encoder[-1] + 32, 512), 512, step=16)
            
    # Define latent_dim with a valid range

    ## overcomplete
    latent_dim = trial.suggest_int(
        'latent_dim', 
        min(neurons_per_layer_encoder[-1] + 8, neurons_per_layer_encoder[-1] * 1.1), 
        neurons_per_layer_decoder[0] * 2.0
    )
    
    ## undercomplte
    #latent_dim = trial.suggest_int(
    #    'latent_dim', 
    #    min(neurons_per_layer_encoder[-1] , int(neurons_per_layer_encoder[-1] / 5) ),  # Ensure the low value is valid
    #    min(neurons_per_layer_decoder[0], int(neurons_per_layer_decoder[0] / 1.5) ),  # Ensure the low value is valid
    #    step=32
    #)
            
    # Debugging prints (optional)
    #print('latent_dim', latent_dim)
    
    if neurons_per_layer_encoder[-1] + 32 > 512:
        print("Warning: `latent_dim` range is limited.")

    # Save the layer configurations as trial attributes for analysis later
    trial.set_user_attr('first_layer_neurons', fln)
    trial.set_user_attr('neurons_per_layer_encoder', neurons_per_layer_encoder)
    trial.set_user_attr('neurons_per_layer_decoder', neurons_per_layer_decoder)
    
    # Optional: Log as part of trial parameters
    trial.set_user_attr(
        'encoder_decoder_config',
        {
            'neurons_per_layer_encoder': neurons_per_layer_encoder,
            'neurons_per_layer_decoder': neurons_per_layer_decoder
        }
    )
    
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
    
    #encoder.summary()
    #decoder.summary()      
    FD_EncoderDecoder.summary()
    #pause
    
    return FD_EncoderDecoder

# %% LOAD DATA

split_num = [0,1,2,3,4]

for idx in split_num:

    res_folder = "results"+str(idx)
    os.makedirs(res_folder, exist_ok=True)
    os.makedirs(res_folder+"/scales", exist_ok=True)
    
    fl_name = 'split_'+str(idx)+'_test.csv'
    test_data = pd.read_csv('../data-Alvi/alloy_splits/'+str(fl_name))
    fl_name = 'split_0_train.csv'
    train_data = pd.read_csv('../data-Alvi/alloy_splits/'+str(fl_name))
    
    # df
    test_data  = test_data[test_data["Iteration"].isin(["AAA","AAB","AAC","AAD","AAE","BBA", "BBB", "BBC", "CBA"])]
    train_data = train_data[train_data["Iteration"].isin(["AAA","AAB","AAC","AAD","AAE","BBA", "BBB", "BBC", "CBA"])]
    
    # Define input and output columns
    input_columns = [
        'Al','Co','Cr','Cu','Fe','Mn','Ni','V'
    ]
    
    output_columns = [
        #'Yield Strength (MPa)',
        #'UTS_True (Mpa)',
        'Elong_T (%)',                          
        #'Hardness (GPa) SRJT', 
        #'Modulus (GPa) SRJT'
    ]

    # Drop columns with all zeros
    test_data = test_data.loc[:, ~(test_data == 0).all()]
    train_data = train_data.loc[:, ~(train_data == 0).all()]
    
    #columns_to_keep = input_columns.tolist() + output_columns.tolist()
    columns_to_keep = input_columns + output_columns
    
    #test_data = test_data[columns_to_keep]
    #train_data = train_data[columns_to_keep]
    
    from sklearn.preprocessing import MinMaxScaler
    
    # Fit scaler only on training data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Drop rows where target column is NaN
    train_data = train_data.dropna(subset=output_columns)
    test_data = test_data.dropna(subset=output_columns)
    
    # Scale inputs
    train_data_scaled = train_data.copy()
    test_data_scaled = test_data.copy()

    train_data_scaled[input_columns] = scaler_X.fit_transform(train_data[input_columns])
    test_data_scaled[input_columns] = scaler_X.transform(test_data[input_columns])
    
    train_data_scaled[output_columns] = scaler_y.fit_transform(train_data[output_columns])
    test_data_scaled[output_columns] = scaler_y.transform(test_data[output_columns])
    
    # Then split into inputs and outputs
    X_train = train_data_scaled[input_columns].to_numpy()
    y_train = train_data_scaled[output_columns].to_numpy()
    X_test = test_data_scaled[input_columns].to_numpy()
    y_test = test_data_scaled[output_columns].to_numpy()
    
    import joblib
    # Save input and output scaler
    joblib.dump(scaler_X, res_folder+'/scales/input_scaler.save')
    joblib.dump(scaler_y, res_folder+'/scales/output_scaler.save')
    
    print("\nDataFrame after dropping all-zero columns:")
    print(X_train)
    print("\nInput Columns:")
    print(input_columns)
    print("\nOutput Columns:")
    print(output_columns)

# %%

    study_infos = [
        {"name": 'distributed-FDN-Optuna', "url": "sqlite:///../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Elon T-SMAPE/distributed-FDN-Optuna.db"}
    ]
    
    storage = RDBStorage(url=study_infos[0]["url"])
    study = optuna.load_study(study_name=study_infos[0]["name"], storage=storage)
    
    trial_numbers_sorted, objective_values_sorted = collect_objective_values(study, sort_values=True)

    # %%
        
    # Rebuild the best model and get predictions for parity plot
    best_trial = study.best_trial  # Access the best trial
    
    # Extract the best parameters
    best_params = best_trial.params
    print("Best trial parameters:", best_params)
    
    #import json
    # Save best_params to JSON
    #with open(res_folder+"/best_params.json", "w") as f:
    #    json.dump(best_params, f, indent=4)
    
    # Choose the trial with the smallest loss (or another criterion)
    #best_tradeoff_trial = pareto_trials_sorted[0]
    #print("Best trade-off trial:")
    #print(f"  Values: {best_tradeoff_trial.values}")
    #print(f"  Params: {best_tradeoff_trial.params}")
    
    
    # Recreate the model using the best parameters
    def rebuild_best_model(best_params, X_train, y_train):
        trial_mock = optuna.trial.FixedTrial(best_params)  # Use a FixedTrial to pass parameters
        return create_fdn_model(trial_mock)  # Reuse your model creation function
    
    best_model = rebuild_best_model(best_params, X_train, y_train)
    
    # Compile the best model
    if best_params['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    elif best_params['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=best_params['learning_rate'])
    elif best_params['optimizer'] == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=best_params['learning_rate'])
    
    best_model.compile(optimizer=optimizer, loss='mse')
    
    # Fit the best model on the training data
    batch_size = best_params['batch_size']
    epochs = 1500 #best_params['epochs']*5
        
    callbacks = [
        EarlyStopping(monitor='loss', patience=25, restore_best_weights=False),
        ModelCheckpoint(filepath='bestmodel_wieghts.keras', monitor='val_loss', save_best_only=True, mode='min')  # Save the best weights
        #SaveAtLastEpoch('autoencoder_model_final.keras')  # Save model at the last epoch
    ]
    
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callbacks],
        verbose=1  # Suppress training logs for brevity
    )

    # %%
    
    os.makedirs(res_folder, exist_ok=True)
    os.makedirs(res_folder+"/plots", exist_ok=True)
    
    # Plot the losses
    plt.figure(figsize=(8, 4))
    plt.style.use("default")  # This resets to Matplotlib's default, which has a white background
    epochs = range(len(history.history['loss']))
    
    # Validation losses (if available)
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['loss'], '-', label='Training Total Loss', color='black', lw=3)
        plt.plot(epochs, history.history['val_loss'], '--', label='Validation Total Loss', color='blue', lw=3)
    
        plt.xlabel('Epochs', fontsize=25)
        plt.ylabel('Loss', fontsize=25)
        
        plt.tick_params(axis='both', which='major', labelsize=25)  # 'both' for x and y, 'major' for major ticks
    
        plt.legend(fontsize=25)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(res_folder+'/plots/loss.jpg', dpi=300, transparent=True)
        plt.show()
    
    # Get predictions on the test data
    y_pred = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)
    
    
    mse_test = mean_squared_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    smape_test = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) * 100
    
    mse = mean_squared_error(y_train, y_pred_train)
    r2 = r2_score(y_train, y_pred_train)
    smape = np.mean(2 * np.abs(y_pred_train - y_train) / (np.abs(y_pred_train) + np.abs(y_train))) * 100
    
    # Plot the parity plot
    plt.figure(figsize=(8, 8))
    
    plt.text(0.05, 0.95, f'MSE-Test: {mse_test:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.92, f'r$^2$-Test : {r2_test :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.88, f'SMAPE-Test: {smape_test:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    #
    plt.text(0.05, 0.80, f'MSE-Train: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.77, f'r$^2$-Train : {r2 :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.73, f'SMAPE-Train: {smape:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.5)
    plt.scatter(y_train.flatten(), y_pred_train.flatten(), alpha=0.75, c='blue')
    plt.plot([min(y_test.flatten()), max(y_test.flatten())],
             [min(y_test.flatten()), max(y_test.flatten())], color='red', linestyle='--', linewidth=2)
    plt.plot([min(y_train.flatten()), max(y_train.flatten())],
             [min(y_train.flatten()), max(y_train.flatten())], color='blue', linestyle=':', linewidth=2)
    
    plt.xlabel("Actual: YS")
    plt.ylabel("Predicted")
    #plt.title("Parity Plot")
    #plt.grid(True)
    plt.savefig(res_folder+'/plots/parity-plot.png',dpi=300)
    plt.show()
    
    # %% Repeats training 5 to 10 times (you can change n_runs).
        
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
    n_runs = 10  # Number of repeated runs
    results = []
    
    #os.makedirs("plots", exist_ok=True)
    os.makedirs(res_folder+"/weights", exist_ok=True)
    
    for run in range(1, n_runs + 1):
        print(f"\n🔁 Run {run}/{n_runs}")
    
        best_model = rebuild_best_model(best_params, X_train, y_train)
        
        # Compile the best model
        if best_params['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
        elif best_params['optimizer'] == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=best_params['learning_rate'])
        elif best_params['optimizer'] == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=best_params['learning_rate'])
        
        best_model.compile(optimizer=optimizer, loss='mse')
        
        # Fit the best model on the training data
        batch_size = best_params['batch_size']
        epochs = 1500 #best_params['epochs']*5
        
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            ModelCheckpoint(filepath=res_folder+f'/weights/bestmodel_weights_run{run}.keras', monitor='val_loss', save_best_only=True, mode='min')
        ]
                
        history = best_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callbacks],
            verbose=1  # Suppress training logs for brevity
        )
    
        # Plot loss
        plt.figure(figsize=(8, 4))
        epoch_range = range(len(history.history['loss']))
        plt.plot(epoch_range, history.history['loss'], '-', label='Train Loss', lw=2)
        if 'val_loss' in history.history:
            plt.plot(epoch_range, history.history['val_loss'], '--', label='Val Loss', lw=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f"Loss Curve - Run {run}")
        plt.tight_layout()
        
        plt.savefig(res_folder+f'/plots/loss_run{run}.png')
        plt.close()
    
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)
    
        # %%
        
        # Inverse transform predictions and true values
        y_pred_rescaled = scaler_y.inverse_transform(y_pred)
        y_test_rescaled = scaler_y.inverse_transform(y_test)
        
        y_pred_train_rescaled = scaler_y.inverse_transform(y_pred_train)
        y_train_rescaled = scaler_y.inverse_transform(y_train)
    
        X_test_rescaled = scaler_X.inverse_transform(X_test)
        X_train_rescaled = scaler_X.inverse_transform(X_train)
        
        # Create test results DataFrame
        df_test_results = pd.DataFrame(X_test_rescaled, columns=input_columns)
        df_test_results[['Actual_' + col for col in output_columns]] = y_test_rescaled
        df_test_results[['Predicted_' + col for col in output_columns]] = y_pred_rescaled
        
        # Create train results DataFrame
        df_train_results = pd.DataFrame(X_train_rescaled, columns=input_columns)
        df_train_results[['Actual_' + col for col in output_columns]] = y_train_rescaled
        df_train_results[['Predicted_' + col for col in output_columns]] = y_pred_train_rescaled
        
        with pd.ExcelWriter(res_folder+'/model_predictions'+str(run)+'.xlsx') as writer:
            df_train_results.to_excel(writer, sheet_name='Train Results', index=False)
            df_test_results.to_excel(writer, sheet_name='Test Results', index=False)

        # %%     
    
        # Metrics
        mse_test = mean_squared_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)
        smape_test = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test) + 1e-9)) * 100
    
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        smape_train = np.mean(2 * np.abs(y_pred_train - y_train) / (np.abs(y_pred_train) + np.abs(y_train) + 1e-9)) * 100
    
        results.append({
            'Run': run,
            'MSE_Train': mse_train,
            'R2_Train': r2_train,
            'SMAPE_Train': smape_train,
            'MSE_Test': mse_test,
            'R2_Test': r2_test,
            'SMAPE_Test': smape_test
        })
    
        # Parity plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, label='Test', color='red')
        plt.scatter(y_train, y_pred_train, alpha=0.5, label='Train', color='blue')
        plt.plot([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())],
                 [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())],
                 linestyle='--', color='gray', linewidth=2)
        
        plt.text(0.05, 0.95, f'MSE-Test: {mse_test:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.92, f'r$^2$-Test : {r2_test :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.88, f'SMAPE-Test: {smape_test:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        #
        plt.text(0.05, 0.80, f'MSE-Train: {mse_train:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.77, f'r$^2$-Train : {r2_train :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.73, f'SMAPE-Train: {smape_train:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Parity Plot - Run {run}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_folder+f'/plots/parity_run{run}.png')
        plt.close()
        
        # Save loss history for this run
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(res_folder+f'/plots/loss_history_run{run}.csv', index=False)
        
        tf.keras.backend.clear_session()
    
    
    # Save all results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(res_folder+"/plots/run_metrics_summary.csv", index=False)
    
    # Summary statistics
    summary = results_df.describe().loc[['min', 'max', 'mean']]
    summary.to_csv(res_folder+"/plots/run_metrics_summary_stats.csv")
    print("\n📊 Summary of Runs:")
    print(summary)
    
          
    all_history = []
    
    for run in range(1, n_runs + 1):
        df = pd.read_csv(res_folder+f'/plots/loss_history_run{run}.csv')
        df['Run'] = run
        df['Epoch'] = df.index
        all_history.append(df)
    
    combined_loss_history = pd.concat(all_history, ignore_index=True)
    combined_loss_history.to_csv(res_folder+"/plots/all_loss_histories_combined.csv", index=False)
    
    # %%
    
    import seaborn as sns
    
    # Load and combine all loss histories
    all_loss = []
    for run in range(1, n_runs + 1):
        df = pd.read_csv(res_folder+f'/plots/loss_history_run{run}.csv')
        df['Run'] = run
        df['Epoch'] = df.index
        all_loss.append(df)
    
    loss_df = pd.concat(all_loss, ignore_index=True)
    
    # Pivot to get each run's loss per epoch
    loss_pivot = loss_df.pivot(index='Epoch', columns='Run', values='loss')
    
    # Compute min, max, and mean across runs
    min_loss = loss_pivot.min(axis=1)
    max_loss = loss_pivot.max(axis=1)
    mean_loss = loss_pivot.mean(axis=1)
    
    # Plot all loss curves with shaded min-max region
    plt.figure(figsize=(10, 6))
    for run in range(1, n_runs + 1):
        plt.plot(loss_pivot.index, loss_pivot[run], alpha=0.4, label=f'Run {run}')
    
    plt.plot(loss_pivot.index, mean_loss, color='black', linewidth=2, label='Mean Loss')
    plt.fill_between(loss_pivot.index, min_loss, max_loss, color='gray', alpha=0.3, label='Min-Max Envelope')
    
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Training Loss", fontsize=14)
    plt.title("Training Loss Across Runs", fontsize=16)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(res_folder+"/plots/combined_loss_with_band.png", dpi=300)
    plt.show()