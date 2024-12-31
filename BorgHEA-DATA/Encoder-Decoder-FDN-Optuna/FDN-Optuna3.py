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
        
        input_dim = X_train.shape[1]  # Number of input features
        output_dim = y_train.shape[1]  # Number of output features
    
        # Sample the number of layers for encoder and decoder
        num_layers_encoder = trial.suggest_int('num_layers_encoder', 1, 5)  # Dynamically choose the number of layers for encoder
        num_layers_decoder = trial.suggest_int('num_layers_decoder', 1, 5)  # Dynamically choose the number of layers for decoder
        
        # Sample the number of neurons in the first encoder layer with constraint
        fln = trial.suggest_int(
            'encoder_neurons_layer_1', 
            16,  # Minimum neurons in the first layer
            128,  # Maximum constrained by latent_dim
            step=16
        )
                                
        # Dynamically generate the number of neurons per layer for encoder and decoder
        neurons_per_layer_encoder = [
            int(fln) * (2 ** i) for i in range(num_layers_encoder)
        ]
                
        # Reverse the encoder layers for the decoder to ensure symmetry
        # neurons_per_layer_decoder = neurons_per_layer_encoder[::-1]
        neurons_per_layer_decoder = [
            max(8, neurons_per_layer_encoder[-1] // (2 ** i)) for i in range(num_layers_decoder)
        ]
        
        # Debugging prints (optional)
        print('fln',fln,num_layers_encoder,num_layers_decoder)
        print('encoder', neurons_per_layer_encoder)
        print('decoder', neurons_per_layer_decoder)

        #latent_dim = trial.suggest_int('latent_dim', neurons_per_layer_encoder[-1]+32, 512, step=16)
        #latent_dim = trial.suggest_int('latent_dim', min(neurons_per_layer_encoder[-1] + 32, 512), 512, step=16)
                
        # Define latent_dim with a valid range
        latent_dim = trial.suggest_int(
            'latent_dim', 
            min(neurons_per_layer_encoder[-1] + 32, 512),  # Ensure the low value is valid
            1024, 
            step=16
        )
                
        # Debugging prints (optional)
        print('latent_dim', latent_dim)
        
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
        
        #encoder.summary()
        #decoder.summary()        
        #pause
        
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
    
        batch_size = trial.suggest_int('batch_size', 32, 128, step=8)
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
    df = pd.read_csv('../Borg_df_updated.csv')
    
    # Define input and output columns
#    input_columns = df.columns[:30]
#    output_columns = df.columns[36:37] # Remaining columns
    
    # Define input and output columns
    input_columns = df.columns[:30]
    input_columns = input_columns.drop(['B', 'Nd', 'Ga', 'Ag', 'Cu', 'C'])
    output_columns = df.columns[ 33:38 ] # Remaining columns
    output_columns = output_columns.drop(['PROPERTY: Test temperature ($^\circ$C)'])

    # Drop columns with all zeros
    df = df.loc[:, ~(df == 0).all()]

    columns_to_keep = input_columns.tolist() + output_columns.tolist()

    df = df[columns_to_keep]
    df = df.dropna()

    print("\nDataFrame after dropping all-zero columns:")
    print(df)
    
    #input_columns = ['Nb', 'Cr', 'V', 'W', 'Zr']
    #output_columns = ['Kou Criteria']
    X_train, X_test, y_train, y_test = process_and_split_data(
        df, 
        input_columns, 
        output_columns, 
        threshold=1e-9, 
        apply_sc=True, scaling_method='minmax', 
        apply_qt=False, qt_method='uniform', 
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
    total_trials  = 6*12
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
    
    # List neuron configurations for each trial
    neurons_configurations = []
    for trial in study.trials:
        encoder_config = trial.user_attrs.get('neurons_per_layer_encoder', [])
        decoder_config = trial.user_attrs.get('neurons_per_layer_decoder', [])
        latent_dim = trial.params.get('latent_dim', None)  # Extract latent_dim from trial parameters
        neurons_configurations.append({
            'trial_id': trial.number,
            'neurons_encoder': encoder_config,
            'latent_dim': latent_dim,  # Add latent_dim here
            'neurons_decoder': decoder_config
        })
    
    # Convert to a DataFrame for easier visualization or export
    neurons_df = pd.DataFrame(neurons_configurations)
    neurons_df.to_csv('neurons_configurations_with_latent_dim.csv', index=False)
    
    print("Neuron configurations with latent_dim saved to 'neurons_configurations_with_latent_dim.csv'")
    
    import seaborn as sns
    
    # Plot encoder neurons across trials
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=neurons_df['neurons_encoder'].explode().dropna().astype(int))
    #sns.boxplot(data=neurons_df['neurons_encoder'].explode().astype(int))
    plt.title('Distribution of Encoder Neurons Across Trials')
    plt.xlabel('Neuron Count')
    plt.show()

    # %%
    
    # Ensure wide display for all columns side by side
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping to the next line
    print(neurons_df)
    
    # %%
    
    # Define the list of parameters you want to visualize in the contour plot
    params = ['latent_dim',
              'lambda',
              'alpha',
              'drop_out_rate',
              'learning_rate',
              'encoder_neurons_layer_1',
              'num_layers_encoder',
              'num_layers_decoder',
              ]
    
    # Generate the contour plot (returns an Axes object)
    # Generate the contour plot
    axes = optuna.visualization.matplotlib.plot_contour(study, params=params)
    
    # Access the figure object and set the figure size
    fig = plt.gcf()
    fig.set_size_inches(20, 18)  # Set the desired figure size
    
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