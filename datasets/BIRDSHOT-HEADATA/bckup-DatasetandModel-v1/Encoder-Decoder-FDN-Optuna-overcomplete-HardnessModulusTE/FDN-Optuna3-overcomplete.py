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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
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
        print('fln',fln,num_layers_encoder,num_layers_decoder)
        print('encoder', neurons_per_layer_encoder)
        print('decoder', neurons_per_layer_decoder)

        #latent_dim = trial.suggest_int('latent_dim', neurons_per_layer_encoder[-1]+32, 512, step=16)
        #latent_dim = trial.suggest_int('latent_dim', min(neurons_per_layer_encoder[-1] + 32, 512), 512, step=16)
                
        # Define latent_dim with a valid range

        ## overcomplete
        latent_dim = trial.suggest_int(
            'latent_dim', 
            min(neurons_per_layer_encoder[-1] + 8, neurons_per_layer_encoder[-1]*1.1),  # Ensure the low value is valid
            neurons_per_layer_decoder[0]*3, # or a fixed number 
            step=16
        )
        
        ## undercomplte
        #latent_dim = trial.suggest_int(
        #    'latent_dim', 
        #    min(neurons_per_layer_encoder[-1] , int(neurons_per_layer_encoder[-1] / 5) ),  # Ensure the low value is valid
        #    min(neurons_per_layer_decoder[0], int(neurons_per_layer_decoder[0] / 1.5) ),  # Ensure the low value is valid
        #    step=32
        #)
                
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
        epochs = trial.suggest_int('epochs', 100, 100, step=20)
    
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=False)
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

        # Calculate metrics
        y_pred = model.predict(X_test)
        test_loss = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)  # Maximize
        
        mse_per_feature = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        r2_per_feature = r2_score(y_test, y_pred, multioutput='raw_values')
        
        # Log per-feature R² scores and MSE dynamically
        for i, (r2) in enumerate(zip(r2_per_feature)):
            #trial.set_user_attr(f'mse_feature_{i + 1}', mse)
            trial.set_user_attr(f'r2_feature_{i + 1}', r2)
        
        # Combine objectives (Weighted Approach)
        # Adjust the weights as per your priority
        #weight_loss = 0.3  # Weight for minimizing loss
        #weight_r2 = 0.7    # Weight for maximizing R² (convert to minimizing -R²)
        #combined_objective = weight_loss * test_loss - weight_r2 * r2
        
        # # Log metrics
        # trial.set_user_attr('test_loss', test_loss)
        # trial.set_user_attr('r2_score1', r2_1)
        # trial.set_user_attr('r2_score2', r2_2)
        # trial.set_user_attr('r2_score3', r2_3)
        # trial.set_user_attr('r2_score4', r2_4)
        # trial.set_user_attr('r2_score5', r2_5)
        # #trial.set_user_attr('r2_score6', r2_6)

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
        
        # Return separate objectives
        return tuple(r2_per_feature)  # Dynamically return all R² scores as objectives
    
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
    
    # Define input and output columns
    df = pd.read_csv('../data/HTMDEC_MasterTable_Interpolated_Orange_Iterations_BBC_with_SFEcalc.csv')

    # Define input and output columns
    input_columns = df.columns[3:11]
    output_columns = df.columns[15:21] # Remaining columns
    #output_columns = output_columns.drop(['SFE_calc','Predicted Yield Strength at 298K (MPa)',
    #output_columns = output_columns.drop(['UTS/YS','Hardness (GPa)', 'Modulus (GPa)', 'Tension Elongation (%)'])
    output_columns = output_columns.drop(['Yield Strength (MPa)', 'Ultimate Tensile Strength (MPa)', 'UTS/YS'])

    # 'Computed Stacking Fault Energy',
    # 'Computed Valence Electron Concentration',
    # 'Computed Pugh Ratio'])
    
    # Drop columns with all zeros
    df = df.loc[:, ~(df == 0).all()]
    
    columns_to_keep = input_columns.tolist() + output_columns.tolist()
    
    df = df[columns_to_keep]
    df = df.dropna()
    
    print("\nDataFrame after dropping all-zero columns:")
    print(df)
    print("\nInput Columns:")
    print(input_columns)
    print("\nOutput Columns:")
    print(output_columns)
        
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


#### begin code
#### begin code
#### begin code
        
if __name__ == "__main__":
    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data()

    start_time = time.time()
    
    # Initialize database and create table once before multiprocessing
    storage = RDBStorage(
        url="sqlite:///distributed-FDN-Optuna.db"
    )
    
    study_name = 'distributed-FDN-Optuna'
        
    # Define the number of objectives dynamically based on R² scores
    num_objectives = np.size(y_train,1)  # Number of R² scores corresponds to the number of features
    
    # Create or load the study with dynamic directions
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,  # Create or load the study
            directions=["maximize"] * num_objectives  # Maximize for each objective (e.g., R² for each feature)
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
    
    # Split the number of trials across multiple processes
    total_trials  = 6*30
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


    # Retrieve Pareto-optimal trials
    pareto_trials = studies[0].best_trials  # Use best_trials instead of best_params
    
    # Print details of Pareto-optimal trials
    for i, trial in enumerate(pareto_trials):
        print(f"Pareto-optimal Trial {i}:")
        print(f"  Values: {trial.values}")  # Objective values
        print(f"  Params: {trial.params}")  # Hyperparameters
        print()

    # Sort Pareto-optimal trials by a specific objective
    pareto_trials_sorted = sorted(pareto_trials, key=lambda t: t.values[0])  # Sort by the first objective (e.g., loss)
    
    # Choose the trial with the smallest loss (or another criterion)
    best_tradeoff_trial = pareto_trials_sorted[0]
    print("Best trade-off trial:")
    print(f"  Values: {best_tradeoff_trial.values}")
    print(f"  Params: {best_tradeoff_trial.params}")
        
    # %%
        
    df = save_study_results(studies[0])
    
    # %% Plots
    
    # Extract metrics for all trials
    losses = [trial.values[0] for trial in study.trials if trial.values is not None]  # First objective: Test Loss
    r2_scores = [-trial.values[1] for trial in study.trials if trial.values is not None]  # Second objective: Negative R² (convert back to R²)
    
    # Plot Loss Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Test Loss (MSE)')
    plt.xlabel('Trial')
    plt.ylabel('MSE')
    plt.title('Loss Evolution')
    plt.legend()
    plt.show()
    
    # Plot R² Score Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(r2_scores, label='R² Score', color='orange')
    plt.xlabel('Trial')
    plt.ylabel('R²')
    plt.title('R² Score Evolution')
    plt.legend()
    plt.show()
    
    
    # Plot contour for the first objective (e.g., Loss)
    plt.figure(figsize=(10, 6))
    ax = optuna.visualization.matplotlib.plot_contour(
        study,
        params=["latent_dim", "batch_size"],
        target=lambda t: t.values[0],  # First objective (e.g., Test Loss)
        target_name="Loss (MSE)"
    )
    plt.tight_layout()
    plt.savefig('optimization_contour_Loss.jpg')
    plt.show()
    
    # Plot contour for the second objective (e.g., R²)
    plt.figure(figsize=(10, 6))
    ax = optuna.visualization.matplotlib.plot_contour(
        study,
        params=["latent_dim", "batch_size"],
        target=lambda t: -t.values[1],  # Second objective (Negative R² converted to positive)
        target_name="R² Score"
    )
    plt.tight_layout()
    plt.savefig('optimization_contour_R2.jpg')
    plt.show()
    
     
    import optuna.visualization as vis
    
    # Plot Pareto front for the first two objectives
    pareto_fig = vis.plot_pareto_front(
        study,
        targets=lambda t: (t.values[0], t.values[1]),  # Select specific objectives (e.g., first two)
        target_names=["R² - 1", "R² - 2"]  # Customize target names for your objectives
    )
    
    # Plot Pareto front for the study
#    pareto_fig = vis.plot_pareto_front(
#        study,
#        target_names=["R² - 1", "R² - 2", "R² - 3", "R² - 4"]  # Customize target names for your objectives
#    )
    
    # Show the Pareto front plot
    pareto_fig.show()
    
    # Optional: Save the Pareto front as an image
    pareto_fig.write_image("pareto_front_plot.png")  # Requires `plotly` and `kaleido` for saving    
    

    # Plot Contour Plot for Selected Hyperparameters
    #plt.figure(figsize=(10, 6))
    #ax = optuna.visualization.matplotlib.plot_contour(study, params=["latent_dim", "batch_size"])
    #plt.tight_layout()
    #plt.savefig('optimization_contour_XY-FDN.jpg')
    #plt.show()
    
    # Parameter Importance (for individual objectives, Optuna doesn't yet support multi-objective parameter importance)
    # If you want parameter importance per objective, create separate single-objective studies or analyze manually.
        
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
    
    # # Define the list of parameters you want to visualize in the contour plot
    # params = ['latent_dim',
    #           'lambda',
    #           'alpha',
    #           'drop_out_rate',
    #           'learning_rate',
    #           'encoder_neurons_layer_1',
    #           'num_layers_encoder',
    #           'num_layers_decoder',
    #           ]
    
    # # Generate the contour plot (returns an Axes object)
    # # Generate the contour plot
    # axes = optuna.visualization.matplotlib.plot_contour(study, params=params)
    
    # # Access the figure object and set the figure size
    # fig = plt.gcf()
    # fig.set_size_inches(20, 18)  # Set the desired figure size
    
    # # Remove the title
    # fig.suptitle("")  # Set to empty string to remove the title
    
    #   # Customize each subplot in the axes array
    # for ax in axes.flatten():
    #     # Customize the spines (axes box) to be black
    #     for spine in ax.spines.values():
    #         spine.set_color('black')
    
    #     # Customize ticks and labels to black
    #     ax.tick_params(axis='both', colors='black')
    #     ax.xaxis.label.set_color('black')
    #     ax.yaxis.label.set_color('black')
    
    #     # Remove grid lines inside the plot
    #     ax.grid(False)
    
    # # Save the customized contour plot to a file
    # plt.savefig('customized_contour_plot_XGBoost.jpg')
    
    # # Show the plot
    # plt.show()

    # %%
    
    # End timer
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")