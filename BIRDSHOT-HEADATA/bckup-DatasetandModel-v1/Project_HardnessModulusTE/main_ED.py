#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 00:23:59 2024

@author: attari.v
"""

import sys
import os
print("Current working directory:", os.getcwd())
print("Python module search paths:", sys.path)

import pandas as pd
from modules.preprocessing import preprocess_data, scale_data
from sklearn.model_selection import train_test_split

# Create necessary directories
os.makedirs('results', exist_ok=True)
os.makedirs('results/QQplot', exist_ok=True)
os.makedirs('results/parityplots-scaled', exist_ok=True)
os.makedirs('results/parityplots-original', exist_ok=True)
os.makedirs('results/weights', exist_ok=True)
os.makedirs('scales', exist_ok=True)

# Load dataset
# Define input and output columns
df = pd.read_csv('../data/HTMDEC_MasterTable_Interpolated_Orange_Iterations_BBC_with_SFEcalc.csv')
    
# Define input and output columns
input_columns = df.columns[3:11]
output_columns = df.columns[15:21] # Remaining columns
output_columns = output_columns.drop(['Yield Strength (MPa)', 'Ultimate Tensile Strength (MPa)','UTS/YS'])
#output_columns = output_columns.drop(['Yield Strength (MPa)', 'Ultimate Tensile Strength (MPa)', 'UTS/YS', 'Tension Elongation (%)'])

# Preprocess data
df = preprocess_data(df, input_columns, output_columns)
#inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt_inputs, pt_outputs, qt_inputs, qt_outputs = scale_data(
scalers = scale_data(
    df, 
    input_columns, 
    output_columns, 
    apply_sc=True, scaling_method='minmax', 
    apply_pt=False, pt_method='yeo-johnson', 
    apply_qt=False, qt_method='uniform', 
    apply_sigmoid=False, 
    apply_sqrt=False, sqrt_constant=10, 
    apply_cbrt=False, cbrt_constant=50
)


# Split data
#X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(scalers['inputs_scaled'], scalers['outputs_scaled'], test_size=0.1, random_state=42)

# %%

from modules.models import read_optuna_parameters, build_autoencoder, print_model_summary

# Path to Optuna trial file
optuna_file_path = '../Encoder-Decoder-FDN-Optuna-overcomplete-HardnessModulusTE/optuna_trials.csv'

# Extract parameters
params = read_optuna_parameters(optuna_file_path, trial_index=303) # 817, 303, 500, 452, 530

# Define input and output dimensions
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# Build the autoencoder
autoencoder, encoder, decoder = build_autoencoder(params, input_dim, output_dim)

# Print model summaries
print_model_summary(encoder, decoder, autoencoder)

# Build the autoencoder
autoencoder, encoder, decoder = build_autoencoder(params, input_dim=X_train.shape[1], output_dim=y_train.shape[1])

# %% Train the model

from modules.training import train_autoencoder

    
# Train the model
history = train_autoencoder(
    model=autoencoder,
    X_train=X_train,
    y_train=y_train,
    epochs=550,
    batch_size=params["batch_size"],
    validation_split=0.1,
    learning_rate=params["learning_rate"],
    patience=30,
    checkpoint_filepath="results/weights/autoencoder_model_final.keras"
)

# %% Evaluate model

from modules.evaluations import evaluate_model, make_predictions, calculate_metrics, save_metrics

# Evaluate the model
test_loss = evaluate_model(autoencoder, X_test, y_test)

# Make predictions
predictions, predictions_train, predictions_scaled, predictions_scaled_train = make_predictions(
    model=autoencoder,
    X_test=X_test,
    X_train=X_train,
    input_scaler=scalers["input_scaler"],
    output_scaler=scalers["output_scaler"],
    apply_dsc=True,  # Enable descaling
    data_type="output"
)

y_test_original = scalers["descale"](y_test, data_type='output')
y_train_original = scalers["descale"](y_train, data_type='output')

# Calculate metrics
mse, mse_per_feature, r2_per_feature = calculate_metrics(predictions_scaled, y_test )
mse, mse_per_feature, r2_per_feature = calculate_metrics(predictions, y_test_original )

# Save metrics
save_metrics(predictions_scaled, y_test, "results/metrics_scaled.txt")
save_metrics(predictions, y_test_original, "results/metrics.txt")

# %% Visualizations

from modules.visualizations import plot_loss, generate_qq_plots, generate_parity_plots

# plot history
plot_loss(history, save_path='results/training_loss_plot.jpg')
    
# Generate QQ plots
generate_qq_plots(y_test, predictions_scaled, y_test_original, predictions)

# Generate scatter plots
#generate_parity_plots(y_test, predictions_scaled, y_test_original, predictions, output_columns)

generate_parity_plots(
    y_test, predictions_scaled, y_test_original, predictions, output_columns,
    y_train=y_train, predictions_scaled_train=predictions_scaled_train, 
    y_train_original=y_train_original, predictions_train=predictions_train
)

