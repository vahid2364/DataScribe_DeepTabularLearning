#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 23:30:37 2024

@author: attari.v
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error,mean_squared_error

from scaling_utils2 import scale_data, descale_data


def load_data(csv_file_path, input_columns, output_columns):
    """Load and preprocess the dataset."""
    df = pd.read_csv(csv_file_path)
    df = df.loc[:, ~(df == 0).all()]  # Drop columns with all zeros
    df = df[input_columns + output_columns].dropna()
    print("\nDataFrame after dropping all-zero columns:")
    print(df)
    return df


def load_model_and_scalers(weight_folder, scale_folder):
    """Load the trained encoder-decoder model and scalers."""
    autoencoder_model_path = os.path.join(weight_folder, 'bestmodel_wieghts.keras')
    input_scaler_path = os.path.join(scale_folder, 'input_scaler.save')
    output_scaler_path = os.path.join(scale_folder, 'output_scaler.save')
    
    model = tf.keras.models.load_model(autoencoder_model_path)
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)
    
    return model, input_scaler, output_scaler


def make_predictions(model, input_scaler, output_scaler, conditional_parameters):
    """Make predictions using the encoder-decoder model and inverse-transform the outputs."""
    scaled_input = input_scaler.transform(conditional_parameters)
    predictions_scaled = model.predict(scaled_input)
    predictions_descaled = descale_data(
        predictions_scaled, 
        input_scaler=input_scaler, output_scaler=output_scaler,
        apply_dsc=True, apply_qt=False, apply_pt=False, apply_log1p=False, apply_sigmoid=False, 
        data_type='output'
    )
    return predictions_scaled, predictions_descaled

def save_and_display_results(df, predictions_descaled, output_columns, file_name, num_rows=20):
    """Save predictions to a CSV and display a comparison."""
    # Ensure indices of predictions match the original dataset
    predictions_descaled_df = pd.DataFrame(
        predictions_descaled[:num_rows], 
        columns=[f'Predicted {col}' for col in output_columns],
        index=df[output_columns].index[:num_rows]  # Align indices
    )
    
    # Combine original and predicted DataFrames side by side
    comparison_df = pd.concat([df[output_columns][:num_rows], predictions_descaled_df], axis=1)
    
    # Display and save the comparison DataFrame
    print(comparison_df)
    comparison_df.to_csv(file_name, index=False)
    print(f"Saved comparison results to {file_name}")

def process_target(csv_file_path, input_columns, output_columns, weight_folder, scale_folder, file_name):
    """Process a single target using the specified parameters."""
    # Load data
    df = load_data(csv_file_path, input_columns, output_columns)
    
    # Load model and scalers
    model, input_scaler, output_scaler = load_model_and_scalers(weight_folder, scale_folder)
    
    # Prepare input data
    conditional_parameters = np.array(df[input_columns][:])
    
    # Make predictions
    _, predictions_descaled = make_predictions(model, input_scaler, output_scaler, conditional_parameters)
    
    # Save and display results
    save_and_display_results(df, predictions_descaled, output_columns, file_name)
    
    return df, predictions_descaled


if __name__ == "__main__":
    # Common configurations
    CSV_FILE_PATH = '../data/data_LIQUID_variable_temprange9_processed.csv'
    INPUT_COLUMNS = ['Al','Cu','Cr','Nb','Ni','Fe','Mo']
    
    # Process Yield Strength (YS)
    YS_OUTPUT_COLUMNS = ['$/kg']
    YS_WEIGHT_FOLDER = '../optimization_contour/Encoder-Decoder-FDN-Optuna-overcomplete-COST-SMAPE'
    YS_SCALE_FOLDER = os.path.join(YS_WEIGHT_FOLDER, 'scales')
    df, predictions_descaled = process_target(CSV_FILE_PATH, INPUT_COLUMNS, YS_OUTPUT_COLUMNS, YS_WEIGHT_FOLDER, YS_SCALE_FOLDER, 'comparison_df1_YS.csv')
    
    # Calculate MAE
    mae1 = mean_absolute_error(df[YS_OUTPUT_COLUMNS][:], predictions_descaled)
    mse1 = mean_squared_error(df[YS_OUTPUT_COLUMNS][:], predictions_descaled)
        

    
    
    print("MAE and MSE:", mae1,mse1)

    
    
    