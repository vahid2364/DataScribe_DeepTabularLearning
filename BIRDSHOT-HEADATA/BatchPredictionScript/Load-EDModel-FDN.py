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
    conditional_parameters = np.array(df[input_columns][:20])
    
    # Make predictions
    _, predictions_descaled = make_predictions(model, input_scaler, output_scaler, conditional_parameters)
    
    # Save and display results
    save_and_display_results(df, predictions_descaled, output_columns, file_name)


if __name__ == "__main__":
    # Common configurations
    CSV_FILE_PATH = '../data/HTMDEC_MasterTable_Iterations_v3_processed.csv'
    INPUT_COLUMNS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
    
    # Process Yield Strength (YS)
    YS_OUTPUT_COLUMNS = ['Yield Strength (MPa)']
    YS_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-YS-SMAPE/'
    YS_SCALE_FOLDER = os.path.join(YS_WEIGHT_FOLDER, 'scales')
    process_target(CSV_FILE_PATH, INPUT_COLUMNS, YS_OUTPUT_COLUMNS, YS_WEIGHT_FOLDER, YS_SCALE_FOLDER, 'comparison_df1_YS.csv')
    
    # Process Ultimate Tensile Strength (UTS)
    UTS_OUTPUT_COLUMNS = ['UTS_True (Mpa)']
    UTS_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-UTS-SMAPE/'
    UTS_SCALE_FOLDER = os.path.join(UTS_WEIGHT_FOLDER, 'scales')
    process_target(CSV_FILE_PATH, INPUT_COLUMNS, UTS_OUTPUT_COLUMNS, UTS_WEIGHT_FOLDER, UTS_SCALE_FOLDER, 'comparison_df2_UTS.csv')
    
    # Process Elongation T
    ELON_OUTPUT_COLUMNS = ['Elong_T (%)']
    ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Elon T-SMAPE/'
    ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df3_Elon.csv')
    
    # Process Hardness 
    ELON_OUTPUT_COLUMNS = ['Hardness (GPa) SRJT']
    ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Hardness-SMAPE/'
    ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df4_Hardness.csv')
    
    # Process Modulus 
    ELON_OUTPUT_COLUMNS = ['Modulus (GPa) SRJT']
    ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Modulus-SMAPE/'
    ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df4_Modulus.csv')
    
    # Process Modulus 
    ELON_OUTPUT_COLUMNS = ['Avg HDYN/HQS']
    ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Avg HDYNHQSRatio-SMAPE/'
    ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df4_AvgHDYNHQSRatio.csv')