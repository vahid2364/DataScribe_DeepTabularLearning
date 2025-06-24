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


def load_data(csv_file_path, input_columns):
    """Load and preprocess the dataset."""
    df = pd.read_csv(csv_file_path)
    #df = df.loc[:, ~(df == 0).all()]  # Drop columns with all zeros
    #df = df[input_columns + output_columns].dropna()
    print('dataframe columns are:')
    print(df.columns)

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
    df = load_data(csv_file_path, input_columns)
    
    df = df[input_columns]*100 # Reorder explicitly
            
    # Load model and scalers
    model, input_scaler, output_scaler = load_model_and_scalers(weight_folder, scale_folder)

    # Prepare input data
    conditional_parameters = np.array(df[input_columns][:])
    
    # Make predictions
    _, predictions_descaled = make_predictions(model, input_scaler, output_scaler, conditional_parameters)
    
    # Save and display results
    #save_and_display_results(df, predictions_descaled, output_columns, file_name)
    
    return df, predictions_descaled


if __name__ == "__main__":
    # Common configurations
    CSV_FILE_PATH = 'input_data/filtered_htmdecy3_n7_d25_s6_7_t27000_TCeq.csv'
    INPUT_COLUMNS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V']
    #Al	Co	Cr	Cu	Fe	Mn	Ni	V
    # 'Al',  'Cu' = 0
    
    # Process Yield Strength (YS)
    YS_OUTPUT_COLUMNS = ['Yield Strength (MPa)']
    YS_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-YS-SMAPE/'
    YS_SCALE_FOLDER = os.path.join(YS_WEIGHT_FOLDER, 'scales')
    df, predictions_descaled1 = process_target(CSV_FILE_PATH, INPUT_COLUMNS, YS_OUTPUT_COLUMNS, YS_WEIGHT_FOLDER, YS_SCALE_FOLDER, 'comparison_df1_YS.csv')
            
    # Process Ultimate Tensile Strength (UTS)
    UTS_OUTPUT_COLUMNS = ['UTS_True (Mpa)']
    UTS_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-UTS-SMAPE/'
    UTS_SCALE_FOLDER = os.path.join(UTS_WEIGHT_FOLDER, 'scales')
    df, predictions_descaled2 = process_target(CSV_FILE_PATH, INPUT_COLUMNS, UTS_OUTPUT_COLUMNS, UTS_WEIGHT_FOLDER, UTS_SCALE_FOLDER, 'comparison_df2_UTS.csv')
    
    ##
    predictions_UTSYS_ratio = predictions_descaled2/predictions_descaled1
    
    # Process Elongation T
    ELON_OUTPUT_COLUMNS = ['Elong_T (%)']
    ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Elon T-SMAPE/'
    ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    df, predictions_descaled3 = process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df3_Elon.csv')
        
    # Process Hardness 
    #ELON_OUTPUT_COLUMNS = ['Hardness (GPa) SRJT']
    #ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Hardness-SMAPE/'
    #ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    #process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df4_Hardness.csv')
    
    # Process Modulus 
    #ELON_OUTPUT_COLUMNS = ['Modulus (GPa) SRJT']
    #ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Modulus-SMAPE/'
    #ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    #process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df4_Modulus.csv')
    
    # Process Modulus 
    ELON_OUTPUT_COLUMNS = ['Avg HDYN/HQS']
    ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Avg HDYNHQSRatio-SMAPE/'
    ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    df, predictions_descaled4 = process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df5_AvgHDYNHQSRatio.csv')

    # Process Modulus 
    ELON_OUTPUT_COLUMNS = ['Depth of Penetration (mm) FE_Sim']
    ELON_WEIGHT_FOLDER = '../Hyperparameter-optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Depth of Penetration (mm) FE_Sim-SMAPE/'
    ELON_SCALE_FOLDER = os.path.join(ELON_WEIGHT_FOLDER, 'scales')
    df, predictions_descaled5 = process_target(CSV_FILE_PATH, INPUT_COLUMNS, ELON_OUTPUT_COLUMNS, ELON_WEIGHT_FOLDER, ELON_SCALE_FOLDER, 'comparison_df5_AvgHDYNHQSRatio.csv')

    
    #####
    
    # Assuming predictions_descaled1, predictions_descaled2, predictions_descaled3, predictions_descaled4 are available
    # Example: YS_OUTPUT_COLUMNS corresponds to feature names for each prediction set
    all_predictions = [predictions_descaled1, predictions_descaled2, predictions_UTSYS_ratio, predictions_descaled3, predictions_descaled4, predictions_descaled5]
    all_output_columns = ['Yield Strength (MPa)', 'Ultimate Tensile Strength (MPa)', 'predictions_UTSYS_ratio', 'Elongation (%)', 'Avg HDYN/HQS', 'Depth of Penetration (mm) FE_Sim']
    
    # Create a DataFrame for each set of predictions
    prediction_dfs = [
        pd.DataFrame(predictions, columns=[f'Predicted {col}']) 
        for predictions, col in zip(all_predictions, all_output_columns)
    ]
    
    # Combine all prediction DataFrames into one
    predictions_df = pd.concat(prediction_dfs, axis=1)
    
    # Merge the original DataFrame and predictions DataFrame
    merged_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
    
    # Display and save the merged DataFrame
    print("Merged DataFrame:")
    print(merged_df)
    
    merged_df.to_csv('merged_comparison_all_predictions.csv', index=False)
    print("Saved merged DataFrame to 'merged_comparison_all_predictions.csv'")
    
    
    
    