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

# %% Load Origianl Data

#csv_file_path = 'IQR_dataframe-.csv'  # Replace with your CSV file path
csv_file_path = '../data/HTMDEC_MasterTable_Interpolated_Orange_Iterations_BBC_with_SFEcalc.csv'  # Replace with your CSV file path

df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

# Define input and output columns
input_columns = df.columns[3:11]
output_columns = df.columns[15:21] # Remaining columns
output_columns = output_columns.drop(['UTS/YS','Hardness (GPa)', 'Modulus (GPa)', 'Tension Elongation (%)'])
#output_columns = output_columns.drop(['Yield Strength (MPa)', 'Ultimate Tensile Strength (MPa)', 'UTS/YS', 'Tension Elongation (%)'])

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


# %% EncoderDecoder Input:    alloy composition

# Example alloy chemistry - replace this with desired alloys
conditional_parameters = np.array(df[input_columns][:10])

# %% UTS and YS

#import FullyDense_Model and scales

WeightFolder = '../ED_regression_FDN_UTSandYS/'
ScaleFolder = '../ED_regression_FDN_UTSandYS/scales/'

# Define file paths dynamically
autoencoder_model_path = os.path.join(WeightFolder, 'autoencoder_model_final.keras')
input_scaler_path = os.path.join(ScaleFolder, 'input_scaler.save')
output_scaler_path = os.path.join(ScaleFolder, 'output_scaler.save')

# Load models and scalers
EncoderDecoderI = tf.keras.models.load_model(autoencoder_model_path)
input_scalerI = joblib.load(input_scaler_path)
output_scalerI = joblib.load(output_scaler_path)

# Use the encoder-Decoder to transform the input data to the output space
predictions_scaled = EncoderDecoderI.predict(input_scalerI.transform(conditional_parameters))

# Inverse transform the predictions to original scale
predictions_descaled = descale_data(
    predictions_scaled, 
    input_scaler=input_scalerI, output_scaler=output_scalerI,
    apply_dsc=True, 
    apply_qt=False, qt_inputs=None, qt_outputs=None, 
    apply_pt=False, pt_inputs=None, pt_outputs=None, 
    apply_log1p=False, 
    apply_sigmoid=False,
    data_type='output'
    )

print("Decoded data shape:", predictions_scaled.shape)

min_length = 10
# Trimming both arrays to match the minimum length
predictions_scaled_trimmed = predictions_scaled[:min_length,:]
predictions_descaled_trimmed = predictions_descaled[:min_length,:]


# Convert both arrays to DataFrames for easy comparison of features side by side
df_scaled = pd.DataFrame(predictions_scaled_trimmed, columns=[f'Scaled Feature {i+1}' for i in range(predictions_scaled.shape[1])])
df_descaled = pd.DataFrame(predictions_descaled_trimmed, columns=[f'Original Feature {i+1}' for i in range(predictions_descaled.shape[1])])

# Combine both DataFrames side by side
comparison_df1 = pd.concat([df[output_columns][:10], df_descaled], axis=1)

# Display the comparison DataFrame
print(comparison_df1)

comparison_df1.to_csv('comparison_df1_UTS_YS.csv', index=False)  # Set index=False to avoid saving the index

# %% Origianl Data - 2nd Calculations

#csv_file_path = 'IQR_dataframe-.csv'  # Replace with your CSV file path
csv_file_path = '../data/HTMDEC_MasterTable_Interpolated_Orange_Iterations_BBC_with_SFEcalc.csv'  # Replace with your CSV file path

df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

# Define input and output columns
#input_columns = df.columns[3:11]
output_columns = df.columns[15:21] # Remaining columns
output_columns = output_columns.drop(['Yield Strength (MPa)', 'Ultimate Tensile Strength (MPa)', 'UTS/YS'])

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


# %% TE, Hardness, and Modulus 

#import FullyDense_Model and scales

WeightFolder = '../ED_regression_FDN_over_HardnessModulusTE/'
ScaleFolder = '../ED_regression_FDN_over_HardnessModulusTE/scales/'

# Define file paths dynamically
autoencoder_model_path = os.path.join(WeightFolder, 'autoencoder_model_final.keras')
input_scaler_path = os.path.join(ScaleFolder, 'input_scaler.save')
output_scaler_path = os.path.join(ScaleFolder, 'output_scaler.save')

# Load models and scalers
EncoderDecoderI = tf.keras.models.load_model(autoencoder_model_path)
input_scalerI = joblib.load(input_scaler_path)
output_scalerI = joblib.load(output_scaler_path)

# Use the encoder-Decoder to transform the input data to the output space
predictions_scaled = EncoderDecoderI.predict(input_scalerI.transform(conditional_parameters))

# Inverse transform the predictions to original scale
predictions_descaled = descale_data(
    predictions_scaled, 
    input_scaler=input_scalerI, output_scaler=output_scalerI,
    apply_dsc=True, 
    apply_qt=False, qt_inputs=None, qt_outputs=None, 
    apply_pt=False, pt_inputs=None, pt_outputs=None, 
    apply_log1p=False, 
    apply_sigmoid=False,
    data_type='output'
    )

print("Decoded data shape:", predictions_scaled.shape)

min_length = 10
# Trimming both arrays to match the minimum length
predictions_scaled_trimmed = predictions_scaled[:min_length,:]
predictions_descaled_trimmed = predictions_descaled[:min_length,:]


# Convert both arrays to DataFrames for easy comparison of features side by side
df_scaled = pd.DataFrame(predictions_scaled_trimmed, columns=[f'Scaled Feature {i+1}' for i in range(predictions_scaled.shape[1])])
df_descaled = pd.DataFrame(predictions_descaled_trimmed, columns=[f'Original Feature {i+1}' for i in range(predictions_descaled.shape[1])])

# Combine both DataFrames side by side
comparison_df2 = pd.concat([df[output_columns][:10], df_descaled], axis=1)

# Display the comparison DataFrame
print(comparison_df2)

comparison_df2.to_csv('comparison_df1_TE_Hardness_Modulus.csv', index=False)  # Set index=False to avoid saving the index



