#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:03:35 2024

@author: attari.v
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# %% Origianl Data

csv_file_path = 'IQR_dataframe-cobalt.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

df['PROP 500C CTE (1/K)'] = df['PROP 500C CTE (1/K)']*1e6
df['PROP 1000C CTE (1/K)'] = df['PROP 1000C CTE (1/K)']*1e6
df['PROP 1500C CTE (1/K)'] = df['PROP 1500C CTE (1/K)']*1e6

# Define the remaining features
columns_to_keep = [
    'Nb', 'Cr', 'V', 'W', 'Zr',
    'PROP LT (K)', 'PROP ST (K)', 
     'PROP 500C CTE (1/K)', 'PROP 1000C CTE (1/K)', 'PROP 1500C CTE (1/K)', 
     'EQ 1273K MAX BCC', 'EQ 1523K MAX BCC',
     'EQ 1273K SUM BCC', 'EQ 1523K SUM BCC',
     'EQ 1273K THCD (W/mK)', 'EQ 1523K THCD (W/mK)', 
     'EQ 1273K Density (g/cc)', 'EQ 1523K Density (g/cc)',
     'EQ 2/3*ST THCD (W/mK)', 'EQ 2/3*ST Density (g/cc)',  
     'YS 1000C PRIOR','YS 1500C PRIOR',
     'Pugh_Ratio_PRIOR', 
     #'1000 Min Creep NH [1/s]', '1300 Min Creep NH [1/s]','1500 Min Creep NH [1/s]', 
    'SCHEIL ST',
    'SCHEIL LT', 
    'Kou Criteria'
]

df = df[columns_to_keep]

# %% EncoderDecoder Input:     'Nb', 'Cr', 'V', 'W', 'Zr',

# Example alloy chemistry - replace this with desired alloys
conditional_parameters = np.array(df.iloc[0:2,0:5])

print(conditional_parameters)

# %% Load Model I: Outputs: 19 features
    
#    'PROP LT (K)': 
#    'PROP ST (K)':
#    'PROP 500C CTE (1/K)', 'PROP 1000C CTE (1/K)', 'PROP 1500C CTE (1/K)', 
#    'EQ 1273K MAX BCC', 'EQ 1523K MAX BCC',
#    'EQ 1273K SUM BCC', 'EQ 1523K SUM BCC',
#    'EQ 1273K THCD (W/mK)', 'EQ 1523K THCD (W/mK)', 
#    'EQ 1273K Density (g/cc)', 'EQ 1523K Density (g/cc)',
#    'EQ 2/3*ST THCD (W/mK)', 'EQ 2/3*ST Density (g/cc)',  
#    'YS 1000C PRIOR','YS 1500C PRIOR',
#    'Pugh_Ratio_PRIOR', 
#    'SCHEIL ST'

EncoderDecoderI = tf.keras.models.load_model('AutoEncoder-single-AllExceptCreepScheilKou/autoencoder_AllExceptCreepScheilKou_model.h5')

input_scalerI  = joblib.load('AutoEncoder-single-AllExceptCreepScheilKou/input_scaler.save')
output_scalerI = joblib.load('AutoEncoder-single-AllExceptCreepScheilKou/output_scaler.save')

# Use the encoder-Decoder to transform the input data to the output space
predictions_scaled = EncoderDecoderI.predict(input_scalerI.transform(conditional_parameters))

# Inverse transform the predictions to original scale
predictionsI = output_scalerI.inverse_transform(predictions_scaled)

#print("Decoded data shape:", predictions.shape)
#print("Decoded data (scale):", predictions_scaled)
print("(19 features) (original scale):", predictionsI)

# %% Load Model II: Outputs: Kou

EncoderDecoderII = tf.keras.models.load_model('AutoEncoder-single-Kou2/autoencoder_KouCriteria2_model.h5')

# # Load scalers
input_scalerII  = joblib.load('AutoEncoder-single-Kou2/input_scaler.save')
output_scalerII = joblib.load('AutoEncoder-single-Kou2/output_scaler.save')

# Use the encoder-Decoder to transform the input data to the output space
predictions_scaled = EncoderDecoderII.predict(input_scalerII.transform(conditional_parameters))

# Inverse transform the predictions to original scale
predictionsII = output_scalerII.inverse_transform(predictions_scaled)

#print("Decoded data shape:", predictions.shape)
#print("Decoded data (scale):", predictions_scaled)
print("Kou Criterion (original scale):", predictionsII)

# %% Load Model III: Outputs: Scheil LT
    
EncoderDecoderIII = tf.keras.models.load_model('AutoEncoder-single-Scheil/autoencoder_ScheilLT_model.h5')

input_scalerIII  = joblib.load('AutoEncoder-single-Scheil/input_scaler.save')
output_scalerIII = joblib.load('AutoEncoder-single-Scheil/output_scaler.save')

# Use the encoder to transform the input data to the latent space
decoded_dataIII  = EncoderDecoderIII.predict(input_scalerIII.transform(conditional_parameters))
predictionsIII   = output_scalerIII.inverse_transform(decoded_dataIII)

#print("Decoded Data Shape:", decoded_data_realIII.shape)
print("Scheil LT (original scale):", predictionsIII)

# %% Load Model IV: Outputs: Cobalt Creep

#    'Pugh_Ratio_PRIOR', 

EncoderDecoderI = tf.keras.models.load_model('Encoder-Decoder-DNNF-NewData/autoencoder_model_epoch_629_loss.keras')

input_scalerI  = joblib.load('Encoder-Decoder-DNNF-NewData/scales/input_scaler.save')
output_scalerI = joblib.load('Encoder-Decoder-DNNF-NewData/scales/output_scaler.save')

# Use the encoder-Decoder to transform the input data to the output space
predictions_scaled = EncoderDecoderI.predict(input_scalerI.transform(conditional_parameters))

# Inverse transform the predictions to original scale
predictionsI = output_scalerI.inverse_transform(predictions_scaled)

#print("Decoded data shape:", predictions.shape)
#print("Decoded data (scale):", predictions_scaled)
print("(19 features) (original scale):", predictionsI)

# %% Combine results into one DataFrame

predictions_df = pd.DataFrame(conditional_parameters, columns=['Nb', 'Cr', 'V', 'W', 'Zr'])
predictions_df = pd.concat([predictions_df, pd.DataFrame(predictionsI, columns=[
    'PROP LT (K)', 'PROP ST (K)', 
    'PROP 500C CTE (1/K)', 'PROP 1000C CTE (1/K)', 'PROP 1500C CTE (1/K)', 
    'EQ 1273K MAX BCC', 'EQ 1523K MAX BCC',
    'EQ 1273K SUM BCC', 'EQ 1523K SUM BCC',
    'EQ 1273K THCD (W/mK)', 'EQ 1523K THCD (W/mK)', 
    'EQ 1273K Density (g/cc)', 'EQ 1523K Density (g/cc)',
    'EQ 2/3*ST THCD (W/mK)', 'EQ 2/3*ST Density (g/cc)',  
    'YS 1000C PRIOR','YS 1500C PRIOR',
    'Pugh_Ratio_PRIOR', 
    'SCHEIL ST'
])], axis=1)
predictions_df['Kou Criteria'] = predictionsII
predictions_df['SCHEIL LT'] = predictionsIII

print(predictions_df)

# %% Calculate MAE between two selected columns in results_df and df

column = 'PROP LT (K)'  # Replace with the column name you want to compare

mae = mean_absolute_error(df[column], predictions_df[column])
r2 = r2_score(df[column], predictions_df[column])
print(f"Mean Absolute Error (MAE) between calculated {column} and orignal {column}: {mae}")
print(f"R² score between calculated {column} and original {column}: {r2}")
