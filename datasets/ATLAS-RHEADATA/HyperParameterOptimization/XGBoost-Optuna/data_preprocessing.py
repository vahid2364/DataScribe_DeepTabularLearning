#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:03:16 2024

@author: attari.v
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer
from scipy.special import expit
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats

def process_and_split_data(df, input_columns, output_columns, threshold=1e-9,
                           apply_sc=True, scaling_method='minmax', 
                           apply_pt=False, pt_method='yeo-johnson', 
                           apply_qt=False, qt_method='normal', 
                           apply_log1p=False, apply_sigmoid=False, 
                           apply_sqrt=False, sqrt_constant=10, 
                           apply_cbrt=False, cbrt_constant=50, 
                           test_size=0.10, random_state=42):
    
    # Function to scale data and apply transformations
    def scale_data(df, input_columns, output_columns, apply_sc, scaling_method, apply_pt, pt_method, apply_qt, qt_method, apply_log1p, apply_sigmoid, apply_sqrt, sqrt_constant, apply_cbrt, cbrt_constant):
        transformed_data = df.copy()

        # Apply log1p transformation if requested
        if apply_log1p:
            for idx in df.columns:
                transformed_data[idx] = np.log1p(df[idx])

        # Apply Sigmoid transformation if requested
        if apply_sigmoid:
            for idx in df.columns:
                transformed_data[idx] = expit(df[idx])

        # Apply PowerTransformer if requested
        if apply_pt:
            pt = PowerTransformer(method=pt_method)
            for idx in df.columns:
                transformed_data[idx] = pt.fit_transform(df[[idx]]).ravel()

        # Apply QuantileTransformer if requested
        if apply_qt:
            qt = QuantileTransformer(output_distribution=qt_method, n_quantiles=1000)
            for idx in df.columns:
                transformed_data[idx] = qt.fit_transform(transformed_data[[idx]]).ravel()

        # Apply square root transformation if requested
        if apply_sqrt:
            for idx in df.columns:
                transformed_data[idx] = np.sqrt(np.maximum(0, sqrt_constant - df[idx]))

        # Apply cube root transformation if requested
        if apply_cbrt:
            for idx in df.columns:
                transformed_data[idx] = np.cbrt(cbrt_constant - df[idx])

        inputs = transformed_data[input_columns].to_numpy()
        outputs = transformed_data[output_columns].to_numpy()

        # Apply scaling if requested
        if apply_sc:
            if scaling_method == 'minmax':
                input_scaler = MinMaxScaler()
                output_scaler = MinMaxScaler()
            elif scaling_method == 'standard':
                input_scaler = StandardScaler()
                output_scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}. Choose 'minmax' or 'standard'.")
            
            inputs_scaled = input_scaler.fit_transform(inputs)
            outputs_scaled = output_scaler.fit_transform(outputs)
            
            os.makedirs('scales', exist_ok=True)
            joblib.dump(input_scaler, 'scales/input_scaler.save')
            joblib.dump(output_scaler, 'scales/output_scaler.save')
        else:
            inputs_scaled = inputs
            outputs_scaled = outputs
        
        return inputs_scaled, outputs_scaled

    # Apply threshold and filtering
    threshold_series = df[output_columns[0]]
    df = df[threshold_series > threshold]
    df[output_columns] = df[output_columns] * 1e6

    # Scale and transform data
    inputs_scaled, outputs_scaled = scale_data(df, input_columns, output_columns, apply_sc, scaling_method, apply_pt, pt_method, apply_qt, qt_method, apply_log1p, apply_sigmoid, apply_sqrt, sqrt_constant, apply_cbrt, cbrt_constant)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# Example Usage:
# Load your dataset and call the function
if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv('../input_data/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv')

    # Define input and output columns
    input_columns = ['Nb', 'Cr', 'V', 'W', 'Zr']
    output_columns = ['1000 Min Creep NH [1/s]']

    # Call the function
    X_train, X_test, y_train, y_test = process_and_split_data(df, input_columns, output_columns)
    
    # Print the shapes of the resulting data splits
    print("Training inputs shape:", X_train.shape)
    print("Test inputs shape:", X_test.shape)
    print("Training outputs shape:", y_train.shape)
    print("Test outputs shape:", y_test.shape)