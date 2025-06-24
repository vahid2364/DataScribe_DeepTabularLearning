#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:03:02 2024

@author: attari.v
"""

import numpy as np
import os
import joblib
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler, StandardScaler
from scipy.special import expit  # Sigmoid function

def scale_data(df, input_columns, output_columns, 
               apply_sc=False, scaling_method='minmax', 
               apply_pt=False, pt_method='yeo-johnson', 
               apply_qt=False, qt_method='uniform', 
               apply_log=False,
               apply_log10=False, 
               apply_log1p=False, 
               apply_sigmoid=False, 
               apply_sqrt=False, sqrt_constant=1, 
               apply_cbrt=False, cbrt_constant=1):
    
    # Copy the data
    transformed_data = df.copy()

    # Apply log1p transformation if requested
    if apply_log:
        for idx in df.columns:
            transformed_data[idx] = np.log(df[idx])

    # Apply log1p transformation if requested
    if apply_log10:
        for idx in df.columns:
            transformed_data[idx] = np.log10(df[idx])
    
    # Apply log1p transformation if requested
    if apply_log1p:
        for idx in df.columns:
            transformed_data[idx] = np.log1p(df[idx])

    # Apply Sigmoid transformation if requested
    if apply_sigmoid:
        for idx in df.columns:
            transformed_data[idx] = expit(df[idx])  # Sigmoid function

    # Apply PowerTransformer if requested
    if apply_pt:
        if pt_method == 'yeo-johnson':
            pt = PowerTransformer(method='yeo-johnson')
        elif pt_method == 'box-cox':
            pt = PowerTransformer(method='box-cox')
        else:
            raise ValueError(f"Unknown PowerTransformer Method: {pt_method}. Choose 'yeo-johnson' or 'box-cox'.")
        
        # Apply Power Transformer to selected columns
        for idx in df.columns:
            transformed_data[idx] = pt.fit_transform(df[[idx]]).ravel()

    # Apply QuantileTransformer if requested
    if apply_qt:
        if qt_method == 'normal':
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        elif qt_method == 'uniform':
            qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        else:
            raise ValueError(f"Unknown QuantileTransformer method: {qt_method}. Choose 'normal' or 'uniform'.")
        
        # Apply Quantile Transformer to selected columns
        for idx in df.columns:
            transformed_data[idx] = qt.fit_transform(transformed_data[[idx]]).ravel()

    # Apply square root transformation √(constant - x) if requested
    if apply_sqrt:
        #for col in input_columns + output_columns:
        for idx in df.columns:
            transformed_data[idx] = np.sqrt(np.maximum(0, sqrt_constant - df[idx]))  # Ensure non-negative inside square root

    # Apply cube root transformation ³√(constant - x) if requested
    if apply_cbrt:
        for idx in df.columns:
            transformed_data[idx] = np.cbrt(cbrt_constant - df[idx])

    # Separate inputs and outputs
    inputs = transformed_data[input_columns].to_numpy()
    outputs = transformed_data[output_columns].to_numpy()

    if apply_sc:
        # Initialize the scalers based on the scaling method chosen
        if scaling_method == 'minmax':
            input_scaler = MinMaxScaler()
            output_scaler = MinMaxScaler()
        elif scaling_method == 'standard':
            input_scaler = StandardScaler()
            output_scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}. Choose 'minmax' or 'standard'.")
    
        # Scale the inputs and outputs
        inputs_scaled = input_scaler.fit_transform(inputs)
        outputs_scaled = output_scaler.fit_transform(outputs)
    
        # Create the 'scales' directory if it doesn't exist
        os.makedirs('scales', exist_ok=True)
        
        # Save the scalers and transformers if applied
        joblib.dump(input_scaler, 'scales/input_scaler.save')
        joblib.dump(output_scaler, 'scales/output_scaler.save')
        if apply_pt:
            joblib.dump(pt, 'scales/power_transformer.save')
        if apply_qt:
            joblib.dump(qt, 'scales/quantile_transformer.save')
            
    else:
        inputs_scaled = inputs
        outputs_scaled = outputs
        input_scaler = []
        output_scaler = []
        
    return inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt if apply_pt else None, qt if apply_qt else None

# Inverse Sigmoid function
def sigmoid_inverse_transform(x):
    return -np.log((1 / x) - 1)

# Function to descale data with flexible options for transformers and sigmoid/log1p
def descale_data(scaled_data, scaler=None, 
                 apply_dsc=False,
                 apply_qt=False, qt=None, 
                 apply_pt=False, pt=None, 
                 apply_log1p=False, 
                 apply_sigmoid=False):
    
    descaled_data = scaled_data
    
    # Inverse transform using scaler (e.g., MinMaxScaler or StandardScaler) if requested
    if apply_dsc and scaler is not None:
        descaled_data = scaler.inverse_transform(scaled_data)
    
    # If QuantileTransformer was applied, inverse transform using the passed QuantileTransformer
    if apply_qt and qt is not None:
        print('qt descaled applied')
        descaled_data = qt.inverse_transform(descaled_data)
    
    # If PowerTransformer was applied, inverse transform using the passed PowerTransformer
    if apply_pt and pt is not None:
        descaled_data = pt.inverse_transform(descaled_data)

    # If log1p was applied, reverse the log1p transformation using expm1
    if apply_log1p:
        descaled_data = np.expm1(descaled_data)

    # If Sigmoid transformation was applied, reverse the Sigmoid transformation
    if apply_sigmoid:
        descaled_data = sigmoid_inverse_transform(descaled_data)

    return descaled_data
