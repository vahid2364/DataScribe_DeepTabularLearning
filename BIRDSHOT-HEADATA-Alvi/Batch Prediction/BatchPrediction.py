#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:20:02 2025

@author: attari.v
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

import re
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error,mean_squared_error

from scaling_utils2 import scale_data, descale_data
import time

from concurrent.futures import ProcessPoolExecutor

def load_model_and_scalers(weight_folder, weight_file, scale_folder):
    """Load the trained encoder-decoder model and scalers."""
    autoencoder_model_path = os.path.join(weight_folder, weight_file)
    input_scaler_path = os.path.join(scale_folder, 'input_scaler.save')
    output_scaler_path = os.path.join(scale_folder, 'output_scaler.save')
    
    model = tf.keras.models.load_model(autoencoder_model_path)
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)
    
    return model, input_scaler, output_scaler


def make_predictions(model, input_scaler, output_scaler, conditional_parameters):
    """Make predictions using the encoder-decoder model and inverse-transform the outputs."""
    scaled_input = input_scaler.transform(conditional_parameters)
    predictions_scaled = model.predict(scaled_input,verbose=0)
    predictions_descaled = descale_data(
        predictions_scaled, 
        input_scaler=input_scaler, output_scaler=output_scaler,
        apply_dsc=True, apply_qt=False, apply_pt=False, apply_log1p=False, apply_sigmoid=False, 
        data_type='output'
    )
    return predictions_scaled, predictions_descaled



# def process_target(csv_file_path, input_columns, output_columns, weight_folder, weight_file, scale_folder):
#     """Process a single target using the specified parameters."""
    
#     # Load model and scalers
#     model, input_scaler, output_scaler = load_model_and_scalers(weight_folder, weight_file, scale_folder)
    
#     # Prepare input data
#     conditional_parameters = np.array(df[input_columns][:])
    
#     # Make predictions
#     _, predictions_descaled = make_predictions(model, input_scaler, output_scaler, conditional_parameters)
    
#     num_rows = 75
#     # Ensure indices of predictions match the original dataset
#     predictions_descaled_df = pd.DataFrame(
#         predictions_descaled[:num_rows], 
#         columns=[f'Predicted {col}' for col in output_columns],
#         index=df[output_columns].index[:num_rows]  # Align indices
#     )
    
#     # Combine original and predicted DataFrames side by side
#     actual = df[output_columns][:num_rows]

#     mae_values = [mean_absolute_error(actual.iloc[i], predictions_descaled_df.iloc[i]) for i in range(num_rows)]
            
#     return actual, predictions_descaled, mae_values

def predict_all_targets(input_values):
    INPUT_COLUMNS = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V',
                     'Recrystallization Temp(℃)', 'Holding time (h)', 'Cold Work (%Reduction)']
    
    model_paths = [
        #'../Hyperparameter-optimization2/R2_Optimization/Encoder-Decoder-FDN-Optuna-overcomplete-YS/results/',
        #'../Hyperparameter-optimization2/R2_optimization/Encoder-Decoder-FDN-Optuna-overcomplete-UTS/results',
        '../Hyperparameter-optimization2/SMAPE_optimization/Encoder-Decoder-FDN-Optuna-overcomplete-YS-TrainSet/results/',
        '../Hyperparameter-optimization2/SMAPE_optimization/Encoder-Decoder-FDN-Optuna-overcomplete-UTS-TrainSet/results',
        '../Hyperparameter-optimization2/SMAPE_optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Elongation-TrainSet/results',
        '../Hyperparameter-optimization2/SMAPE_optimization/Encoder-Decoder-FDN-Optuna-overcomplete-Avg-TrainSet/results',
        '../Hyperparameter-optimization2/SMAPE_optimization/Encoder-Decoder-FDN-Optuna-overcomplete-DoP-TrainSet/results'
    ]

    TARGETS = [
        ['Yield Strength (MPa)'],
        ['UTS_True (Mpa)'],
        ['Elong_T (%)'],
        ['Avg HDYN/HQS'],
        ['Depth of Penetration (mm) FE_Sim']
    ]
    
    run_idx = range(1, 11)
    predictions = {}

    # Convert input to 2D numpy array
    input_array = np.array(input_values).reshape(1, -1)
    
    for target, model_dir in zip(TARGETS, model_paths):
        weight_folder = os.path.join(model_dir, 'weights')
        scale_folder = os.path.join(model_dir, '../scales')
        
        # Load scalers
        #x_scaler, y_scaler = load_scalers(scale_folder, INPUT_COLUMNS, target[0])
        #x_scaled = x_scaler.transform(input_array)

        all_preds = []
        for idx in run_idx:
            weight_file = 'bestmodel_weights_run'+str(idx)+'.keras'
            
            # Load and predict
            #model = load_model(weight_file, input_dim=x_scaled.shape[1], output_dim=1)
            model, input_scaler, output_scaler = load_model_and_scalers(weight_folder, weight_file, scale_folder)
            y_pred_scaled, y_pred = make_predictions(model, input_scaler, output_scaler, input_array)
            all_preds.append(y_pred.flatten()[0])

        # Average prediction
        predictions[target[0]] = np.mean(all_preds)
        
    # Calculate and append the UTS/YS ratio
    uts = predictions['UTS_True (Mpa)']
    ys = predictions['Yield Strength (MPa)']
    predictions['UTS/YS Ratio'] = uts / ys
        
    # Remove the original UTS entry
    #del result['UTS_True (Mpa)']
        
    return predictions, all_preds
    
# Wrapper function to process a row
def process_row(row):
    input_list = row.tolist()
    result, _ = predict_all_targets(input_list)
    combined = dict(row)
    combined.update(result)
    return combined
    
## Example CALL
if __name__ == "__main__":
    
    start_time = time.time()
    
# %% LOAD DATA

    split_num = [0,1,2,3,4]
    
    for idx in split_num:
    
        res_folder = "results"+str(idx)
    
        fl_name = 'split_'+str(idx)+'_test.csv'
        test_data = pd.read_csv('../data-Alvi/alloy_splits/'+str(fl_name))
        fl_name = 'split_0_train.csv'
        train_data = pd.read_csv('../data-Alvi/alloy_splits/'+str(fl_name))
        
        # df
        test_data  = test_data[test_data["Iteration"].isin(["AAA","AAB","AAC","AAD","AAE","BBA", "BBB", "BBC", "CBA"])]
        train_data = train_data[train_data["Iteration"].isin(["AAA","AAB","AAC","AAD","AAE","BBA", "BBB", "BBC", "CBA"])]
        
        # Define input and output columns
        input_columns = [
            'Al','Co','Cr','Cu','Fe','Mn','Ni','V'
        ]
        
        
        output_columns = [
            #'Yield Strength (MPa)',
            #'UTS_True (Mpa)',
            'Elong_T (%)',                          
            #'Hardness (GPa) SRJT', 
            #'Modulus (GPa) SRJT'
        ]
                
        # Drop columns with all zeros
        test_data = test_data.loc[:, ~(test_data == 0).all()]
        train_data = train_data.loc[:, ~(train_data == 0).all()]
        
        #columns_to_keep = input_columns.tolist() + output_columns.tolist()
        columns_to_keep = input_columns + output_columns
        
        #test_data = test_data[columns_to_keep]
        #train_data = train_data[columns_to_keep]
        
        from sklearn.preprocessing import MinMaxScaler
        
        # Fit scaler only on training data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Drop rows where target column is NaN
        train_data = train_data.dropna(subset=output_columns)
        test_data = test_data.dropna(subset=output_columns)
        
        train_data[input_columns] = scaler_X.fit_transform(train_data[input_columns])
        test_data[input_columns] = scaler_X.transform(test_data[input_columns])
        
        train_data[output_columns] = scaler_y.fit_transform(train_data[output_columns])
        test_data[output_columns] = scaler_y.transform(test_data[output_columns])
        
        # Then split into inputs and outputs
        X_train = train_data[input_columns].to_numpy()
        y_train = train_data[output_columns].to_numpy()
        X_test = test_data[input_columns].to_numpy()
        y_test = test_data[output_columns].to_numpy()
                
        print("\nDataFrame after dropping all-zero columns:")
        print(X_train)
        print("\nInput Columns:")
        print(input_columns)
        print("\nOutput Columns:")
        print(output_columns)


        pause


    #input_sample = [4, 8, 4, 4, 16, 12, 48, 4, 950, 0.5, 60] 
    #input_sample = [4,16, 0, 4, 12,  8, 52, 4, 950, 0.5, 60] 
    
    # %% Define the feasible subset for benchmark calculations    

    ### Test for 200 points --- Update the number
    df_subset = df_feasile.iloc[0:200,:]

    # %% Parallel run

    combined_results = []
    # Using map to preserve order
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_row, [row for _, row in df_subset.iterrows()]))

    # Create the final DataFrame with inputs + predictions
    df_combined = pd.DataFrame(results)        
    df_combined.to_excel('combined_predictions_feasibleSpace_all.xlsx', sheet_name='Predictions', index=False)
    
    ## 96    4   8.0   4   4  16.0  12  48.0   4                      950.0               0.5                    60.0            284.200000
    ## 97    4  16.0   0   4  12.0   8  52.0   4                      950.0               0.5                    60.0            301.000000
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    
    