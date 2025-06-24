#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 21:55:39 2025

@author: attari.v
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

# -----------------------------
# Metrics Functions
# -----------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def spearman_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

# -----------------------------
# Process a Single Excel File
# -----------------------------

def process_excel_file(filepath):
    df = pd.read_excel(filepath)
    # Automatically detect actual and predicted columns
    actual_cols = [col for col in df.columns if col.startswith('Actual_')]
    predicted_cols = [col for col in df.columns if col.startswith('Predicted_')]
    
    if len(actual_cols) != 1 or len(predicted_cols) != 1:
        raise ValueError(f"Expected 1 actual and 1 predicted column, found {actual_cols} and {predicted_cols}")
    
    y_true = df[actual_cols[0]]
    y_pred = df[predicted_cols[0]]
    
    #print(rmse(y_true, y_pred))
    #pause
    
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'R2': r2(y_true, y_pred),
        'Spearman': spearman_corr(y_true, y_pred),
        'SMAPE (%)': smape(y_true, y_pred)
    }

# -----------------------------
# Process All Files in Results
# -----------------------------

def process_all_training_folders(base_path):
    all_results = []
    
    folders = ['Training-DoP2', 'Training-Avg2', 'Training-Elongation2', 
               'Training-Hardness2', 'Training-Modulus2', 
               'Training-UTS2', 'Training-YS2']

    for training_folder in folders:
        training_path = os.path.join(base_path, training_folder)
        if not os.path.isdir(training_path):
            continue

        for results_folder in os.listdir(training_path):
            results_path = os.path.join(training_path, results_folder)
            if not os.path.isdir(results_path) or not results_folder.startswith("results"):
                continue

            excel_files = glob.glob(os.path.join(results_path, 'model_predictions*.xlsx'))
            for excel_file in excel_files:
                # Process both sheets
                xls = pd.ExcelFile(excel_file)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)

                    # Find actual and predicted columns
                    actual_cols = [col for col in df.columns if col.startswith('Actual_')]
                    predicted_cols = [col for col in df.columns if col.startswith('Predicted_')]

                    if len(actual_cols) != 1 or len(predicted_cols) != 1:
                        print(f"Skipping {excel_file} [{sheet_name}]: found {actual_cols} and {predicted_cols}")
                        continue

                    y_true = df[actual_cols[0]]
                    y_pred = df[predicted_cols[0]]

                    metrics = {
                        'TrainingFolder': training_folder,
                        'ResultsFolder': results_folder,
                        'File': os.path.basename(excel_file),
                        'Sheet': sheet_name,
                        'RMSE': rmse(y_true, y_pred),
                        'MAE': mae(y_true, y_pred),
                        'R2': r2(y_true, y_pred),
                        'Spearman': spearman_corr(y_true, y_pred),
                        'SMAPE (%)': smape(y_true, y_pred)
                    }
                    all_results.append(metrics)

    df_metric = pd.DataFrame(all_results)
    
    # Save different sheets for each TrainingFolder + Sheet type
    with pd.ExcelWriter("all_training_metrics.xlsx") as writer:
        for (folder, sheet), group in df_metric.groupby(['TrainingFolder', 'Sheet']):
            sheet_name = f"{folder}_{sheet}"[:31]  # Excel sheet names max length = 31
            group.to_excel(writer, sheet_name=sheet_name, index=False)

    return df_metric

# -----------------------------
# Execute
# -----------------------------

if __name__ == "__main__":
    base_dir = "."  # 👈 Change this
    results_df = process_all_training_folders(base_dir)
    print(results_df)
    # Optionally save it
    #results_df.to_excel("all_model_metrics.xlsx", index=False)