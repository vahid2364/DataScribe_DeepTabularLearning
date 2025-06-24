#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 01:32:06 2024

@author: attari.v
"""

import os
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    mean_squared_log_error,
)
from modules.preprocessing import descale_data

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and returns the loss.
    """
    loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss}")
    return loss

def make_predictions(model, X_test, X_train, input_scaler, output_scaler, 
                     apply_dsc=False, data_type='output'):
    """
    Makes predictions and descaling on test and train data using descale_data.
    """
    # Scaled predictions
    predictions_scaled = model.predict(X_test)
    predictions_scaled_train = model.predict(X_train)

    # Descale predictions using descale_data
    predictions = descale_data(predictions_scaled, 
                               input_scaler=input_scaler, 
                               output_scaler=output_scaler, 
                               apply_dsc=apply_dsc, 
                               data_type=data_type)
    
    predictions_train = descale_data(predictions_scaled_train, 
                                     input_scaler=input_scaler, 
                                     output_scaler=output_scaler, 
                                     apply_dsc=apply_dsc, 
                                     data_type=data_type)

    return predictions, predictions_train, predictions_scaled, predictions_scaled_train

def calculate_metrics(predictions, y_test_original, metric_fn=mean_squared_error):
    """
    Calculates evaluation metrics for predictions.
    """
    mse = metric_fn(y_test_original, predictions)
    print(f"Mean Squared Error (MSE): {mse}")

    mse_per_feature = metric_fn(y_test_original, predictions, multioutput="raw_values")
    r2_per_feature = r2_score(y_test_original, predictions, multioutput="raw_values")

    for i, mse_feature in enumerate(mse_per_feature):
        print(f"Mean Squared Error for feature {i}: {mse_feature}")
    for i, r2_feature in enumerate(r2_per_feature):
        print(f"R^2 for feature {i}: {r2_feature}")

    return mse, mse_per_feature, r2_per_feature


def save_metrics(predictions, y_test_original, file_path):
    """
    Saves evaluation metrics to a file.
    """
    metrics = {
        "Test MSE": mean_squared_error(y_test_original, predictions),
        "Test MAE": mean_absolute_error(y_test_original, predictions),
        "Test R²": r2_score(y_test_original, predictions),
        "Test Explained Variance": explained_variance_score(y_test_original, predictions),
    }

    with open(file_path, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

    print(f"Metrics saved to {file_path}")