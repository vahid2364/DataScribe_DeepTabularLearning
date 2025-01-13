#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 01:48:28 2024

@author: attari.v
"""

import matplotlib.pyplot as plt
#import numpy as np
#from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
#from scipy.stats import probplot

# Ensure necessary directories exist
#os.makedirs('results/QQplot', exist_ok=True)
#os.makedirs('results/scatterplots-scaled', exist_ok=True)
#os.makedirs('results/scatterplots-original', exist_ok=True)

# Import functions from the script
from modules.Parity_Plots import plot_qq, plot_qq_all, plot_scatter  # Adjust the import path as needed

# Function to plot training and validation loss
def plot_loss(history, save_path='results/loss_plot.jpg'):
    """
    Plot training and validation loss over epochs.
    
    Parameters:
    history : History object
        Training history from Keras model fitting.
    save_path : str
        Path to save the loss plot.
    """
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    
    # Increase font size for axis numbers
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    
    plt.legend(fontsize=15)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Wrapper function to generate QQ plots
def generate_qq_plots(y_test, predictions_scaled, y_test_original, predictions):
    """
    Generate QQ plots for scaled and original data.

    Parameters:
    y_test : array-like
        Actual outputs (scaled).
    predictions_scaled : array-like
        Predicted outputs (scaled).
    y_test_original : array-like
        Actual outputs (original scale).
    predictions : array-like
        Predicted outputs (original scale).
    """
    plot_qq(y_test, predictions_scaled, 'results/QQplot/qq_scaled_data.jpg')
    plot_qq(y_test_original, predictions, 'results/QQplot/qq_original_data.jpg')
    plot_qq_all(y_test_original, predictions, 'results/QQplot/qq_original_data_all.jpg')

# Wrapper function to generate parity plots
def generate_parity_plots(
    y_test, predictions_scaled, y_test_original, predictions, output_columns,
    y_train=None, predictions_scaled_train=None, y_train_original=None, predictions_train=None
):
    """
    Generate scatter plots for scaled and original data, with optional train data plots.

    Parameters:
    y_test : array-like
        Actual outputs (scaled) for test data.
    predictions_scaled : array-like
        Predicted outputs (scaled) for test data.
    y_test_original : array-like
        Actual outputs (original scale) for test data.
    predictions : array-like
        Predicted outputs (original scale) for test data.
    output_columns : list or array-like
        List of output column names for labeling.
    y_train : array-like, optional
        Actual outputs (scaled) for train data. Default is None.
    predictions_scaled_train : array-like, optional
        Predicted outputs (scaled) for train data. Default is None.
    y_train_original : array-like, optional
        Actual outputs (original scale) for train data. Default is None.
    predictions_train : array-like, optional
        Predicted outputs (original scale) for train data. Default is None.
    """
    # Scatter plots for scaled data
    plot_scatter(
        y_test, predictions_scaled, output_columns, 
        'results/parityplots-scaled', log_scale=False, 
        y_train=y_train, predictions_train=predictions_scaled_train
    )
    
    # Scatter plots for original data
    plot_scatter(
        y_test_original, predictions, output_columns, 
        'results/parityplots-original', log_scale=False, 
        y_train=y_train_original, predictions_train=predictions_train
    )