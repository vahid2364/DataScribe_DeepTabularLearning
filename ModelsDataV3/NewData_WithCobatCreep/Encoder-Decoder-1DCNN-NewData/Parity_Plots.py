#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:33:33 2024

@author: attari.v
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error

# Create directories for saving plots
os.makedirs('QQplot', exist_ok=True)
os.makedirs('parityplots-scaled', exist_ok=True)
os.makedirs('parityplots-original', exist_ok=True)

# Function to plot QQ plots
def plot_qq(original, reconstructed, filename):
    """
    Plot QQ plot for original and reconstructed data and save the plot.
    
    Parameters:
    original : array-like
        Original data points.
    reconstructed : array-like
        Reconstructed data points.
    filename : str
        Filename to save the plot.
    """
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(1, 1, 1)

    # Plot the QQ plot for the original data
    stats.probplot(original.flatten(), plot=ax)
    ax.get_lines()[0].set_label('Original Data')

    # Plot the QQ plot for the reconstructed data
    stats.probplot(reconstructed.flatten(), plot=ax)
    ax.get_lines()[2].set_color('red')
    ax.get_lines()[2].set_label('Reconstructed Data')

    # Add the legend
    plt.legend()

    # Set the Y-axis limit
    plt.ylim([0, original.flatten().max() * 1.05])

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Function to plot scatter plots for scaled and original data
def plot_parity(y_true, y_pred, output_columns, filename_prefix, log_scale=False):
    """
    Plot scatter plots comparing actual vs predicted data.

    Parameters:
    y_true : array-like
        Actual outputs (ground truth).
    y_pred : array-like
        Predicted outputs.
    output_columns : list or array-like
        List of output column names for labeling.
    filename_prefix : str
        Directory prefix for saving the plots (e.g., 'scatterplots-scaled/' or 'scatterplots-original/').
    log_scale : bool, optional
        Whether to plot the data in log scale. Default is False.
    """
    for idx in range(y_true.shape[1]):
        plt.figure(figsize=(7, 7))



        if log_scale:
            #msle = mean_squared_log_error(y_true[:, idx], y_pred[:, idx])
            #log_r2 = r2_score(np.log(y_true[:, idx]), np.log(y_pred[:, idx]))
            #plt.text(0.05, 0.90, f'MSLE: {msle:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            #plt.text(0.05, 0.87, f'Log r$^2$: {log_r2:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            # Safeguard log against zero or negative values
            y_test_log_safe = np.where(y_true[:, idx] <= 0, 1e-20, y_true[:, idx])
            predictions_log_safe = np.where(y_pred[:, idx] <= 0, 1e-20, y_pred[:, idx])
            
            # Calculate MSLE and log R²
            mse = mean_squared_error(y_test_log_safe, predictions_log_safe)
            msle = mean_squared_log_error(y_test_log_safe, predictions_log_safe)
            log_r2 = r2_score(np.log(y_test_log_safe), np.log(predictions_log_safe))
            
            # Calculate GMAE, SMAPE, MASE, and RMSPE
            gmae = np.mean(np.abs(np.log(y_test_log_safe) - np.log(predictions_log_safe)))
            smape = np.mean(2 * np.abs(y_pred[:, idx] - y_true[:, idx]) / (np.abs(y_pred[:, idx]) + np.abs(y_true[:, idx]))) * 100
            mase = mse / np.mean(np.abs(np.diff(y_true[:, idx])))  # Simplified MASE
            rmspe = np.sqrt(np.mean(np.square((y_true[:, idx] - y_pred[:, idx]) / y_true[:, idx])))
            
            # Calculate RMSLE
            rmsle = np.sqrt(msle)
    
            # Annotate MSLE, RMSLE, and log R² on the plot
            plt.text(0.05, 0.95, f'MSLE: {msle:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
            plt.text(0.05, 0.91, f'RMSLE: {rmsle:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
            plt.text(0.05, 0.87, f'log R$^2$ : {log_r2:.3f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
            plt.text(0.05, 0.82, f'GMAE: {gmae:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
            plt.text(0.05, 0.78, f'SMAPE: {smape:.2f}%', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
            plt.text(0.05, 0.74, f'MASE: {mase:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
            plt.text(0.05, 0.70, f'RMSPE: {rmspe:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

        else:
            # Annotate MSE, MSLE, or other metrics on the plot
            mse = mean_squared_error(y_true[:, idx], y_pred[:, idx])
            r2 = r2_score(y_true[:, idx], y_pred[:, idx])
            
            plt.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.text(0.05, 0.92, f'r$^2$ : {r2 :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


        # Plot predictions vs actual outputs for each output feature
        plt.scatter(y_true[:, idx], y_pred[:, idx])
        plt.plot([np.min(y_true[:, idx]), np.max(y_true[:, idx])], [np.min(y_true[:, idx]), np.max(y_true[:, idx])], c='black')

        # Set axes to log scale if requested
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')

        plt.xlabel(f'Actual Outputs: {output_columns[idx]}')
        plt.ylabel('Predicted Outputs')
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'{filename_prefix}/scatterplot_{idx}.jpg')
        plt.show()

# # Example usage for QQ plots
# plot_qq(y_test, predictions_scaled, 'QQplot/qq_scaled_data.jpg')
# plot_qq(y_test_original, predictions, 'QQplot/qq_original_data.jpg')

# # Example usage for scatter plots (scaled data)
# plot_scatter(y_test, predictions_scaled, output_columns, 'scatterplots-scaled', log_scale=False)

# # Example usage for scatter plots (original data in log scale)
# plot_scatter(y_test_original, predictions, output_columns, 'scatterplots-original', log_scale=True)