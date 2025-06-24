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
#os.makedirs('QQplot', exist_ok=True)
#os.makedirs('scatterplots-scaled', exist_ok=True)
#os.makedirs('scatterplots-original', exist_ok=True)

    
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
    print(filename)
    plt.savefig(filename, dpi=300)
    plt.show()


# Function to plot QQ plot for original and reconstructed data
def plot_qq_all(original, reconstructed, filename):
    plt.figure(figsize=(7, 6))
    ax = plt.subplot(1, 1, 1)

    # Plot the QQ plot for the original data
    stats.probplot(original.flatten(), plot=ax)
    # Get the first line (corresponding to original data) and set its label
    ax.get_lines()[0].set_label('Original Data')

    # Plot the QQ plot for the reconstructed data
    stats.probplot(reconstructed.flatten(), plot=ax)
    # Get the second line (corresponding to reconstructed data) and set its label

    # Modify the markers for reconstructed data (unfilled red circles)
    ax.get_lines()[2].set_color('red')
    ax.get_lines()[2].set_marker('o')   # Use 'o' for circles
    ax.get_lines()[2].set_markerfacecolor('None')  # Make the face blank
    ax.get_lines()[2].set_markeredgecolor('red')  # Red outline
    ax.get_lines()[2].set_label('Reconstructed Data (Log Scale)')

    # Add label for the theoretical line (ideal fit for a normal distribution)
    ax.get_lines()[1].set_color('green')  # Change color to differentiate if desired
    ax.get_lines()[1].set_label('Ideal Fit (Normal Distribution)')

    # Add the legend
    # Add the legend
    plt.legend(loc='upper left', fontsize=13)

    # Set the Y-axis limit
    plt.ylim([0, original.flatten().max() * 1.05])

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_scatter(y_true, y_pred, output_columns, filename_prefix, log_scale=False, y_train=None, predictions_train=None):
    """
    Plot scatter plots comparing actual vs predicted data, with optional train data plots.

    Parameters:
    y_true : array-like
        Actual outputs (ground truth) for test data.
    y_pred : array-like
        Predicted outputs for test data.
    output_columns : list or array-like
        List of output column names for labeling.
    filename_prefix : str
        Directory prefix for saving the plots (e.g., 'parityplots-scaled' or 'parityplots-original').
    log_scale : bool, optional
        Whether to plot the data in log scale. Default is False.
    y_train : array-like, optional
        Actual outputs (ground truth) for train data. Default is None.
    predictions_train : array-like, optional
        Predicted outputs for train data. Default is None.
    """
    for idx in range(y_true.shape[1]):
        plt.figure(figsize=(7, 7))

        # Annotate MSE and R² for test data
        mse = mean_squared_error(y_true[:, idx], y_pred[:, idx])
        r2 = r2_score(y_true[:, idx], y_pred[:, idx])
        plt.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.92, f'r$^2$ : {r2 :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Plot predictions vs actual outputs for test data
        plt.scatter(y_true[:, idx], y_pred[:, idx], label="Test Data", color="blue")

        # Plot the diagonal (ideal line)
        plt.plot([np.min(y_true[:, idx]), np.max(y_true[:, idx])], 
                 [np.min(y_true[:, idx]), np.max(y_true[:, idx])], 
                 c='black', linestyle='-.', label="Ideal test-set")


        # Plot train data if available
        if y_train is not None and predictions_train is not None:
            plt.scatter(y_train[:, idx], predictions_train[:, idx], label="Train Data", color="orange", alpha=0.7)
            # Plot the diagonal (ideal line)
            plt.plot([np.min(y_train[:, idx]), np.max(y_train[:, idx])], 
                     [np.min(y_train[:, idx]), np.max(y_train[:, idx])], 
                     c='black', linestyle='--', label="Ideal train-set")
            

        # Set axes to log scale if requested
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            
        # Increase font size for axis numbers
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=12)

        # Add labels and legend
        plt.xlabel(f'Actual Outputs: {output_columns[idx]}', fontsize=14)
        plt.ylabel('Predicted Outputs', fontsize=14)
        plt.legend(loc='lower right')
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