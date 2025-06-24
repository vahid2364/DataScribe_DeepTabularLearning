#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:43:54 2024

@author: attari.v
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs('results_heatmap',exist_ok=True)

# %%

def get_top_correlated_pairs(corr_matrix, method_name, top_n=40, output_dir="results"):
    """
    Get the top N highly correlated pairs from the correlation matrix and save to an Excel file,
    including the correlation method name in the output file name.

    Parameters:
        corr_matrix (pd.DataFrame): The correlation matrix.
        method_name (str): The name of the correlation method used.
        top_n (int): Number of top correlated pairs to return. Default is 40.
        output_dir (str): Directory to save the output Excel file. Default is 'results'.

    Returns:
        pd.DataFrame: DataFrame of the top N correlated pairs.
    """
    # Get the upper triangle of the correlation matrix without the diagonal
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Stack the upper triangle matrix and sort by absolute correlation values
    sorted_corr_pairs_abs = upper_triangle.unstack().dropna().abs().sort_values(ascending=False)
    
    # Stack the original upper triangle matrix (to retain the original sign of the correlation values)
    sorted_corr_pairs = upper_triangle.unstack().dropna()
    
    # Get the top N highly correlated pairs based on the absolute value
    top_pairs = sorted_corr_pairs_abs.head(top_n).index
    
    # Create a DataFrame to show both the absolute values and the original signed correlations
    top_corr_df = pd.DataFrame({
        'Absolute Correlation': sorted_corr_pairs_abs.loc[top_pairs],
        'Signed Correlation': sorted_corr_pairs.loc[top_pairs]
    })
    
    # Print the top correlated pairs
    print(f"Top {top_n} highly correlated pairs using {method_name} correlation:")
    print(top_corr_df)
    
    # Save the top correlated pairs to an Excel file
    output_file = f"{output_dir}/top_{top_n}_correlated_pairs_{method_name}.xlsx"
    top_corr_df.to_excel(output_file, index=True)
    print(f"Top {top_n} highly correlated pairs have been saved to '{output_file}'.")
    
    return top_corr_df

# Example usage
# top_40_df = get_top_correlated_pairs(corr, top_n=40, output_file="results/top_40_correlated_pairs.xlsx")

# %%


def plot_with_sfe_markers(df, x_col, y_col, hue_col, sfe_col, custom_palette, output_filename):
    """
    Plots a scatterplot of x_col vs y_col with different markers based on SFE values.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the x-axis column.
        y_col (str): The name of the y-axis column.
        hue_col (str): The name of the column used for color coding.
        sfe_col (str): The name of the column containing SFE values.
        custom_palette (list): The color palette for the scatterplot.
        output_filename (str): The filename to save the plot.
    """
    
    # Check if the SFE column exists
    if sfe_col not in df.columns:
        raise KeyError(f"Column '{sfe_col}' not found in DataFrame")

    # Discretize SFE_calc into categories (e.g., low, medium, high)
    df['SFE_category'] = pd.cut(df[sfe_col], bins=[-np.inf, 50, np.inf], labels=['Low', 'High'])
    
    # Use different markers for each category
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=x_col, y=y_col, data=df, 
                    hue='Iteration', style='SFE_category',  # Use 'style' for marker type
                    s=140,
                    palette=custom_palette, alpha=0.7)


    # Set layout and save the plot
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()
    print(f"Plot saved to {output_filename}")

# %% Load and inspect Data

df = pd.read_csv('data/HTMDEC_MasterTable_Interpolated_Orange_Iterations_BBC_with_SFEcalc.csv')

# Define input and output columns
input_columns = df.columns[3:11]
output_columns = df.columns[ 11: ] # Remaining columns
output_columns = output_columns.drop(['SFE_calc'])

# Drop columns with all zeros
df = df.loc[:, ~(df == 0).all()]

print("\nDataFrame after dropping all-zero columns:")
print(df)

columns_to_keep = input_columns.tolist() + output_columns.tolist()

df = df[columns_to_keep]
df = df.dropna()
output_column_names = output_columns.tolist()

dataset = df

# %%

# # Select only those columns from the DataFrame
df_sliced = dataset

print(df_sliced)

# %% Data 

# # Select only those columns from the DataFrame
df_sliced2 = dataset
print(df_sliced2.columns)

# Normalize each column of df_sliced2 using Min-Max scaling
df_normalized = df_sliced2.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

# Display the normalized DataFrame
print(df_normalized.head())

# List of correlation methods
correlation_methods = ['pearson']#, 'kendall', 'spearman']

# Loop through each method and calculate/save the results
for method in correlation_methods:
    # Calculate the correlation matrix
    corr = df_sliced2.corr(method=method)
    
    #print(df_sliced2)
    #print(corr)
    
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(25, 23))
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, 
                mask=mask, 
                annot=True,              # Enable annotations
                cmap='coolwarm', 
                square=True, 
                linewidths=0.5, 
                annot_kws={"size": 14},  # Set font size for annotations
                cbar_kws={"shrink": 0.75})
    
    # Rotate the x and y axis labels by 45 degrees
    plt.xticks(rotation=90, ha='right', fontsize=14)  # ha='right' aligns the labels better
    plt.yticks(rotation=0, ha='right', fontsize=14)
    
    # Set a title for the heatmap
    plt.title(f"{method.capitalize()} Correlation Heatmap", fontsize=16, pad=20)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'results_heatmap/{method}_correlation_heatmap.jpg', dpi=300)
    plt.show()
    
    # Use the updated function to save top correlated pairs
    top_40_df = get_top_correlated_pairs(corr, method_name=method, top_n=40, output_dir="results_heatmap")

    print(f"{method.capitalize()} correlation heatmap saved successfully.")


    
    
