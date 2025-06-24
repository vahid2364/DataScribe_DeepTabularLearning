#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:35:56 2024

@author: attari.v
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

# Function to check if log scale is appropriate
def should_use_log_scale(series: pd.Series, threshold: int = 3) -> bool:
    """
    Determines whether a pandas Series should be plotted on a log scale.
    
    Parameters:
    - series: pd.Series, the data series to evaluate.
    - threshold: int, the threshold for the range of orders of magnitude.
    
    Returns:
    - bool: True if log scale should be used, False otherwise.
    """
    # Ensure all values are positive
    positive_values = series[series > 0]  # Only consider positive values for log scale
    
    # Return False if no positive values exist
    if positive_values.empty:
        return False
    
    # Calculate the range of values (order of magnitude)
    min_value = positive_values.min()
    max_value = positive_values.max()
    
    # Check if the ratio of max to min spans several orders of magnitude
    order_of_magnitude = np.log10(max_value / min_value)
    
    # Use log scale if the data spans more than the threshold orders of magnitude
    return order_of_magnitude >= threshold

# Load the CSV file
csv_file_path = 'processed_data/IQR_dataframe-NbCrVWZr_data_stoic_creep_equil_v3.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

print(df.describe())

# Convert the DataFrame to a LaTeX table
latex_table = df.iloc[:, 32:37].describe().applymap(lambda x: f'{x:.2g}').to_latex(index=False)

# Print or save the LaTeX table
print(latex_table)

# %%

# Plotting section
plt.rcParams.update({'font.size': 22})
matplotlib.rcParams.update({'font.size': 22})

folder = 'images_log_IQRData'
os.makedirs(folder, exist_ok=True)

i = 0
#plt.figure(figsize=(8,6))

for idx in df.columns:
    print(i)
    
    plt.figure(figsize=(8,6))
    
    # Automatically determine if log scale is needed
    use_log_scale = should_use_log_scale(df[idx])
    
    # Remove non-positive values if using log scale and handle extremely small values
    if use_log_scale:
        filtered_data = df[idx][df[idx] > 0]  # Log scale requires positive values only
        sns.kdeplot(filtered_data, label=idx, fill=True, log_scale=(True, False))  # log scale on x-axis
    else:
        sns.kdeplot(df[idx], label=idx, fill=True, log_scale=False)
    
    plt.ylabel('Density', fontsize=32)
    plt.xlabel(idx, fontsize=32)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(False)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(folder + f'/kde_{i}.jpg')
    plt.show()
    
    i += 1
    
    
# %% For Paper - data exploration

os.makedirs('FeatureExploration',exist_ok=True)

plt.figure(figsize=(10,8))
sns.kdeplot(df['EQ 1273K Density (g/cc)'], label='EQ 1273K Density (g/cc)', fill=True, log_scale=(False, False))  # log scale on x-axis
sns.kdeplot(df['EQ 1523K Density (g/cc)'], label='EQ 1523K Density (g/cc)', fill=True, log_scale=(False, False))  # log scale on x-axis
plt.xlabel('Data')
plt.legend()
plt.tight_layout()
plt.savefig('FeatureExploration/Density.jpg',dpi=300)

plt.figure(figsize=(10,8))
sns.kdeplot(df['PROP 500C CTE (1/K)'], label='PROP 500C CTE (1/K)', fill=True, log_scale=(True, False))  # log scale on x-axis
sns.kdeplot(df['PROP 1000C CTE (1/K)'], label='PROP 1000C CTE (1/K)', fill=True, log_scale=(True, False))  # log scale on x-axis
sns.kdeplot(df['PROP 1500C CTE (1/K)'], label='PROP 1500C CTE (1/K)', fill=True, log_scale=(True, False))  # log scale on x-axis
plt.xlabel('Data')
plt.legend()
plt.tight_layout()
plt.savefig('FeatureExploration/CTE.jpg',dpi=300)

plt.figure(figsize=(10,8))
df_clean = df['EQ 1523K THCD (W/mK)'][df['EQ 1523K THCD (W/mK)'] > 0]
sns.kdeplot(df['EQ 1273K THCD (W/mK)'].replace([np.inf, -np.inf], np.nan).dropna(), label='EQ 1273K THCD (W/mK)', fill=True, log_scale=(True, False))  # log scale on x-axis
sns.kdeplot(df_clean, label='EQ 1523K THCD (W/mK)', fill=True, log_scale=(True, False))  # log scale on x-axis
plt.xlabel('Data')
plt.legend(loc='upper left', fontsize=20)  # Example location 'upper right'
plt.tight_layout()
plt.savefig('FeatureExploration/THCD.jpg',dpi=300)

plt.figure(figsize=(10,8))
sns.kdeplot(df['YS 1000C PRIOR'], label='YS 1000C PRIOR', fill=True, log_scale=(False, False))  # log scale on x-axis
sns.kdeplot(df['YS 1500C PRIOR'], label='YS 1500C PRIOR', fill=True, log_scale=(False, False))  # log scale on x-axis
plt.xlabel('Data')
plt.legend()
plt.tight_layout()
plt.savefig('FeatureExploration/YS.jpg',dpi=300)


plt.figure(figsize=(10,8))
sns.kdeplot(df['500 Min Creep CB [1/s]'], label='500 Min Creep CB [1/s]', fill=True, log_scale=(True, False))  # log scale on x-axis
sns.kdeplot(df['1000 Min Creep CB [1/s]'], label='1000 Min Creep CB [1/s]', fill=True, log_scale=(True, False))  # log scale on x-axis
sns.kdeplot(df['1300 Min Creep CB [1/s]'], label='1300 Min Creep CB [1/s]', fill=True, log_scale=(True, False))  # log scale on x-axis
sns.kdeplot(df['1500 Min Creep CB [1/s]'], label='1500 Min Creep CB [1/s]', fill=True, log_scale=(True, False))  # log scale on x-axis
sns.kdeplot(df['2000 Min Creep CB [1/s]'], label='2000 Min Creep CB [1/s]', fill=False, log_scale=(True, False))  # log scale on x-axis
plt.xlabel('Data')
plt.legend(loc='upper left', fontsize=20)  # Example location 'upper right'
plt.tight_layout()
plt.savefig('FeatureExploration/Creep_CB.jpg',dpi=300)

plt.figure(figsize=(10,8))
#sns.kdeplot(df['Pugh_Ratio_PRIOR'], label='Pugh Ratio', fill=True, log_scale=(False, False))  # log scale on x-axis
#sns.kdeplot(df['SCHEIL LT'], label='SCHEIL LT', fill=True, log_scale=(False, False))  # log scale on x-axis
sns.kdeplot(df['Creep Merit'], label='Creep Merit', fill=True, log_scale=(True, False))  # log scale on x-axis
plt.xlabel('Data')
plt.legend(loc='upper left', fontsize=20)  # Example location 'upper right'
plt.tight_layout()
plt.savefig('FeatureExploration/Criteria_CB.jpg',dpi=300)


    